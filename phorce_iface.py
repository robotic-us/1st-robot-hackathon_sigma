#!/usr/bin/env python3
"""The only file in this project that knows phorce or ROS 2 exist.

Two implementations behind one interface:

* :class:`MockRobot`  -- synthetic 4-axis feedback, logs ``play()`` calls.
  Runs on a laptop with no ROS, no robot, no phorce package installed.
* :class:`PhorceRobot` -- the real thing, via the ``phorce`` facade.  Every
  phorce import happens lazily inside :meth:`start`, so merely importing this
  module never fails on a machine without the SDK.

Pick one with :func:`make_robot`.

Things the hackathon docs are emphatic about, encoded here so we cannot forget:

* ``/phorce/feedback`` is 1 kHz and **must** be subscribed with
  ``qos_profile_sensor_data``.  With the default reliable QoS you get zero
  messages and *no error* -- a silent failure that costs hours.
* ``result.ok`` is an **attribute**, not a method.
* Only reject code 5 (BUSY / QUEUE_FULL) is worth retrying.  Codes 12
  (NOT_READY_FOR_MOTION) and 13 (RECOVERY_REQUIRED) need a *human* to press a
  physical button -- looping on them accomplishes nothing.
* An axis value may be trusted only when that axis's ``valid`` flag is true.
  ``not stale`` is explicitly **not** a substitute: an axis that has never
  reported is not stale either.
* The robot plays one motion at a time and has no queue.

Smoke-test the mock::

    python3 phorce_iface.py --seconds 4
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Protocol, Sequence

LOGGER = logging.getLogger("phorce_iface")

# Motion-slot contract, straight from the manual.
MIN_MOTION_ID = 1
MAX_MOTION_ID = 50
NO_MOTION_ID = 0   # sentinel -- never play this

# The robot has 12 axes; this project uses the first four.
NUM_AXES = 12
DEFAULT_AXES: tuple[int, ...] = (0, 1, 2, 3)


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AxisState:
    """One axis of a feedback frame (the subset of fields we actually use)."""

    index: int
    position_rad: float
    velocity_rad_s: float
    current_a: float
    dob_a: float          # disturbance observer: estimated external force
    temp_c: float
    valid: bool           # the ONLY positive evidence this axis may be trusted


@dataclass(frozen=True)
class JointState:
    """A snapshot of the robot's body sense."""

    axes: tuple[AxisState, ...]
    ts: float

    def by_index(self, index: int) -> Optional[AxisState]:
        for axis in self.axes:
            if axis.index == index:
                return axis
        return None

    def positions(self, indices: Sequence[int]) -> Optional[list[float]]:
        """Positions for ``indices``, or ``None`` if any of them is untrustworthy.

        Returning ``None`` rather than a partly-garbage vector is deliberate: the
        decider scores slots by how close their start pose is to where we are,
        and a wrong pose is worse than no pose (which it handles explicitly).
        """
        out: list[float] = []
        for index in indices:
            axis = self.by_index(index)
            if axis is None or not axis.valid:
                return None
            out.append(axis.position_rad)
        return out

    def velocities(self, indices: Sequence[int]) -> Optional[list[float]]:
        """Velocities for ``indices``, or ``None`` if any is untrustworthy."""
        return self._gather(indices, "velocity_rad_s")

    def external_force(self, indices: Sequence[int]) -> Optional[list[float]]:
        """Disturbance-observer estimate per axis -- "what is pushing on me".

        DREAM-Chunk's collision-resistance term is built on this: it is the only
        contact signal the participant API exposes, and it needs no extra sensor.
        """
        return self._gather(indices, "dob_a")

    def _gather(self, indices: Sequence[int], attribute: str) -> Optional[list[float]]:
        out: list[float] = []
        for index in indices:
            axis = self.by_index(index)
            if axis is None or not axis.valid:
                return None
            out.append(float(getattr(axis, attribute)))
        return out

    def age_s(self, now: Optional[float] = None) -> float:
        return (time.monotonic() if now is None else now) - self.ts


class PlayOutcome(Enum):
    """What came of a ``play()`` request.

    ``OK`` from :meth:`RobotInterface.play` means *accepted and started* -- see
    that method's docstring for why the call is asynchronous.  ``OK`` delivered
    to the completion callback means the motion actually finished.
    """

    OK = "ok"
    BUSY = "busy"                        # reject code 5 -- retry next tick
    NEEDS_OPERATOR = "needs_operator"    # codes 12/13 -- a human must intervene
    ERROR = "error"


PlayCallback = Callable[[int, PlayOutcome], None]


class MockMotionModel(Protocol):
    """How the mock robot should move while a slot is playing.

    Supplied from outside so this module stays ignorant of the motion format:
    ``phorce_iface`` is the hardware boundary, and teaching it about P-Vectors
    would put the world model on the wrong side of that line.  ``main.py``
    hands in an adapter over the chunk dictionary (see dream.ChunkMockModel).

    Without one the mock free-runs on a synthetic drift, which is fine for
    smoke-testing plumbing but useless for testing DREAM-Chunk -- the dream
    would diverge constantly because nothing is following it.
    """

    def duration_s(self, slot_id: int) -> float: ...

    def pose_at(
        self, slot_id: int, start_rad: Sequence[float], elapsed_s: float
    ) -> Optional[list[float]]: ...


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #
class RobotInterface(ABC):
    """Feedback in, motion-slot requests out.  Nothing else is available."""

    def __init__(self, on_play_result: Optional[PlayCallback] = None) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[JointState] = None
        self._motion_active = False
        self._play_thread: Optional[threading.Thread] = None
        self._last_outcome: Optional[PlayOutcome] = None
        self._on_play_result = on_play_result
        self._closing = threading.Event()

    # -- lifecycle --------------------------------------------------------- #
    @abstractmethod
    def start(self) -> None:
        """Connect and begin receiving feedback."""

    def close(self) -> None:
        """Stop everything.  Safe to call twice."""
        self._closing.set()
        thread = self._play_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def __enter__(self) -> "RobotInterface":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- feedback (the fast path) ------------------------------------------ #
    def latest(self) -> Optional[JointState]:
        """Most recent feedback frame, or ``None`` if nothing has arrived yet."""
        with self._lock:
            return self._latest

    def _store(self, state: JointState) -> None:
        """Called from the feedback thread at up to 1 kHz -- store only, never decide."""
        with self._lock:
            self._latest = state

    # -- motion (the slow path) -------------------------------------------- #
    def is_motion_active(self) -> bool:
        with self._lock:
            return self._motion_active

    def last_outcome(self) -> Optional[PlayOutcome]:
        with self._lock:
            return self._last_outcome

    def play(self, slot_id: int) -> PlayOutcome:
        """Request a motion slot.  **Returns immediately.**

        The real ``robot.play()`` blocks until the motion completes -- up to 30
        seconds.  Calling that inline would stall the decision loop for the whole
        motion, so the request is handed to a worker thread and this returns
        straight away:

        * ``BUSY``  -- a motion is already running; the robot has no queue, so
          there is nothing to do but try again on the next tick.
        * ``OK``    -- accepted and started.  The terminal result arrives via the
          ``on_play_result`` callback and :meth:`last_outcome`.
        * ``ERROR`` -- the slot id is outside the contract (1..50).

        This does not weaken the two-rate rule: ``play()`` must still never be
        called from the 1 kHz feedback path.  It just stops the 2 Hz decision
        loop from blocking too.
        """
        if not (MIN_MOTION_ID <= slot_id <= MAX_MOTION_ID):
            LOGGER.error("slot %r outside the contract range %d..%d (0 = no-motion sentinel)",
                         slot_id, MIN_MOTION_ID, MAX_MOTION_ID)
            return PlayOutcome.ERROR

        with self._lock:
            if self._motion_active:
                return PlayOutcome.BUSY
            self._motion_active = True

        self._play_thread = threading.Thread(
            target=self._play_worker, args=(slot_id,),
            name=f"play-{slot_id}", daemon=True,
        )
        self._play_thread.start()
        return PlayOutcome.OK

    def _play_worker(self, slot_id: int) -> None:
        try:
            outcome = self._play_blocking(slot_id)
        except Exception:  # pragma: no cover - defensive; never kill the thread
            LOGGER.exception("play(%d) raised", slot_id)
            outcome = PlayOutcome.ERROR
        finally:
            with self._lock:
                self._motion_active = False

        with self._lock:
            self._last_outcome = outcome
        if outcome is PlayOutcome.NEEDS_OPERATOR:
            LOGGER.warning(
                "play(%d) REJECTED -- a human must intervene: hold the zero button "
                "(button 1) for 0.6s and wait ~3s, or park with button 2 first. "
                "Retrying in software will not clear this.", slot_id,
            )
        else:
            LOGGER.debug("play(%d) finished: %s", slot_id, outcome.value)
        if self._on_play_result is not None:
            self._on_play_result(slot_id, outcome)

    @abstractmethod
    def _play_blocking(self, slot_id: int) -> PlayOutcome:
        """Actually run the motion.  Called on the worker thread; may block."""


# --------------------------------------------------------------------------- #
# Mock -- no ROS, no robot, no phorce package
# --------------------------------------------------------------------------- #
class MockRobot(RobotInterface):
    """Synthetic feedback plus a ``play()`` that logs and takes realistic time.

    Deliberately imperfect on purpose: it reports BUSY while a motion is running
    (because the real robot has no queue) and can be told to reject, so the retry
    and needs-operator paths get exercised long before the robot arrives.
    """

    def __init__(
        self,
        axes: Sequence[int] = DEFAULT_AXES,
        feedback_hz: float = 200.0,
        motion_duration_s: float = 1.6,
        reject_rate: float = 0.0,
        seed: int = 20260805,
        on_play_result: Optional[PlayCallback] = None,
        jam_after_s: float = 0.0,
        jam_axis: int = 0,
        motion_model: Optional[MockMotionModel] = None,
    ) -> None:
        super().__init__(on_play_result)
        # 200 Hz, not 1 kHz: a mock does not need to burn a core to be useful.
        self.axes = tuple(axes)
        self.feedback_hz = feedback_hz
        self.motion_duration_s = motion_duration_s
        self.reject_rate = reject_rate
        # Obstacle simulation, for testing DREAM-Chunk with no robot: after
        # jam_after_s, one axis stops moving and its disturbance observer reads
        # a large external force -- exactly what a hand on the arm looks like.
        self.jam_after_s = jam_after_s
        self.jam_axis = jam_axis
        self._jam_hold: Optional[float] = None
        self.motion_model = motion_model
        self._rng = random.Random(seed)   # seeded -> reproducible runs
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0
        # What the mock is currently "executing", for motion_model playback.
        self._active_slot: Optional[int] = None
        self._active_t0: float = 0.0
        self._active_start: Optional[list[float]] = None
        self._drift: Optional[list[float]] = None  # last pose, for continuity

    def start(self) -> None:
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._feedback_loop, name="mock-feedback", daemon=True)
        self._thread.start()
        LOGGER.info("MockRobot started: %d axes at %.0f Hz (no robot, no ROS)",
                    len(self.axes), self.feedback_hz)

    def close(self) -> None:
        super().close()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        LOGGER.info("MockRobot stopped")

    def _feedback_loop(self) -> None:
        period = 1.0 / self.feedback_hz
        while not self._closing.is_set():
            now = time.monotonic()
            t = now - self._t0
            active = self.is_motion_active()

            # With a motion model the mock is a digital twin: while a slot is
            # playing it tracks that slot's real trajectory, so a dream of the
            # same slot matches and only a genuine fault makes it diverge.
            commanded: Optional[list[float]] = None
            if self.motion_model is not None and active and self._active_slot is not None:
                if self._active_start is not None:
                    commanded = self.motion_model.pose_at(
                        self._active_slot, self._active_start, now - self._active_t0,
                    )

            states = []
            for slot, index in enumerate(self.axes):
                if commanded is not None and slot < len(commanded):
                    base = commanded[slot]
                    previous = self._drift[slot] if self._drift else base
                    velocity = (base - previous) * self.feedback_hz
                else:
                    # Free-run: slow drift per axis, phase-shifted so the axes
                    # are not clones.
                    phase = 0.7 * slot
                    base = 0.30 * math.sin(2.0 * math.pi * 0.05 * t + phase)
                    velocity = 0.30 * 2.0 * math.pi * 0.05 * math.cos(
                        2.0 * math.pi * 0.05 * t + phase)
                    if active and self.motion_model is None:
                        # While "playing", the joints actually move a bit more.
                        base += 0.15 * math.sin(2.0 * math.pi * 0.8 * t + phase)
                        velocity += 0.15 * 2.0 * math.pi * 0.8 * math.cos(
                            2.0 * math.pi * 0.8 * t + phase)
                noise = self._rng.gauss(0.0, 0.002)
                position = base + noise
                dob = self._rng.gauss(0.0, 0.03)

                jammed = self.jam_after_s > 0.0 and t >= self.jam_after_s and index == self.jam_axis
                if jammed:
                    # Freeze where the obstacle caught it and report the force
                    # the servo is now fighting.
                    if self._jam_hold is None:
                        self._jam_hold = position
                    position = self._jam_hold + noise
                    velocity = 0.0
                    dob = 2.5 + self._rng.gauss(0.0, 0.05)

                states.append(
                    AxisState(
                        index=index,
                        position_rad=position,
                        velocity_rad_s=velocity,
                        current_a=0.4 + 0.2 * abs(velocity) + self._rng.gauss(0.0, 0.01),
                        dob_a=dob,
                        temp_c=35.0 + 2.0 * math.sin(2.0 * math.pi * 0.01 * t) + slot,
                        valid=True,
                    )
                )
            self._drift = [s.position_rad for s in states]
            self._store(JointState(axes=tuple(states), ts=now))
            time.sleep(period)

    def _play_blocking(self, slot_id: int) -> PlayOutcome:
        if self._rng.random() < self.reject_rate:
            LOGGER.info("MOCK play(%d) -> rejected (simulated NOT_READY)", slot_id)
            return PlayOutcome.NEEDS_OPERATOR

        duration = self.motion_duration_s
        if self.motion_model is not None:
            duration = self.motion_model.duration_s(slot_id) or duration
            # Launch from wherever the mock currently is, exactly as the real
            # robot would -- that is what the dream is anchored on too.
            state = self.latest()
            self._active_start = (
                state.positions(self.axes) if state is not None else [0.0] * len(self.axes)
            ) or [0.0] * len(self.axes)
            self._active_t0 = time.monotonic()
            self._active_slot = slot_id

        LOGGER.info("MOCK play(%d) -> running for %.1fs", slot_id, duration)
        deadline = time.monotonic() + duration
        try:
            while time.monotonic() < deadline:
                if self._closing.is_set():
                    return PlayOutcome.ERROR
                time.sleep(0.02)
        finally:
            self._active_slot = None
            self._active_start = None
        return PlayOutcome.OK


# --------------------------------------------------------------------------- #
# Real -- the phorce facade
# --------------------------------------------------------------------------- #
class PhorceRobot(RobotInterface):
    """The real robot, on the Jetson.

    ``import phorce`` happens inside :meth:`start` so this module still imports
    cleanly on a laptop.
    """

    def __init__(
        self,
        target: str = "robot",
        axes: Sequence[int] = DEFAULT_AXES,
        # "rclpy", not "facade": the installed SDK's Robot exposes only
        # close/doctor/play/play_async/status/motions.  robot.watch() appears in
        # the older (RH_Guide_Angel) tutorial but does not exist in the shipped
        # package -- calling it is an AttributeError at start().  Verified with
        # `python3 -c "import phorce; print(dir(phorce.Robot))"` on the golden image.
        feedback_source: str = "rclpy",
        on_play_result: Optional[PlayCallback] = None,
    ) -> None:
        super().__init__(on_play_result)
        self.target = target            # "robot" (real) or e.g. "sim:demo"
        self.axes = tuple(axes)
        self.feedback_source = feedback_source  # "facade" | "rclpy"
        self._phorce = None             # the module, once imported
        self._connection = None         # the context manager from connect()
        self._robot = None              # what connect() yields
        self._ros_thread: Optional[threading.Thread] = None
        self._ros_node = None

    # -- lifecycle --------------------------------------------------------- #
    def start(self) -> None:
        import phorce  # noqa: PLC0415 -- lazy on purpose; absent on a laptop

        self._phorce = phorce
        # connect() is a context manager. We are not inside a `with`, so drive it
        # by hand and unwind in close().
        self._connection = phorce.connect(self.target)
        self._robot = self._connection.__enter__()
        LOGGER.info("connected to phorce target=%s", self.target)

        try:
            # Real Status fields (checked against the installed SDK): state_name,
            # primary_state, physical_idle, recovery_required, contract_active,
            # age_ms, is_fresh().  There is no ethercat_operational/estop_active
            # here -- that spelling is from the older guide generation.
            status = self._robot.status()
            LOGGER.info("state=%s fresh=%s physical_idle=%s recovery_required=%s",
                        status.state_name, status.is_fresh(),
                        status.physical_idle, status.recovery_required)
            if status.recovery_required:
                LOGGER.warning("robot needs RECOVERY -- park with button 2, then hold "
                               "the zero button (button 1) for 0.6s")
            if not status.contract_active:
                LOGGER.warning("motion contract is not active -- the robot will reject plays")
        except Exception:  # pragma: no cover - diagnostics only
            LOGGER.exception("status() failed (continuing anyway)")

        if self.feedback_source == "facade":
            # Kept only for a future SDK that grows a push API. Today there is
            # none, so refuse loudly instead of dying on an AttributeError.
            raise RuntimeError(
                "feedback_source='facade' needs robot.watch(), which the installed "
                "phorce SDK does not provide. Use feedback_source='rclpy'."
            )
        self._start_rclpy_feedback()

    def close(self) -> None:
        super().close()
        if self._ros_node is not None:  # pragma: no cover - needs ROS
            try:
                self._ros_node.destroy_node()
            except Exception:
                LOGGER.exception("destroy_node failed")
        if self._connection is not None:
            try:
                self._connection.__exit__(None, None, None)
            except Exception:  # pragma: no cover
                LOGGER.exception("error closing the phorce connection")
            finally:
                self._connection = None
                self._robot = None
        LOGGER.info("phorce connection closed")

    # -- feedback ---------------------------------------------------------- #
    def _on_feedback(self, frame: object) -> None:
        """Called at 1 kHz.  Store the latest frame and return -- nothing else.

        Do not decide, log or send from here: the robot takes one motion at a
        time with no queue, so a play() call at this rate would be discarded as
        BUSY roughly a thousand times a second.
        """
        now = time.monotonic()
        states: list[AxisState] = []
        try:
            for index in self.axes:
                axis = frame.axis[index]  # type: ignore[attr-defined]
                states.append(
                    AxisState(
                        index=index,
                        position_rad=float(axis.position_rad),
                        velocity_rad_s=float(axis.velocity_rad_s),
                        current_a=float(axis.current_a),
                        dob_a=float(axis.dob_a),
                        temp_c=float(axis.temp_c),
                        # Trust nothing else about this axis if valid is false.
                        valid=bool(axis.valid),
                    )
                )
        except Exception:  # pragma: no cover - malformed frame
            LOGGER.exception("could not parse a feedback frame")
            return
        self._store(JointState(axes=tuple(states), ts=now))

    def _start_rclpy_feedback(self) -> None:  # pragma: no cover - needs ROS
        """Subscribe to /phorce/feedback directly, bypassing the facade.

        Only needed if you want the raw message; the facade's ``watch()`` is the
        easier path.  Kept because it is where the single nastiest failure mode
        lives, and it should be visible in our own code rather than assumed.
        """
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data  # <-- THE critical import
        from agx_msgs.msg import PhorceFeedback

        if not rclpy.ok():
            rclpy.init()
        node = Node("perception_decider_feedback")
        # qos_profile_sensor_data (best effort) is MANDATORY here. The publisher
        # is a 1 kHz best-effort sensor stream; subscribing with the default
        # reliable QoS is not an error -- you simply receive nothing, forever,
        # in silence. If feedback is "not arriving", suspect this line first.
        node.create_subscription(
            PhorceFeedback, "/phorce/feedback", self._on_feedback, qos_profile_sensor_data
        )
        self._ros_node = node
        self._ros_thread = threading.Thread(
            target=rclpy.spin, args=(node,), name="rclpy-spin", daemon=True
        )
        self._ros_thread.start()
        LOGGER.info("feedback via direct rclpy subscription (qos_profile_sensor_data)")

    # -- motion ------------------------------------------------------------ #
    def _play_blocking(self, slot_id: int) -> PlayOutcome:
        phorce = self._phorce
        robot = self._robot
        if phorce is None or robot is None:
            LOGGER.error("play(%d) before start()", slot_id)
            return PlayOutcome.ERROR

        try:
            result = robot.play(slot_id)
            # .ok is an ATTRIBUTE, not a method. `result.ok()` would be a truthy
            # bound method and would silently "succeed" every time.
            return PlayOutcome.OK if result.ok else PlayOutcome.ERROR
        except phorce.MotionBusy:
            # Reject code 5. The only code worth retrying -- it clears on its own.
            LOGGER.debug("play(%d) busy", slot_id)
            return PlayOutcome.BUSY
        except phorce.MotionRejected:
            # Codes 12/13: NOT_READY_FOR_MOTION / RECOVERY_REQUIRED. Waiting does
            # nothing; someone has to press a physical button.
            return PlayOutcome.NEEDS_OPERATOR
        except phorce.MotionAborted:
            LOGGER.error("play(%d) aborted mid-motion -- follow the recovery steps "
                         "in the message", slot_id)
            return PlayOutcome.ERROR
        except phorce.PhorceError:
            LOGGER.exception("play(%d) failed", slot_id)
            return PlayOutcome.ERROR


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def make_robot(mock: bool = True, **kwargs: object) -> RobotInterface:
    """The config flag that picks real vs mock.

    ``mock=True`` needs nothing installed; ``mock=False`` needs the Jetson.
    """
    if mock:
        allowed = {"axes", "feedback_hz", "motion_duration_s", "reject_rate",
                   "seed", "on_play_result", "jam_after_s", "jam_axis",
                   "motion_model"}
        return MockRobot(**{k: v for k, v in kwargs.items() if k in allowed})  # type: ignore[arg-type]
    allowed = {"target", "axes", "feedback_source", "on_play_result"}
    return PhorceRobot(**{k: v for k, v in kwargs.items() if k in allowed})  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Mock smoke test
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the mock robot.")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--reject-rate", type=float, default=0.0)
    parser.add_argument("--slot", type=int, default=1)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    with make_robot(mock=True, reject_rate=args.reject_rate) as robot:
        deadline = time.monotonic() + args.seconds
        next_play = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            state = robot.latest()
            if state is not None:
                positions = state.positions(DEFAULT_AXES)
                print(f"t={state.ts:10.3f} valid_positions={positions} "
                      f"age={state.age_s() * 1000:.1f}ms active={robot.is_motion_active()}")
            if time.monotonic() >= next_play:
                print(f"  -> play({args.slot}) returned {robot.play(args.slot).value}")
                next_play += 1.0
            time.sleep(0.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

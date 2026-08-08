#!/usr/bin/env python3
"""The only file in this project that knows phorce or ROS 2 exist.

MockRobot (no ROS/robot/SDK needed) and PhorceRobot (real, lazy imports)
behind one interface; pick with :func:`make_robot`.  Contract: feedback is
1 kHz and needs qos_profile_sensor_data (default QoS = silence, no error);
``result.ok`` is an attribute, not a method; only reject 5 (BUSY) is worth
retrying -- 12/13 need a human; trust an axis only when its ``valid`` flag
is true; one motion at a time, no queue.

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
        """Positions for ``indices``, or ``None`` if any axis is untrustworthy."""
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
        """Disturbance-observer estimate per axis -- the API's only contact signal."""
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
    """What came of a ``play()`` request.  ``OK`` from ``play()`` = accepted
    and started; ``OK`` at the completion callback = actually finished."""

    OK = "ok"
    BUSY = "busy"                        # reject code 5 -- retry next tick
    NEEDS_OPERATOR = "needs_operator"    # codes 12/13 -- a human must intervene
    ERROR = "error"


PlayCallback = Callable[[int, PlayOutcome], None]


class MockMotionModel(Protocol):
    """How the mock robot moves while a slot plays; supplied from outside so
    this module stays ignorant of the motion format."""

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
        """Request a motion slot.  **Returns immediately** (worker thread; the
        real ``robot.play()`` blocks up to ~30 s).  BUSY = already running, no
        queue, retry next tick; OK = accepted and started (terminal result via
        the callback / :meth:`last_outcome`); ERROR = slot outside 1..50.
        Still never call this from the 1 kHz feedback path.
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


class CliRobot(RobotInterface):
    """Plays slots through the organizer's own ``phorce play N --json`` CLI --
    the path proven on the hardware.  Needs (ROS_DOMAIN_ID=21, sourced
    workspace) are inherited from the environment.  Measured contract:
    success = ``ok`` + ``status_name == "SUCCEEDED"``; rejects ride
    ``decision_reason``/``recovery_required`` (codes 12/13 set the latter).
    """

    def __init__(self, target: str = "robot", binary: str = "phorce",
                 timeout_s: float = 150.0,
                 on_play_result: Optional[PlayCallback] = None) -> None:
        super().__init__(on_play_result)
        self.target = target
        self.binary = binary
        self.timeout_s = timeout_s

    def start(self) -> None:
        import shutil
        if shutil.which(self.binary) is None:
            raise RuntimeError(
                f"{self.binary!r} is not on PATH -- source the ROS workspace "
                "(the same shell that can run `phorce play N` by hand)")

    def _play_blocking(self, slot_id: int) -> PlayOutcome:
        import json
        import subprocess
        cmd = [self.binary, "play", str(slot_id),
               "--target", self.target, "--json"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            LOGGER.error("phorce play %d: no reply in %.0f s",
                         slot_id, self.timeout_s)
            return PlayOutcome.ERROR
        data = None
        for line in reversed(proc.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                except ValueError:
                    pass
                break
        if data is None:
            LOGGER.error("phorce play %d: no JSON in reply (rc=%d): %s",
                         slot_id, proc.returncode,
                         (proc.stdout + proc.stderr).strip()[:300])
            return PlayOutcome.ERROR
        if data.get("ok") and data.get("status_name") == "SUCCEEDED":
            return PlayOutcome.OK
        reason = str(data.get("decision_reason", "")).lower()
        if data.get("recovery_required") or "operator" in reason:
            return PlayOutcome.NEEDS_OPERATOR
        if "busy" in reason:
            return PlayOutcome.BUSY
        LOGGER.error("phorce play %d refused: %s", slot_id, data)
        return PlayOutcome.ERROR


# --------------------------------------------------------------------------- #
# Mock -- no ROS, no robot, no phorce package
# --------------------------------------------------------------------------- #
class MockRobot(RobotInterface):
    """Synthetic feedback plus a ``play()`` that logs and takes realistic time;
    reports BUSY while playing and can be told to reject, so the retry and
    needs-operator paths get exercised without the robot."""

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
        # Obstacle sim: after jam_after_s one axis freezes with a big DOB force.
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

            # With a motion model the mock tracks the playing slot's trajectory.
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
                    # Free-run: slow phase-shifted drift per axis.
                    phase = 0.7 * slot
                    base = 0.30 * math.sin(2.0 * math.pi * 0.05 * t + phase)
                    velocity = 0.30 * 2.0 * math.pi * 0.05 * math.cos(
                        2.0 * math.pi * 0.05 * t + phase)
                    if active and self.motion_model is None:
                        base += 0.15 * math.sin(2.0 * math.pi * 0.8 * t + phase)
                        velocity += 0.15 * 2.0 * math.pi * 0.8 * math.cos(
                            2.0 * math.pi * 0.8 * t + phase)
                noise = self._rng.gauss(0.0, 0.002)
                position = base + noise
                dob = self._rng.gauss(0.0, 0.03)

                jammed = self.jam_after_s > 0.0 and t >= self.jam_after_s and index == self.jam_axis
                if jammed:
                    # Freeze at the obstacle; report the force being fought.
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
            # Launch from wherever the mock currently is, as the robot would.
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
    """The real robot, on the Jetson; ``import phorce`` is lazy inside
    :meth:`start` so this module still imports cleanly on a laptop."""

    def __init__(
        self,
        target: str = "robot",
        axes: Sequence[int] = DEFAULT_AXES,
        # "rclpy", not "facade": the shipped SDK has no robot.watch() push API.
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
        # connect() is a context manager: driven by hand, unwound in close().
        self._connection = phorce.connect(self.target)
        self._robot = self._connection.__enter__()
        LOGGER.info("connected to phorce target=%s", self.target)

        try:
            status = self._robot.status()
            # is_fresh is a method in one SDK generation, a plain bool in another.
            fresh = status.is_fresh() if callable(status.is_fresh) else status.is_fresh
            LOGGER.info("state=%s fresh=%s physical_idle=%s recovery_required=%s",
                        status.state_name, fresh,
                        status.physical_idle, status.recovery_required)
            if status.recovery_required:
                LOGGER.warning("robot needs RECOVERY -- park with button 2, then hold "
                               "the zero button (button 1) for 0.6s")
            if not status.contract_active:
                LOGGER.warning("motion contract is not active -- the robot will reject plays")
        except Exception:  # pragma: no cover - diagnostics only
            LOGGER.exception("status() failed (continuing anyway)")

        if self.feedback_source == "facade":
            # No push API in today's SDK -- refuse loudly.
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
        """Called at 1 kHz.  Store the latest frame and return -- never decide,
        log or send from here."""
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
        """Subscribe to /phorce/feedback directly, bypassing the facade."""
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data  # <-- THE critical import
        from agx_msgs.msg import PhorceFeedback

        if not rclpy.ok():
            rclpy.init()
        node = Node("perception_decider_feedback")
        # qos_profile_sensor_data is MANDATORY: the default reliable QoS
        # receives nothing, forever, in silence.  Suspect this line first.
        node.create_subscription(
            PhorceFeedback, "/phorce/feedback", self._on_feedback, qos_profile_sensor_data
        )
        self._ros_node = node
        # A dedicated executor, NOT rclpy.spin(): the SDK already spins the
        # global executor; contending starves its watchdog (PhorceUnavailable).
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        self._ros_thread = threading.Thread(
            target=executor.spin, name="rclpy-spin", daemon=True
        )
        self._ros_thread.start()
        LOGGER.info("feedback via dedicated-executor rclpy subscription "
                    "(qos_profile_sensor_data)")

    # -- motion ------------------------------------------------------------ #
    def _play_blocking(self, slot_id: int) -> PlayOutcome:
        phorce = self._phorce
        robot = self._robot
        if phorce is None or robot is None:
            LOGGER.error("play(%d) before start()", slot_id)
            return PlayOutcome.ERROR

        try:
            result = robot.play(slot_id)
            # .ok is an ATTRIBUTE; result.ok() would be truthy every time.
            return PlayOutcome.OK if result.ok else PlayOutcome.ERROR
        except phorce.MotionBusy:
            # Reject code 5 -- the only code worth retrying.
            LOGGER.debug("play(%d) busy", slot_id)
            return PlayOutcome.BUSY
        except phorce.MotionRejected:
            # Codes 12/13: a human must press a physical button.
            return PlayOutcome.NEEDS_OPERATOR
        except phorce.MotionAborted:
            LOGGER.error("play(%d) aborted mid-motion -- follow the recovery steps "
                         "in the message", slot_id)
            return PlayOutcome.ERROR
        except phorce.PhorceError:
            LOGGER.exception("play(%d) failed", slot_id)
            return PlayOutcome.ERROR


# --------------------------------------------------------------------------- #
# Bridge -- the decision loop's motion, replayed as PCM slots
# --------------------------------------------------------------------------- #
class SlotBridge:
    """Keeps a :class:`RobotInterface` playing the slot the engine is in.

    The robot has no queue, so "continuous rocking" = re-requesting the slot
    whenever the robot goes idle (``repeat=False``: one episode per decision).
    ``None`` stops *requesting* only -- the API has no abort; the physical
    stop is the robot's own contract and the E-stop.  NEEDS_OPERATOR and
    ERROR hold off retries.  Time is injected (``tick(now, ...)``) so the
    hold-offs are testable with a fake clock.
    """

    RETRY_OPERATOR_S = 10.0
    RETRY_ERROR_S = 2.0

    def __init__(self, robot: RobotInterface,
                 log: Optional[Callable[[str], None]] = None,
                 rest_s: float = 0.0,
                 max_slot: Optional[int] = None,
                 repeat: bool = True) -> None:
        self.robot = robot
        # repeat=False: a *completed* slot is not re-requested while still
        # named; a new slot or a park-and-recommand plays.  Rejects/errors
        # still retry under the hold-offs below.
        self.repeat = repeat
        # Rest between completed slots; counts from a fully finished motion.
        self.rest_s = max(0.0, rest_s)
        # Slots above this are not on the card: screen only, warned once each.
        self.max_slot = max_slot
        self._off_card: set = set()
        self._log = log if log is not None else (
            lambda text: LOGGER.info("%s", text))
        self._hold_until = 0.0
        self._awaiting = False      # a play we issued has not finished yet
        self._last_slot: Optional[int] = None
        self._req_slot: Optional[int] = None   # slot of the outstanding play
        self._done_slot: Optional[int] = None  # completed; held while renamed
        self._warned_no_abort = False

    def tick(self, now: float, slot_id: Optional[int]) -> None:
        if (slot_id is not None and self.max_slot is not None
                and slot_id > self.max_slot):
            if slot_id not in self._off_card:
                self._off_card.add(slot_id)
                self._log(f"robot: slot {slot_id} is not on the card "
                          f"(1..{self.max_slot}) -- screen only")
            slot_id = None
        if slot_id is None:
            if self.robot.is_motion_active() and not self._warned_no_abort:
                self._warned_no_abort = True
                self._log("robot: mode parked -- current slot finishes "
                          "(the API has no abort)")
            self._last_slot = None
            self._done_slot = None
            return
        self._warned_no_abort = False

        if self.robot.is_motion_active():
            return
        if self._awaiting:
            # Our play completed; its outcome decides whether to re-request.
            self._awaiting = False
            outcome = self.robot.last_outcome()
            if outcome is PlayOutcome.NEEDS_OPERATOR:
                self._hold_until = now + self.RETRY_OPERATOR_S
                self._log("robot: NOT READY -- hold the zero button (button 1) "
                          f"0.6 s; retrying in {self.RETRY_OPERATOR_S:.0f} s")
            elif outcome is PlayOutcome.ERROR:
                self._hold_until = now + self.RETRY_ERROR_S
            else:
                if not self.repeat:
                    self._done_slot = self._req_slot
                    self._log(f"robot: slot {self._req_slot} done -- "
                              "quiet until a new decision")
                if self.rest_s > 0.0:
                    self._hold_until = max(self._hold_until,
                                           now + self.rest_s)
        if now < self._hold_until:
            return
        if not self.repeat and slot_id == self._done_slot:
            return

        if self.robot.play(slot_id) is PlayOutcome.OK:
            self._awaiting = True
            self._req_slot = slot_id
            if slot_id != self._last_slot:
                self._log(f"robot: playing slot {slot_id}")
                self._last_slot = slot_id
        # BUSY = someone else owns the robot -- try again next tick.


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def make_robot(mock: bool = True, **kwargs: object) -> RobotInterface:
    """The config flag that picks real vs mock; ``mock=True`` needs nothing
    installed, ``mock=False`` needs the Jetson."""
    if mock:
        allowed = {"axes", "feedback_hz", "motion_duration_s", "reject_rate",
                   "seed", "on_play_result", "jam_after_s", "jam_axis",
                   "motion_model"}
        return MockRobot(**{k: v for k, v in kwargs.items() if k in allowed})  # type: ignore[arg-type]
    allowed = {"target", "axes", "feedback_source", "on_play_result"}
    kw = {k: v for k, v in kwargs.items() if k in allowed}
    # target "cli" (or "cli:sim:demo") plays via the organizer's own CLI.
    tgt = str(kw.get("target") or "")
    if tgt == "cli" or tgt.startswith("cli:"):
        return CliRobot(target=tgt[4:] or "robot",
                        on_play_result=kw.get("on_play_result"))  # type: ignore[arg-type]
    return PhorceRobot(**kw)  # type: ignore[arg-type]


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

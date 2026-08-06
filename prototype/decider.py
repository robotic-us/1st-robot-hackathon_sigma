#!/usr/bin/env python3
"""Decider: turn what the camera sees into a motion-slot number.

This is the "brain".  It consumes a :class:`~perception.Perception` and the
robot's current joint state and answers one question: *which slot, if any,
should we play right now?*

The mapping, in one paragraph:

    Where the circle is (``x``, with the dead-band scaled by ``distance``) picks
    a **direction** -- left, center or right.  How much it is moving (``motion``)
    picks an **amplitude** -- small, medium or large.  Those two coordinates
    select a cell of the 3x3 slot grid.  If several slots live in that cell, the
    one whose ``start_pose`` is closest to where the robot actually is wins, so
    the transition into the motion is smooth.

Two behaviours sit on top of that grid:

    * **stirring trigger** -- when ``motion`` is very high the baby is stirring;
      force the large amplitude and react on a shortened dwell.
    * **settle ramp** -- when motion decays to ~0 after having been restless,
      play a descending sequence of slots (center large -> medium -> small ->
      the dedicated settle slot).  This is our discrete stand-in for a
      continuous wind-down: we cannot modulate torque, so we step down through
      progressively gentler pre-recorded motions instead.

Nothing here knows about ROS, cameras or BUSY replies.  It is a pure mapping
from perception + pose to a slot number, which is what makes it testable:

    python3 decider.py --replay demo.csv     # drive it from ground truth, no hardware
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
# This module lives in prototype/ but imports the shared robot layer from the
# project root (dream, pvector, phorce_iface, slot_table).
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import csv
import logging
import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence

from dream import ChunkMatcher, DivergenceReport
from perception import Perception, clamp01
from phorce_iface import JointState
from slot_table import (
    AMPLITUDE_VALUE, Amplitude, Direction, Slot, SlotTable, load_slot_table,
)

LOGGER = logging.getLogger("decider")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class DeciderConfig:
    """Every threshold, in one place.  Defaults are tuned for stimulus.py's demo."""

    # --- direction dead-band ---------------------------------------------- #
    dir_enter: float = 0.25       # |x| beyond this leaves CENTER
    dir_exit: float = 0.15        # |x| must fall back inside this to return
    dir_distance_k: float = 0.40  # how strongly distance narrows the dead-band
    dir_min_threshold: float = 0.08   # never let the dead-band collapse entirely
    x_smoothing_s: float = 1.5    # time constant for the x used to pick direction.
    #                               Direction should track where the baby *is*,
    #                               not where a fast oscillation happens to be at
    #                               the instant we sampled it -- the oscillation
    #                               is already accounted for, as motion.

    # --- amplitude thresholds (on motion) ---------------------------------- #
    amp_medium_at: float = 0.18   # motion >= this is MEDIUM
    amp_large_at: float = 0.55    # motion >= this is LARGE
    amp_hysteresis: float = 0.07  # thresholds shift by this against the current bucket

    # --- special cases ----------------------------------------------------- #
    stir_threshold: float = 0.75      # "the baby is stirring"
    settle_arm_motion: float = 0.55   # must get at least this restless before a
    #                                   wind-down means anything
    settle_threshold: float = 0.12    # motion at or below this reads as "still".
    #                                   Also the "no action needed" floor: a baby
    #                                   this calm does not want to be handled.
    settle_hold_s: float = 2.0        # ...sustained this long before winding down

    # --- anti-thrash -------------------------------------------------------- #
    min_dwell_s: float = 2.5      # minimum gap between any two plays
    repeat_dwell_s: float = 6.0   # longer gap before replaying the SAME slot
    urgent_scale: float = 0.5     # both are scaled by this while stirring

    # --- scoring weights ---------------------------------------------------- #
    w_pose: float = 1.0           # weight on "starts where we already are"
    w_amp: float = 0.5            # weight on "is the right size for this motion"

    # --- optional override --------------------------------------------------- #
    settle_ramp: Optional[tuple[int, ...]] = None  # else derived from the table


# --------------------------------------------------------------------------- #
# Pure bucketing / scoring helpers
# --------------------------------------------------------------------------- #
def direction_threshold(distance: float, cfg: DeciderConfig) -> float:
    """Dead-band half-width for the current apparent distance.

    ``thr = dir_enter * (1 - k * distance)``.  A *near* circle (large radius,
    ``distance`` -> 1) means the same normalised x offset corresponds to a bigger
    real displacement, so we narrow the dead-band and react sooner.  A far circle
    widens it, which also suppresses jitter when the target is small and noisy.
    """
    thr = cfg.dir_enter * (1.0 - cfg.dir_distance_k * clamp01(distance))
    return max(cfg.dir_min_threshold, thr)


def bucket_direction(
    x: float,
    distance: float,
    cfg: DeciderConfig,
    current: Optional[Direction] = None,
) -> Direction:
    """Schmitt trigger on ``x``: it takes more to leave CENTER than to stay out.

    Pure: pass the previous bucket in as ``current`` and the hysteresis comes
    along for free, with no hidden state.
    """
    enter = direction_threshold(distance, cfg)
    # Preserve the enter/exit ratio as the dead-band scales with distance.
    release = enter * (cfg.dir_exit / cfg.dir_enter) if cfg.dir_enter else cfg.dir_exit

    if current is Direction.LEFT and x < -release:
        return Direction.LEFT
    if current is Direction.RIGHT and x > release:
        return Direction.RIGHT
    if x < -enter:
        return Direction.LEFT
    if x > enter:
        return Direction.RIGHT
    return Direction.CENTER


def bucket_amplitude(
    motion: float,
    cfg: DeciderConfig,
    current: Optional[Amplitude] = None,
) -> Amplitude:
    """Bucket ``motion``, with the thresholds biased against leaving ``current``."""
    medium_at, large_at = cfg.amp_medium_at, cfg.amp_large_at
    h = cfg.amp_hysteresis

    if current is Amplitude.SMALL:      # make it harder to climb out
        medium_at, large_at = medium_at + h, large_at + h
    elif current is Amplitude.LARGE:    # make it harder to drop out
        medium_at, large_at = medium_at - h, large_at - h

    if motion >= large_at:
        return Amplitude.LARGE
    if motion >= medium_at:
        return Amplitude.MEDIUM
    return Amplitude.SMALL


def pose_distance(slot: Slot, joints: Optional[Sequence[float]]) -> Optional[float]:
    """Mean absolute joint error between where a slot starts and where we are.

    ``None`` when we have no trustworthy joint data -- the caller decides what
    that means rather than silently pretending the error is zero.
    """
    if joints is None or not slot.start_pose:
        return None
    n = min(len(slot.start_pose), len(joints))
    if n == 0:
        return None
    return sum(abs(slot.start_pose[i] - joints[i]) for i in range(n)) / n


def score_slot(
    slot: Slot,
    joints: Optional[Sequence[float]],
    target_amplitude: float,
    cfg: DeciderConfig,
) -> tuple[float, float, float]:
    """Cost of choosing ``slot`` right now.  Lower is better.

        cost = w_pose * mean(|start_pose - current_joint_positions|)
             + w_amp  * |slot_amplitude_value - observed_motion|

    The first term buys smooth transitions: a motion that begins near where the
    arm already is will not snap.  The second breaks ties toward the slot whose
    size best fits what we are actually seeing, which matters when the exact
    (direction, amplitude) cell is empty and we fall back to a neighbour.

    Returns ``(cost, pose_term, amp_term)`` so the choice can be explained in the
    log rather than being an unexplained number.
    """
    pose = pose_distance(slot, joints)
    pose_term = 0.0 if pose is None else pose   # no joint data -> term drops out
    amp_term = abs(AMPLITUDE_VALUE[slot.amplitude] - target_amplitude)
    return cfg.w_pose * pose_term + cfg.w_amp * amp_term, pose_term, amp_term


# --------------------------------------------------------------------------- #
# The reasoning record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Decision:
    """Why the decider did what it did -- everything main.py needs to log."""

    slot_id: Optional[int]
    reason: str
    direction: Optional[Direction] = None
    amplitude: Optional[Amplitude] = None
    motion: float = 0.0
    x: float = 0.0          # smoothed x -- what actually picked the direction
    x_raw: float = 0.0      # this frame's x, for cross-checking against the video
    distance: float = 0.0
    candidates: tuple[int, ...] = ()
    cost: Optional[float] = None
    pose_term: Optional[float] = None
    amp_term: Optional[float] = None
    fallback: bool = False
    stirring: bool = False
    settling: bool = False
    have_joints: bool = False
    # --- DREAM-Chunk ------------------------------------------------------- #
    dreamed: bool = False       # the chunk matcher, not score_slot(), ranked this
    preempted: bool = False     # dwell cut short because the dream diverged
    dream_error: float = 0.0    # last chunk's measured-vs-dreamed error, rad
    external_force: float = 0.0 # worst-axis |dob_a| at decision time, amps
    vetoed: tuple[int, ...] = ()  # candidates the excursion guard rejected

    def describe(self) -> str:
        """One line explaining the decision, for the log."""
        head = (f"x={self.x:+.2f}(raw {self.x_raw:+.2f}) "
                f"d={self.distance:.2f} m={self.motion:.2f}")
        if self.direction is None:
            return f"{head} | {self.reason}"

        flags = []
        if self.stirring:
            flags.append("stir trigger")
        if self.settling:
            flags.append("settle ramp")
        if self.fallback:
            flags.append("cell empty -> nearest amplitude")
        if not self.have_joints:
            flags.append("no joint data, pose term inactive")
        if self.dreamed:
            flags.append(f"dream-matched (err {self.dream_error:.3f} rad, "
                         f"dob {self.external_force:.2f}A)")
        if self.preempted:
            flags.append("DWELL PRE-EMPTED: dream diverged")
        if self.vetoed:
            flags.append(f"vetoed {list(self.vetoed)}")
        suffix = f" ({', '.join(flags)})" if flags else ""

        body = f"dir={self.direction.value.upper()} amp={self.amplitude.value.upper()}{suffix}"
        if self.slot_id is None:
            return f"{head} | {body} | {self.reason}"

        detail = f"cand={list(self.candidates)} pick={self.slot_id}"
        if self.cost is not None:
            detail += (f" cost={self.cost:.3f} "
                       f"(pose {self.pose_term:.3f} + amp {self.amp_term:.3f})")
        return f"{head} | {body} | {detail} | {self.reason}"


# --------------------------------------------------------------------------- #
# The decider
# --------------------------------------------------------------------------- #
class Decider:
    """Stateful wrapper around the pure helpers above.

    All mutable state (hysteresis latches, dwell timer, wind-down queue) lives
    here; the bucketing and scoring functions stay pure and independently
    testable.
    """

    def __init__(
        self,
        table: SlotTable,
        config: Optional[DeciderConfig] = None,
        matcher: Optional[ChunkMatcher] = None,
    ) -> None:
        self.table = table
        self.cfg = config or DeciderConfig()
        # When a matcher is supplied, candidate ranking moves from the static
        # score_slot() heuristic to DREAM-Chunk's dreamed-state matching. The
        # bucketing above it is unchanged -- direction/amplitude still choose
        # *which* chunks are candidates; the dream chooses between them.
        self.matcher = matcher
        self.settle_ramp = self._derive_settle_ramp()

        self._direction = Direction.CENTER
        self._amplitude = Amplitude.SMALL
        self._last_play_ts: float = -math.inf
        self._last_slot: Optional[int] = None
        self._settle_queue: list[int] = []
        self._armed = False                       # has been restless recently
        self._calm_since: Optional[float] = None
        self._x_avg: Optional[float] = None       # smoothed x (see x_smoothing_s)
        self._last_ts: Optional[float] = None
        self._last_decision = Decision(None, "nothing decided yet")

        LOGGER.info("settle ramp: %s", " -> ".join(str(s) for s in self.settle_ramp) or "(none)")

    # -- public ------------------------------------------------------------ #
    def decide(
        self,
        perc: Perception,
        joint_state: Optional[JointState],
        now: Optional[float] = None,
        divergence: Optional[DivergenceReport] = None,
    ) -> Optional[int]:
        """Which slot to play, or ``None`` when no action is warranted."""
        now = time.monotonic() if now is None else now
        cfg = self.cfg

        # Positions come back as None if ANY axis we need is flagged invalid --
        # the manual is explicit that valid is the only trustworthy evidence.
        joints = joint_state.positions(self.table.axes) if joint_state is not None else None
        forces = (joint_state.external_force(self.table.axes)
                  if joint_state is not None else None)

        # DREAM-Chunk's reactive trigger. The robot cannot be interrupted
        # mid-motion, so a diverged dream cannot stop what is playing -- but it
        # can and does collapse the dwell, so the switch happens at the first
        # boundary the hardware permits instead of one full dwell later.
        diverged = bool(divergence is not None and divergence.diverged)
        dream_error = float(divergence.error_rad) if divergence is not None else 0.0
        worst_force = max((abs(f) for f in forces), default=0.0) if forces else 0.0

        if not perc.present:
            # Nothing to react to. Keep the latches as they are; the circle may
            # come back in a moment.
            self._last_ts = now
            return self._record(Decision(None, "no target in view", motion=perc.motion))

        # --- 1. buckets, with hysteresis ---------------------------------- #
        x_avg = self._smooth_x(perc.x, now)
        self._direction = bucket_direction(x_avg, perc.distance, cfg, self._direction)
        self._amplitude = bucket_amplitude(perc.motion, cfg, self._amplitude)

        # --- 2. stirring / settle state ------------------------------------ #
        stirring = perc.motion >= cfg.stir_threshold
        if perc.motion >= cfg.settle_arm_motion:
            # Restless enough that a later wind-down is meaningful.
            self._armed = True
        if perc.motion <= cfg.settle_threshold:
            if self._calm_since is None:
                self._calm_since = now
        else:
            self._calm_since = None

        if stirring:
            # Stirring again: abandon any wind-down in progress.
            if self._settle_queue:
                LOGGER.info("wind-down cancelled -- motion rose to %.2f", perc.motion)
                self._settle_queue.clear()
        elif (
            self._armed
            and not self._settle_queue
            and self._calm_since is not None
            and (now - self._calm_since) >= cfg.settle_hold_s
        ):
            self._settle_queue = self._ramp_from_here()
            self._armed = False
            if self._settle_queue:
                LOGGER.info(
                    "settling: motion held below %.2f for %.1fs -> winding down %s",
                    cfg.settle_threshold, cfg.settle_hold_s,
                    " -> ".join(str(s) for s in self._settle_queue),
                )
            else:
                LOGGER.info("settling: already at the gentlest motion, nothing to wind down")

        direction = self._direction
        # The stirring trigger overrides the amplitude bucket outright.
        amplitude = Amplitude.LARGE if stirring else self._amplitude

        # A baby this still does not want to be handled. This covers both a calm
        # start (do nothing until something happens) and the quiet after a
        # completed wind-down (stop, rather than rocking forever).
        if not self._settle_queue and perc.motion <= cfg.settle_threshold:
            return self._record(Decision(
                None, f"calm (motion {perc.motion:.2f} <= {cfg.settle_threshold:.2f}), "
                      "no action needed",
                direction=direction, amplitude=amplitude,
                motion=perc.motion, x=x_avg, x_raw=perc.x, distance=perc.distance,
                have_joints=joints is not None,
            ))

        # --- 3. pick what we would play ------------------------------------ #
        settling = bool(self._settle_queue)
        if settling:
            slot_id = self._settle_queue[0]
            slot = self.table.slots.get(slot_id)
            if slot is None:  # defensive: ramp refers to a slot that vanished
                self._settle_queue.pop(0)
                return self._record(Decision(None, f"settle ramp slot {slot_id} missing"))
            candidates = (slot_id,)
            cost = pose_term = amp_term = None
            fallback = False
            dreamed = False
            vetoed: tuple[int, ...] = ()
        else:
            matches, fallback = self._candidates(direction, amplitude)
            if not matches:
                return self._record(Decision(
                    None, "no slot matches this bucket pair",
                    direction=direction, amplitude=amplitude,
                    motion=perc.motion, x=x_avg, x_raw=perc.x, distance=perc.distance,
                    stirring=stirring, have_joints=joints is not None,
                    dream_error=dream_error, external_force=worst_force,
                ))

            if self.matcher is not None:
                # --- DREAM-Chunk: rank by dreaming each candidate from here --
                ranked = self.matcher.rank(
                    [s.slot_id for s in matches],
                    joints,
                    external_force_a=forces,
                    target_amplitude=perc.motion,
                    amplitude_of={s.slot_id: AMPLITUDE_VALUE[s.amplitude] for s in matches},
                    recent_error_rad=dream_error,
                )
                vetoed = tuple(s.slot_id for s in ranked if s.vetoed)
                survivors = [s for s in ranked if not s.vetoed]
                if not survivors:
                    return self._record(Decision(
                        None, "every candidate vetoed by the excursion guard",
                        direction=direction, amplitude=amplitude,
                        motion=perc.motion, x=x_avg, x_raw=perc.x, distance=perc.distance,
                        stirring=stirring, have_joints=joints is not None,
                        dreamed=True, dream_error=dream_error,
                        external_force=worst_force, vetoed=vetoed,
                    ))
                pick = survivors[0]
                slot_id = pick.slot_id
                candidates = tuple(s.slot_id for s in survivors)
                cost, pose_term, amp_term = pick.total, pick.resistance, pick.task
                dreamed = True
            else:
                # Sort by (cost, slot_id): the id keeps ties deterministic, which
                # matters because placeholder poses make ties common.
                scored = sorted(
                    ((score_slot(s, joints, perc.motion, cfg), s) for s in matches),
                    key=lambda pair: (pair[0][0], pair[1].slot_id),
                )
                (cost, pose_term, amp_term), slot = scored[0]
                slot_id = slot.slot_id
                candidates = tuple(s.slot_id for _, s in scored)
                dreamed = False
                vetoed = ()

        # --- 4. anti-thrash: dwell ------------------------------------------ #
        required = cfg.repeat_dwell_s if slot_id == self._last_slot else cfg.min_dwell_s
        if stirring:
            required *= cfg.urgent_scale
        waited = now - self._last_play_ts
        preempted = False
        if diverged:
            # The dream broke: something is physically in the way. Waiting out
            # the anti-thrash dwell would mean ignoring a live contact for
            # seconds, which is the opposite of reactive.
            preempted = True
            required = 0.0
        if waited < required:
            return self._record(Decision(
                None, f"dwell: {required - waited:.1f}s to go (need {required:.1f}s)",
                direction=direction, amplitude=amplitude,
                motion=perc.motion, x=x_avg, x_raw=perc.x, distance=perc.distance,
                candidates=candidates, stirring=stirring, settling=settling,
                have_joints=joints is not None, dreamed=dreamed,
                dream_error=dream_error, external_force=worst_force, vetoed=vetoed,
            ))

        # --- 5. commit ------------------------------------------------------ #
        if settling:
            self._settle_queue.pop(0)
            step = len(self.settle_ramp) - len(self._settle_queue)
            reason = f"wind-down step {step}/{len(self.settle_ramp)}"
        else:
            reason = "selected"
        self._last_play_ts = now
        self._last_slot = slot_id

        if preempted:
            reason = f"{reason} (reacting to a diverged dream)"

        return self._record(Decision(
            slot_id, reason,
            direction=direction, amplitude=amplitude,
            motion=perc.motion, x=x_avg, x_raw=perc.x, distance=perc.distance,
            candidates=candidates, cost=cost, pose_term=pose_term, amp_term=amp_term,
            fallback=fallback, stirring=stirring, settling=settling,
            have_joints=joints is not None, dreamed=dreamed, preempted=preempted,
            dream_error=dream_error, external_force=worst_force, vetoed=vetoed,
        ))

    def explain(self) -> Decision:
        """The full reasoning behind the most recent :meth:`decide` call."""
        return self._last_decision

    def reset(self) -> None:
        self._direction = Direction.CENTER
        self._amplitude = Amplitude.SMALL
        self._last_play_ts = -math.inf
        self._last_slot = None
        self._settle_queue.clear()
        self._armed = False
        self._calm_since = None
        self._x_avg = None
        self._last_ts = None

    # -- internals --------------------------------------------------------- #
    def _record(self, decision: Decision) -> Optional[int]:
        self._last_decision = decision
        return decision.slot_id

    def _smooth_x(self, x: float, now: float) -> float:
        """Low-pass ``x`` over ``x_smoothing_s``.

        Time-constant based rather than a fixed alpha, so the smoothing is the
        same whether the decision loop runs at 2 Hz or 5 Hz.  Without this, a
        circle oscillating fast about the centre makes the direction bucket flip
        on every tick purely from where we happened to sample it -- the classic
        way to make a robot look confused.
        """
        if self._x_avg is None or self._last_ts is None:
            self._x_avg = x
        else:
            dt = max(0.0, now - self._last_ts)
            tau = max(1e-3, self.cfg.x_smoothing_s)
            alpha = 1.0 - math.exp(-dt / tau)
            self._x_avg += alpha * (x - self._x_avg)
        self._last_ts = now
        return self._x_avg

    def _candidates(
        self, direction: Direction, amplitude: Amplitude
    ) -> tuple[list[Slot], bool]:
        """Slots for this cell; if it is empty, the nearest amplitude in the same
        direction.  The bool says whether we fell back."""
        exact = self.table.cell(direction, amplitude)
        if exact:
            return exact, False

        same_direction = [
            s for s in self.table.slots.values()
            if s.direction is direction and s.amplitude is not Amplitude.SETTLE
        ]
        if not same_direction:
            return [], True

        target = AMPLITUDE_VALUE[amplitude]
        nearest = min(
            same_direction,
            key=lambda s: (abs(AMPLITUDE_VALUE[s.amplitude] - target), s.slot_id),
        )
        return self.table.cell(direction, nearest.amplitude), True

    def _ramp_from_here(self) -> list[int]:
        """The wind-down steps that are gentler than whatever we last played.

        Starting the ramp at its top would mean answering a baby who has just
        gone quiet with the *largest* motion in the table -- exactly backwards.
        Instead the wind-down picks up from wherever the robot already is and
        only ever steps down, which is what makes a sequence of discrete slots
        read as one continuous easing-off.
        """
        last = self.table.slots.get(self._last_slot) if self._last_slot is not None else None
        if last is None:
            return list(self.settle_ramp)

        ceiling = AMPLITUDE_VALUE[last.amplitude]
        steps: list[int] = []
        for slot_id in self.settle_ramp:   # already ordered large -> settle
            slot = self.table.slots.get(slot_id)
            if slot is not None and AMPLITUDE_VALUE[slot.amplitude] < ceiling:
                steps.append(slot_id)
        return steps

    def _derive_settle_ramp(self) -> tuple[int, ...]:
        """center/large -> center/medium -> center/small -> the settle slot.

        Derived from the table rather than hard-coded, so renumbering slots.json
        does not silently break the wind-down.
        """
        if self.cfg.settle_ramp:
            return tuple(self.cfg.settle_ramp)

        ramp: list[int] = []
        for amplitude in (Amplitude.LARGE, Amplitude.MEDIUM, Amplitude.SMALL):
            cell = self.table.cell(Direction.CENTER, amplitude)
            if cell:
                ramp.append(min(s.slot_id for s in cell))
        settle_slots = self.table.with_amplitude(Amplitude.SETTLE)
        if settle_slots:
            ramp.append(min(s.slot_id for s in settle_slots))
        else:
            LOGGER.warning("no slot has amplitude 'settle' -- the wind-down will "
                           "end on the smallest center motion instead")
        return tuple(ramp)


# --------------------------------------------------------------------------- #
# Replay: drive the decider from a stimulus ground-truth CSV
# --------------------------------------------------------------------------- #
def replay(
    csv_path: str,
    decider: Decider,
    decide_hz: float = 2.0,
    distance: float = 0.35,
) -> int:
    """Run the decider over a recorded scenario -- no camera, no robot, no clock.

    Uses the CSV's own timestamps, so this is deterministic and instant: the
    ideal way to check bucket transitions and the wind-down before wiring
    anything up.
    """
    with open(csv_path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        LOGGER.error("%s is empty", csv_path)
        return 1

    period = 1.0 / decide_hz
    next_tick = 0.0
    plays = 0

    print(f"{'t':>6}  {'phase':<12} {'slot':>4}  reasoning")
    print("-" * 100)
    for row in rows:
        t = float(row["t"])
        if t < next_tick:
            continue
        next_tick = t + period

        perc = Perception(
            present=True,
            x=float(row["x_true"]),
            y=float(row["y_true"]),
            distance=distance,   # the stimulus does not change apparent size
            motion=float(row["motion_true"]),
            ts=t,
        )
        slot = decider.decide(perc, None, now=t)
        decision = decider.explain()
        if slot is not None:
            plays += 1
            print(f"{t:6.2f}  {row['phase']:<12} {slot:>4}  {decision.describe()}")
        else:
            print(f"{t:6.2f}  {row['phase']:<12} {'-':>4}  {decision.describe()}")

    print("-" * 100)
    print(f"{plays} play() calls over {float(rows[-1]['t']):.1f}s")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slot-table", default="slots.json")
    parser.add_argument("--replay", metavar="GROUND_TRUTH.CSV",
                        help="drive the decider from a stimulus.py --ground-truth file")
    parser.add_argument("--decide-hz", type=float, default=2.0)
    parser.add_argument("--distance", type=float, default=0.35,
                        help="apparent distance to assume during replay")
    parser.add_argument("--dream", action="store_true",
                        help="rank candidates with DREAM-Chunk instead of score_slot()")
    parser.add_argument("--motion-map", help="the robot's MotionMap.csv (implies --dream)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    table = load_slot_table(args.slot_table)
    matcher = None
    if args.dream or args.motion_map:
        from dream import DreamConfig
        from pvector import load_chunk_dictionary
        matcher = ChunkMatcher(
            load_chunk_dictionary(motion_map=args.motion_map,
                                  slot_table=args.slot_table, axes=table.axes),
            DreamConfig(),
        )
    decider = Decider(table, matcher=matcher)

    if args.replay:
        return replay(args.replay, decider, args.decide_hz, args.distance)

    # No replay file: just describe the loaded table.
    print(f"axes: {list(table.axes)}")
    for direction in Direction:
        for amplitude in Amplitude:
            cell = table.cell(direction, amplitude)
            if cell:
                ids = ", ".join(f"{s.slot_id} ({s.desc})" for s in cell)
                print(f"  {direction.value:<7} {amplitude.value:<7} -> {ids}")
    print(f"settle ramp: {' -> '.join(str(s) for s in decider.settle_ramp)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

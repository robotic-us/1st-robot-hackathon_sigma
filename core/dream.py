#!/usr/bin/env python3
"""DREAM-Chunk: dreamed-state reactive action matching, on motion slots.

The paper's loop is: keep a dictionary of candidate action chunks, at test time
*dream* the future states each one would produce, and switch to whichever chunk
stays dynamically consistent with what the sensors are actually reporting -- no
fine-tuning, no gradient step, decided live.

Mapping that onto this robot, term by term:

    action chunk        a motion slot (1..50).  Already pre-recorded, already
                        safe, already on the SD card.  The dictionary is the
                        slot table.
    world model         the P-Vector quintic (see pvector.py).  Closed form, so
                        dreaming ten candidates over a 2 s horizon costs well
                        under a millisecond -- no diffusion model, no NPU budget.
    dreamed state       chunk.dream(measured_pose) -- anchored on where the arm
                        *actually is*, which is what makes it a prediction about
                        this moment rather than a replay of a nominal plan.
    sensor feedback     /phorce/feedback at 1 kHz: position_rad for consistency,
                        dob_a (disturbance observer) for external force.
    reactive matching   rank candidates by dynamic consistency + collision
                        resistance + task fit; play the winner.

One honest limitation, stated up front because it shapes the design: the paper
switches chunks *mid-execution*.  This robot cannot.  It plays one motion at a
time, has no queue, and the manual is explicit that a real (non-sim) play cannot
be cancelled once started.  So:

    * :class:`DreamMonitor` still watches divergence continuously at feedback
      rate, and still decides that a switch is warranted the instant the
      measured state leaves the dreamed tube.
    * :class:`ChunkMatcher` then acts on that decision at the earliest boundary
      the hardware allows -- it pre-empts the dwell timer instead of waiting it
      out, so the reaction lands one chunk-tail later rather than never.

That is the faithful implementation of the mechanism under this contract, and
the gap (mid-chunk pre-emption) is a hardware limit, not a shortcut.  Say so in
the demo; do not draw the ideal loop and imply it runs.

Run standalone::

    python3 core/dream.py --demo           # rank the whole dictionary from a pose
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

if __package__ in (None, ""):   # direct run: put the repo root on sys.path
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pvector import MotionChunk, load_chunk_dictionary

LOGGER = logging.getLogger("dream")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class DreamConfig:
    """Everything tunable about dreaming and matching."""

    # --- dreaming ---------------------------------------------------------- #
    dt: float = 0.02              # dream grid.  50 Hz is plenty: we are
    #                               comparing envelopes, not tracking a 1 kHz
    #                               servo loop, and a finer grid buys nothing.
    horizon_s: float = 2.0        # how far ahead to dream when ranking

    # --- ranking weights (cost, lower is better) --------------------------- #
    w_consistency: float = 1.0    # does the recent past agree with this chunk?
    w_resistance: float = 2.0     # external force x how far it would push
    w_continuity: float = 0.5     # would starting it here yank the arm?
    w_task: float = 0.8           # does its size match what we asked for?

    # --- divergence monitor ------------------------------------------------ #
    diverge_rad: float = 0.12     # measured-vs-dreamed error that counts as
    #                               "the dream broke".  Roughly 7 degrees: well
    #                               outside servo tracking error, well inside
    #                               a real collision.
    diverge_hold_s: float = 0.15  # ...sustained this long.  A single 1 kHz
    #                               sample over threshold is noise; 150 ms of it
    #                               is an obstacle.
    force_threshold_a: float = 0.35   # |dob_a| above this reads as real contact

    # --- safety ------------------------------------------------------------- #
    max_excursion_rad: float = 1.2    # veto any chunk that would swing further
    #                                   than this from the current pose


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChunkScore:
    """Why a candidate ranked where it did.  Every term is separately loggable."""

    slot_id: int
    total: float
    consistency: float
    resistance: float
    continuity: float
    task: float
    peak_excursion: float
    vetoed: bool = False
    veto_reason: str = ""

    def describe(self) -> str:
        if self.vetoed:
            return f"slot {self.slot_id}: VETOED ({self.veto_reason})"
        return (f"slot {self.slot_id}: cost={self.total:.4f} "
                f"[consist {self.consistency:.3f} | resist {self.resistance:.3f} | "
                f"contin {self.continuity:.3f} | task {self.task:.3f}] "
                f"peak={self.peak_excursion:.3f} rad")


@dataclass(frozen=True)
class DivergenceReport:
    """How badly the executing chunk's dream is failing right now."""

    active_slot: Optional[int]
    progress: float          # 0..1 through the chunk
    error_rad: float         # current |measured - dreamed|, worst axis
    rms_rad: float           # over the whole window so far
    diverged: bool           # sustained past the threshold
    external_force_a: float  # worst-axis |dob_a|
    samples: int

    def describe(self) -> str:
        if self.active_slot is None:
            return "no chunk executing"
        flag = " DIVERGED" if self.diverged else ""
        return (f"slot {self.active_slot} {self.progress * 100:5.1f}%  "
                f"err={self.error_rad:.4f} rms={self.rms_rad:.4f} "
                f"dob={self.external_force_a:.3f}A ({self.samples} samples){flag}")


# --------------------------------------------------------------------------- #
# The divergence monitor -- the "reactive" half
# --------------------------------------------------------------------------- #
class DreamMonitor:
    """Compares the executing chunk's dream against live feedback.

    Fed from the fast loop (feedback rate).  Stores only -- it never decides to
    play anything, in keeping with the two-rate rule.  The decider reads its
    verdict on the slow tick.
    """

    def __init__(self, config: Optional[DreamConfig] = None) -> None:
        self.cfg = config or DreamConfig()
        self._chunk: Optional[MotionChunk] = None
        self._start_rad: Optional[np.ndarray] = None
        self._t0: float = 0.0
        self._dream_t: Optional[np.ndarray] = None
        self._dream_y: Optional[np.ndarray] = None
        self._sq_error_sum = 0.0
        self._samples = 0
        self._over_since: Optional[float] = None
        self._last = DivergenceReport(None, 0.0, 0.0, 0.0, False, 0.0, 0)

    # -- lifecycle --------------------------------------------------------- #
    def begin(self, chunk: MotionChunk, start_rad: Sequence[float], now: float) -> None:
        """A chunk just started.  Dream it once, from the pose we launched at."""
        self._chunk = chunk
        self._start_rad = np.asarray(start_rad, dtype=float)
        self._t0 = now
        self._dream_t, self._dream_y = chunk.dream(
            start_rad, dt=self.cfg.dt, horizon_s=chunk.duration_s
        )
        self._sq_error_sum = 0.0
        self._samples = 0
        self._over_since = None
        # A fresh chunk gets a clean verdict -- this is the one place the
        # carried-over divergence from the previous chunk is discarded.
        self._last = DivergenceReport(chunk.slot_id, 0.0, 0.0, 0.0, False, 0.0, 0)
        LOGGER.debug("dreaming slot %d over %.2fs from %s",
                     chunk.slot_id, chunk.duration_s, np.round(self._start_rad, 3))

    def end(self) -> None:
        """Stop comparing, but *keep* the verdict.

        The decision loop runs slower than the chunk, so by the time it asks
        "did that diverge?" the chunk is already over.  Clearing the report here
        would throw away the very signal the reaction is built on -- the switch
        would never happen.  :meth:`begin` clears it instead.
        """
        self._chunk = None
        self._dream_t = self._dream_y = None
        self._over_since = None

    @property
    def active(self) -> bool:
        return self._chunk is not None

    # -- fast path ---------------------------------------------------------- #
    def update(
        self,
        measured_rad: Optional[Sequence[float]],
        now: float,
        external_force_a: Optional[Sequence[float]] = None,
    ) -> DivergenceReport:
        """One feedback frame.  Cheap: an interpolation and a subtraction."""
        if self._chunk is None or self._dream_y is None or measured_rad is None:
            return self._last

        elapsed = now - self._t0
        duration = max(1e-6, self._chunk.duration_s)
        progress = min(1.0, max(0.0, elapsed / duration))

        # Where the dream says we should be, right now.
        row = min(len(self._dream_y) - 1, int(round(elapsed / self.cfg.dt)))
        predicted = self._dream_y[row]
        measured = np.asarray(measured_rad, dtype=float)
        if measured.shape != predicted.shape:
            return self._last

        errors = np.abs(measured - predicted)
        worst = float(errors.max())
        self._sq_error_sum += float(np.mean(errors ** 2))
        self._samples += 1
        rms = math.sqrt(self._sq_error_sum / max(1, self._samples))

        # Sustained, not instantaneous: a lone spike is sensor noise.
        if worst >= self.cfg.diverge_rad:
            if self._over_since is None:
                self._over_since = now
        else:
            self._over_since = None
        diverged = (
            self._over_since is not None
            and (now - self._over_since) >= self.cfg.diverge_hold_s
        )

        force = 0.0
        if external_force_a is not None and len(external_force_a):
            force = float(np.max(np.abs(np.asarray(external_force_a, dtype=float))))

        self._last = DivergenceReport(
            active_slot=self._chunk.slot_id,
            progress=progress,
            error_rad=worst,
            rms_rad=rms,
            diverged=diverged,
            external_force_a=force,
            samples=self._samples,
        )
        return self._last

    def latest(self) -> DivergenceReport:
        return self._last


# --------------------------------------------------------------------------- #
# The matcher -- the "action matching" half
# --------------------------------------------------------------------------- #
class ChunkMatcher:
    """Ranks candidate chunks by dreaming each one from the measured pose."""

    def __init__(
        self,
        chunks: dict[int, MotionChunk],
        config: Optional[DreamConfig] = None,
    ) -> None:
        self.chunks = chunks
        self.cfg = config or DreamConfig()
        self._residual = np.zeros(0)   # ASAP-lite, see note_residual()
        if any(c.synthesised for c in chunks.values()):
            LOGGER.warning(
                "chunk dictionary is SYNTHESISED from slots.json -- the dream is a "
                "one-segment approximation. Point --motion-map at the robot's "
                "MotionMap.csv for the real trajectories."
            )

    # -- ASAP-lite ---------------------------------------------------------- #
    def note_residual(self, residual_rad: Sequence[float]) -> None:
        """Record the measured-minus-dreamed offset from the last chunk.

        This is ASAP's delta idea, reduced to what this contract allows.  ASAP
        adds its residual to the *actuator command*; the participant API has no
        such channel, so we instead fold the residual into the *world model* --
        the dream is corrected, the command is not.  It makes the next
        prediction honest about this robot's real friction and backlash, which
        is the part of ASAP we can actually run here.
        """
        residual = np.asarray(residual_rad, dtype=float)
        if self._residual.shape != residual.shape:
            self._residual = residual.copy()
        else:
            self._residual += 0.3 * (residual - self._residual)

    def corrected_dream(
        self, chunk: MotionChunk, start_rad: Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        t, y = chunk.dream(start_rad, dt=self.cfg.dt, horizon_s=self.cfg.horizon_s)
        if self._residual.shape == (y.shape[1],):
            # Ramp the correction in over the chunk: the residual accumulates
            # with travel, so applying it at t=0 would be wrong.
            ramp = np.linspace(0.0, 1.0, y.shape[0])[:, None]
            y = y + ramp * self._residual[None, :]
        return t, y

    # -- ranking ------------------------------------------------------------ #
    def rank(
        self,
        candidate_ids: Sequence[int],
        measured_rad: Optional[Sequence[float]],
        external_force_a: Optional[Sequence[float]] = None,
        target_amplitude: float = 0.0,
        amplitude_of: Optional[dict[int, float]] = None,
        recent_error_rad: float = 0.0,
    ) -> list[ChunkScore]:
        """Score every candidate.  Returns them sorted, cheapest first.

        With no joint data the pose-dependent terms drop out rather than being
        faked -- same rule the rest of this project follows.
        """
        scores: list[ChunkScore] = []
        force = (np.abs(np.asarray(external_force_a, dtype=float))
                 if external_force_a is not None and len(external_force_a) else None)

        for slot_id in candidate_ids:
            chunk = self.chunks.get(slot_id)
            if chunk is None:
                continue

            if measured_rad is None or len(measured_rad) != len(chunk.programs):
                # No trustworthy pose: rank on task fit alone.
                task = abs((amplitude_of or {}).get(slot_id, 0.0) - target_amplitude)
                scores.append(ChunkScore(
                    slot_id=slot_id, total=self.cfg.w_task * task,
                    consistency=0.0, resistance=0.0, continuity=0.0,
                    task=task, peak_excursion=0.0,
                ))
                continue

            t, y = self.corrected_dream(chunk, measured_rad)
            start = np.asarray(measured_rad, dtype=float)
            excursion = np.abs(y - start[None, :])
            peak = float(excursion.max())

            if peak > self.cfg.max_excursion_rad:
                scores.append(ChunkScore(
                    slot_id=slot_id, total=math.inf, consistency=0.0,
                    resistance=0.0, continuity=0.0, task=0.0, peak_excursion=peak,
                    vetoed=True,
                    veto_reason=f"peak {peak:.2f} rad > {self.cfg.max_excursion_rad} rad",
                ))
                continue

            # --- collision resistance ---------------------------------------
            # "the candidate whose external-force state and collision
            # resistance are lowest": weight each axis's travel by the external
            # force already measured on that axis.  Pushing hard into something
            # that is already pushing back is exactly what to avoid.
            if force is not None and force.shape == (y.shape[1],):
                per_axis_travel = excursion.max(axis=0)
                resistance = float(np.sum(force * per_axis_travel))
            else:
                resistance = 0.0

            # --- continuity -------------------------------------------------
            # Peak predicted speed: a chunk that would yank the arm away from
            # where it sits scores worse than one that eases out of it.
            speed = float(np.abs(np.diff(y, axis=0)).max() / max(1e-6, self.cfg.dt))
            continuity = speed

            # --- dynamic consistency ----------------------------------------
            # How well the *last* chunk's dream held up, scaled by how far this
            # candidate commits.  A model we have just seen fail should not be
            # trusted to justify a big move.
            consistency = recent_error_rad * peak

            task = abs((amplitude_of or {}).get(slot_id, 0.0) - target_amplitude)

            total = (
                self.cfg.w_consistency * consistency
                + self.cfg.w_resistance * resistance
                + self.cfg.w_continuity * continuity
                + self.cfg.w_task * task
            )
            scores.append(ChunkScore(
                slot_id=slot_id, total=total, consistency=consistency,
                resistance=resistance, continuity=continuity, task=task,
                peak_excursion=peak,
            ))

        # (cost, slot_id): the id keeps ties deterministic, which matters while
        # the dictionary is still synthesised and ties are common.
        scores.sort(key=lambda s: (s.total, s.slot_id))
        return scores

    def best(self, *args, **kwargs) -> Optional[ChunkScore]:
        ranked = [s for s in self.rank(*args, **kwargs) if not s.vetoed]
        return ranked[0] if ranked else None


# --------------------------------------------------------------------------- #
# Mock playback -- makes the mock robot a digital twin of the chunk dictionary
# --------------------------------------------------------------------------- #
class ChunkMockModel:
    """Drives :class:`~phorce_iface.MockRobot` along the same P-Vectors we dream.

    Without this the mock free-runs on a synthetic sine, so the dream diverges
    on every chunk and the divergence monitor cries wolf continuously -- which
    makes it impossible to tell a real fault from the mock being the mock.

    With it, the two agree to within sensor noise, and the *only* thing that
    breaks the agreement is an injected fault (``--mock-jam-after``).  That is
    what makes the divergence test meaningful rather than decorative.

    Note this is deliberately the *same* model on both sides, so a clean run
    proves the plumbing, not the physics.  Real sim-to-real error only shows up
    against the actual robot -- that is precisely the gap ASAP's residual is
    for, and why note_residual() exists.
    """

    def __init__(self, chunks: dict[int, MotionChunk], dt: float = 0.005) -> None:
        self.chunks = chunks
        self.dt = dt
        self._cache: dict[tuple[int, tuple[float, ...]], np.ndarray] = {}

    def duration_s(self, slot_id: int) -> float:
        chunk = self.chunks.get(slot_id)
        return chunk.duration_s if chunk is not None else 0.0

    def pose_at(
        self, slot_id: int, start_rad: Sequence[float], elapsed_s: float
    ) -> Optional[list[float]]:
        chunk = self.chunks.get(slot_id)
        if chunk is None or len(start_rad) != len(chunk.programs):
            return None
        key = (slot_id, tuple(round(v, 6) for v in start_rad))
        table = self._cache.get(key)
        if table is None:
            _, table = chunk.dream(start_rad, dt=self.dt, horizon_s=chunk.duration_s)
            self._cache[key] = table
            if len(self._cache) > 64:       # bounded: one entry per launch pose
                self._cache.pop(next(iter(self._cache)))
        row = min(len(table) - 1, max(0, int(round(elapsed_s / self.dt))))
        return [float(v) for v in table[row]]

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def run_demo(args: argparse.Namespace) -> int:
    chunks = load_chunk_dictionary(motion_map=args.motion_map, slot_table=args.slot_table)
    if not chunks:
        print("no chunks to rank")
        return 1
    matcher = ChunkMatcher(chunks, DreamConfig())

    width = len(next(iter(chunks.values())).programs)
    pose = [float(v) for v in args.pose.split(",")] if args.pose else [0.0] * width
    force = [float(v) for v in args.force.split(",")] if args.force else None

    print(f"pose  = {pose}")
    print(f"dob_a = {force if force else '(none)'}")
    print(f"recent dream error = {args.recent_error:.3f} rad\n")
    for score in matcher.rank(sorted(chunks), pose, external_force_a=force,
                              recent_error_rad=args.recent_error):
        print("  " + score.describe())
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demo", action="store_true", help="rank the dictionary from a pose")
    parser.add_argument("--motion-map", help="the robot's MotionMap.csv")
    parser.add_argument("--slot-table", default="slots.json")
    parser.add_argument("--pose", help="current joint pose, comma-separated radians")
    parser.add_argument("--force", help="per-axis |dob_a|, comma-separated amps")
    parser.add_argument("--recent-error", type=float, default=0.0,
                        help="last chunk's dream error, rad (drives the consistency term)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)-7s %(name)s: %(message)s",
    )
    if args.demo:
        return run_demo(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

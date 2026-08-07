#!/usr/bin/env python3
"""P-Vector: the analytic world model that DREAM-Chunk needs.

DREAM-Chunk asks for a *light* world model that can predict, at test time, the
future states a candidate action chunk would produce.  The obvious reading is
"train a small dynamics net", and the guideline explicitly warns that a
diffusion-class model will not fit the board's compute budget.

We do not need one.  The robot's motions are already stored as P-Vectors, and a
P-Vector *is* a closed-form trajectory:

    y(tau) = a0 + a2*tau^2 + a3*tau^3 + a4*tau^4 + a5*tau^5

    a0 = y0
    a2 = 0.5*s0            * (yd - y0)
    a3 = (10 - 1.5*s0 + 0.5*sd) * (yd - y0)
    a4 = (-15 + 1.5*s0 - sd)    * (yd - y0)
    a5 = (6 - 0.5*s0 + 0.5*sd)  * (yd - y0)

    tau = k / L_traj  in [0, 1]

So "dreaming" a chunk costs one polynomial evaluation per axis per sample --
microseconds, exactly, and with no training data at all.  That is the whole
trick: the world model was handed to us in the motion format.

Where the numbers come from:

``MotionMap.csv`` on the robot's SD card is the source: one row per (motion
slot, axis), the P-Vector columns being that axis's segment list.

Run standalone::

    python3 core/pvector.py --motion-map MotionMap.csv  # same, from the real file
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

LOGGER = logging.getLogger("pvector")

# The pcm records teaching data at 1 kHz, and L_traj is a count of those
# samples.  In the P-Vector deck, P(1) = [200, 1000, 0, 5] reaches its target at
# t = 1.0 s, which pins this down.
PCM_SAMPLE_HZ = 1000.0

# P-Vector positions are int16 in "output-axis degrees" per the spec sheet, but
# the worked example plots yd=200 as 1.0 on its position axis, so there is a
# fixed scale between the stored integer and a physical degree that the deck
# does not state outright.  Keep it in one named place: calibrate it once
# against real feedback (see --calibrate-scale) rather than sprinkling magic
# numbers through the matcher.
UNITS_PER_DEG = 1.0


def units_to_rad(value: float, units_per_deg: float = UNITS_PER_DEG) -> float:
    """P-Vector storage units -> radians (what /phorce/feedback reports)."""
    return math.radians(value / units_per_deg)


def rad_to_units(value: float, units_per_deg: float = UNITS_PER_DEG) -> float:
    return math.degrees(value) * units_per_deg


# --------------------------------------------------------------------------- #
# One segment
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PVector:
    """One unit trajectory: rest at ``y0`` -> rest at ``yd``.

    Both endpoints have zero velocity by construction (that is why the
    polynomial has no linear term), which is what lets segments be concatenated
    without a velocity discontinuity at the join.
    """

    yd: float        # target position, storage units
    l_traj: int      # length in PCM samples
    s0: float = 0.0  # acceleration shaping, int8 range
    sd: float = 0.0  # deceleration shaping, int8 range

    def __post_init__(self) -> None:
        if self.l_traj <= 0:
            raise ValueError(f"P-Vector needs a positive L_traj, got {self.l_traj}")

    @property
    def duration_s(self) -> float:
        return self.l_traj / PCM_SAMPLE_HZ

    def coefficients(self, y0: float) -> tuple[float, float, float, float, float, float]:
        """``(a0, a1, a2, a3, a4, a5)``.  ``a1`` is always 0 -- see the class doc."""
        span = self.yd - y0
        return (
            y0,
            0.0,
            0.5 * self.s0 * span,
            (10.0 - 1.5 * self.s0 + 0.5 * self.sd) * span,
            (-15.0 + 1.5 * self.s0 - self.sd) * span,
            (6.0 - 0.5 * self.s0 + 0.5 * self.sd) * span,
        )

    def evaluate(self, y0: float, tau: float | np.ndarray) -> np.ndarray:
        """Position at normalised progress ``tau`` in [0, 1]."""
        a0, _, a2, a3, a4, a5 = self.coefficients(y0)
        t = np.clip(np.asarray(tau, dtype=float), 0.0, 1.0)
        return a0 + t * t * (a2 + t * (a3 + t * (a4 + t * a5)))

    def velocity(self, y0: float, tau: float | np.ndarray) -> np.ndarray:
        """d y / d tau.  Divide by ``duration_s`` for units per second."""
        _, _, a2, a3, a4, a5 = self.coefficients(y0)
        t = np.clip(np.asarray(tau, dtype=float), 0.0, 1.0)
        return t * (2.0 * a2 + t * (3.0 * a3 + t * (4.0 * a4 + t * 5.0 * a5)))

    def sample(self, y0: float, dt: float) -> np.ndarray:
        """Positions on a uniform ``dt`` grid covering the whole segment."""
        n = max(2, int(round(self.duration_s / max(1e-6, dt))) + 1)
        return self.evaluate(y0, np.linspace(0.0, 1.0, n))

    @classmethod
    def parse(cls, text: str) -> Optional["PVector"]:
        """Parse one ``"yd, L_traj, s0, sd"`` cell from MotionMap.csv.

        Empty cells and ``-`` mean "this axis has no further segment", which is
        normal: axes finish at different times.
        """
        cleaned = text.strip()
        if not cleaned or cleaned in {"-", "--"}:
            return None
        parts = [p.strip() for p in cleaned.split(",") if p.strip() != ""]
        if len(parts) < 2:
            return None
        try:
            values = [float(p) for p in parts]
        except ValueError:
            LOGGER.debug("skipping unparseable P-Vector cell %r", text)
            return None
        # Real files carry trailing zeros (e.g. "2389.4,1500,0,0,0.0,0.0").
        yd, l_traj = values[0], int(round(values[1]))
        s0 = values[2] if len(values) > 2 else 0.0
        sd = values[3] if len(values) > 3 else 0.0
        if l_traj <= 0:
            return None
        return cls(yd=yd, l_traj=l_traj, s0=s0, sd=sd)


# --------------------------------------------------------------------------- #
# One axis's program, and one whole chunk
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AxisProgram:
    """The ordered segments one axis plays during one motion."""

    axis_index: int
    segments: tuple[PVector, ...]

    @property
    def duration_s(self) -> float:
        return sum(s.duration_s for s in self.segments)

    def dream(self, y0: float, dt: float, horizon_s: Optional[float] = None) -> np.ndarray:
        """Predicted positions on a ``dt`` grid, starting from ``y0``.

        Segments chain: each one begins where the previous ended.  Past the end
        of the program the axis holds its final value -- that is what the robot
        does while slower axes finish, and pretending otherwise would make the
        divergence monitor fire at every chunk tail.
        """
        span = self.duration_s if horizon_s is None else horizon_s
        n = max(2, int(round(span / max(1e-6, dt))) + 1)
        out = np.empty(n, dtype=float)

        cursor = y0
        sample = 0
        elapsed = 0.0
        for segment in self.segments:
            end = elapsed + segment.duration_s
            while sample < n and (sample * dt) < end - 1e-12:
                tau = ((sample * dt) - elapsed) / segment.duration_s
                out[sample] = segment.evaluate(cursor, tau)
                sample += 1
            cursor = segment.evaluate(cursor, 1.0).item()
            elapsed = end
        out[sample:] = cursor  # hold after the program ends
        return out


@dataclass(frozen=True)
class MotionChunk:
    """One motion slot, as a dictionary entry DREAM-Chunk can dream about."""

    slot_id: int
    name: str
    programs: tuple[AxisProgram, ...]
    units_per_deg: float = UNITS_PER_DEG

    @property
    def axes(self) -> tuple[int, ...]:
        return tuple(p.axis_index for p in self.programs)

    @property
    def duration_s(self) -> float:
        return max((p.duration_s for p in self.programs), default=0.0)

    def dream(
        self,
        start_rad: Sequence[float],
        dt: float = 0.02,
        horizon_s: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict this chunk's joint trajectory from a measured start pose.

        ``start_rad`` is where the arm *actually is* right now, in radians, in
        the same order as :attr:`axes`.  Anchoring the dream on measured state
        rather than on the chunk's nominal start is the point: it is what makes
        the prediction answer "what would happen if I played this, from here?"

        Returns ``(t, y)`` with ``y`` shaped ``(len(t), n_axes)``, in radians.
        """
        if len(start_rad) != len(self.programs):
            raise ValueError(
                f"chunk {self.slot_id} has {len(self.programs)} axes but got "
                f"{len(start_rad)} start positions"
            )
        span = self.duration_s if horizon_s is None else horizon_s
        n = max(2, int(round(span / max(1e-6, dt))) + 1)
        t = np.arange(n, dtype=float) * dt

        y = np.empty((n, len(self.programs)), dtype=float)
        for col, program in enumerate(self.programs):
            y0_units = rad_to_units(start_rad[col], self.units_per_deg)
            traj = program.dream(y0_units, dt, horizon_s=span)
            y[:, col] = np.vectorize(units_to_rad)(traj[:n], self.units_per_deg)
        return t, y

    def nominal_end_rad(self) -> list[float]:
        """Where this chunk finishes if started from its own nominal origin."""
        out = []
        for program in self.programs:
            cursor = 0.0
            for segment in program.segments:
                cursor = segment.evaluate(cursor, 1.0).item()
            out.append(units_to_rad(cursor, self.units_per_deg))
        return out

    def peak_excursion_rad(self, start_rad: Sequence[float], dt: float = 0.02) -> float:
        """Largest distance any axis travels from its start during the chunk.

        The collision-resistance term uses this: a big swing into a space where
        the disturbance observer already reports an external force is exactly
        the move DREAM-Chunk is supposed to veto.
        """
        _, y = self.dream(start_rad, dt=dt)
        return float(np.max(np.abs(y - np.asarray(start_rad, dtype=float))))


# --------------------------------------------------------------------------- #
# Loading a dictionary
# --------------------------------------------------------------------------- #
def load_motion_map(
    path: str,
    axes: Optional[Sequence[int]] = None,
    units_per_deg: float = UNITS_PER_DEG,
) -> dict[int, MotionChunk]:
    """Parse the robot's ``MotionMap.csv`` into a chunk dictionary.

    Expected columns (the deck's layout): ``MS ID``, ``MS NAME``, ``MD ID``,
    ``C-Vector``, then one column per P-Vector slot.  Blank ``MS ID`` cells
    inherit from the row above -- the file merges those cells visually.
    """
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"{path} is empty")

    header = [c.strip().lower().replace(" ", "").replace("-", "") for c in rows[0]]

    def column(*names: str) -> Optional[int]:
        for name in names:
            if name in header:
                return header.index(name)
        return None

    ms_col = column("msid", "motionsetid", "msi d")
    name_col = column("msname", "name")
    md_col = column("mdid", "axis", "axisid")
    if ms_col is None or md_col is None:
        raise ValueError(
            f"{path}: could not find 'MS ID' / 'MD ID' columns in header {rows[0]!r}"
        )
    first_pv = max(c for c in (ms_col, name_col, md_col, column("cvector")) if c is not None) + 1

    collected: dict[int, dict[int, list[PVector]]] = {}
    names: dict[int, str] = {}
    last_ms: Optional[int] = None

    for row in rows[1:]:
        if not any(cell.strip() for cell in row):
            continue
        raw_ms = row[ms_col].strip() if ms_col < len(row) else ""
        if raw_ms:
            try:
                last_ms = int(float(raw_ms))
            except ValueError:
                continue
        if last_ms is None:
            continue
        if name_col is not None and name_col < len(row) and row[name_col].strip():
            names.setdefault(last_ms, row[name_col].strip())

        raw_md = row[md_col].strip() if md_col < len(row) else ""
        axis_index = _parse_axis_index(raw_md)
        if axis_index is None:
            continue

        segments = [pv for pv in (PVector.parse(c) for c in row[first_pv:]) if pv is not None]
        if segments:
            collected.setdefault(last_ms, {})[axis_index] = segments

    chunks: dict[int, MotionChunk] = {}
    for slot_id, per_axis in collected.items():
        wanted = tuple(axes) if axes is not None else tuple(sorted(per_axis))
        programs = tuple(
            AxisProgram(axis_index=a, segments=tuple(per_axis.get(a, ())))
            for a in wanted
            if per_axis.get(a)
        )
        if not programs:
            LOGGER.warning("slot %d has no segments for axes %s -- skipped", slot_id, wanted)
            continue
        chunks[slot_id] = MotionChunk(
            slot_id=slot_id,
            name=names.get(slot_id, f"slot {slot_id}"),
            programs=programs,
            units_per_deg=units_per_deg,
        )

    LOGGER.info("loaded %d chunks from %s", len(chunks), path)
    return chunks


def _parse_axis_index(text: str) -> Optional[int]:
    """``MD5`` / ``5`` / ``0x06`` -> a zero-based feedback axis index.

    The wiring sheet numbers the actuators 0x02..0x0D while /phorce/feedback
    indexes axis[0..11], so a hex phact id is shifted down by 2.  ``MD``
    numbering in the motion map is 1-based.
    """
    cleaned = text.strip().upper()
    if not cleaned:
        return None
    if cleaned.startswith("MD"):
        try:
            return int(cleaned[2:]) - 1
        except ValueError:
            return None
    if cleaned.startswith("0X"):
        try:
            return int(cleaned, 16) - 2
        except ValueError:
            return None
    try:
        return int(float(cleaned))
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def describe(chunks: dict[int, MotionChunk]) -> None:
    if not chunks:
        print("no chunks")
        return
    print(f"{len(chunks)} chunks (MotionMap.csv)\n")
    print(f"{'slot':>4}  {'dur(s)':>7}  {'axes':>12}  {'peak(rad)':>9}  name")
    print("-" * 78)
    for slot_id in sorted(chunks):
        chunk = chunks[slot_id]
        start = [0.0] * len(chunk.programs)
        print(f"{slot_id:>4}  {chunk.duration_s:>7.3f}  {str(list(chunk.axes)):>12}  "
              f"{chunk.peak_excursion_rad(start):>9.4f}  {chunk.name}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--motion-map", required=True,
                        help="the robot's MotionMap.csv")
    parser.add_argument("--units-per-deg", type=float, default=UNITS_PER_DEG)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    describe(load_motion_map(args.motion_map, units_per_deg=args.units_per_deg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

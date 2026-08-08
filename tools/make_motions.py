#!/usr/bin/env python3
"""core/cradle.py -> motion_NN.csv, the MotionMap format pcm and simulator read.

Loader rules: the MS ID inside the file must match the filename number; ``MD
ID`` is 1-based (``MD1`` = feedback ``axis[0]``); a P-Vector cell is
``"yd, L_traj, s0, sd"`` with ``yd`` an *absolute* target in degrees (int16,
-360..360) and ``L_traj`` in 1 kHz samples.  Schedules are millimetres of plate
travel solved into crank angles through the rig IK (core/rig.py), one solve per
P-Vector, so the four cranks carry different magnitudes.

Usage::

    python3 tools/make_motions.py --library     # write ./motions_m50 from M01-M50
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

if __package__ in (None, ""):   # direct run: put the repo root on sys.path
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def slot_name(entry: dict, slot_id: int) -> str:
    """A short upper-case name; `phorce list` shows this."""
    direction = str(entry.get("direction", "")).upper()
    amplitude = str(entry.get("amplitude", "")).upper()
    return f"{direction}_{amplitude}" if direction and amplitude else f"SLOT_{slot_id}"


def write_motion(
    path: Path, slot_id: int, name: str, end_pose_rad: list[float],
    l_traj: int, s0: float, sd: float,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["MS ID", "MS NAME", "MD ID", "C-Vector", "P-Vector 1"])
        for i, target_rad in enumerate(end_pose_rad):
            writer.writerow([
                slot_id, name, f"MD{i + 1}", 0,
                f"{math.degrees(target_rad):.2f},{l_traj},{s0:g},{sd:g}",
            ])


# --------------------------------------------------------------------------- #
# --library: the evidence report's M01-M50, compiled to pcm slots
# --------------------------------------------------------------------------- #
AXES = 4
REFERENCE = (0.5, 10.0)  # (Hz, mm) sway the C0 transition episodes demonstrate

# Three channels (two five-bar linkages, see core/rig.py):
#   sway   all four cranks the same way          -> horizontal, ~4.3 mm/deg
#   heave  each pair's two cranks opposed        -> VERTICAL, ~3.1 mm/deg
#   pitch  one pair up while the other goes down -> a see-saw

# Which channel each library axis rides.  AP has no translation axis (every
# joint turns about Y); rendered as pitch rather than dropped.  See CLAUDE.md.
CHANNEL = {"ML": "sway", "Z": "heave", "AP": "pitch", "": "sway"}


def per_axis(schedule: list[tuple[float, int]], channel: str = "sway",
             second: list[tuple[float, int]] | None = None,
             second_channel: str = "pitch") -> list[list[tuple[float, int]]]:
    """A plate-travel schedule (mm) -> one crank-angle row per axis, MD1..MD4.

    Each P-Vector is solved through the linkage; ``second`` adds a simultaneous
    channel on the same timing grid (the diagonals).
    """
    if second is not None:
        assert len(schedule) == len(second), "channels must share a timing grid"
    rows: list[list[tuple[float, int]]] = [[] for _ in range(AXES)]
    for i, (mm, ms) in enumerate(schedule):
        kw = {channel + "_mm": mm}
        if second is not None:
            kw[second_channel + "_mm"] = kw.get(second_channel + "_mm", 0.0) + second[i][0]
        for axis, rad in enumerate(cradle_axis_degrees(**kw)):
            rows[axis].append((rad, ms))
    return rows


def scaled(schedule: list[tuple[float, int]], k: float) -> list[tuple[float, int]]:
    """Same timing, travel multiplied -- how one channel is scaled."""
    return [(mm * k, ms) for mm, ms in schedule]


def cradle_axis_degrees(sway_mm: float = 0.0, heave_mm: float = 0.0,
                        pitch_mm: float = 0.0) -> list[float]:
    """The four crank angles in **degrees**, which is what a P-Vector carries."""
    from core.rig import axis_angles
    return [math.degrees(v)
            for v in axis_angles(sway_mm=sway_mm, heave_mm=heave_mm,
                                 pitch_mm=pitch_mm)]


def sway_schedule(amp_mm: float, f_hz: float, ramp_s: float = 5.0,
                  hold_cycles: int = 2,
                  taper_s: float | None = None) -> list[tuple[float, int]]:
    """One amplitude entry per half-cycle: (peak plate travel mm, half-cycle ms).

    Envelope smoothsteps up over ``ramp_s``, holds, down over ``taper_s``
    (defaults to ``ramp_s``; 0 skips a side).
    """
    from core.cradle import smoothstep

    half_ms = max(1, round(500.0 / f_hz))
    if taper_s is None:
        taper_s = ramp_s
    n_up = math.ceil(ramp_s * 2.0 * f_hz)
    n_down = math.ceil(taper_s * 2.0 * f_hz)
    return ([(amp_mm * smoothstep((j + 1) / n_up), half_ms) for j in range(n_up)]
            + [(amp_mm, half_ms)] * (hold_cycles * 2)
            + [(amp_mm * smoothstep(1.0 - (j + 1) / n_down), half_ms)
               for j in range(n_down)])


def compile_halves(schedule: list[tuple[float, int]]) -> list[tuple[float, int]]:
    """Alternate the swing direction, then park: schedule -> P-Vector targets."""
    segments, sign = [], 1.0
    for theta, half_ms in schedule:
        segments.append((sign * theta, half_ms))
        sign = -sign
    segments.append((0.0, schedule[-1][1]))
    return segments


def library_episode(m) -> list[list[tuple[float, int]]]:
    """One library entry -> its slot trajectory, one segment list per axis."""
    ref_f, ref_mm = REFERENCE
    if m.kind == "static":
        return per_axis([(0.0, 1000)])
    if m.kind == "pause":
        return per_axis([(0.0, 5000)])
    if m.kind == "soft_start":       # M03/M04: the ramp is the point
        return per_axis(compile_halves(
            sway_schedule(ref_mm, ref_f, ramp_s=m.ramp_s,
                          hold_cycles=1, taper_s=5.0)))
    if m.kind == "taper":            # M05/M06/M07: the wind-down is the point
        return per_axis(compile_halves(
            sway_schedule(ref_mm, ref_f, hold_cycles=1, taper_s=m.ramp_s)))
    if m.kind == "micro_resume":     # M08: the reference sway at 50%
        return per_axis(compile_halves(
            sway_schedule(0.5 * ref_mm, ref_f, ramp_s=m.ramp_s,
                          hold_cycles=1, taper_s=5.0)))
    if m.kind == "adaptive_a":       # A 5 -> 10 -> 15 mm blocks
        return per_axis(compile_halves(
            sway_schedule(5.0, m.f_hz, taper_s=0.0)
            + sway_schedule(10.0, m.f_hz, ramp_s=0.0, taper_s=0.0)
            + sway_schedule(15.0, m.f_hz, ramp_s=0.0)))
    if m.kind == "adaptive_f":       # f 0.3 -> 0.5 -> 0.7 Hz blocks
        return per_axis(compile_halves(
            sway_schedule(m.a_mm, 0.3, taper_s=0.0)
            + sway_schedule(m.a_mm, 0.5, ramp_s=0.0, taper_s=0.0)
            + sway_schedule(m.a_mm, 0.7, ramp_s=0.0)))
    if m.kind == "pseudo_walk":      # band-limited 0.4-0.7: its centre
        return per_axis(compile_halves(sway_schedule(m.a_mm, 0.55)))

    base = compile_halves(sway_schedule(m.a_mm, m.f_hz))
    if m.kind == "diagonal":
        # v = a/sqrt(2): sway and pitch in phase, sign picks the lean.
        v = 1.0 / math.sqrt(2.0)
        return per_axis(scaled(base, m.sign * v), "sway",
                        scaled(base, v), "pitch")
    # Quadrature shapes keep their primary amplitude on their axis's channel;
    # all eight are R-grade, so nothing automatic reaches them.
    return per_axis(base, CHANNEL.get(m.axis, "sway"))


def write_slot(path: Path, slot_id: int, name: str,
               axis_segments: list[list[tuple[float, int]]]) -> None:
    """One CSV per slot, one MD row per axis -- the rows may now differ."""
    width = len(axis_segments[0])
    assert all(len(s) == width for s in axis_segments), \
        f"{path.name}: MD rows must be column-aligned"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["MS ID", "MS NAME", "MD ID", "C-Vector"]
                        + [f"P-Vector {i + 1}" for i in range(width)])
        for axis, segments in enumerate(axis_segments):
            writer.writerow([slot_id, name, f"MD{axis + 1}", 0]
                            + [f"{deg:.2f},{ms},0,0" for deg, ms in segments])


def build_library(out: Path) -> int:
    from core.rig import LEVER_M as lever_m
    from core.cradle import A_HARD_MM, LIBRARY
    # Second guard, in crank-angle space: ceiling taken over all three channels
    # -- heave/pitch cost more angle per mm than sway (~3.1 vs ~4.3 mm/deg).
    hard_deg = max(abs(v)
                   for chan in ("sway_mm", "heave_mm", "pitch_mm")
                   for v in cradle_axis_degrees(**{chan: A_HARD_MM}))
    out.mkdir(exist_ok=True)
    for stale in out.glob("motion_*.csv"):
        stale.unlink()

    for m in LIBRARY:
        slot_id = m.slot
        axis_segments = library_episode(m)
        peak = max(abs(deg) for segs in axis_segments for deg, _ in segs)
        assert peak <= hard_deg + 1e-6, f"{m.id} breaks the envelope"
        for axis, segs in enumerate(axis_segments):
            assert abs(segs[-1][0]) < 1e-9, \
                f"{m.id} MD{axis + 1} must end parked at rest"
        segments = axis_segments[0]
        duration = sum(ms for _, ms in segments) / 1000.0
        # Channel read back off the rows: a pair agrees on sway, opposes on
        # heave/pitch.
        pair = [round(a[0] + b[0], 6) for a, b in
                zip(axis_segments[0], axis_segments[1])]
        chan = "sway" if all(abs(v) > 1e-9 or abs(a[0]) < 1e-9
                             for v, a in zip(pair, axis_segments[0])) else "    "
        opposed = any(a[0] * b[0] < -1e-12
                      for a, b in zip(axis_segments[0], axis_segments[1]))
        chan = "opposed" if opposed else "sway   "
        path = out / f"motion_{slot_id:02d}.csv"
        write_slot(path, slot_id, m.name, axis_segments)
        print(f"  {path}  {m.name:<24} [{m.grade}] peak {peak:5.2f} deg  "
              f"{len(segments):>2} segs  {duration:6.1f}s  {chan}")

    print(f"\n50 motion slots in {out}/  (1 deg = "
          f"{lever_m * 1000 * math.radians(1):.1f} mm of sway)")
    print(f"launch the simulator with them:\n  ./sim.sh {out}")
    return 0


def build_n34(out: Path) -> int:
    """Compile the team's N01-N34 system (docs/motion-system.png) to slots.

    Each entry is sampled from the live MotionEngine every 250 ms, each sample
    solved through the rig IK; ramps in over 4 s, tapers via N01, ends parked.
    """
    from core.cradle import A_HARD_MM, VIBE_MM, N_LIBRARY, MotionEngine
    hard_deg = max(abs(v)
                   for chan in ("sway_mm", "heave_mm", "pitch_mm")
                   for v in cradle_axis_degrees(**{chan: A_HARD_MM + VIBE_MM}))
    out.mkdir(exist_ok=True)
    for stale in out.glob("motion_*.csv"):
        stale.unlink()

    sample_ms = 250
    for m in N_LIBRARY:
        if m.kind == "static":                       # N01: the parked state
            rows = [[(0.0, 1000)] for _ in range(AXES)]
        else:
            rows = [[] for _ in range(AXES)]
            eng = MotionEngine()
            eng.command(m.id, 0.0, ramp_s=4.0)
            t, t_taper = 0.0, 4.0 + (26.0 if m.decay else 12.0)
            while t < t_taper + 4.5:
                if t >= t_taper and eng.active and not eng.tapering:
                    eng.command("N01", t, ramp_s=4.0)   # the system's own stop
                for _ in range(5):                      # engine clamps dt<=0.1
                    t += sample_ms / 5000.0
                    eng.tick(t)
                _, ml, z = eng.offsets_mm()
                for axis, deg in enumerate(
                        cradle_axis_degrees(sway_mm=ml, heave_mm=z)):
                    rows[axis].append((deg, sample_ms))
            for r in rows:                              # end exactly parked
                r.append((0.0, 400))
        peak = max(abs(deg) for segs in rows for deg, _ in segs)
        assert peak <= hard_deg + 1e-6, f"{m.id} breaks the crank envelope"
        path = out / f"motion_{m.slot:02d}.csv"
        write_slot(path, m.slot, m.name, rows)
        duration = sum(ms for _, ms in rows[0]) / 1000.0
        print(f"  {path}  {m.name:<16} [{m.grade}] peak {peak:5.2f} deg  "
              f"{len(rows[0]):>3} segs  {duration:5.1f}s")

    print(f"\n34 N-system slots in {out}/\nlaunch the simulator with them:"
          f"\n  ./sim.sh {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None,
                        help="output dir (default: motions, or motions_m50 with --library)")
    parser.add_argument("--library", action="store_true",
                        help="compile core/cradle.py's M01-M50 to motions_m50/")
    parser.add_argument("--n34", action="store_true",
                        help="compile the N01-N34 system (docs/motion-system.png) "
                             "to motions_n34/ by sampling the live engine")
    args = parser.parse_args(argv)
    if args.n34:
        return build_n34(Path(args.out or "motions_n34"))
    if args.library:
        return build_library(Path(args.out or "motions_m50"))
    parser.error("nothing to compile: pass --library (M01-M50) or --n34.\n"
                 "  (the old slots.json path went with the tag stack in 2026-08)")


if __name__ == "__main__":
    raise SystemExit(main())

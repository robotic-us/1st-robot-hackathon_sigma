#!/usr/bin/env python3
"""slots.json -> motion_NN.csv, the format the pcm and the simulator both read.

Discovered empirically against the shipped simulator (the error messages are
good teachers).  A motion file is one CSV per slot, in the MotionMap layout from
the P-Vector guide:

    MS ID,MS NAME,MD ID,C-Vector,P-Vector 1
    1,LEFT_SMALL,MD1,0,"-14.32,1600,0,0"
    ...one row per axis...

Rules the loader enforces, learned the hard way:

* **The MS ID inside the file must match the filename number.** ``motion_01.csv``
  with MS ID 0 is rejected with "파일명과 내용의 MS_ID 불일치" -- and the message
  says the real pcm rejects it too, not just the sim.
* ``MD ID`` is 1-based (``MD1`` = feedback ``axis[0]``).
* A P-Vector cell is ``"yd, L_traj, s0, sd"``: target position in **degrees**
  (int16, -360..360 per the spec sheet), length in 1 kHz samples, then the
  acceleration and deceleration shaping terms.

``yd`` is an *absolute* target, not a delta -- the unit trajectory runs from
wherever the axis currently is to ``yd``.  So we write ``end_pose`` converted to
degrees, and the arm reaches the same posture regardless of where it started.

This exists so we can populate a catalog **without a robot and without teaching
anything** -- which is what makes the simulator useful and gives DREAM-Chunk a
dictionary with more than one entry per cell.  Motions taught for real in phorce
Studio will overwrite these; treat them as scaffolding, not as the final poses.

**--library** compiles the evidence report's M01-M50 (core/cradle.py) into 50
pcm slots instead -- the full-simulation catalog, MS ID k = M{k:02d}.  A sway
becomes a chain of rest-to-rest quintic half-cycles (both a sine and a quintic
have zero velocity at the extremes), with the amplitude envelope ramped in and
out so every slot starts soft and ends parked at rest.  Millimetres become arm
degrees through the CAD lever (apps/demo.py's PIVOT - AXIS0).  Notes for the
honest small print: this rig sways one horizontal axis, so ML and AP land on
the same joints, two-axis research modes (diagonal/ellipse/circle/Lissajous)
are compiled at their primary amplitude, pseudo-walk as its band centre, and
the Z modes as flat holds -- there is no vertical DOF to play them on.  C0
transition commands compile as reference episodes around a 0.5 Hz / A10 sway
(soft starts ramp in over their 30/60 s, tapers ramp out over theirs).

Usage::

    python3 tools/make_motions.py               # write ./motions from slots.json
    python3 tools/make_motions.py --library     # write ./motions_m50 from M01-M50
    python3 tools/make_motions.py --duration-s 2.0   # slower slots.json motions
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
AXES = 4                 # the parallelogram: every axis gets the same command
REFERENCE = (0.5, 10.0)  # (Hz, mm) sway the C0 transition episodes demonstrate


def sway_schedule(theta_deg: float, f_hz: float, ramp_s: float = 5.0,
                  hold_cycles: int = 2,
                  taper_s: float | None = None) -> list[tuple[float, int]]:
    """One amplitude entry per half-cycle: (peak deg, half-cycle ms).

    The amplitude envelope smoothsteps up over ``ramp_s``, holds, and back
    down over ``taper_s`` (defaults to ``ramp_s``; 0 skips a side) -- the same
    shape the live MotionEngine plays, coarsened to half-cycle resolution.
    """
    from core.cradle import smoothstep

    half_ms = max(1, round(500.0 / f_hz))
    if taper_s is None:
        taper_s = ramp_s
    n_up = math.ceil(ramp_s * 2.0 * f_hz)
    n_down = math.ceil(taper_s * 2.0 * f_hz)
    return ([(theta_deg * smoothstep((j + 1) / n_up), half_ms) for j in range(n_up)]
            + [(theta_deg, half_ms)] * (hold_cycles * 2)
            + [(theta_deg * smoothstep(1.0 - (j + 1) / n_down), half_ms)
               for j in range(n_down)])


def compile_halves(schedule: list[tuple[float, int]]) -> list[tuple[float, int]]:
    """Alternate the swing direction, then park: schedule -> P-Vector targets."""
    segments, sign = [], 1.0
    for theta, half_ms in schedule:
        segments.append((sign * theta, half_ms))
        sign = -sign
    segments.append((0.0, schedule[-1][1]))
    return segments


def library_episode(m, to_deg) -> list[tuple[float, int]]:
    """One library entry -> its slot trajectory (same command on every axis)."""
    ref_f, ref_mm = REFERENCE
    ref_deg = to_deg(ref_mm)
    if m.kind == "static":
        return [(0.0, 1000)]
    if m.kind == "pause":
        return [(0.0, 5000)]
    if m.kind == "soft_start":       # M03/M04: the ramp is the point
        return compile_halves(sway_schedule(ref_deg, ref_f, ramp_s=m.ramp_s,
                                            hold_cycles=1, taper_s=5.0))
    if m.kind == "taper":            # M05/M06/M07: the wind-down is the point
        return compile_halves(sway_schedule(ref_deg, ref_f, hold_cycles=1,
                                            taper_s=m.ramp_s))
    if m.kind == "micro_resume":     # M08: the reference sway at 50%
        return compile_halves(sway_schedule(0.5 * ref_deg, ref_f,
                                            ramp_s=m.ramp_s, hold_cycles=1,
                                            taper_s=5.0))
    if m.kind == "adaptive_a":       # A 5 -> 10 -> 15 mm blocks
        return compile_halves(
            sway_schedule(to_deg(5.0), m.f_hz, taper_s=0.0)
            + sway_schedule(to_deg(10.0), m.f_hz, ramp_s=0.0, taper_s=0.0)
            + sway_schedule(to_deg(15.0), m.f_hz, ramp_s=0.0))
    if m.kind == "adaptive_f":       # f 0.3 -> 0.5 -> 0.7 Hz blocks
        a = to_deg(m.a_mm)
        return compile_halves(sway_schedule(a, 0.3, taper_s=0.0)
                              + sway_schedule(a, 0.5, ramp_s=0.0, taper_s=0.0)
                              + sway_schedule(a, 0.7, ramp_s=0.0))
    if m.axis == "Z":                # no vertical DOF on this rig: flat hold
        return [(0.0, 2000)]
    if m.kind == "pseudo_walk":      # band-limited 0.4-0.7: its centre
        return compile_halves(sway_schedule(to_deg(m.a_mm), 0.55))
    # sine / diagonal / ellipse / circle / lissajous: primary amplitude
    return compile_halves(sway_schedule(to_deg(m.a_mm), m.f_hz))


def write_slot(path: Path, slot_id: int, name: str,
               segments: list[tuple[float, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["MS ID", "MS NAME", "MD ID", "C-Vector"]
                        + [f"P-Vector {i + 1}" for i in range(len(segments))])
        for axis in range(AXES):
            writer.writerow([slot_id, name, f"MD{axis + 1}", 0]
                            + [f"{deg:.2f},{ms},0,0" for deg, ms in segments])


def build_library(out: Path) -> int:
    from apps.demo import AXIS0, PIVOT
    from core.cradle import A_HARD_MM, LIBRARY

    lever_m = float(PIVOT[2] - AXIS0[2])
    to_deg = lambda mm: math.degrees(mm / 1000.0 / lever_m)
    out.mkdir(exist_ok=True)
    for stale in out.glob("motion_*.csv"):
        stale.unlink()

    for m in LIBRARY:
        slot_id = int(m.id[1:])
        segments = library_episode(m, to_deg)
        peak = max(abs(deg) for deg, _ in segments)
        assert peak <= to_deg(A_HARD_MM) + 1e-9, f"{m.id} breaks the envelope"
        assert abs(segments[-1][0]) < 1e-9, f"{m.id} must end parked at rest"
        duration = sum(ms for _, ms in segments) / 1000.0
        path = out / f"motion_{slot_id:02d}.csv"
        write_slot(path, slot_id, m.name, segments)
        print(f"  {path}  {m.name:<24} [{m.grade}] peak {peak:5.2f} deg  "
              f"{len(segments):>2} segs  {duration:6.1f}s")

    print(f"\n50 motion slots in {out}/  (1 deg = "
          f"{lever_m * 1000 * math.radians(1):.1f} mm of sway)")
    print(f"launch the simulator with them:\n  ./sim.sh {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slot-table", default="slots.json")
    parser.add_argument("--out", default=None,
                        help="output dir (default: motions, or motions_m50 with --library)")
    parser.add_argument("--library", action="store_true",
                        help="compile core/cradle.py's M01-M50 instead of slots.json")
    parser.add_argument("--duration-s", type=float, default=1.6)
    parser.add_argument("--s0", type=float, default=0.0, help="acceleration shaping")
    parser.add_argument("--sd", type=float, default=0.0, help="deceleration shaping")
    args = parser.parse_args(argv)
    if args.library:
        return build_library(Path(args.out or "motions_m50"))
    args.out = args.out or "motions"

    raw = json.loads(Path(args.slot_table).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    for stale in out.glob("motion_*.csv"):
        stale.unlink()

    # Vigour maps to speed: a big soothing sway is a faster one, and the
    # settle glides. --duration-s is the fallback for unknown amplitudes.
    duration_by_amplitude = {"small": 2.2, "medium": 1.6, "large": 1.1, "settle": 2.8}
    written = 0
    for key, entry in sorted(raw.get("slots", {}).items(), key=lambda kv: int(kv[0])):
        slot_id = int(key)
        end_pose = [float(v) for v in entry.get("end_pose", ())]
        if not end_pose:
            print(f"  slot {slot_id}: no end_pose -- skipped")
            continue
        name = slot_name(entry, slot_id)
        duration = duration_by_amplitude.get(str(entry.get("amplitude")), args.duration_s)
        l_traj = max(1, int(round(duration * 1000)))  # pcm records at 1 kHz
        path = out / f"motion_{slot_id:02d}.csv"
        write_motion(path, slot_id, name, end_pose, l_traj, args.s0, args.sd)
        degrees = ", ".join(f"{math.degrees(v):+.2f}" for v in end_pose)
        print(f"  {path}  {name:<15} -> [{degrees}] deg over {duration:.2f}s")
        written += 1

    print(f"\n{written} motion files in {out}/")
    print(f"launch the simulator with them:\n"
          f"  ros2 launch agx_bringup motion.launch.py motion_dir:={out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

Usage::

    python3 make_motions.py                    # write ./motions from slots.json
    python3 make_motions.py --duration-s 2.0   # slower motions
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slot-table", default="slots.json")
    parser.add_argument("--out", default="motions")
    parser.add_argument("--duration-s", type=float, default=1.6)
    parser.add_argument("--s0", type=float, default=0.0, help="acceleration shaping")
    parser.add_argument("--sd", type=float, default=0.0, help="deceleration shaping")
    args = parser.parse_args(argv)

    raw = json.loads(Path(args.slot_table).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    for stale in out.glob("motion_*.csv"):
        stale.unlink()

    l_traj = max(1, int(round(args.duration_s * 1000)))  # pcm records at 1 kHz
    written = 0
    for key, entry in sorted(raw.get("slots", {}).items(), key=lambda kv: int(kv[0])):
        slot_id = int(key)
        end_pose = [float(v) for v in entry.get("end_pose", ())]
        if not end_pose:
            print(f"  slot {slot_id}: no end_pose -- skipped")
            continue
        name = slot_name(entry, slot_id)
        path = out / f"motion_{slot_id:02d}.csv"
        write_motion(path, slot_id, name, end_pose, l_traj, args.s0, args.sd)
        degrees = ", ".join(f"{math.degrees(v):+.2f}" for v in end_pose)
        print(f"  {path}  {name:<15} -> [{degrees}] deg over {args.duration_s:.2f}s")
        written += 1

    print(f"\n{written} motion files in {out}/")
    print(f"launch the simulator with them:\n"
          f"  ros2 launch agx_bringup motion.launch.py motion_dir:={out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

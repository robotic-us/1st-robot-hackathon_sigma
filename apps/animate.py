#!/usr/bin/env python3
"""RViz joint driver: /joint_states fed by the real MotionEngine.

The rig is two five-bar linkages, so library modes publish four *different*
crank angles; --tour/--motion exaggerate amplitude like serve.py --viz-gain.

    python3 apps/animate.py --tour         # walk the real M-library -- the demo
    python3 apps/animate.py --motion M16   # hold one library entry
    python3 apps/animate.py --rock         # one plain sine, until Ctrl-C
    python3 apps/animate.py --sweep        # one joint at a time, a wiring check

Needs the model up first::  ./cad/view.sh
"""

from __future__ import annotations

import argparse
import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

if __package__ in (None, ""):   # direct run: put the repo root on sys.path
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rig import (JOINT_NAMES, LEVER_M, cradle_cranks,
                       cradle_joint_state, joint_state)
from core.cradle import LIBRARY_BY_ID, MotionEngine
VIZ_GAIN = 5.0          # display exaggeration, exactly like serve.py --viz-gain

# Every ML rate 0.2-0.8 Hz, the three A20 entries, then taper home.  M08 after
# M15 on purpose: micro-resume at half amplitude buys a third amplitude level.
# Nothing here is R-grade; tests enforce that.
TOUR = ("M09", "M10", "M11", "M12", "M13", "M14", "M15", "M08",
        "M16", "M17", "M18", "M22", "M26")
TOUR_END = "M05"        # always finish at rest

# R-grade shapes, added only under --research.  M40-M42/M44 deliberately
# absent as near-duplicates on one plane.
TOUR_RESEARCH = ("M43", "M45", "M46", "M35", "M37", "M33", "M39", "M49")

TOUR_DWELL_S = 10.0     # seconds per entry; --dwell overrides
# RAMP_MIN_S (5 s) floors every start/taper; --dwell never shortens a ramp.
TOUR_RAMP_S = 5.0

# The URDF joint names, imported from core/rig.py -- one list, one owner
# (a stale local copy once left three arms without TF in RViz).
JOINTS = JOINT_NAMES
RATE_HZ = 50.0


class Animator(Node):
    def __init__(self) -> None:
        super().__init__("sigma_animator")
        self.pub = self.create_publisher(JointState, "/joint_states", 10)

    def send(self, positions) -> None:
        """``positions`` is a full joint_state() -- all 9 joints, always."""
        pos = [float(v) for v in positions]
        assert len(pos) == len(JOINTS), \
            f"publish all {len(JOINTS)} joints or arms detach, got {len(pos)}"
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINTS
        msg.position = pos
        self.pub.publish(msg)


# --rock is a screen animation, deliberately louder than the real envelope
# (0.8 Hz, +-2.53 deg at A10); the node says so on every start.
ROCK_HZ = 1.5
ROCK_DEG = 20.0

# Aggregate lever: mm of sway per degree of crank.
MM_PER_DEG = LEVER_M * 1000.0 * math.radians(1.0)


def sway_state(amp_deg: float, phase: float) -> list[float]:
    """A plain sway at ``amp_deg``, solved through the linkage as a full joint state."""
    return joint_state(sway_mm=amp_deg * MM_PER_DEG * math.sin(phase))


def library_angles(engine: MotionEngine, script, gain: float = VIZ_GAIN,
                   ramp_s: float = TOUR_RAMP_S, dt: float = 1.0 / RATE_HZ,
                   solve=cradle_joint_state):
    """Walk a ``(motion_id, dwell_s)`` script through the real MotionEngine.

    Yields ``(motion_id, angles_rad, note)`` per sample -- four angles, not one.
    ``note`` carries the engine's reply on an entry's first sample, else None.
    Free of ROS, so a test can walk a whole tour headlessly.
    """
    t = 0.0
    for motion_id, dwell in script:
        _, note = engine.command(motion_id, t, ramp_s=ramp_s)
        end = t + dwell
        while t < end:
            engine.tick(t)
            ap_mm, ml_mm, z_mm = engine.offsets_mm()
            # Gain multiplies plate travel, not joint angles (linkage is
            # nonlinear); ``solve`` picks full joint state vs cradle_cranks.
            yield motion_id, solve(ap_mm, ml_mm, z_mm, gain), note
            note = None
            t += dt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tour", action="store_true",
                        help="walk the real library: rungs, resume, pause, taper")
    parser.add_argument("--dwell", type=float, default=TOUR_DWELL_S,
                        help=f"seconds per --tour entry (default {TOUR_DWELL_S:.0f})")
    parser.add_argument("--motion", help="hold one library motion, e.g. M16")
    parser.add_argument("--gain", type=float, default=VIZ_GAIN,
                        help=f"--tour/--motion display exaggeration (default {VIZ_GAIN})")
    parser.add_argument("--research", action="store_true",
                        help="allow R-grade entries (M29-M50), never automatic")
    parser.add_argument("--rock", action="store_true",
                        help="rock continuously until Ctrl-C -- one plain sine")
    parser.add_argument("--hz", type=float, default=ROCK_HZ,
                        help=f"--rock frequency (default {ROCK_HZ}, the library's fastest)")
    parser.add_argument("--deg", type=float, default=ROCK_DEG,
                        help=f"--rock amplitude in degrees (default {ROCK_DEG}, exaggerated)")
    parser.add_argument("--sweep", action="store_true", help="sine on each joint in turn")
    parser.add_argument("--loop", action="store_true", help="repeat forever")
    args = parser.parse_args(argv)

    rclpy.init()
    node = Animator()
    try:
        if args.tour or args.motion:
            if args.motion:
                m = LIBRARY_BY_ID.get(args.motion.upper())
                if m is None:
                    node.get_logger().error(f"unknown motion {args.motion!r} -- "
                                            "M01..M50, see /motions or the report")
                    return 1
                # The engine would refuse this anyway, but silently.
                if m.grade == "R" and not args.research:
                    node.get_logger().error(
                        f"{m.id} {m.name} is research-only (R) -- "
                        "re-run with --research (report 7.6)")
                    return 1
            engine = MotionEngine(allow_research=args.research)
            dwell = max(1.0, args.dwell)
            ids = TOUR + (TOUR_RESEARCH if args.research else ()) + (TOUR_END,)
            script = (tuple((mid, dwell) for mid in ids) if args.tour
                      else ((args.motion.upper(), 1e9),))
            node.get_logger().info(
                f"library playback at {args.gain:.0f}x display gain -- the real "
                f"envelope is +-2.53 deg at A10"
                + (f"; {len(TOUR_RESEARCH)} R-grade shapes included"
                   if args.tour and args.research else ""))
            while rclpy.ok():
                for motion_id, angles, note in library_angles(
                        engine, script, args.gain):
                    if not rclpy.ok():
                        break
                    if note:
                        node.get_logger().info(f"{motion_id}  {note}")
                    node.send(angles)
                    time.sleep(1.0 / RATE_HZ)
                if not args.tour:       # --motion runs until Ctrl-C anyway
                    break
            return 0

        if args.rock:
            node.get_logger().info(
                f"rocking {args.hz:.2f} Hz +-{args.deg:.1f} deg -- display "
                f"amplitude; the real envelope is +-2.53 deg at A10")
            # rclpy.ok(), not True: publishing into a torn-down context throws.
            t = 0.0
            while rclpy.ok():
                node.send(sway_state(args.deg, 2 * math.pi * args.hz * t))
                time.sleep(1.0 / RATE_HZ)
                t += 1.0 / RATE_HZ
            return 0

        if args.sweep:
            # Wiring check: one joint at a time, linkage deliberately open.
            node.get_logger().info(
                f"sweeping each of the {len(JOINTS)} joints +-45 deg")
            while True:
                for j, name in enumerate(JOINTS):
                    node.get_logger().info(f"  {name}")
                    for i in range(int(2.0 * RATE_HZ)):
                        pose = [0.0] * len(JOINTS)
                        pose[j] = math.radians(45) * math.sin(2 * math.pi * i / (2.0 * RATE_HZ))
                        node.send(pose)
                        time.sleep(1.0 / RATE_HZ)
                if not args.loop:
                    break
            return 0

        parser.error("pick a mode: --tour, --motion, --rock or --sweep")
    except KeyboardInterrupt:
        print()
    finally:
        node.destroy_node()
        if rclpy.ok():      # a signal may have shut the context down already
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

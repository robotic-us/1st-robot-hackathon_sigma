#!/usr/bin/env python3
"""Drive the RViz model's joints -- this is what makes the actuators move.

``robot_state_publisher`` turns joint angles into link poses, but something has
to publish the angles.  That is this: a ``/joint_states`` publisher fed by the
real MotionEngine, so what you see in RViz is not an animation someone
keyframed -- it is the library's own trajectory, solved through the linkage:

    python3 apps/animate.py --tour          # walk the real M-library -- the demo
    python3 apps/animate.py --motion M16          # hold one library entry
    python3 apps/animate.py --rock          # one plain sine, until Ctrl-C
    python3 apps/animate.py --sweep          # one joint at a time, a wiring check

The rig is two five-bar linkages, not a parallelogram: all four cranks the same
way sways the plate, a pair's two cranks opposed lifts it, and pair against pair
pitches it.  So the library modes publish four *different* crank angles, and ML,
AP and Z look genuinely different on screen.  ``--tour`` and ``--motion``
exaggerate amplitude the way ``serve.py --viz-gain`` does; the timing is the
engine's own.  (``--rock`` is a plain screen sine on the sway channel only.)

Needs the model up first::

    ./cad/view.sh          # robot_state_publisher + RViz
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

# A pass through what the rig can actually do: slow to fast, double amplitude,
# a half-amplitude resume, the AP entries (which pitch the plate rather than
# sway it -- a genuinely different motion now that the compiler drives the real
# linkage), and the taper home.  Nothing here is R-grade; tests enforce that.
#
# M02 (pause) and M03 (soft start) used to sit in the middle and were the whole
# reason the tour felt gappy: M02 parks the engine at zero, and M03's ramp is
# fixed at 30 s by the library, so in a short slot it never climbs off the
# floor.  Oscillation entries have no such problem -- the engine ramps env from
# 1.0 to 1.0 and only the frequency changes, so those switches are seamless.
# Every ML rate the report defines, 0.2 Hz to 0.8 Hz, then the three A20
# entries at double reach.  M08 sits after M15 on purpose: micro-resume runs
# the *current* mode at half amplitude, so placing it after an A10 entry buys
# a third amplitude level (~6 deg) that the library has no other way to show.
TOUR = ("M09", "M10", "M11", "M12", "M13", "M14", "M15", "M08",
        "M16", "M17", "M18", "M22", "M26")
TOUR_END = "M05"        # always finish at rest

# R-grade shapes, added only under --research.  Chosen by measuring what a
# single horizontal DOF actually renders, which is not what the names suggest:
#
#   M43 lissajous    two frequencies summed -- the one genuinely compound wave
#   M45 pseudo-walk  band-limited 0.4-0.7 Hz, the least periodic thing here
#   M46 adaptive-A   stepped amplitude, the smallest reach in the library
#   M35/M37 ellipse  AP+ML sum to a sine at reaches P1 cannot reach
#   M33 diagonal     the widest travel of anything that runs
#   M39 circle       AP+ML in quadrature
#
#   M49 Z sine       the vertical channel -- both pairs counter-rotating
#
# Deliberately absent as near-duplicates on one plane: M40-M42 (CW and CCW
# trace the same path) and M44 (identical to M43).
TOUR_RESEARCH = ("M43", "M45", "M46", "M35", "M37", "M33", "M39", "M49")

TOUR_DWELL_S = 10.0     # seconds per entry; --dwell overrides
# RAMP_MIN_S (5 s) is the engine's floor on any start or taper and --dwell does
# not touch it: shortening a ramp is the one thing here that would misrepresent
# how the cradle actually behaves.
TOUR_RAMP_S = 5.0

# The URDF's joints, by name, in the order joint_state() emits them.  This is
# imported from core/rig.py rather than written out again: after the tree was
# re-rooted (every upper arm on the holder), a stale local copy here kept
# publishing joint_axis_1/2/3 -- which no longer exist -- and never published
# the joint_bearing_N that do, so robot_state_publisher had no TF for three
# arms and RViz showed a broken robot.  One list, one owner.
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


# --rock is a screen animation, not a robot command: nothing in this mode goes
# near the engine, the library or the envelope, so both numbers are chosen to
# read well on a projector rather than to match the hardware.  The real cradle
# tops out at 0.8 Hz and +-2.53 deg (A10); this is deliberately louder, and the
# node says so on every start.  For the honest motion, use --tour.
ROCK_HZ = 1.5
ROCK_DEG = 20.0


# The aggregate lever, mm of sway per degree of crank -- for turning a screen
# amplitude in degrees into the plate travel joint_state() wants.
MM_PER_DEG = LEVER_M * 1000.0 * math.radians(1.0)


def sway_state(amp_deg: float, phase: float) -> list[float]:
    """A plain sway at ``amp_deg``, as a full joint state.

    Solved through the linkage rather than copied to four joints, so even the
    exaggerated screen modes keep every arm on the holder.
    """
    return joint_state(sway_mm=amp_deg * MM_PER_DEG * math.sin(phase))


def library_angles(engine: MotionEngine, script, gain: float = VIZ_GAIN,
                   ramp_s: float = TOUR_RAMP_S, dt: float = 1.0 / RATE_HZ,
                   solve=cradle_joint_state):
    """Walk a ``(motion_id, dwell_s)`` script through the real MotionEngine.

    This is where the variety comes from: rather than inventing waveforms, it
    plays the report's own entries and publishes whatever the engine produces
    -- the ramps, the frequency changes, the half-amplitude resume and the
    tapers are all the engine's doing, not ours.

    Yields ``(motion_id, angles_rad, note)`` per sample -- four angles, not one:
    ML reaches every arm, AP opposes the front and rear pairs.  Publishing a
    single summed angle to all four joints, which this used to do, made every
    AP entry render as a copy of its ML twin and the arms always swing together.
    ``note`` carries the engine's reply on the first sample of an entry and is
    None after that.  Free of ROS, so a test can walk a whole tour headlessly.
    """
    t = 0.0
    for motion_id, dwell in script:
        _, note = engine.command(motion_id, t, ramp_s=ramp_s)
        end = t + dwell
        while t < end:
            engine.tick(t)
            ap_mm, ml_mm, z_mm = engine.offsets_mm()
            # Gain multiplies the plate travel, not the joint angles: the
            # linkage is nonlinear, so scaling angles would lift the rods off
            # their bearings and the arms would come apart on screen.
            # ``solve`` decides what comes out: the full joint state for
            # publishing, or cradle_cranks when a caller wants amplitude.
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
                # The engine would refuse this anyway, but silently: say it
                # here, or the operator just watches a statue and wonders.
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
            # Visualisation only.  Say so out loud: an onlooker watching RViz
            # swing +-12 deg should not walk away thinking the cradle does.
            node.get_logger().info(
                f"rocking {args.hz:.2f} Hz +-{args.deg:.1f} deg -- display "
                f"amplitude; the real envelope is +-2.53 deg at A10")
            # rclpy.ok() rather than True: a SIGTERM tears the context down
            # under us, and publishing into a dead context throws.
            t = 0.0
            while rclpy.ok():
                node.send(sway_state(args.deg, 2 * math.pi * args.hz * t))
                time.sleep(1.0 / RATE_HZ)
                t += 1.0 / RATE_HZ
            return 0

        if args.sweep:
            # No world model involved -- just proves each joint is wired up and
            # turns the way we think.  One joint at a time, so the linkage is
            # deliberately open here; every other mode keeps it closed.
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

#!/usr/bin/env python3
"""Drive the RViz model's joints -- this is what makes the actuators move.

``robot_state_publisher`` turns joint angles into link poses, but something has
to publish the angles.  That is this: a ``/joint_states`` publisher fed by the
same P-Vector world model DREAM-Chunk dreams with (see pvector.py).

So what you see in RViz is not an animation someone keyframed.  It is the exact
trajectory the pcm would play for that slot, evaluated from the quintic:

    python3 apps/animate.py --slot 3          # play one motion slot, once
    python3 apps/animate.py --all          # cycle through every slot
    python3 apps/animate.py --sweep          # slow sine on each joint, for a sanity check

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

from core.pvector import load_chunk_dictionary

# URDF joint names, in the same order as slots.json's "axes", plus the passive
# platform bearing (driven below, never commanded by a slot).
JOINTS = ["joint_axis_0", "joint_axis_1", "joint_axis_2", "joint_axis_3",
          "joint_platform"]
RATE_HZ = 50.0


class Animator(Node):
    def __init__(self) -> None:
        super().__init__("sigma_animator")
        self.pub = self.create_publisher(JointState, "/joint_states", 10)

    def send(self, positions) -> None:
        pos = [float(v) for v in positions][:4]
        # The platform bearing counter-rotates the arm that carries it, so the
        # plate stays level -- the closed parallelogram enforces this for free
        # in hardware; here we enforce it in software.
        pos.append(-pos[0])
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINTS
        msg.position = pos
        self.pub.publish(msg)

    def hold(self, positions, seconds: float) -> None:
        """Keep publishing a pose. RViz needs a steady stream, not one message."""
        steps = max(1, int(seconds * RATE_HZ))
        for _ in range(steps):
            self.send(positions)
            time.sleep(1.0 / RATE_HZ)


def play_chunk(node: Animator, chunk, start) -> list[float]:
    """Walk one motion slot's dreamed trajectory at wall-clock speed."""
    t, y = chunk.dream(start, dt=1.0 / RATE_HZ)
    node.get_logger().info(
        f"slot {chunk.slot_id} '{chunk.name}' -- {chunk.duration_s:.2f}s")
    for row in y:
        node.send(row)
        time.sleep(1.0 / RATE_HZ)
    return list(y[-1])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slot", type=int, help="play one slot")
    parser.add_argument("--all", action="store_true", help="cycle through every slot")
    parser.add_argument("--sweep", action="store_true", help="sine on each joint in turn")
    parser.add_argument("--motion-map", help="MotionMap.csv (else motions/ or slots.json)")
    parser.add_argument("--slot-table", default="slots.json")
    parser.add_argument("--loop", action="store_true", help="repeat forever")
    args = parser.parse_args(argv)

    rclpy.init()
    node = Animator()
    try:
        if args.sweep:
            # No world model involved -- just proves the joints are wired up and
            # the axes point where we think they do.
            node.get_logger().info("sweeping each joint +-45 deg")
            while True:
                for j in range(4):
                    for i in range(int(2.0 * RATE_HZ)):
                        pose = [0.0] * 4
                        pose[j] = math.radians(45) * math.sin(2 * math.pi * i / (2.0 * RATE_HZ))
                        node.send(pose)
                        time.sleep(1.0 / RATE_HZ)
                if not args.loop:
                    break
            return 0

        chunks = load_chunk_dictionary(motion_map=args.motion_map,
                                       slot_table=args.slot_table)
        if not chunks:
            node.get_logger().error("no chunks -- run: python3 tools/make_motions.py")
            return 1

        while True:
            pose = [0.0] * 4
            ids = sorted(chunks) if args.all else [args.slot or min(chunks)]
            for slot_id in ids:
                chunk = chunks.get(slot_id)
                if chunk is None:
                    node.get_logger().error(f"slot {slot_id} not in the dictionary")
                    return 1
                pose = play_chunk(node, chunk, pose)
                node.hold(pose, 0.4)
            if not args.loop:
                break
    except KeyboardInterrupt:
        print()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

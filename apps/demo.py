#!/usr/bin/env python3
"""The visible DREAM-Chunk demo: shake an AprilTag, watch the cradle think.

    webcam ─► tag.py ─► (x, shake) ─► DREAM-Chunk ranks the candidate slots
                                          │
              RViz: cradle plays the winner; every candidate's *dreamed*
              platform arc is drawn -- green = chosen, grey = ranked out,
              red = vetoed.  The terminal prints the cost table per decision.

What you are looking at, mapped to the paper:

    dreamed state       each arc is chunk.dream(measured pose) -- the P-Vector
                        quintic evaluated from where the cradle actually is
    action matching     the cost table: task fit + continuity + collision
                        resistance + consistency ranks the dictionary
    reactive            press ``j`` to jam the cradle mid-motion.  The dream
                        keeps going, the "measured" pose does not; the monitor
                        flags divergence, the dwell collapses, and -- because
                        the jam also raises dob_a -- the next ranking visibly
                        flips from the big sway to a gentle one.

Where the person was in the earlier pipeline, the tag is now:

    WHERE you hold it  -> direction    left / center / right
    HOW HARD you shake -> target amplitude for the matcher

Run::

    ./cad/view.sh                   # terminal 1: RViz (arcs display included)
    python3 apps/demo.py            # terminal 2: this

Keys in the camera window: ``j`` jam on/off, ``q`` quit.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

if __package__ in (None, ""):   # direct run: put the repo root on sys.path
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dream import ChunkMatcher, DreamConfig, DreamMonitor
from core.pvector import chunks_from_slot_table, load_motion_map
from core.slot_table import AMPLITUDE_VALUE, load_slot_table
from perception.tag import TagReading, TagTracker, draw_overlay

# --- the cradle's geometry, from make_urdf's output (metres) ---------------- #
AXIS0 = np.array([0.060, 0.037, 0.050])     # axis_0 joint centre
PIVOT = np.array([0.0536, 0.008, 0.3308])   # platform bearing at rest
PLATE = np.array([0.141, 0.092, 0.392])     # plate centre at rest

# --- behaviour -------------------------------------------------------------- #
CALM_FLOOR = 0.12      # shake below this: nobody needs soothing
DEAD_BAND = 0.25       # |x| inside this counts as centred
DWELL_S = 1.2          # pause between motions...
DWELL_PREEMPTED_S = 0.2  # ...collapsed when the dream diverged
JAM_DOB_A = 2.5        # external force the jam pretends to measure
PLAY_DT = 0.02         # playback grid, matches chunk.dream()


def plate_point(theta: float) -> np.ndarray:
    """Where the plate centre sits when the arms are at ``theta``.

    The coupler translates without rotating (the bearing cancels the arm
    angle), carried by the bearing's orbit around axis_0:  t = R_y(th)r - r.
    """
    r = PIVOT - AXIS0
    c, s = math.cos(theta), math.sin(theta)
    t = np.array([c * r[0] + s * r[2] - r[0], 0.0, -s * r[0] + c * r[2] - r[2]])
    return PLATE + t


def load_chunks(motions_dir: str = "motions", slot_table: str = "slots.json"):
    """The same motion files the simulator plays, or slots.json as fallback."""
    chunks = {}
    for path in sorted(Path(motions_dir).glob("motion_*.csv")):
        chunks.update(load_motion_map(str(path)))
    return chunks if chunks else chunks_from_slot_table(slot_table)


class Brain:
    """Decision + playback state.  Pure (time injected): the selftest runs it."""

    def __init__(self, chunks, table, config: Optional[DreamConfig] = None) -> None:
        self.chunks = chunks
        self.table = table
        self.cfg = config or DreamConfig()
        self.matcher = ChunkMatcher(chunks, self.cfg)
        self.monitor = DreamMonitor(self.cfg)
        self.pose = [0.0, 0.0, 0.0, 0.0]   # the cradle's joints, radians
        self.jam = False
        self.playing = None                # (chunk, t0, trajectory)
        self.next_ok = 0.0
        self.last_ranked = []              # for the overlay / markers
        self.preempted = False

    # -- fast path: advance playback, watch the dream ----------------------- #
    def tick(self, now: float) -> None:
        if self.playing is None:
            return
        chunk, t0, traj = self.playing
        row = int((now - t0) / PLAY_DT)
        if row >= len(traj):
            report = self.monitor.latest()
            if report.samples:
                # ASAP-lite: fold what we just watched into the next dream.
                self.matcher.note_residual([report.rms_rad] * len(self.pose))
            self.preempted = report.diverged
            self.monitor.end()
            self.playing = None
            self.next_ok = now + (DWELL_PREEMPTED_S if self.preempted else DWELL_S)
            return
        if not self.jam:
            # Jammed = the cradle physically stopped: the pose freezes while
            # the dream marches on.  That gap is what the monitor measures.
            self.pose = [float(v) for v in traj[row]]
        self.monitor.update(self.pose, now, external_force_a=self.dob())

    def dob(self) -> list[float]:
        return [JAM_DOB_A, 0.0, 0.0, 0.0] if self.jam else [0.0] * 4

    # -- slow path: decide ---------------------------------------------------- #
    def decide(self, reading: TagReading, now: float) -> Optional[int]:
        if self.playing is not None or now < self.next_ok:
            return None
        if not reading.present or reading.motion < CALM_FLOOR:
            return None

        direction = ("left" if reading.x < -DEAD_BAND else
                     "right" if reading.x > DEAD_BAND else "center")
        candidates = [s.slot_id for s in self.table.slots.values()
                      if s.direction.value == direction
                      and s.amplitude.value != "settle"]
        amplitude_of = {s.slot_id: AMPLITUDE_VALUE[s.amplitude]
                        for s in self.table.slots.values()}

        # DREAM-Chunk decides the amplitude: rank the whole direction column.
        self.last_ranked = self.matcher.rank(
            candidates, self.pose,
            external_force_a=self.dob(),
            target_amplitude=reading.motion,
            amplitude_of=amplitude_of,
            recent_error_rad=self.monitor.latest().error_rad,
        )
        survivors = [s for s in self.last_ranked if not s.vetoed]
        if not survivors:
            return None

        chunk = self.chunks[survivors[0].slot_id]
        _, traj = chunk.dream(self.pose, dt=PLAY_DT)
        self.monitor.begin(chunk, self.pose, now)
        self.playing = (chunk, now, traj)
        return chunk.slot_id

    def describe_ranking(self) -> str:
        lines = []
        for i, s in enumerate(self.last_ranked):
            mark = "PLAY <--" if (i == 0 and not s.vetoed) else ""
            lines.append(f"    {s.describe()}  {mark}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# ROS wrapper: joints + dream arcs (only imported when actually running live)
# --------------------------------------------------------------------------- #
class RosSide:
    JOINTS = ["joint_axis_0", "joint_axis_1", "joint_axis_2", "joint_axis_3",
              "joint_platform"]

    def __init__(self) -> None:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from visualization_msgs.msg import Marker, MarkerArray
        self.rclpy, self.JointState = rclpy, JointState
        self.Marker, self.MarkerArray = Marker, MarkerArray
        rclpy.init()
        self.node = Node("sigma_demo")
        self.joint_pub = self.node.create_publisher(JointState, "/joint_states", 10)
        self.marker_pub = self.node.create_publisher(MarkerArray, "/dream_markers", 10)

    def send_joints(self, pose) -> None:
        msg = self.JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = self.JOINTS
        msg.position = [float(v) for v in pose] + [-float(pose[0])]
        self.joint_pub.publish(msg)

    def send_arcs(self, ranked, chunks, pose) -> None:
        """One dreamed platform arc per candidate.  This is the dream, drawn."""
        arr = self.MarkerArray()
        stamp = self.node.get_clock().now().to_msg()
        for i, score in enumerate(ranked):
            chunk = chunks.get(score.slot_id)
            if chunk is None:
                continue
            _, y = chunk.dream(pose, dt=0.05)
            m = self.Marker()
            m.header.frame_id, m.header.stamp = "base_link", stamp
            m.ns, m.id, m.type = "dream", score.slot_id, self.Marker.LINE_STRIP
            m.scale.x = 0.012 if i == 0 and not score.vetoed else 0.004
            if score.vetoed:
                m.color.r, m.color.g, m.color.b, m.color.a = 0.9, 0.2, 0.2, 0.9
            elif i == 0:
                m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.9, 0.3, 0.95
            else:
                m.color.r, m.color.g, m.color.b, m.color.a = 0.6, 0.6, 0.65, 0.5
            m.lifetime.sec = 4
            from geometry_msgs.msg import Point
            for theta in y[:, 0]:
                p = plate_point(float(theta))
                m.points.append(Point(x=p[0], y=p[1], z=p[2]))
            arr.markers.append(m)

            label = self.Marker()
            label.header.frame_id, label.header.stamp = "base_link", stamp
            label.ns, label.id = "dream_label", score.slot_id
            label.type = self.Marker.TEXT_VIEW_FACING
            end = plate_point(float(y[-1, 0]))
            label.pose.position.x, label.pose.position.y = end[0], end[1]
            label.pose.position.z = end[2] + 0.03 + 0.025 * i
            label.scale.z = 0.025
            label.color = m.color
            label.text = (f"slot {score.slot_id}  " +
                          ("VETO" if score.vetoed else f"cost {score.total:.2f}"))
            label.lifetime.sec = 4
            arr.markers.append(label)
        self.marker_pub.publish(arr)

    def close(self) -> None:
        self.node.destroy_node()
        self.rclpy.shutdown()

# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args(argv)

    chunks = load_chunks()
    table = load_slot_table("slots.json")
    brain = Brain(chunks, table)
    ros = RosSide()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"could not open camera {args.camera_index}")
        return 1
    tracker = TagTracker()
    print("show the tag (tags/tag_0.png), shake it -- 'j' jams the cradle, 'q' quits")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.monotonic()
            reading = tracker.update(frame, now)

            brain.tick(now)
            slot = brain.decide(reading, now)
            if slot is not None:
                print(f"\nx={reading.x:+.2f} shake={reading.motion:.2f} "
                      f"dob={brain.dob()[0]:.1f}A -> DREAM ranking:")
                print(brain.describe_ranking())
                ros.send_arcs(brain.last_ranked, chunks, brain.pose)
            ros.send_joints(brain.pose)

            if not args.no_window:
                canvas = draw_overlay(frame, reading, tracker)
                status = "JAMMED" if brain.jam else \
                    (f"playing slot {brain.playing[0].slot_id}" if brain.playing else "idle")
                report = brain.monitor.latest()
                if report.active_slot is not None and report.diverged:
                    status += f"  DIVERGED err={report.error_rad:.2f}"
                cv2.putText(canvas, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255) if brain.jam else (255, 255, 255), 2)
                cv2.imshow("SIGMA demo -- j: jam, q: quit", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("j"):
                    brain.jam = not brain.jam
                    print(f"\n*** JAM {'ON -- cradle held, dob=2.5A' if brain.jam else 'off'} ***")
    except KeyboardInterrupt:
        print()
    finally:
        cap.release()
        ros.close()
        if not args.no_window:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

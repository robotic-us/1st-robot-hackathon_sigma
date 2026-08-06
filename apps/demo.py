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
AXIS0 = np.array([0.0600, 0.1070, 0.0560])  # axis_0 joint centre
PIVOT = np.array([0.0355, 0.0748, 0.2829])  # platform bearing (upper_0 lap)
PLATE = np.array([0.1292, 0.1610, 0.3123])  # platform centre at rest

# --- the rig's real kinematics: two five-bar linkages ----------------------- #
# Measured off docs/udrf_assembly.stl, not assumed.  The contact graph gives
#
#     base -> axis_N (actuator) -> [disc_N, rear only] -> lower_N -> upper_N
#                                                                     -> platform
#
# so every leg is a *two-link chain* -- a driven crank and a passive rod with a
# knee between them -- and all four legs are the same part (identical triangle
# counts, ~152 mm links).  Legs 0 and 1 share one base pivot in XZ and meet at
# ONE platform point with mirrored elbows (+35.6 deg / -47.0 deg); legs 2 and 3
# do the same further along.  Two cranks, two rods, one shared endpoint: each
# pair is a closed five-bar with two degrees of freedom, and the passive knee is
# what absorbs the reach change that a rigid parallelogram could not.
#
# This is NOT the parallelogram rocker the URDF and the old constants assumed.
# The consequences, straight from the Jacobian (and cross-checked: common mode
# comes out at 4.00 mm/deg, matching the independently documented 1 deg ~ 4 mm):
#
#   all four cranks the same way   -> +4.28 mm horizontal, pitch unchanged
#   the two cranks of a pair opposed -> -3.09 mm VERTICAL   <- the Z axis exists
#   front pair against rear pair   -> +0.17 deg of pitch
#
# Every joint turns about Y, so all motion lives in the XZ plane: there is no
# second horizontal axis.  One horizontal channel, one vertical, one pitch.
PAIR_A = (0, 1)             # the cranks that share the x=60 mm base pivot
PAIR_B = (2, 3)             # ...and the x=260 mm one
CRANK_MM = 152.0            # lower_N, pivot to knee
ROD_MM = 152.5              # upper_N, knee to platform bearing
BASE_A = (60.0, 56.0)       # pair A base pivot, XZ (mm)
BASE_B = (260.0, 56.0)
# The neutral pose is NOT read off the export.  In docs/udrf_assembly.stl the
# top bearings are not yet mated to the table, so the platform is only resting
# in place: the two pairs sit at unequal reach (228.3 vs 262.0 mm) and the table
# is tilted 9.7 deg -- confirmed independently by its deck normal (-0.17,0,0.99)
# and by the line through its mounts.  That is an unfinished assembly, not a
# design rest position, and taking the rest pose from it baked the tilt in.
#
# What the export *does* pin down, because it does not depend on the arms:
# the table is rigid and its mounts are coplanar, 200.4 / 199.6 mm apart --
# matching the 200.0 mm base-pivot separation.  So the neutral pose is the
# symmetric one: both pairs at equal reach with each endpoint directly above
# its own base, which puts the table level and the mounts over the pivots.
# Ride height is the mean of the two observed reaches; it sets where in the
# linkage's travel we sit, and is the one number here still worth measuring on
# the finished rig.
RIDE_MM = 245.2             # equal reach for both pairs -> cranks at +-36.5 deg
REST_A = (BASE_A[0], BASE_A[1] + RIDE_MM)
REST_B = (BASE_B[0], BASE_B[1] + RIDE_MM)
# The report's AP is a horizontal axis this rig does not have.  Rather than
# compile those 11 motions to nothing, they drive the pitch channel: 1 mm of
# AP becomes 1 mm of see-saw between the two pairs.  That is a deliberate
# substitution, not the report's motion -- see CLAUDE.md.
AP_AS_PITCH = 1.0
LEVER_M = float(PIVOT[2] - AXIS0[2])   # kept: apps/animate.py and the CAD notes


def _pair_cranks(cx: float, cz: float, base: tuple[float, float],
                 crank: float = CRANK_MM, rod: float = ROD_MM
                 ) -> tuple[float, float]:
    """Exact five-bar IK: a platform point -> that pair's two crank angles.

    Angles are measured from vertical (+Z) toward +X.  The pair's two legs are
    assembled with mirrored elbows, which is the +/- on the interior angle --
    reproduces the CAD's own crank angles to within 0.6 deg.
    """
    dx, dz = cx - base[0], cz - base[1]
    d = math.hypot(dx, dz)
    d = min(max(d, abs(crank - rod) + 1e-6), crank + rod - 1e-6)   # keep reachable
    phi = math.atan2(dx, dz)
    cos_a = (crank * crank + d * d - rod * rod) / (2.0 * crank * d)
    a = math.acos(min(1.0, max(-1.0, cos_a)))
    return phi + a, phi - a


def _neutral_cranks() -> tuple[float, ...]:
    """The four crank angles at the level neutral, solved per leg like the rest.

    Deferred into a function because it needs the mount table defined further
    down; computed once below, right after that table exists.
    """
    mounts, _ = holder_pose()
    return tuple(_leg_ik(m, BASE_A if i < 2 else BASE_B, _ELBOW[i],
                         _CRANK[i], _ROD[i])[0]
                 for i, m in enumerate(mounts))


def axis_angles(sway_mm: float = 0.0, heave_mm: float = 0.0,
                pitch_mm: float = 0.0) -> list[float]:
    """Platform motion -> the four crank angles, radians, relative to rest.

    ``sway`` moves both pairs along X together; ``heave`` lifts both (which is
    the pairs counter-rotating internally -- the channel the old code had no way
    to express, and why the Z modes compiled to flat zeros); ``pitch`` is a
    see-saw, one pair up and the other down.

    Solved through the real linkage rather than a lever, so the cross-coupling
    between channels (a sway also lifts ~0.4 mm, a heave also drifts ~0.3 mm)
    comes out of the geometry instead of being ignored.
    """
    mounts, _ = holder_pose(sway_mm, heave_mm, pitch_mm)
    out = []
    for leg, mount in enumerate(mounts):
        theta, _ = _leg_ik(mount, BASE_A if leg < 2 else BASE_B, _ELBOW[leg],
                           _CRANK[leg], _ROD[leg])
        out.append(theta - _REST[leg])
    return out


# The export pose, straight off the assembly, in the XZ plane: base pivot, knee
# pin, top bearing per leg.  The URDF's zero is this pose, so every joint value
# published is a delta from here -- which is why these have to be written down
# even though the neutral pose above is a different, level configuration.
EXPORT_LEG = (
    ((60.0, 56.0), (148.3, 180.1), (35.5, 282.9)),      # leg 0
    ((60.0, 56.0), (-51.5, 159.9), (39.5, 283.4)),      # leg 1
    ((260.0, 56.0), (322.8, 194.1), (233.0, 316.6)),    # leg 2
    ((260.0, 56.0), (171.7, 179.3), (236.3, 316.9)),    # leg 3
)


def _ang(p: tuple[float, float], q: tuple[float, float]) -> float:
    """Angle of q-p from vertical (+Z) toward +X -- the convention throughout."""
    return math.atan2(q[0] - p[0], q[1] - p[1])


# Per leg at the URDF's zero: crank angle, and the rod's angle in the world.
_EXPORT_CRANK = tuple(_ang(a, k) for a, k, _ in EXPORT_LEG)
_EXPORT_ROD = tuple(_ang(k, c) for _, k, c in EXPORT_LEG)
# ...and the holder's own orientation there, taken through _ang() like every
# other angle in this file.  Measuring it from +X instead -- which is the
# natural way to write a "tilt" -- silently flips its sign against the joint
# convention, and the holder then rotates the wrong way and doubles its tilt
# instead of levelling.  Same convention everywhere, or nothing lines up.
_EXPORT_TILT = _ang(EXPORT_LEG[0][2], EXPORT_LEG[2][2])


# The URDF's joint order, and the tree it describes.  Leg 0 runs up to the
# holder and carries it; legs 1-3 hang down from it, so every upper arm is
# joined to the holder.  The open end is at the actuators, which are bolted to
# the base -- so when the angles are right, axis_1/2/3 land back on their own
# pivots, and tests.py::animate checks exactly that.
JOINT_NAMES = ["joint_axis_0", "joint_knee_0", "joint_platform",
               "joint_bearing_1", "joint_knee_1",
               "joint_bearing_2", "joint_knee_2",
               "joint_bearing_3", "joint_knee_3"]


# The holder's four mounts, in the export pose, and the midpoint of each pair.
# Each leg is solved against its OWN mount: the two mounts of a pair sit a few
# millimetres apart, and pretending they coincide leaves every actuator ~4 mm
# off its bolted-down pivot no matter what the joints do.
_MOUNT = tuple(c for _, _, c in EXPORT_LEG)
_MID_A = ((_MOUNT[0][0] + _MOUNT[1][0]) / 2.0, (_MOUNT[0][1] + _MOUNT[1][1]) / 2.0)
_MID_B = ((_MOUNT[2][0] + _MOUNT[3][0]) / 2.0, (_MOUNT[2][1] + _MOUNT[3][1]) / 2.0)
_ELBOW = (+1.0, -1.0, +1.0, -1.0)   # mirrored assembly, read off the export
# Per-leg link lengths, measured rather than averaged.  The legs are the same
# part, but the export's contact centroids put them at 151.7-152.5 (crank) and
# 151.9-153.4 (rod); using one rounded pair for all four leaves every actuator
# about a millimetre off its pivot, which is small but is pure model error.
_CRANK = tuple(math.dist(k, a) for a, k, _ in EXPORT_LEG)
_ROD = tuple(math.dist(c, k) for _, k, c in EXPORT_LEG)
_EXPORT_TILT = _ang(_MID_A, _MID_B)


def _turn(p, centre, a):
    """Rotate p about centre by a, in the same sense as a URDF joint about Y."""
    dx, dz = p[0] - centre[0], p[1] - centre[1]
    c, s = math.cos(a), math.sin(a)
    return (centre[0] + c * dx + s * dz, centre[1] - s * dx + c * dz)


def _leg_ik(mount, base, elbow, crank=CRANK_MM, rod=ROD_MM):
    """One leg: its mount -> (crank angle, knee point).  Two links, two DOF, so
    each leg reaches its own mount on its own -- no pair constraint needed."""
    d = math.dist(mount, base)
    d = min(max(d, abs(crank - rod) + 1e-6), crank + rod - 1e-6)
    phi = _ang(base, mount)
    a = math.acos(min(1.0, max(-1.0, (crank * crank + d * d - rod * rod)
                               / (2.0 * crank * d))))
    theta = phi + elbow * a
    return theta, (base[0] + crank * math.sin(theta),
                   base[1] + crank * math.cos(theta))


def holder_pose(sway_mm: float = 0.0, heave_mm: float = 0.0,
                pitch_mm: float = 0.0):
    """Where the holder's four mounts sit, and how far it has turned."""
    ends = ((REST_A[0] + sway_mm, REST_A[1] + heave_mm - pitch_mm),
            (REST_B[0] + sway_mm, REST_B[1] + heave_mm + pitch_mm))
    d_tilt = _ang(ends[0], ends[1]) - _EXPORT_TILT
    shift = (ends[0][0] - _MID_A[0], ends[0][1] - _MID_A[1])
    mounts = [tuple(v + o for v, o in zip(_turn(m, _MID_A, d_tilt), shift))
              for m in _MOUNT]
    return mounts, d_tilt


JOINT_NAMES = ["joint_axis_0", "joint_knee_0", "joint_platform",
               "joint_bearing_1", "joint_knee_1",
               "joint_bearing_2", "joint_knee_2",
               "joint_bearing_3", "joint_knee_3"]


def joint_state(sway_mm: float = 0.0, heave_mm: float = 0.0,
                pitch_mm: float = 0.0) -> list[float]:
    """Every URDF joint, in ``JOINT_NAMES`` order.

    Leg 0 runs up to the holder and carries it; legs 1-3 hang down from it, so
    each value is that link's turn relative to a different parent.  A wrong sign
    here does not raise -- it silently detaches an arm on screen -- which is why
    tests.py::animate walks these back through the URDF and checks every
    actuator lands on its own pivot.
    """
    mounts, d_tilt = holder_pose(sway_mm, heave_mm, pitch_mm)
    d_crank, d_rod = [], []
    for leg, mount in enumerate(mounts):
        base = BASE_A if leg < 2 else BASE_B
        theta, knee = _leg_ik(mount, base, _ELBOW[leg], _CRANK[leg], _ROD[leg])
        d_crank.append(theta - _EXPORT_CRANK[leg])
        d_rod.append(_ang(knee, mount) - _EXPORT_ROD[leg])

    joints = [d_crank[0],                      # base   -> axis_0
              d_rod[0] - d_crank[0],           # crank  -> rod      (leg 0 knee)
              d_tilt - d_rod[0]]               # rod    -> holder
    for leg in (1, 2, 3):
        joints += [d_rod[leg] - d_tilt,        # holder -> rod      (bearing)
                   d_crank[leg] - d_rod[leg]]  # rod    -> crank    (knee)
    return joints


_REST = _neutral_cranks()


def cradle_angles(ap_mm: float, ml_mm: float, z_mm: float = 0.0) -> list[float]:
    """The engine's (ap, ml, z) offsets -> the four crank angles, radians.

    ML is the one horizontal axis the rig has, Z is real, and AP -- which it
    cannot translate along -- is rendered as pitch.
    """
    return axis_angles(sway_mm=ml_mm, heave_mm=z_mm,
                       pitch_mm=AP_AS_PITCH * ap_mm)


def cradle_cranks(ap_mm: float, ml_mm: float, z_mm: float = 0.0,
                  gain: float = 1.0) -> list[float]:
    """Just the four crank angles -- what "how far did it swing" means.

    ``joint_state`` interleaves knees and bearings, so its first four entries
    are *not* the cranks; anything measuring amplitude wants this instead.
    """
    return axis_angles(sway_mm=ml_mm * gain, heave_mm=z_mm * gain,
                       pitch_mm=AP_AS_PITCH * ap_mm * gain)


def cradle_joint_state(ap_mm: float, ml_mm: float, z_mm: float = 0.0,
                       gain: float = 1.0) -> list[float]:
    """The engine's offsets -> every URDF joint, ready to publish.

    ``gain`` exaggerates the *plate travel* before solving, not the joint angles
    afterwards.  Scaling the angles would pull the rods off their bearings,
    because the linkage is nonlinear -- the exaggerated pose has to be a real
    pose of the mechanism, or the arms visibly come apart on screen.
    """
    return joint_state(sway_mm=ml_mm * gain, heave_mm=z_mm * gain,
                       pitch_mm=AP_AS_PITCH * ap_mm * gain)

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
    # Cranks, then knees, then the bearing -- the order joint_state() returns.
    # One owner: JOINT_NAMES, defined beside joint_state().  A stale copy here
    # is exactly how RViz broke after the tree was re-rooted.
    JOINTS = JOINT_NAMES

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

    def send_joints(self, joints) -> None:
        """``joints`` is a full joint_state(): 4 cranks, 4 knees, the bearing."""
        msg = self.JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = self.JOINTS
        msg.position = [float(v) for v in joints]
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

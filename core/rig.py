#!/usr/bin/env python3
"""The rig: measured kinematics of the two five-bar linkages, plus RViz I/O.

The one home of the geometry convention -- the compiler, serve.py's RViz
mirror and apps/animate.py all import this module (history: CLAUDE.md).
Regenerate constants with ``python3 tools/make_urdf.py`` after CAD changes.
"""

from __future__ import annotations

import math

import numpy as np

# --- the cradle's geometry, from make_urdf's output (metres) ---------------- #
AXIS0 = np.array([0.0600, 0.1070, 0.0560])  # axis_0 joint centre
PIVOT = np.array([0.0355, 0.0748, 0.2829])  # platform bearing (upper_0 lap)
PLATE = np.array([0.1292, 0.1610, 0.3123])  # platform centre at rest

# --- the rig's real kinematics: two five-bar linkages ----------------------- #
# Measured off docs/urdf_assembly.stl.  Each leg is a two-link chain -- driven
# crank, passive knee, rod -- and legs 0/1 (likewise 2/3) close a five-bar
# sharing one platform point with mirrored elbows.  Per degree of crank:
# all four the same way -> +4.28 mm horizontal; a pair's cranks opposed ->
# -3.09 mm VERTICAL; pair against pair -> +0.17 deg pitch.  Every joint turns
# about Y, so all motion is in XZ: one horizontal, one vertical, one pitch.
PAIR_A = (0, 1)             # the cranks that share the x=60 mm base pivot
PAIR_B = (2, 3)             # ...and the x=260 mm one
CRANK_MM = 152.0            # lower_N, pivot to knee
ROD_MM = 152.5              # upper_N, knee to platform bearing
BASE_A = (60.0, 56.0)       # pair A base pivot, XZ (mm)
BASE_B = (260.0, 56.0)
# The neutral pose is NOT read off the export (the assembly is unfinished --
# CLAUDE.md): neutral is the symmetric pose, equal reach with each endpoint
# above its own base, table level.  RIDE_MM is the one number here still
# worth measuring on the finished rig.
RIDE_MM = 245.2             # equal reach for both pairs -> cranks at +-36.5 deg
REST_A = (BASE_A[0], BASE_A[1] + RIDE_MM)
REST_B = (BASE_B[0], BASE_B[1] + RIDE_MM)
# AP is a horizontal axis this rig does not have; the report's 11 AP motions
# drive pitch instead -- a deliberate substitution (CLAUDE.md).
AP_AS_PITCH = 1.0
LEVER_M = float(PIVOT[2] - AXIS0[2])   # kept: apps/animate.py and the CAD notes


def _neutral_cranks() -> tuple[float, ...]:
    """The four crank angles at the level neutral; computed once below, after
    the mount table it needs exists."""
    mounts, _ = holder_pose()
    return tuple(_leg_ik(m, BASE_A if i < 2 else BASE_B, _ELBOW[i],
                         _CRANK[i], _ROD[i])[0]
                 for i, m in enumerate(mounts))


def axis_angles(sway_mm: float = 0.0, heave_mm: float = 0.0,
                pitch_mm: float = 0.0) -> list[float]:
    """Platform motion -> the four crank angles, radians, relative to rest.
    sway = both pairs along X; heave = vertical (a pair's cranks opposed);
    pitch = see-saw.  Solved through the real linkage, cross-coupling and all."""
    mounts, _ = holder_pose(sway_mm, heave_mm, pitch_mm)
    out = []
    for leg, mount in enumerate(mounts):
        theta, _ = _leg_ik(mount, BASE_A if leg < 2 else BASE_B, _ELBOW[leg],
                           _CRANK[leg], _ROD[leg])
        out.append(theta - _REST[leg])
    return out


# The export pose (XZ: base pivot, knee pin, top bearing per leg).  The URDF's
# zero is this pose, so every published joint value is a delta from here.
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

# The holder's four mounts at the export pose.  Each leg is solved against its
# OWN mount -- a pair's mounts sit a few millimetres apart.
_MOUNT = tuple(c for _, _, c in EXPORT_LEG)
_MID_A = ((_MOUNT[0][0] + _MOUNT[1][0]) / 2.0, (_MOUNT[0][1] + _MOUNT[1][1]) / 2.0)
_MID_B = ((_MOUNT[2][0] + _MOUNT[3][0]) / 2.0, (_MOUNT[2][1] + _MOUNT[3][1]) / 2.0)
_ELBOW = (+1.0, -1.0, +1.0, -1.0)   # mirrored assembly, read off the export
# Per-leg link lengths, measured per leg (151.7-153.4 mm), never averaged.
_CRANK = tuple(math.dist(k, a) for a, k, _ in EXPORT_LEG)
_ROD = tuple(math.dist(c, k) for _, k, c in EXPORT_LEG)
# Holder tilt via _ang() -- angles measure from +Z toward +X everywhere;
# any other convention flips the sign and nothing lines up.
_EXPORT_TILT = _ang(_MID_A, _MID_B)


def _turn(p, centre, a):
    """Rotate p about centre by a, in the same sense as a URDF joint about Y."""
    dx, dz = p[0] - centre[0], p[1] - centre[1]
    c, s = math.cos(a), math.sin(a)
    return (centre[0] + c * dx + s * dz, centre[1] - s * dx + c * dz)


def _leg_ik(mount, base, elbow, crank=CRANK_MM, rod=ROD_MM):
    """One leg: its mount -> (crank angle, knee point)."""
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


# The ONE joint-name list; both publishers reference it, never a copy.
# Re-rooted tree: leg 0 runs up to the holder, legs 1-3 hang down from it;
# closure = each actuator lands on its own pivot (tests.py::animate).
JOINT_NAMES = ["joint_axis_0", "joint_knee_0", "joint_platform",
               "joint_bearing_1", "joint_knee_1",
               "joint_bearing_2", "joint_knee_2",
               "joint_bearing_3", "joint_knee_3"]


def joint_state(sway_mm: float = 0.0, heave_mm: float = 0.0,
                pitch_mm: float = 0.0) -> list[float]:
    """Every URDF joint, in ``JOINT_NAMES`` order -- interleaved per leg, NOT
    the four cranks first (use ``cradle_cranks`` for amplitude)."""
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
    ML is the one horizontal axis, Z is real, AP is rendered as pitch."""
    return axis_angles(sway_mm=ml_mm, heave_mm=z_mm,
                       pitch_mm=AP_AS_PITCH * ap_mm)


def cradle_cranks(ap_mm: float, ml_mm: float, z_mm: float = 0.0,
                  gain: float = 1.0) -> list[float]:
    """Just the four crank angles -- what "how far did it swing" means
    (``joint_state`` interleaves; its first four entries are NOT the cranks)."""
    return axis_angles(sway_mm=ml_mm * gain, heave_mm=z_mm * gain,
                       pitch_mm=AP_AS_PITCH * ap_mm * gain)


def cradle_joint_state(ap_mm: float, ml_mm: float, z_mm: float = 0.0,
                       gain: float = 1.0) -> list[float]:
    """The engine's offsets -> every URDF joint, ready to publish.
    ``gain`` scales the *plate travel* before solving, never the joint angles
    -- the exaggerated pose must be a real pose of the nonlinear mechanism."""
    return joint_state(sway_mm=ml_mm * gain, heave_mm=z_mm * gain,
                       pitch_mm=AP_AS_PITCH * ap_mm * gain)

class RosSide:
    # One owner: JOINT_NAMES -- a stale copy here is how RViz once broke.
    JOINTS = JOINT_NAMES

    def __init__(self) -> None:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        self.rclpy, self.JointState = rclpy, JointState
        rclpy.init()
        self.node = Node("sigma_rig")
        self.joint_pub = self.node.create_publisher(JointState, "/joint_states", 10)

    def send_joints(self, joints) -> None:
        """``joints`` is a full joint_state(): 4 cranks, 4 knees, the bearing."""
        msg = self.JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = self.JOINTS
        msg.position = [float(v) for v in joints]
        self.joint_pub.publish(msg)

    def close(self) -> None:
        self.node.destroy_node()
        self.rclpy.shutdown()

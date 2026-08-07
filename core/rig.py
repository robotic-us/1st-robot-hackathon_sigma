#!/usr/bin/env python3
"""The rig: measured kinematics of the two five-bar linkages, plus RViz I/O.

Everything that knows the cradle's geometry lives here -- the compiler
(tools/make_motions.py), the server's RViz mirror (serve.py) and the player
(apps/animate.py) all import this one module, because the last three bugs in
this area were three copies of the same convention drifting apart.  Derivation
and the hard-won facts: CLAUDE.md.  Regenerate the constants with
``python3 tools/make_urdf.py`` whenever the CAD changes.
"""

from __future__ import annotations

import math

import numpy as np

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
# The holder's orientation at the export pose, taken through _ang() like every
# other angle in this file.  Measuring it from +X instead -- the natural way to
# write a "tilt" -- silently flips its sign against the joint convention, and
# the holder then rotates the wrong way and doubles its tilt instead of
# levelling.  Same convention everywhere, or nothing lines up.
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


# The URDF's joint order, and the tree it describes.  Leg 0 runs up to the
# holder and carries it; legs 1-3 hang down from it, so every upper arm is
# joined to the holder.  The open end is at the actuators, which are bolted to
# the base -- so when the angles are right, axis_1/2/3 land back on their own
# pivots, and tests.py::animate checks exactly that.  The ONE joint-name list;
# both publishers reference it, never a copy.
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

class RosSide:
    # Cranks, then knees, then the bearing -- the order joint_state() returns.
    # One owner: JOINT_NAMES, defined beside joint_state().  A stale copy here
    # is exactly how RViz broke after the tree was re-rooted.
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

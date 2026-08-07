#!/usr/bin/env python3
"""One assembly STL -> per-part meshes + a URDF we can show in RViz.

Fusion exported the whole robot as one STL: 188k triangles, no joints, no part
names, no kinematics.  A URDF needs the opposite -- separate meshes per link and
an explicit parent/child chain.  This bridges the two as far as geometry alone
allows.

What it can work out on its own:

* **Separate rigid bodies**, by welding shared vertices and taking connected
  components.  Triangles that touch belong to the same part.
* **Which bodies are phact actuators**, by size: the catalogue says phact-401 is
  Phi85 x 40 mm, and exactly four bodies measure 85 x 34 x 85.  Their centres are
  the joint origins.
* **Each joint's rotation axis.**  A pancake actuator turns about its thin axis,
  and these are thin in Y -- so the axes are Y.
* **Fasteners**, by volume.  The ~99 tiny bodies are M3/M4 bolts and 6807ZZ
  bearings; they get merged into whichever large body they sit nearest so they
  do not become links of their own.

What geometry *cannot* tell us on its own is which touching part moves the
other, so the roles are classified from the contact graph plus two anchors:
the base sits on the ground, and the platform is the one large body the arm
tops all meet.  Each arm is then actuator -> lower link -> upper link, paired
by contact and assigned to the actuator column it stands on.

Built for the complete export (docs/udrf_assembly.stl): all four arms exist
as real bodies, and the platform IS the coupler -- the upper links lap it
directly, which is where the 6807ZZ bearings sit.  The coupler
counter-rotates the arm angle, so it translates while staying level -- a
classic parallelogram rocker.  (The earlier half-built export needed its
right side synthesised; that code is gone, see git history if it returns.)
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

MM_TO_M = 0.001
ACTUATOR_SIZE_MM = (85.0, 85.0)   # phact-401 face, Phi85 (the 3rd dim is depth)
ACTUATOR_TOL_MM = 6.0
FASTENER_VOL_CM3 = 15.0           # below this it is a bolt/bearing/washer, not a link

# The kinematic tree, filled in by build() from the contact graph:
#
#   base_link ─┬─ axis_0 → lower_0 → upper_0 ─(bearing: joint_platform)─ platform
#              ├─ axis_1 → lower_1 → upper_1
#              ├─ axis_2 → lower_2 → upper_2   (+ disc_N mounting plates,
#              └─ axis_3 → lower_3 → upper_3      fixed to their actuator)
#
# Only ONE bearing can be an explicit joint (URDF is a tree); the other three
# laps stay closed because the linkage is a true parallelogram: with equal
# actuator angles and the coupler counter-rotated, every arm top tracks its
# lap exactly, at any angle.  Unequal angles split the laps -- which is also
# what they would do to the real hardware.
PARENT: dict[str, str] = {}

# Links that rotate. The joint sits at the actuator's centre, about its thin axis.
# Every joint on this rig turns about Y.  The table below is keyed by CHILD
# link and gives the joint name -- which also decides revolute vs fixed: a link
# absent from it gets a fixed joint.
#
# The knee (crank meets rod) used to be welded, and only upper_0 reached the
# holder, so each leg was one rigid arm and three of the four top bearings
# joined nothing.  Both are real pins, and both are joints now.
JOINT_NAME = {
    "axis_0": "joint_axis_0",            # the only actuator still on the base
    "upper_0": "joint_knee_0",           # leg 0 runs UP to the holder
    "platform": "joint_platform",        # ...and carries it
    "upper_1": "joint_bearing_1",        # legs 1-3 hang DOWN from the holder,
    "upper_2": "joint_bearing_2",        # so every upper arm is joined to it
    "upper_3": "joint_bearing_3",
    "lower_1": "joint_knee_1",
    "lower_2": "joint_knee_2",
    "lower_3": "joint_knee_3",
}
PLATFORM = "platform"
PLATFORM_JOINT = "joint_platform"
FRAME_OVERRIDE: dict[str, "np.ndarray"] = {}   # filled in build(): bearing centre

# file:// keeps RViz working with no ROS package to install. Swap for
# package://sigma_description/meshes when this becomes a real package.
# Set in build() from --out, so it survives this script moving around.
MESH_URI = ""


def read_stl(path: Path) -> np.ndarray:
    """Binary STL -> (n, 3, 3) float64 triangles."""
    raw = path.read_bytes()
    n = int(np.frombuffer(raw[80:84], dtype="<u4")[0])
    rec = np.frombuffer(
        raw[84:84 + n * 50],
        dtype=np.dtype([("nrm", "<3f4"), ("v", "<3,3f4"), ("attr", "<u2")]),
    )
    return rec["v"].astype(np.float64)


def write_stl(path: Path, tri: np.ndarray) -> None:
    """(n, 3, 3) triangles -> binary STL with recomputed normals."""
    n = len(tri)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    with path.open("wb") as fh:
        fh.write(b"SIGMA split mesh".ljust(80, b" "))
        fh.write(struct.pack("<I", n))
        for i in range(n):
            fh.write(struct.pack("<12fH", *normals[i], *tri[i].ravel(), 0))


def split_bodies(tri: np.ndarray) -> np.ndarray:
    """Label each triangle with the rigid body it belongs to."""
    flat = np.round(tri.reshape(-1, 3), 3)      # weld coincident vertices
    _, vid = np.unique(flat, axis=0, return_inverse=True)
    vid = vid.reshape(-1, 3)
    a = np.concatenate([vid[:, 0], vid[:, 1], vid[:, 2]])
    b = np.concatenate([vid[:, 1], vid[:, 2], vid[:, 0]])
    graph = coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(vid.max() + 1,) * 2)
    _, labels = connected_components(graph, directed=False)
    return labels[vid[:, 0]]


def volume_cm3(tri: np.ndarray) -> float:
    """Enclosed volume via the divergence theorem, in cm^3 (input mm)."""
    return abs(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0) / 1000.0


def is_actuator(size: np.ndarray) -> bool:
    """Two of the three dimensions ~85 mm and the third clearly thinner."""
    ordered = np.sort(size)[::-1]
    return (
        abs(ordered[0] - ACTUATOR_SIZE_MM[0]) < ACTUATOR_TOL_MM
        and abs(ordered[1] - ACTUATOR_SIZE_MM[1]) < ACTUATOR_TOL_MM
        and ordered[2] < ordered[1] * 0.7
    )


def thin_axis(size: np.ndarray) -> int:
    """Index of the shortest dimension -- a pancake actuator spins about it."""
    return int(np.argmin(size))


class Part:
    def __init__(self, index: int, tri: np.ndarray) -> None:
        pts = tri.reshape(-1, 3)
        self.index = index
        self.tri = tri
        self.lo, self.hi = pts.min(0), pts.max(0)
        self.size = self.hi - self.lo
        self.centre = (self.lo + self.hi) / 2.0
        self.volume = volume_cm3(tri)
        self.actuator = is_actuator(self.size)
        self.name = ""


DISC_VOL_CM3 = 40.0   # below this an actuator-face body is a mounting disc


def contact_points(a_tri: np.ndarray, b_tri: np.ndarray, slack_mm: float = 3.0):
    """Points of A near B: the lap where two parts meet (or None if apart)."""
    a = np.unique(np.round(a_tri.reshape(-1, 3), 2), axis=0)
    b = np.unique(np.round(b_tri.reshape(-1, 3), 2), axis=0)
    dist, _ = cKDTree(b).query(a)
    near = a[dist < float(dist.min()) + slack_mm]
    return near if dist.min() < 2.5 else None


def build(stl: Path, out_dir: Path, report_only: bool) -> int:
    global MESH_URI
    MESH_URI = f"file://{(out_dir / 'meshes').resolve()}"
    tri = read_stl(stl)
    labels = split_bodies(tri)
    parts = [Part(c, tri[labels == c]) for c in range(labels.max() + 1)]
    parts.sort(key=lambda p: -p.volume)

    big = [p for p in parts if p.volume >= FASTENER_VOL_CM3]
    small = [p for p in parts if p.volume < FASTENER_VOL_CM3]

    # Fasteners join whichever structural part they sit nearest.
    merged: dict[int, list[Part]] = {p.index: [] for p in big}
    for s in small:
        nearest = min(big, key=lambda p: np.linalg.norm(p.centre - s.centre))
        merged[nearest.index].append(s)

    # Anchors: actuators are axis_N by (x, y); the base is whatever sits on
    # the ground plane (break ties by bulk -- the top face would pick a
    # bearing, because bearings are short); the platform is the biggest body
    # left, riding on top of the arms.
    actuators = sorted([p for p in big if p.actuator], key=lambda p: (p.centre[0], p.centre[1]))
    for i, p in enumerate(actuators):
        p.name = f"axis_{i}"
    base = min((p for p in big if not p.actuator),
               key=lambda p: (round(float(p.lo[2]), 1), -p.volume))
    base.name = "base_link"
    rest = [p for p in big if not p.name]
    plate = max(rest, key=lambda p: p.volume)
    plate.name = PLATFORM

    # Roles from contact: uppers lap the platform, each lower laps its upper,
    # small leftovers are the actuator-face mounting discs.
    uppers = [p for p in rest if p is not plate
              and contact_points(p.tri, plate.tri) is not None]
    others = [p for p in rest if p is not plate and p not in uppers]
    discs = [p for p in others if p.volume < DISC_VOL_CM3]
    lowers = [p for p in others if p not in discs]
    assert len(uppers) == 4 and len(lowers) == 4, \
        f"expected 4 arms, found {len(uppers)} uppers / {len(lowers)} lowers"

    # Name the arms first: each lower laps exactly one upper, and stands on the
    # actuator column nearest it in plan view.
    for lower in lowers:
        upper = next(u for u in uppers
                     if contact_points(lower.tri, u.tri) is not None)
        axis = min(actuators,
                   key=lambda a: np.linalg.norm(a.centre[:2] - lower.centre[:2]))
        i = axis.name[-1]
        lower.name, upper.name = f"lower_{i}", f"upper_{i}"
    for disc in discs:
        axis = min(actuators,
                   key=lambda a: np.linalg.norm(a.centre - disc.centre))
        disc.name = f"disc_{axis.name[-1]}"

    # The tree.  A URDF is a tree, so the holder can have exactly one parent --
    # but every upper arm carries it in hardware, and a model where three of
    # them join nothing is not the machine.  So leg 0 runs up to the holder and
    # legs 1-3 hang DOWN from it:
    #
    #   base_link - axis_0 - lower_0 - upper_0 -+
    #                                           +- platform (the holder)
    #               axis_1 - lower_1 - upper_1 -+
    #               axis_2 - lower_2 - upper_2 -+
    #               axis_3 - lower_3 - upper_3 -+
    #
    # Every upper arm is now joined to the holder, which is the connection that
    # has to hold. The open end moves to the actuators -- and those are bolted
    # to the base, so with the angles solved they land back on their pivots.
    PARENT.clear()
    PARENT["axis_0"] = "base_link"
    PARENT["lower_0"] = "axis_0"
    PARENT["upper_0"] = "lower_0"
    PARENT[PLATFORM] = "upper_0"
    for i in (1, 2, 3):
        PARENT[f"upper_{i}"] = PLATFORM        # <- joined to the holder
        PARENT[f"lower_{i}"] = f"upper_{i}"
        PARENT[f"axis_{i}"] = f"lower_{i}"
    for disc in discs:
        PARENT[disc.name] = f"axis_{disc.name[-1]}"

    # Everything each link owns: its own shell + merged fasteners.
    by_name = {p.name: p for p in big}
    combined_tri = {p.name: np.concatenate([p.tri] + [s.tri for s in merged[p.index]])
                    for p in big}

    # Every pin, measured from the meshes.  Each link's frame sits on the joint
    # that attaches it to its parent, which is what makes the tree above work:
    # leg 0's rod is framed on its knee (it hangs off the crank), while legs
    # 1-3's rods are framed on their bearings (they hang off the holder).
    knee, bearing = {}, {}
    for i in range(4):
        lap = contact_points(combined_tri[f"lower_{i}"], combined_tri[f"upper_{i}"])
        assert lap is not None, f"lower_{i} must lap upper_{i}"
        knee[i] = lap.mean(axis=0)
        lap = contact_points(combined_tri[f"upper_{i}"], combined_tri[PLATFORM])
        assert lap is not None, f"upper_{i} must lap the platform"
        bearing[i] = lap.mean(axis=0)
        print(f"leg {i}: knee {np.round(knee[i], 1)}  "
              f"bearing {np.round(bearing[i], 1)} mm")

    # Every actuator is framed on its own axis of rotation, including the three
    # that now hang off the holder -- otherwise they inherit the knee's frame,
    # which sits a whole crank length away and makes any closure check nonsense.
    for i in range(4):
        FRAME_OVERRIDE[f"axis_{i}"] = by_name[f"axis_{i}"].centre
    FRAME_OVERRIDE["upper_0"] = knee[0]        # leg 0 runs up: framed on the knee
    FRAME_OVERRIDE[PLATFORM] = bearing[0]      # the holder enters through leg 0
    for i in (1, 2, 3):
        FRAME_OVERRIDE[f"upper_{i}"] = bearing[i]   # framed where it meets the holder
        FRAME_OVERRIDE[f"lower_{i}"] = knee[i]
    print()

    print(f"{len(parts)} rigid bodies -> {len(big)} links "
          f"({len(small)} fasteners merged in)\n")
    print(f"{'link':<12} {'vol cm3':>8} {'size mm':<22} {'centre mm':<22} axis")
    for p in big:
        axis = "XYZ"[thin_axis(p.size)] if p.actuator else "-"
        print(f"{p.name:<12} {p.volume:>8.1f} {str(np.round(p.size, 1)):<22} "
              f"{str(np.round(p.centre, 1)):<22} {axis}")
    if report_only:
        return 0

    meshes = out_dir / "meshes"
    meshes.mkdir(parents=True, exist_ok=True)
    # Wipe first: a previous run with different thresholds leaves stale
    # part_N.stl files behind, and they silently pollute any later analysis.
    for old in meshes.glob("*.stl"):
        old.unlink()
    for p in big:
        write_stl(meshes / f"{p.name}.stl", combined_tri[p.name])

    urdf = out_dir / "sigma.urdf"
    urdf.write_text(render_urdf(big, base), encoding="utf-8")
    print(f"\nwrote {urdf} and {len(big)} meshes in {meshes}/")

    # core/rig.py carries these three as its geometry constants -- paste
    # them there whenever the CAD changes.
    metres = lambda v: ", ".join(f"{x * MM_TO_M:.4f}" for x in v)
    print("\ngeometry for core/rig.py (metres):")
    print(f"  AXIS0 = np.array([{metres(by_name['axis_0'].centre)}])")
    print(f"  PIVOT = np.array([{metres(FRAME_OVERRIDE[PLATFORM])}])")
    print(f"  PLATE = np.array([{metres(plate.centre)}])")
    return 0


def frame_origin(part, by_name):
    """World position of the frame a link is expressed in, in mm.

    A revolute link gets its own frame at the actuator centre -- that is where
    the rotation axis has to pass through. The platform's frame sits at its
    bearing. Everything else inherits its parent's frame, so rigid
    sub-assemblies keep sharing one origin.
    """
    if part.name in FRAME_OVERRIDE:
        return FRAME_OVERRIDE[part.name]
    parent = PARENT.get(part.name)
    return frame_origin(by_name[parent], by_name) if parent else np.zeros(3)


def render_urdf(parts: list[Part], base: Part) -> str:
    """Links in their own frames, joints on the actuator axes."""
    by_name = {p.name: p for p in parts}
    out = ['<?xml version="1.0"?>', '<robot name="sigma">', "",
           '  <material name="grey"><color rgba="0.6 0.6 0.62 1"/></material>',
           '  <material name="accent"><color rgba="0.2 0.6 0.9 1"/></material>', ""]

    for p in parts:
        # The mesh keeps its CAD coordinates, so shift the visual back by the
        # frame origin -- otherwise every rotated link jumps to the origin.
        off = -frame_origin(p, by_name) * MM_TO_M
        colour = "accent" if p.actuator else "grey"
        out += [
            f'  <link name="{p.name}">', "    <visual>",
            f'      <origin xyz="{off[0]:.6f} {off[1]:.6f} {off[2]:.6f}" rpy="0 0 0"/>',
            "      <geometry>",
            f'        <mesh filename="{MESH_URI}/{p.name}.stl"'
            f' scale="{MM_TO_M} {MM_TO_M} {MM_TO_M}"/>',
            "      </geometry>", f'      <material name="{colour}"/>',
            "    </visual>", "  </link>", "",
        ]

    for p in parts:
        if p.name == base.name:
            continue
        parent_name = PARENT.get(p.name, base.name)
        parent = by_name.get(parent_name, base)
        origin = (frame_origin(p, by_name) - frame_origin(parent, by_name)) * MM_TO_M
        xyz = f'{origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}'

        if p.name in JOINT_NAME:
            jname = JOINT_NAME[p.name]
            axis = np.array([0.0, 1.0, 0.0])   # every pin on this rig is about Y
            out += [
                f'  <joint name="{jname}" type="revolute">',
                f'    <parent link="{parent_name}"/>',
                f'    <child link="{p.name}"/>',
                f'    <origin xyz="{xyz}" rpy="0 0 0"/>',
                f'    <axis xyz="{axis[0]:g} {axis[1]:g} {axis[2]:g}"/>',
                # phact-401: 7.2 Nm continuous, 150 rpm = 15.7 rad/s.
                '    <limit lower="-1.57" upper="1.57" effort="7.2" velocity="15.7"/>',
                "  </joint>", "",
            ]
        else:
            out += [
                f'  <joint name="fix_{p.name}" type="fixed">',
                f'    <parent link="{parent_name}"/>',
                f'    <child link="{p.name}"/>',
                f'    <origin xyz="{xyz}" rpy="0 0 0"/>',
                "  </joint>", "",
            ]
    out.append("</robot>")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stl", default="docs/udrf_assembly.stl")
    parser.add_argument("--out", default="cad")
    parser.add_argument("--report", action="store_true", help="print parts, write nothing")
    args = parser.parse_args(argv)
    return build(Path(args.stl), Path(args.out), args.report)


if __name__ == "__main__":
    raise SystemExit(main())

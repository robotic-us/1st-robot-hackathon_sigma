#!/usr/bin/env bash
# Show the robot in RViz.
#
#   ./cad/view.sh
#
# Starts robot_state_publisher with cad/sigma.urdf and opens RViz. Add the
# RobotModel display and set "Description Topic" to /robot_description
# (Fixed Frame: base_link).
#
# Joint angles come from /joint_states -- RViz alone shows a statue.  Drive
# the joints from another terminal:
#
#   python3 apps/animate.py --tour      # walks the real M-library, a new motion
#                                       #   every 10 s: 0.3 -> 0.8 Hz, A10 -> A20,
#                                       #   resume, taper -- run this for a demo
#   python3 apps/animate.py --motion M16   # hold one entry
#   python3 apps/animate.py --rock      # one plain sine, 1.5 Hz +-20 deg
#   python3 serve.py --baby             # the machine's real motions (subtle:
#                                       #   full sway is only ~2.5 deg; add
#                                       #   --viz-gain 5 to exaggerate in RViz)
#   python3 apps/animate.py --sweep     # +-45 deg one joint at a time, wiring check
set -euo pipefail
cd "$(dirname "$0")/.."

URDF="cad/sigma.urdf"
[ -f "$URDF" ] || { echo "missing $URDF -- run: python3 tools/make_urdf.py"; exit 1; }

echo "RViz shows a statue until something publishes /joint_states -- run"
echo "  python3 apps/animate.py --tour           (the real library, varied)"
echo "  python3 apps/animate.py --rock           (one sine; --hz/--deg to taste)"
echo "  python3 serve.py --baby --viz-gain 5     (the real machine, subtler)"
echo "in another terminal to see it move."
ros2 run robot_state_publisher robot_state_publisher \
    --ros-args -p robot_description:="$(cat "$URDF")" &
RSP=$!
trap 'kill $RSP 2>/dev/null || true' EXIT
sleep 2
# -d loads a config that already has the RobotModel display, the right topic
# QoS and a camera framed on a 0.45 m robot. Without it RViz opens empty and
# you have to add all three by hand.
rviz2 -d cad/sigma.rviz

#!/usr/bin/env bash
# Show the robot in RViz.
#
#   ./cad/view.sh
#
# Starts robot_state_publisher with cad/sigma.urdf and opens RViz. Add the
# RobotModel display and set "Description Topic" to /robot_description
# (Fixed Frame: base_link).
#
# Joint angles come from /joint_states. Nothing publishes those yet, so with
# the current all-fixed URDF the robot just stands there -- that is expected.
set -euo pipefail
cd "$(dirname "$0")/.."

URDF="cad/sigma.urdf"
[ -f "$URDF" ] || { echo "missing $URDF -- run: python3 make_urdf.py"; exit 1; }

ros2 run robot_state_publisher robot_state_publisher \
    --ros-args -p robot_description:="$(cat "$URDF")" &
RSP=$!
trap 'kill $RSP 2>/dev/null || true' EXIT
sleep 2
# -d loads a config that already has the RobotModel display, the right topic
# QoS and a camera framed on a 0.45 m robot. Without it RViz opens empty and
# you have to add all three by hand.
rviz2 -d cad/sigma.rviz

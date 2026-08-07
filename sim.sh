#!/usr/bin/env bash
# Start the phorce simulator with this project's motion library.
#
#   ./sim.sh                  # start (stops any previous one first)
#   ./sim.sh <motions_dir>    # a specific library (default: motions_m50
#                             #   if compiled, else the 10 DREAM slots)
#   ./sim.sh --stop           # just stop
#
# Resolves its own absolute path, so it does not matter which directory you run
# it from -- passing motion_dir:=$PWD/motions from the wrong cwd is the single
# easiest way to get "모션 디렉토리를 열 수 없습니다".
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${1:-}" ] && [ "$1" != "--stop" ]; then
    MOTIONS="$(cd "$1" 2>/dev/null && pwd)" || { echo "no such dir: $1"; exit 1; }
elif [ -d "$ROOT/motions_m50" ]; then
    MOTIONS="$ROOT/motions_m50"   # the full M01-M50 library, when compiled
else
    MOTIONS="$ROOT/motions"
fi

# Stop any running simulator. Matched on the installed binary path rather than
# with `pkill -f motion_action_server`, which also matches the shell running
# this script and kills it mid-run.
stop() {
    local pids
    pids=$(pgrep -f "/opt/ros/humble/lib/agx_motion_slot/motion_action_server" || true)
    pids="$pids $(pgrep -f "ros2 launch agx_bringup" | grep -v "^$$\$" || true)"
    pids=$(echo $pids | tr ' ' '\n' | grep -E '^[0-9]+$' | grep -v "^$$\$" | sort -u)
    if [ -n "$pids" ]; then
        echo "stopping: $(echo $pids | tr '\n' ' ')"
        kill $pids 2>/dev/null
        sleep 3
    fi
}

[ "${1:-}" = "--stop" ] && { stop; echo "stopped."; exit 0; }

[ -d "$MOTIONS" ] || { echo "no $MOTIONS -- run: python3 tools/make_motions.py --library"; exit 1; }
count=$(ls "$MOTIONS"/motion_*.csv 2>/dev/null | wc -l)
[ "$count" -gt 0 ] || { echo "no motion_*.csv in $MOTIONS -- run: python3 tools/make_motions.py --library"; exit 1; }

stop
echo "starting simulator with $count motions from $MOTIONS"
ros2 launch agx_bringup motion.launch.py "motion_dir:=$MOTIONS" \
    > /tmp/sigma-sim.log 2>&1 &

for _ in $(seq 1 15); do
    sleep 1
    if phorce list --target sim:demo >/dev/null 2>&1; then
        echo
        phorce list --target sim:demo
        echo
        echo "ready.  log: /tmp/sigma-sim.log"
        echo "  phorce play 3 --target sim:demo"
        echo "  python3 $ROOT/serve.py --baby --robot sim:demo"
        exit 0
    fi
done

echo "simulator did not come up in 15s -- see /tmp/sigma-sim.log"
exit 1

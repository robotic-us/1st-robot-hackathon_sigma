#!/usr/bin/env bash
# Bring up the REAL robot (EtherCAT) with the auto axis profile.
#
#   ./robot.sh            # start phorce_monitor + motion_action_server
#   ./robot.sh --verify   # run the organizer checklist against a running pair
#   ./robot.sh --stop     # stop both
#
# Transcribes docs/changes/2026-08-06-jetson-auto-axis-profile.md from the
# organizer guide: the monitor runs with mode:=op_idle, axes:=auto and adopts
# the axis mask the PCM actually reports (250 consecutive valid frames) instead
# of a fixed profile; the action server drives the hardware with backend:=ecat.
# Every process talking to the robot -- including `phorce play` and serve.py in
# other terminals -- must share ROS_DOMAIN_ID=21 or it silently sees nothing.
#
# What this does NOT change (same doc): PCM firmware, the slots loaded on the
# PCM, or slot<->hardware axis compatibility. `phorce list` is still how you
# check a slot exists before playing it.
set -uo pipefail

export ROS_DOMAIN_ID=21
NIC=eno1
MON_LOG=/tmp/sigma-robot-monitor.log
ACT_LOG=/tmp/sigma-robot-motion.log

# Matched on installed binary paths, not `pkill -f`, for the same reason as
# sim.sh: a looser pattern also matches this script and kills it mid-run.
# motion_action_server is the same binary the simulator uses (different
# backend), so this also clears a leftover sim -- only one may own the action.
stop() {
    local pids
    pids=$(pgrep -f "/opt/ros/humble/lib/agx_phorce_bridge/phorce_monitor" || true)
    pids="$pids $(pgrep -f "/opt/ros/humble/lib/agx_motion_slot/motion_action_server" || true)"
    pids=$(echo $pids | tr ' ' '\n' | grep -E '^[0-9]+$' | grep -v "^$$\$" | sort -u)
    if [ -n "$pids" ]; then
        echo "stopping: $(echo $pids | tr '\n' ' ')"
        kill $pids 2>/dev/null
        sleep 3
    fi
}

doctor_state() {  # READY / NOT READY / ...ERROR -- one notion of "ready" for
    phorce doctor 2>/dev/null |            # the wait loop and --verify both
        grep -oE 'READY|NOT[ _]READY|[A-Z_]*ERROR' | head -1
}

# field <yaml> <key> -- one field out of a captured /phorce/status message.
field() { echo "$1" | awk -F': ' -v k="$2" '$1 == k {print $2; exit}'; }

check() {  # check <label> <got> <want>  -- prints a checklist row, returns 1 on mismatch
    if [ "$2" = "$3" ]; then
        printf '  ok    %-22s %s\n' "$1" "$2"
    else
        printf '  FAIL  %-22s %s (want %s)\n' "$1" "${2:-<none>}" "$3"
        return 1
    fi
}

verify() {
    # Two topic echoes total: one message answers every mask/flag field, and a
    # second sample exists only to see verdict_pass move.  Each `ros2 topic
    # echo` pays 1-2 s of CLI startup + DDS discovery, so per-field calls made
    # the checklist take ~10 s longer than the thing it checks.
    local bad=0 doctor st oper v1 v2
    doctor=$(doctor_state)
    check "phorce doctor" "${doctor:-<no answer>}" "READY" || bad=1
    check "axes" "$(ros2 param get /phorce_monitor axes 2>/dev/null | awk '{print $NF}')" "auto" || bad=1
    check "mode" "$(ros2 param get /phorce_monitor mode 2>/dev/null | awk '{print $NF}')" "op_idle" || bad=1
    st=$(ros2 topic echo /phorce/status --once 2>/dev/null)
    oper=$(field "$st" axis_oper_mask)
    if [ -n "${oper:-}" ] && [ "$oper" != "0" ]; then
        printf '  ok    %-22s %s\n' "axis_oper_mask" "$oper (nonzero)"
    else
        printf '  FAIL  %-22s %s (want nonzero -- auto profile not adopted?)\n' \
            "axis_oper_mask" "${oper:-<none>}"; bad=1
    fi
    check "axis_stale_mask" "$(field "$st" axis_stale_mask)" "0" || bad=1
    check "axis_fault_mask" "$(field "$st" axis_fault_mask)" "0" || bad=1
    check "mbx_veto_active" "$(field "$st" mbx_veto_active)" "false" || bad=1
    check "ethercat_operational" "$(field "$st" ethercat_operational)" "true" || bad=1
    v1=$(field "$st" verdict_pass); sleep 1
    v2=$(field "$(ros2 topic echo /phorce/status --once 2>/dev/null)" verdict_pass)
    if [ -n "${v1:-}" ] && [ -n "${v2:-}" ] && [ "$v2" -gt "$v1" ] 2>/dev/null; then
        printf '  ok    %-22s %s -> %s (increasing)\n' "verdict_pass" "$v1" "$v2"
    else
        printf '  FAIL  %-22s %s -> %s (want increasing)\n' \
            "verdict_pass" "${v1:-<none>}" "${v2:-<none>}"; bad=1
    fi
    return $bad
}

case "${1:-}" in
--stop)   stop; echo "stopped."; exit 0 ;;
--verify)
    if verify; then echo; echo "all checks passed -- ready to play motion slots."; exit 0
    else echo; echo "checklist FAILED -- see $MON_LOG / $ACT_LOG"; exit 1; fi ;;
"") ;;
*)  echo "usage: ./robot.sh [--verify|--stop]"; exit 2 ;;
esac

# The EtherCAT link is physical: no carrier on the NIC means the robot is off
# or unplugged, and everything after this would only fail slower.
if ! grep -qs 1 "/sys/class/net/$NIC/carrier"; then
    echo "no link on $NIC -- is the robot powered and the EtherCAT cable in?"
    exit 1
fi

stop
echo "starting phorce_monitor (mode:=op_idle axes:=auto) + motion_action_server (backend:=ecat)"
ros2 run agx_phorce_bridge phorce_monitor --ros-args \
    -p nic:="$NIC" -p mode:=op_idle -p axes:=auto -p mbx_enabled:=true \
    > "$MON_LOG" 2>&1 &
ros2 run agx_motion_slot motion_action_server --ros-args \
    -p backend:=ecat \
    > "$ACT_LOG" 2>&1 &

# The monitor needs 250 consecutive valid PCM frames before it adopts the mask;
# give the whole stack a generous window to reach READY.
for _ in $(seq 1 30); do
    sleep 1
    if [ "$(doctor_state)" = "READY" ]; then
        echo
        verify || true
        echo
        echo "up.  logs: $MON_LOG  $ACT_LOG"
        echo "every other terminal needs:  export ROS_DOMAIN_ID=21"
        echo "  phorce list          # slots actually on the PCM"
        echo "  phorce play <id>     # play one"
        exit 0
    fi
done

echo "robot did not reach READY in 30s -- see $MON_LOG"
exit 1

# TEST.md — the tested command lines

One sentence per command; all verified on this Jetson (2026-08-08); run from the repo root.

## The demo

```bash
./robot.sh                                   # terminal A: rig bring-up (--verify grades it)
export ROS_DOMAIN_ID=21                      # needed in every robot terminal
python3 serve.py --sense --robot cli         # terminal B: camera → machine → SD slots
python3 serve.py --sense --robot cli --crop 0.25,0.1,0.5,0.8 --policy reflex --personality
python3 serve.py --sense --robot cli --crop 0.25,0.1,0.5,0.8
```

Dashboard `https://<jetson-ip>:8080`, iPad face `/baby` — HTTPS is automatic; install `.local-certs/sigma-ca.crt` on the iPad once.

## One motion only

```bash
python3 serve.py --play 5 --robot cli        # loop SD slot 5 (1..14); auto off, gate on
python3 serve.py --play 5 --once --robot cli # ...or play it a single time, then part
```

## Each layer alone

```bash
python3 tests.py                             # every suite, headless
python3 serve.py --verify                    # scripted episode, no camera
python3 serve.py --baby --personality --policy dream   # virtual infant + planner
python3 perception/nubzuki.py docs/Nubzuki.jpg --grade # recognizer vs the sheet
python3 serve.py --sense                     # camera → machine, no robot
./sim.sh && python3 serve.py --baby --robot cli:sim:demo  # robot path, no hardware
```

`./sim.sh` moves nothing — judge motion in RViz (`./cad/view.sh` + `python3 apps/animate.py --tour`).

## Facts

- Robot plays **one episode per decision** (no replay while the same motion is held; a switch or park+re-command plays again), with **1 s rest** between any two plays. `--play N` without `--once` still loops its slot.
- Card cap = **slots 1..14** (default with `--robot`; lift with `--slots 34`); off-card motions log "screen only".
- Shake the iPad hard (2→7 m/s²) → rage; drag the wheel → any pose; cover the camera → gate in ~2.7 s.
- Vision alone caps at the fuss band (§5); the safety gate runs in every mode.

## Troubleshooting

| Symptom | Fix |
|---|---|
| camera would not open | `fuser -v /dev/video0` — stop the other holder |
| rig silent, screen rocking | event log: "not on the card" or "NOT READY — zero button" |
| phorce refuses everything | `phorce doctor`; `export ROS_DOMAIN_ID=21` |
| iPad face but no sensor | must be https + CA trusted |
| frozen numbers | header dot red = stream dropped; reload |

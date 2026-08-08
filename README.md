# SIGMA — Infant-Responsive Robotic Cradle

> Team **SIGMA** (Seoul National University) at the 1st Robot Hackathon 2026,
> hosted by [Roboticus](https://robotic-us.com) at KAIST, Aug 5–8 2026.
> Members: Sieun Baek · Jeonghwan Kim · Jinmyoung Lee.

A research prototype of a robotic cradle that watches an infant's state and
decides — carefully, reversibly, safety-first — when and how to rock. Every
motion it may play is transcribed from an evidence report
(`docs/infant_robotic_cradle_evidence_report_ko.pdf`); on top of that fixed
safety ladder sits a learning layer that personalises the soothing motion to
*this* baby.

**This is a hackathon research prototype, not a medical device.** Nothing here
may be used with a real infant before the report's own preconditions
(pediatrics, biomechanics, ethics review) are met — see
[docs/VERIFICATION.md](docs/VERIFICATION.md) for how the system is verified *without* one.

## How it works

```
   SENSING                         DECIDING                    MOVING
┌─────────────────────┐   ┌──────────────────────┐   ┌─────────────────────┐
│ --verify  scripted  │   │ CradleMachine        │   │ MotionEngine        │
│ --baby    virtual   │──►│  safety ladder (§5)  │──►│  M01–M50 + N01–N34  │
│ --sense   camera    │   │  + SoothePolicy      │   │  envelope-gated     │
│           reads the │   │    (advises only)    │   │  0.2–0.8 Hz ≤30 mm  │
│           cradle    │   └──────────────────────┘   └──────────┬──────────┘
│           iPad      │                                         │ five-bar IK
└─────────────────────┘                                         ▼ (core/rig.py)
                                          web dashboard · RViz · real robot (phorce)
```

- **Evidence-based motion library.** M01–M50 from the report plus the team's
  N01–N34 system, both gated at import against the kinematic envelope
  (0.2–0.8 Hz, amplitude ≤ 30 mm, a_peak ≤ 0.05 g). Research-grade motions
  are refused without an explicit `--research` flag and never play
  automatically.
- **Safety-first state machine.** A calm infant is left alone; sustained
  distress opens a soft-started trial; a jam or a lost face tapers to zero in
  5 s and alerts the caregiver — the gate runs in every mode.
- **Personalised soothing.** A virtual infant with a hidden motion temperament
  (habituation, mood) closes the loop, and interchangeable decision brains —
  `reflex`, `dream` (the DREAM-Chunk planner), `ollama`, `claude` — advise
  which motion each trial uses. The planner measures **−26.4 % upset / −57.1 %
  crying vs the fixed ladder** over 40 paired simulated nights
  ([docs/DREAM-CHUNK.md](docs/DREAM-CHUNK.md)).
- **A verifiable vision loop.** The cradle iPad draws the infant's state as
  the Nubzuki mascot; a camera pointed at the iPad reads the pose back and
  checks it against what was sent — a vision test with labelled ground truth
  (17/17 on both reference corpora), which abstains rather than guesses.
- **Real hardware.** Two five-bar linkages (sway / heave / pitch channels)
  driven over EtherCAT through the phorce SDK; the same decisions replay as
  PCM slots on the rig, in the simulator, and in RViz.

## Getting started

Python 3 with NumPy and OpenCV is enough for everything except the real robot
(ROS 2 / phorce ship with the Jetson image and are never required on a laptop):

```bash
pip install -r requirements.txt
```

```bash
python3 tests.py                                # all 9 suites, headless, ~30 s
python3 serve.py --verify                       # dashboard on :8080, scripted episode, no camera
python3 serve.py --baby                         # closed loop against the virtual infant
python3 serve.py --baby --personality --policy dream   # + hidden temperament + the planner
```

Open `https://localhost:8080` for the dashboard (an HTTPS certificate is
created automatically under `.local-certs/`). Run everything from the repo
root — data paths resolve from the working directory.

## The full demo (real robot)

The command lines below are the tested set — one sentence each in
[docs/DEMO.md](docs/DEMO.md), including troubleshooting.

```bash
./robot.sh                                   # terminal A: rig bring-up (--verify grades the checklist)
export ROS_DOMAIN_ID=21                      # required in every terminal that talks to the robot
python3 serve.py --sense --robot cli         # terminal B: camera → machine → SD-card slots
```

No hardware? The same robot code path runs against the phorce simulator
(`./sim.sh`, then `--robot cli:sim:demo`) — but the simulator validates the
motion *contract* only and moves nothing; judge trajectories in RViz
(`./cad/view.sh` + `python3 apps/animate.py --tour`).

### The cradle iPad

Open `https://<host>:8080/baby` on a tablet placed in the cradle: it draws the
animated infant face and (after **Start motion sensor**, HTTPS required)
streams real IMU measurements back — so the virtual baby is soothed by the
*measured* motion of the cradle, not the commanded one. Install
`.local-certs/sigma-ca.crt` on the tablet once and trust it under
Settings → General → About → Certificate Trust Settings; the private keys in
`.local-certs/` never leave the machine.

## Repository layout

| Path | Role |
|---|---|
| `serve.py` | ★ Entry point: sensing → CradleMachine → MotionEngine → web dashboard, one process |
| `tests.py` | The only test entry — all 9 suites (`--list` to enumerate) |
| `core/cradle.py` | M01–M50 + N01–N34 libraries, MotionEngine, CradleMachine — the heart |
| `core/policy.py` | The soothing policy: brains (reflex/dream/ollama/claude), WorldModel, ChunkMatcher |
| `core/rig.py` | Measured five-bar kinematics + IK — the one home of the geometry convention |
| `core/pvector.py` · `core/phorce_iface.py` | P-Vector world model · the only file that knows phorce/ROS 2 (+ SlotBridge) |
| `perception/` | `nubzuki.py` (the vision system — reads the iPad, never an infant), `baby.py` (virtual infant) |
| `apps/animate.py` | RViz player for the motion library (`--tour`) |
| `tools/` | Generators: motions, URDF, mascot templates, figures (`learn_report.py`, `state_figure.py`), camera aiming |
| `web/` · `webapp/` | The dashboard (no build step) · the same dashboard as a Next.js client (optional, needs Node) |
| `motions_m50/` · `motions_n34/` | The libraries compiled to phorce PCM slots |
| `cad/` · `stl/` | URDF + meshes for RViz (`./cad/view.sh`) |
| `docs/` | Design docs, the evidence report, competition guides — see below |

## Documentation

| Document | What it covers |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | The whole design + report ↔ code traceability table |
| [docs/DEMO.md](docs/DEMO.md) | The tested command lines for the demo, one sentence each |
| [docs/VERIFICATION.md](docs/VERIFICATION.md) | How the cradle is verified without an infant (layered protocol) |
| [docs/DREAM-CHUNK.md](docs/DREAM-CHUNK.md) | The DREAM-Chunk planner: design, estimator audit, all measurements |
| [docs/IDEA.md](docs/IDEA.md) | The personalisation idea the policy layer implements |
| `docs/infant_robotic_cradle_evidence_report_ko.pdf` | **The spec** (Korean). When motion behaviour is in question, it wins |
| `docs/RH_Guide*/` | Competition-provided material (SDK docs, papers, wiring) |

## Testing

```bash
python3 tests.py             # everything: no robot, no camera, no ROS needed
python3 tests.py cradle m50  # pick suites
python3 tests.py --list      # what exists
```

Suites: `listen` `cradle` `baby` `policy` `m50` `nubzuki` `animate` `serve`
`bridge`. New behaviour goes into `tests.py` — modules carry no self-tests.

## Safety properties (do not weaken)

- The safety gate (jam, or face lost > 0.7 s → taper to zero in 5 s +
  caregiver alert) runs even when automatic care is off.
- Vision alone is capped below the cry threshold: no visual reading can open
  the escalation ladder without audio confirmation.
- R-grade motions (M29–M50) are research-only and refused without
  `--research`; they are never routed into automatic behaviour.
- The default value of automation is *not moving*: a calm infant is left
  alone.

## Team & license

Team **SIGMA** (Seoul National University): Sieun Baek · Jeonghwan Kim ·
Jinmyoung Lee.

MIT — see [LICENSE](LICENSE). The intellectual property of this project
belongs to the SIGMA team (all members); the competition organiser (Roboticus)
uses this repository for archival and promotional purposes only.

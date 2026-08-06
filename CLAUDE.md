# CLAUDE.md

SIGMA — an infant-responsive robotic cradle (research prototype, Robot
Hackathon 2026, KAIST). The motion behaviour is a transcription of
`docs/infant_robotic_cradle_evidence_report_ko.pdf` (Korean): a 50-motion
library (M01–M50), a kinematic envelope, and a safety-first state machine.
That PDF is the spec — when motion behaviour is in question, it wins.
Architecture and the report↔code traceability table: `docs/ARCHITECTURE.md`.

## Commands

| Command | What it does |
|---|---|
| `python3 tests.py` | **The only test entry.** 10 suites, headless, ~30 s. `--list` shows them; suite names as args to filter. |
| `python3 serve.py --fake` | Dashboard on :8080 with a synthetic state-card scenario (no camera). |
| `python3 serve.py` | Real camera + AprilTag state cards (tag_0 calm / tag_1 fuss / tag_2 cry). |
| `python3 serve.py --sense` | Real testing: face emotion + mic cry-band drive the machine. Needs models. |
| `python3 tools/fetch_models.py` | Fetch/verify the 3 ONNX models (SHA-256 pinned). |
| `python3 tools/make_motions.py --library` | Compile M01–M50 → `motions_m50/motion_NN.csv` (pcm slots). |
| `./sim.sh [dir]` | phorce simulator; defaults to `motions_m50/` when present. `--stop` to kill. |
| `./cad/view.sh` | robot_state_publisher + RViz with `cad/sigma.urdf`. |
| `python3 tools/make_urdf.py` | Regenerate URDF + per-link meshes from `docs/udrf_assembly.stl`. |

## Layout

- `serve.py` — the entry point: sensing → CradleMachine → MotionEngine → SSE/MJPEG dashboard (`web/index.html`). 2D only; 3D lives in RViz.
- `core/` — `cradle.py` (M01–M50 library + MotionEngine + CradleMachine — the heart), `dream.py`/`pvector.py` (DREAM-Chunk matcher + P-Vector world model), `slot_table.py`, `phorce_iface.py` (only file that knows phorce/ROS 2).
- `perception/` — `tag.py` (state cards), `sense.py` + `listen.py` (face+audio → distress), `face/` (YuNet/SFace/FER+ package; was top-level `sigma/`).
- `apps/` — runnable demos: `demo.py` (DREAM tag demo; also exports the CAD geometry constants `AXIS0/PIVOT/PLATE`), `care.py`, `run.py`, `animate.py` (RViz player).
- `tools/` — one-off generators: `make_motions.py`, `make_urdf.py`, `fetch_models.py`, `enroll.py`.
- Data: `motions/` (10 DREAM slots — tests expect exactly 10), `motions_m50/` (the 50-slot library), `models/` (ONNX, gitignored), `tags/`, `slots.json`, `cad/`, `faces/` (biometric, gitignored).

## Conventions

- **Run everything from the repo root** — data paths (`slots.json`, `motions/`, `cad/`) resolve from cwd. Scripts in subdirs still run directly thanks to a `sys.path` bootstrap block at the top of each.
- **Tests live only in `tests.py`** — modules carry no selftests and no `--selftest` flags. New behaviour → new/extended suite there, registered in `SUITES`.
- Time is injected everywhere in `core/cradle.py` (no `time.time()` inside) so the machine is testable with a fake clock. Keep it that way.
- Module docstrings follow a house style: what it is, why it exists, run lines. Match it.

## Hard-won facts (do not rediscover)

- **OpenCV is 4.5.4** on this Jetson → YuNet must stay the 2022mar model (2023 head won't parse). `tools/fetch_models.py` pins commit + SHA-256.
- **pcm MotionMap rules**: MS ID inside the CSV must match the filename number; `MD ID` is 1-based; P-Vector cell = `"yd_deg, L_traj_ms(1 kHz), s0, sd"` with `yd` an *absolute* target in degrees; slot ids capped at `MAX_MOTION_ID = 50`.
- **Lever arm**: plate offset ↔ arm angle via `PIVOT[2] - AXIS0[2]` (≈227 mm) → 1° ≈ 4 mm of sway; A=10 mm ≈ 2.53°. Constants live in `apps/demo.py`, printed by `make_urdf.py` on every rebuild — update them when the CAD changes.
- **R-grade motions** (M29–M50) are research-only: engine refuses them without `--research`; never route them into automatic behaviour.
- The rig is a parallelogram rocker with **one horizontal DOF**: all four axes get the same angle, ML/AP land on the same axis, Z modes can't play.
- The safety gate (jam or face/tag lost >0.7 s → taper to zero in 5 s + caregiver alert) runs even when auto mode is off. Don't weaken it.
- `--dream-auto` restores the legacy tag-shake→DREAM-slot autopilot; default auto is the report's machine. The two are mutually exclusive with `--sense`.

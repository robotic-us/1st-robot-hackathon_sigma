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
| `python3 serve.py --baby` | Closed-loop sim: a virtual infant (random state process, `perception/baby.py`) the sway genuinely soothes; drawn as a circle in the dashboard, cradle answers in RViz. `--baby-seed N` for repeatable runs. |
| `python3 tools/fetch_models.py` | Fetch/verify the 3 ONNX models (SHA-256 pinned). |
| `python3 tools/make_motions.py --library` | Compile M01–M50 → `motions_m50/motion_NN.csv` (pcm slots). |
| `./sim.sh [dir]` | phorce simulator; defaults to `motions_m50/` when present. `--stop` to kill. |
| `./cad/view.sh` | robot_state_publisher + RViz with `cad/sigma.urdf`. |
| `python3 apps/animate.py --tour` | Drives RViz from the real M-library, a new motion every 10 s — the demo to run beside `view.sh`. `--motion M16` holds one entry, `--research` adds the R-grade shapes, `--rock` is a plain sine for the projector. |
| `python3 tools/make_urdf.py` | Regenerate URDF + per-link meshes from `docs/udrf_assembly.stl`. |

## Layout

- `serve.py` — the entry point: sensing → CradleMachine → MotionEngine → SSE/MJPEG dashboard (`web/index.html`). 2D only; 3D lives in RViz.
- `core/` — `cradle.py` (M01–M50 library + MotionEngine + CradleMachine — the heart), `dream.py`/`pvector.py` (DREAM-Chunk matcher + P-Vector world model), `slot_table.py`, `phorce_iface.py` (only file that knows phorce/ROS 2).
- `perception/` — `tag.py` (state cards), `sense.py` + `listen.py` (face+audio → distress), `face/` (YuNet/SFace/FER+ package; was top-level `sigma/`).
- `apps/` — runnable demos: `demo.py` (DREAM tag demo; also exports the CAD geometry constants `AXIS0/PIVOT/PLATE`), `care.py`, `run.py`, `animate.py` (RViz player: `--tour`/`--motion` replay the real library through `MotionEngine`; `--slot`/`--all` play DREAM chunks).
- `webapp/` — the dashboard as a Next.js app, a *client* of `serve.py` and not a replacement: `web/index.html` is still what the server serves and what `tests.py` checks. Needs a Node toolchain this repo does not ship; see `webapp/README.md`.
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
- **Lever arm**: `PIVOT[2] - AXIS0[2]` (≈227 mm) → 1° ≈ 4 mm of sway is the *aggregate* gain, and it is still the right number for a back-of-envelope check. It is not a per-crank conversion — see the five-bar facts below. Constants live in `apps/demo.py`, printed by `make_urdf.py` on every rebuild.
- **R-grade motions** (M29–M50) are research-only: engine refuses them without `--research`; never route them into automatic behaviour.
- **The rig is two five-bar linkages, not a parallelogram rocker.** Measured off `docs/udrf_assembly.stl` (contact graph + IK), not assumed. Every leg is a *two-link chain* — `axis_N` → `lower_N` (crank, 152 mm) → knee → `upper_N` (rod, 152 mm) → platform — and all four legs are the same part. Legs 0 and 1 share one base pivot and meet at **one** platform point with mirrored elbows (+35.6°/−47.0°); legs 2 and 3 likewise. Two cranks, two rods, one shared endpoint = a closed five-bar, 2 DOF per pair. The passive knee absorbs the reach change a rigid parallelogram could not.
- **Three channels, and Z is real.** `sway` (all four cranks the same way) → horizontal, **4.17 mm/deg**. `heave` (each pair's two cranks *opposed*) → **vertical, 3.02 mm/deg**. `pitch` (pair against pair) → see-saw. Cross-checked: sway A10 solves to 2.40° against the report's documented 2.53°. The old "Z modes can't play / no vertical DOF" claim was wrong — M48–M50 now compile to real motion.
- **The assembly STL is not finished, so do not read the rest pose off it.** The top four bearings are not yet mated to the table: the platform is only resting in place, the two pairs sit at unequal reach (228.3 vs 262.0 mm) and the table is tilted 9.7° — confirmed twice over, by its deck normal `(-0.17, 0, 0.99)` and by the line through its mounts. Taking `REST_A`/`REST_B` from the export baked that tilt into every motion. What the export *does* pin down (because it does not depend on the arms) is that the table is rigid with coplanar mounts 200.4/199.6 mm apart, matching the 200.0 mm base-pivot separation — so the neutral pose is the symmetric one: equal reach, each endpoint above its own base, table level, cranks at ±36.43°. `RIDE_MM = 245.2` (the mean of the two observed reaches) sets where in the travel that sits and is **the one number still worth measuring on the finished rig**.
- **AP does not exist as a translation.** Every joint turns about Y, so all motion is in the XZ plane; there is no second horizontal axis. The report's 11 AP entries are rendered as **pitch** (`AP_AS_PITCH` in `apps/demo.py`) — a deliberate substitution, not the report's motion.
- **`apps/demo.py::axis_angles()` / `cradle_angles()` is the one definition.** It solves the real five-bar IK (reproduces the CAD's own crank angles to 0.6°). The compiler, `serve.py`'s RViz mirror and `apps/animate.py` all go through it; `tests.py::m50` and `::animate` assert both the CSVs and the live path. Fixing only one of the three silently collapses everything back into the same sway — that has now happened twice.
- **The linkage is nonlinear, so bounds need both directions.** At the level neutral the two pairs are identical and 10 mm of sway costs [2.27, 2.40, 2.27, 2.40]° — but the cranks still differ within a pair, −20 mm costs more crank angle than +20 mm, and pitch/heave cost more per millimetre than sway (3.02 vs 4.17 mm/deg). Any ceiling computed from sway alone, or from one sign, under-counts; `tests.py::animate` derives its ceiling over every channel and both signs for exactly this reason.
- **Every upper arm joins the holder; the tree is re-rooted to make that true.** A URDF is a tree, so the holder can have only one parent — but all four arms carry it in hardware, and a model where three join nothing is not the machine. Leg 0 runs *up* to the holder and legs 1–3 hang *down* from it: `base_link → axis_0 → lower_0 → upper_0 → platform → upper_{1,2,3} → lower_{1,2,3} → axis_{1,2,3}`. The knees are real joints too (`joint_knee_N`), not welds. The open end is therefore at the actuators — and they are bolted down, so **closure means each actuator lands back on its own pivot**: `tests.py::animate` walks the published joints through the real URDF exactly as `robot_state_publisher` does and asserts this (worst 0.025 mm across all channels).
- **`joint_state()`'s first four entries are NOT the four cranks.** The order is `axis_0, knee_0, platform, bearing_1, knee_1, bearing_2, knee_2, bearing_3, knee_3` — it interleaves, because each leg hangs off a different parent. Use `cradle_cranks()` for anything measuring amplitude; `library_angles(..., solve=)` picks which. Indexing `[:4]` for "the cranks" is wrong and reads as a wildly wrong amplitude. **`apps/demo.py::JOINT_NAMES` is the only joint-name list** — both publishers (`RosSide`, `apps/animate.py`) reference it, never a copy; a stale copy is how RViz broke after the re-root (nonexistent `joint_axis_1/2/3` published, real `joint_bearing_1/2/3` never published, three arms with no TF). `tests.py::animate` asserts the list matches the URDF's revolute joints and that both publishers share the one object.
- **Angles are measured from +Z toward +X, everywhere, via `_ang()`.** Writing the holder's tilt the natural way (`atan2(Δz, Δx)`, from +X) flips its sign against the joint convention — the holder then rotates the wrong way and *doubles* its tilt instead of levelling, which put the far actuators 68 mm off their pivots. One convention or nothing lines up.
- **Solve each leg against its own mount, with its own link lengths.** The holder's four mounts sit a few mm apart, and the legs measure 151.7–152.5 (crank) / 151.9–153.4 (rod). Treating each pair as sharing one endpoint, or using one rounded length for all four, each cost about a millimetre of closure error.
- **`--viz-gain` multiplies plate travel, not joint angles.** The linkage is nonlinear, so scaling the angles produces a pose the mechanism cannot reach and lifts the rods off their bearings. `cradle_joint_state(..., gain=)` scales the travel and re-solves, so the exaggerated pose is a real pose. At gain 5 the worst joint reaches 39.6°, well inside the ±90° limit.
- **`./sim.sh` does not move anything.** The phorce sim backend announces itself in `/tmp/sigma-sim.log` as a `mechanism-neutral fake PCM ... 로봇은 움직이지 않습니다 (aggregate 실행 800ms)`: it validates MS IDs, request tokens and completion semantics, then returns SUCCEEDED after a fixed ~0.8 s without walking the P-Vector chain. Measured — a 9-segment slot and a 128-segment 120 s taper both take 0.78 s, and `pvector_index` never leaves 255. Never use it to judge a trajectory; use RViz (`cad/view.sh` + `apps/animate.py --tour`).
- The safety gate (jam or face/tag lost >0.7 s → taper to zero in 5 s + caregiver alert) runs even when auto mode is off. Don't weaken it.
- `--dream-auto` restores the legacy tag-shake→DREAM-slot autopilot; default auto is the report's machine. The two are mutually exclusive with `--sense`.

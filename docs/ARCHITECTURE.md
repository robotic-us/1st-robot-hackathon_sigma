# SIGMA Architecture

An infant-responsive robotic cradle. One sentence of philosophy drives the
whole design, straight from the evidence report
(`infant_robotic_cradle_evidence_report_ko.pdf`): **the default value of
automation is not moving.** Everything below exists to decide, carefully and
reversibly, when that default may be broken.

## The live loop

```
                 SENSING (one of two modes)
  ┌──────────────────────────────┬──────────────────────────────┐
  │ --verify (scripted, default) │ --baby (closed loop)         │
  │ serve.py episode; each phase │ perception/baby.py; random    │
  │ asserts its own state+level  │ infant state, and the engine's│
  │ (machine test, not a         │ sway feeds back as soothing   │
  │  recognizer test)            │                               │
  └──────────────┬───────────────┴───────────────┬──────────────┘
                 └────── present, level 0..1 ────┘

  (--sense and the infant recognizer were removed 2026-08-08.  The only
   camera path left is perception/nubzuki.py, which reads the *iPad* --
   it verifies the display loop and is capped in the fuss band, so it
   never feeds this input.  See docs/VERIFICATION.md.)
                                │
                    CradleMachine  (core/cradle.py, report §5)
        gate_fail > pain > stable_sleep > quiet_awake > cry trial
                                │  M-commands
                    MotionEngine   (core/cradle.py, report §6-7)
        M01-M50 + N01-N34 libraries · phase-continuous sines · S-curve ramps ≥5 s
        envelope asserted at import: 0.2-0.8 Hz, A ≤ 30 mm, a_peak ≤ 0.05 g
                                │  (ap, ml, z) mm → five-bar IK (core/rig.py)
                                ▼
                     joint angles (cradle_joint_state)
              ┌─────────────────┼──────────────────┐
              ▼                 ▼                  ▼
        web dashboard      RViz mirror        real arm / sim
        serve.py SSE/MJPEG /joint_states      SlotBridge (--robot),
        web/index.html     cad/sigma.urdf     core/phorce_iface.py
```

`serve.py` hosts this loop in one process: a sensor thread produces frames
and readings, the machine and engine tick on it, and plain-HTTP endpoints
(`/events` SSE, `/frame` MJPEG, `/history`, `/motions`, `/motion?id=`,
`/auto`, `/jam`, `/policy`, `/taste`, `/baby`, `/motion-sensor`) expose
everything to the browser. serve.py stays 2D; the
3D view is RViz's job.

## The safety ladder (report §5, `CradleMachine`)

1. **Safety gate** — jam (arm desync stand-in) or face/tag lost >0.7 s:
   amplitude to zero in 5 s, caregiver alert. Runs even with auto off.
2. **Stable sleep** — 60 s calm during a trial → TAPER_60 → hold M01.
3. **Quiet awake** — no motion. Ever. A calm infant is left alone.
4. **Cry trial** — sustained distress opens a 30 s soft-started sway
   (M10 for fussing, M12 for crying). Checkpoints every 30 s: improving →
   hold (max 5 min); not improving → one rung up the ladder
   (M10→M12→M13→M16); no improvement in 60 s or getting worse → taper +
   caregiver alert. One MICRO_RESUME (M08, 50 % amplitude) is allowed if
   fussing returns during a taper; after an alert, cooldown blocks restarts.

## The motion library (report §7, `LIBRARY` in core/cradle.py)

| Grade | IDs | Meaning |
|---|---|---|
| C0 | M01–M08 | Stop/transition commands: static hold, pause-observe, soft starts (30/60 s), tapers (30/60/120 s), micro-resume |
| P1 | M09–M28 | Trial candidates: single-axis horizontal sines, 0.2–0.8 Hz, A 10–20 mm (ML and AP sets) |
| R | M29–M50 | Research-only: diagonals, ellipses, circles, Lissajous, pseudo-walk, adaptive, vertical. **Refused without `--research`; never automatic.** |
| N | N01–N34 | The team's motion system (`docs/motion-system.png`: shapes × small-fast/large-slow × vibe × decay), built beside the report library in `N_LIBRARY` and envelope-gated the same way. Its 26 sustained entries (`N_CANDIDATES`) are the policy's whole search space |

The kinematic envelope is asserted when the module imports (the report's V0
gate): frequency band, hard 30 mm amplitude cap, theoretical base
a_peak ≤ 0.05 g per component.

## Two compiled artifacts, one source of truth

- **Live engine** (`MotionEngine`) — continuous, indefinite, retargetable
  mid-flight; what serve.py plays.
- **pcm slot catalog** (`tools/make_motions.py --library` →
  `motions_m50/motion_NN.csv`) — the same 50 motions as finite MotionMap
  episodes (chains of rest-to-rest quintic half-cycles; both a quintic and a
  sine have zero velocity at the extremes). MS ID k = M*k*. `./sim.sh` feeds
  them to the phorce simulator; `phorce play 12` plays M12.

Millimetres become joint degrees through the measured five-bar IK in
`core/rig.py` (`axis_angles()` / `cradle_angles()` — the one definition; the
compiler, serve.py's RViz mirror and `apps/animate.py` all go through it).
The rig is two five-bar linkages with three channels — sway (4.17 mm/deg,
horizontal), heave (3.02 mm/deg, vertical) and pitch (see-saw); there is no
AP translation axis, so the report's AP entries are deliberately rendered as
pitch (`AP_AS_PITCH`). The CAD lever `PIVOT − AXIS0` (≈227 mm → 1° ≈ 4 mm)
remains the right back-of-envelope check. URDF + meshes are regenerated by
`tools/make_urdf.py` from `docs/urdf_assembly.stl` (`cad/sigma.urdf`). The
team's N01–N34 system compiles separately via `make_motions.py --n34` →
`motions_n34/`, sampled through the same IK.

## DREAM-Chunk

The paper's ranking half, re-anchored from the arm onto the infant, lives in
`core/policy.py`: `WorldModel` learns what this baby makes of each motion
(taste, pooled over motion *features*; wear, computed from its own play
history; and transitions), and `ChunkMatcher` dreams every candidate forward
through it and ranks them on task fit + continuity + habituation. `DreamBrain`
wraps that as one of the interchangeable brains (`--policy dream`), and the
dashboard's *What it dreamed* card draws the ranking as a tree of futures.

The paper's monitoring half — a divergence tube cutting a motion that left its
dreamed curve — was implemented, measured at +2.9 % upset, and removed.
`docs/DREAM-CHUNK.md` carries that measurement, the estimator audit behind it,
and the tables for every arm. *(The pre-2026-08 autopilot it grew out of —
`--dream-auto`, `slots.json`, `core/dream.py` — is gone with the tag stack.)*

## Report → code traceability

| Report section | Code |
|---|---|
| §1 core recommendations (default = M01, trials, bans) | `CradleMachine` defaults, `TRIAL_LADDER` |
| §4 sensor states (5-state vocabulary) | kept in `perception/nubzuki.py` (visual level capped in the fuss band); the §4.2 audio features survive in `perception/listen.py` (orphaned since 2026-08-08, still tested) |
| §5 state-motion table + pseudocode + priority | `CradleMachine.tick` |
| §6 envelope, coordinates, 4-arm sync | `Motion.worst_components` import gate; jam→gate in serve.py |
| §7 the 50 motions | `LIBRARY` (`python3 core/cradle.py --catalog`) |
| §9.1 V0 software gate (reproduce 50 ids/units/axes/ramps) | import-time asserts + `tests.py m50` |

## Test matrix (`python3 tests.py`)

| Suite | Proves |
|---|---|
| listen | audio features: cry band vs hiss/noise/quiet |
| cradle | library gate, engine ramps, the whole safety ladder on a fake clock |
| baby | virtual infant dynamics; 20 sim-minutes of closed loop baby↔machine↔engine |
| policy | ranks, brains, the world model + matcher, the advised closed loop |
| m50 | 50 compiled slots round-trip within envelope, end at rest |
| nubzuki | the mascot recognizer, graded against the reference sheet it was built on |
| animate | the rig: one JOINT_NAMES list, real five-bar IK, closure to 0.025 mm |
| serve | every HTTP endpoint against a live server, card→trial end-to-end |
| bridge | SlotBridge: decisions become PCM slots (no phorce, no ROS) |

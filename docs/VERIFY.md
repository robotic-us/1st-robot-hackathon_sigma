# Verifying the recognizer without an infant

No real baby is available (and none should be near this prototype before the
report's own preconditions — pediatrics, biomechanics, ethics review — are
met). This is the layered protocol that stands in. Each layer catches what the
one below cannot; none of them claims what only a clinical study could.

## Layer 1 — synthetic streams (automated, every `python3 tests.py`)

**Watch it live:** `python3 serve.py --verify` acts a scripted nursery episode
(quiet → fussing → soothed → crying → sleep → a rollover → resettled) through
the real AudioTrack → InfantJudge → CradleMachine chain and visualises it on
the dashboard and the webapp — the camera panel shows the drawn baby acting
the script, watermarked as synthetic; the internals drawer shows the judge
state and the current phase.

The judge, audio track and posture rules are pure, so the suite drives them
with scripted signals and asserts the report's own rows:

- §4.2: quiet / fuss / cry decided by **duty cycle + vocal-unit length**,
  never loudness (short sparse whimpers ≠ chained wails).
- §4.3: a closure < 1 s is a blink; closure within 2 s of a vocalisation is
  never sleep; 10 s closed+quiet → SLEEP_TENTATIVE; +60 s low body flow →
  SLEEP_STABLE.
- §5: visual-only distress caps at the gentle band; only audio crosses
  CRY_LEVEL; pain/posture → alarm → the machine's gate.
- Posture: supine+level stays quiet, one rolled frame (a squirm) is ignored,
  each of the three cues (head roll ≥ 60°, shoulder visibility asymmetry,
  shoulder-width collapse) raises risk only when sustained ≥ 2 s.

## Layer 2 — closed loop against the virtual infant (automated)

`perception/baby.py`'s random state process is rendered as the *sensor
signals* a real baby would produce (cry-band duty when crying, closed eyes
when asleep, body flow when agitated), judged by the report layer, fed to the
real CradleMachine — and the machine's sway must actually soothe it. This
exercises recognizer → judge → machine → motion as one loop, seeded and
repeatable (`tests.py watch`, "closed loop").

## Layer 3 — bench protocol (manual, with a doll / photo / an adult)

The detectors (YuNet, FER+, MediaPipe, BlazePose) respond to a printed baby
photo, a doll with a face, or a teammate acting. Run `python3 serve.py
--sense` and walk this table; every row must behave as written **before the
demo**:

| Do this at the bench | Expect |
|---|---|
| Face visible, quiet room | QUIET_AWAKE, cradle still |
| Play a recorded cry from a phone (sustained wails) | CRY within ~3 s, M12 trial |
| Short sparse whimpers instead | FUSS_WEAK, M10 only — never escalates |
| Stop the audio, keep the face | level decays, trial ends, sleep taper |
| Cover the camera / remove the face | gate: taper to zero ≤ 5 s + alert |
| Rotate the photo/doll past ~60° and hold | posture alarm ≥ 2 s → gate + alert |
| Rotate briefly and return | nothing — a squirm is not a rollover |
| Close the doll's eyes (or tester's) 10 s, silent | SLEEP_TENTATIVE, taper begins |

Record one pass of this table as a video; replay it any time with
`python3 perception/watch.py --video clip.mp4 --no-display --jsonl out.jsonl`
and diff the state timeline. A recorded bench pass is the regression baseline
the camera path otherwise lacks.

## Layer 4 — logged recalibration (after the hackathon)

The report's own stance (§5.2 footnote): every numeric threshold is a v0
engineering value that must be recalibrated from logs and a pre-registered
study — not tuned live on an infant. The `--jsonl` stream exists so those
logs accumulate from day one.

## Known domain gaps (do not paper over)

- **FER+ and BlazePose are trained on adults.** Absolute keypoints and
  emotion labels on an infant are unvalidated; that is why posture uses only
  coarse relative cues and why expression alone never escalates motion.
- **A doll is not a baby**: layer 3 verifies the *mechanism* (thresholds,
  timing, gate paths), not clinical validity. Nothing here is a medical
  device claim — see the report's scope box on page 1.

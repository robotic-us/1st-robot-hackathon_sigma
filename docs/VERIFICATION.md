# Verifying the cradle without an infant

> **2026-08-08 — the infant recognizer was removed.** `perception/watch.py`,
> `sense.py`, `face/`, the ONNX weights and `serve.py --sense` are gone, and
> with them the report's §4.2 duty/unit cry test, the §4.3 sleep ladder, FER+
> expression and BlazePose posture. Layers 1 and 3 below described *that* code
> and no longer run; they are kept, struck through, because they record what
> was verified before the removal and what a future infant path would have to
> re-establish. What still runs is **layer 2** (the virtual-infant closed
> loop), **layer 4**, and the **mascot loopback**, which is now the only vision
> verification in the project.
>
> The one safety property that survived intact: vision is capped in the fuss
> band, so no visual channel can open the escalation ladder on its own.

No real baby is available (and none should be near this prototype before the
report's own preconditions — pediatrics, biomechanics, ethics review — are
met). This is the layered protocol that stands in. Each layer catches what the
one below cannot; none of them claims what only a clinical study could.

## ~~Layer 1 — synthetic streams~~ (removed 2026-08-08)

**What survives:** `python3 serve.py --verify` still acts the same scripted
episode (quiet → fussing → soothed → crying → sleep → a rollover → resettled)
and still drives the real CradleMachine, so the ladder, the trials, the taper
and the gate are exercised end to end. What it no longer does is *judge*: each
phase now asserts its own state and level. It is a machine test, not a
recognizer test, and the dashboard says so.

The rows below were the recognizer assertions. They are what a future infant
path would have to re-establish:

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

`perception/baby.py`'s random state process feeds the real CradleMachine, and
the machine's sway must actually soothe it. The judged-signals half of this
went with the recognizer; what remains — state → machine → motion → soothing →
state — is still a genuine closed loop, seeded and repeatable
(`tests.py baby`, and `tests.py policy` for the advised version).

This is now the strongest verification the project has, because it is the only
one where the machine's decisions have to actually work rather than merely be
well-formed.

## ~~Layer 3 — bench protocol~~ (removed 2026-08-08: the detectors are gone)

There are no detectors left to walk this table with, and `--sense` no longer
exists. Kept as the specification a future infant path would be held to:

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

The equivalent bench for what *does* exist is the mascot loopback below, and
it has something this table never could: labelled ground truth.

## Layer 4 — logged recalibration (after the hackathon)

The report's own stance (§5.2 footnote): every numeric threshold is a v0
engineering value that must be recalibrated from logs and a pre-registered
study — not tuned live on an infant. The `--jsonl` stream exists so those
logs accumulate from day one.

## The mascot loopback (automated, `python3 tests.py nubzuki`)

Since the removal this is the project's only vision verification, and it is
worth being exact about what it verifies. It does **not** ask whether anything
understands an infant. It asks whether the *display and camera path* is intact:
the machine sends a distress level, `web/baby.js` draws Nubzuki at that level on
the cradle iPad, and a camera pointed at the iPad should recover the level that
was sent.

`perception/nubzuki.py` is the reader. It names a pose by matching the figure
against 102 reference crops — the 17 renderings `web/baby.js` produces,
each baked at five sizes,
(`docs/poses/`) and the 17 sheet figures — and taking the nearest.

It is graded on **both** corpora, and the distinction matters. The sheet is
labelled by construction: it *is* a circumplex, so where a figure is printed
states its emotion independently of how it is drawn. `docs/poses/` is what a
camera pointed at the iPad will actually see. The recogniser scores **17/17 on
each**, and on the rig corpus it also holds 17/17 through a full camera
simulation — 8° rotation, perspective, glare, blur, noise, JPEG q45.

**It abstains.** Past a measured distance, or when the runner-up is within 8%,
it returns nothing rather than a guess — which reads downstream as UNKNOWN and
stops the cradle. On frames degraded past reading (30% downscale *and* blur
*and* JPEG q25) it refuses all 17 instead of naming any.

    python3 perception/nubzuki.py docs/Nubzuki.jpg --grade

**Running it live.** Start the dashboard, open `/baby` on the cradle iPad, point
the camera at the iPad, and run:

    python3 serve.py --verify                       # or --baby
    python3 perception/nubzuki.py --camera 0 --server http://localhost:8080

The overlay reads *camera sees* / *jetson sent* / *loop MATCH*, and the exit
line gives the match rate. Two things to expect before calling a mismatch a
bug:

- **The recovered level is quantised to the five ladder rungs** (0, .22, .50,
  .78, 1.0). `web/baby.js` blends the two poses either side of the commanded
  level, so at level .35 the iPad is genuinely showing something between
  Neutral and Crying; the reader names the nearer one. `nearest_rung()` is what
  the comparison uses, and anything finer would be a fake accuracy figure.
- **A pose off the ladder means a hand is on the wheel.** Twelve of the
  seventeen are unreachable by the machine, so reading one back says somebody
  dragged the wheel, not that the cradle is in that state. Those report no
  level at all rather than a wrong one.

Verified against the rig's own output, not just the sheet: a headless render of
`/baby?level=0.50` reads back as Crying, with the page's dark "Start motion
sensor" button correctly rejected rather than named.

**What it is not.** A mascot is further from a baby than a doll is. Nothing in
that file may be pointed at an infant — it reads a screen the machine already
wrote. What it buys is a regression test with ground truth, which the bench
table above never could have.

**Where it breaks** (measured over the rig corpus; blur, downscale, exposure,
colour cast, JPEG q20, noise, perspective and the full camera simulation all
pass 17/17 with zero wrong answers):

| Condition | Result |
|---|---|
| Rotation ±8° | 16/17 — a lying figure tilted reads as another lying figure |
| Degraded past reading | 0 wrong, refuses instead — the failure mode we want |

The one confusion worth knowing is **Lounging ↔ Sleeping**: rotated and
blurred, a figure lying with its eyes open is a figure lying with its eyes
shut, and the eyes are the first thing blur takes. It costs nothing here —
Lounging is hand-only, so the machine can never command it.

All three are lens problems, not rule problems: keep the iPad in focus, out of
direct glare, and correctly exposed. They are recorded rather than fixed with a
looser threshold, because a threshold loose enough to pass them stops separating
the poses — the under-exposure row is exactly that trade, tried and reverted
(see the brown rule in `classify()`).

**Detection, not naming, is the weak link.** Naming holds 17/17 on renders
while detection on a *photographed* panel is the part that fails — the failure
reads as "the classifier works but it never finds him". The pipeline is now
page → normalise → mask (see CLAUDE.md), which took simulated panel photos from
2/17 to 12/17 detected with zero wrong answers, degrading to 0/17 under heavy
glare. Those numbers come from a **simulation**, not a photograph, and should be
treated as a direction rather than a measurement.

    python3 perception/nubzuki.py --camera 0 --debug     # page / mask / gate

**Distance** was the other reported failure and it behaved differently from
the glare one: detection held at every distance, and *naming* was what gave
out — abstaining, never guessing. References are now baked at five sizes and
the page is anchored on the brightest surface rather than the 90th percentile.
By figure height, named correctly out of 17:

| figure height | 300 px | 128 px | 72 px | 52 px | 35 px |
|---|---|---|---|---|---|
| before | 13 | 8 | 1 | 0 | 0 |
| after | 14 | 17 | 13 | 11 | 6 |

Below roughly 50 px it degrades into abstention, which is the safe direction.

**The gap that remains** is the glass. Everything above is measured on renders
and simulated distortion; nothing yet is a photograph of an actual panel, with
its moire, its glare and its viewing angle. `--dump` exists to close that: it
saves frames with what the recogniser made of them, and `/baby?pose=<key>`
reaches all 17 poses so a capture can be walked deterministically.

    # on the iPad: /baby?pose=crying   (?pose= reaches all 17, ?level= only 5)
    python3 perception/nubzuki.py --camera 0 --dump data/bench/ --label crying

## Known domain gaps (do not paper over)

- **FER+ and BlazePose are trained on adults.** Absolute keypoints and
  emotion labels on an infant are unvalidated; that is why posture uses only
  coarse relative cues and why expression alone never escalates motion.
- **A doll is not a baby**: layer 3 verifies the *mechanism* (thresholds,
  timing, gate paths), not clinical validity. Nothing here is a medical
  device claim — see the report's scope box on page 1.

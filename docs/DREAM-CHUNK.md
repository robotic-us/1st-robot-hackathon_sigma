# DREAM-Chunk on phorce motion slots

Our implementation of **DREAM-Chunk** (*Dreamed-state REactive Action Matching
for Action Chunking*, arXiv 2026), the third of the three required SOTA papers.

## Why this paper

The prep guideline's suggested shortcuts for GaP and ASAP both assume we can
write into the low-level control loop — GaP retunes control parameters live,
ASAP adds a residual `Δa_t` at the motor driver. **The phorce participant API
does not expose that.** The only downward channel is `play(1..50)`; the
4-tuple recipe command (`목표위치, 피드포워드토크, Kp, Kd`) is documented as
내부/벤치 전용. Any plan built on torque injection dies on contact with the SDK.

DREAM-Chunk is the one paper whose mechanism *is* what the API affords:

| Paper term | Here |
|---|---|
| action chunk | a motion slot (1–50), pre-recorded on the pcm's SD card |
| chunk dictionary | the slot table — already "safe candidates", as the shortcut guide asks |
| world model | the **P-Vector quintic** (`RH_Guide_3`). Closed form. No training, no NPU cost |
| dreamed state | `chunk.dream(measured_pose)` — anchored on where the arm *actually is* |
| sensor feedback | `/phorce/feedback` @1 kHz: `position_rad` for consistency, `dob_a` for contact |
| reactive matching | rank candidates by consistency + collision resistance + task fit |

The key realisation is that **we were handed the world model in the motion
format**. A P-Vector `P(i) = [yd, L_traj, s0, sd]` expands to

```
y(τ) = a0 + a2τ² + a3τ³ + a4τ⁴ + a5τ⁵ ,  τ = k / L_traj
a0 = y0                            a3 = (10 − 1.5·s0 + 0.5·sd)(yd − y0)
a2 = 0.5·s0·(yd − y0)              a4 = (−15 + 1.5·s0 − sd)(yd − y0)
                                   a5 = (6 − 0.5·s0 + 0.5·sd)(yd − y0)
```

so dreaming ten candidates over a 2 s horizon is a handful of polynomial
evaluations. That is the whole trick, and it is why this runs at decision rate
on the Orin with room to spare.

`dob_a` — the disturbance observer, an estimate of external force per axis — is
the second gift. It is the contact signal the papers keep asking for, available
with no extra sensor.

## Honest limitation — read before presenting

The paper switches chunks **mid-execution**. **This robot cannot.** It plays one
motion at a time, has no queue, and a real (non-sim) play cannot be cancelled
once started. So:

* a divergence monitor watches continuously at feedback rate and decides a
  switch is warranted the instant measured state leaves the dreamed tube;
* `ChunkMatcher` acts on that at the **earliest boundary the hardware allows** —
  it collapses the anti-thrash dwell instead of waiting it out, so the reaction
  lands one chunk-tail later rather than never.

*(That is the arm-side design. On the infant side the monitor half was built,
measured and removed — see "Two halves" below. The matcher is what ships.)*

That is the faithful implementation *under this contract*. The gap (mid-chunk
pre-emption) is a hardware limit, not a shortcut. **Say so.** The OT rules
allow presenting unimplemented parts via AI, but explicitly require the team to
state 사용 방식과 한계 — so draw the ideal loop if you like, and label it.

## Re-anchored on the infant (`core/policy.py`)

The version above dreams the **arm**. The arm is not what we are trying to
improve. The same four parts, re-anchored on the baby:

| Paper term | Arm version | Infant version |
|---|---|---|
| action chunk | a motion slot | one advised motion (`CANDIDATES`) |
| world model | P-Vector quintic | first-order soothing response, `y(t) = y0·(1 − g·(1 − e^(−t/τ)))`, gain `g` learned per motion |
| dreamed state | `chunk.dream(pose)` | the comfort curve this motion *should* be producing |
| sensor feedback | `position_rad`, `dob_a` | the judge's live 0..1 level — the same input the machine runs on |
| reactive matching | switch chunk at the boundary | re-open the trial's motion decision early |

The doc's honest limitation inverts here. There, hardware could not pre-empt
mid-chunk. In the cradle it can: `MotionEngine.command` swaps the mode without
dropping the envelope, so a diverging motion is replaced within a tick and the
baby feels no gap. Through `--robot` the old limit returns unchanged — a real
PCM slot play cannot be cancelled — so the reaction again lands at the next
boundary. Say which one is running.

`note_residual()` is the same ASAP-reduced idea as below: the measured drop
corrects the **dream**, never a command.

### Two halves, and only one of them pays

The paper has a monitor **and** a matcher, and they measured very differently.
This is the most useful thing we learned, so present both.

**The matcher (`ChunkMatcher`, `--policy dream`) is the one that works.** It
is the mechanism the paper is named for: dream all 26 candidates forward from
where the infant is, score them, take the best. The shipped configuration, over
40 paired simulated nights (same seed, same hidden temperament in every arm,
`--pace 10`):

| arm | upset time | crying time |
|---|---|---|
| reflex heuristic vs the report ladder | −21.0 % (t=−5.9) | −47.6 % (t=−7.2) |
| **DREAM-Chunk planner vs the ladder** | **−26.4 %** (t=−8.5) | **−57.1 %** (t=−9.0) |
| **planner vs the reflex heuristic** | **−6.8 %** (t=−1.8) | **−18.2 %** (t=−1.9) |

Absolute: ladder 19.9, reflex 15.7, planner **14.7** upset min per 30-minute
night. The margin over the heuristic is the one number here that is only
marginally significant at n=40 — quote it as a direction, not a result.

Its ranking is on the dashboard, not just in this table: the planner is the one
brain that can show its work, so `SoothePolicy.snapshot()` carries the plan and
the *What it dreamed* card draws every candidate it scored, the comfort each was
predicted to reach, and which cost term moved it. Cold start is one big tie —
an untried candidate inherits its features' gain, so with nothing measured they
all score alike; the tie breaks on `EXPLORE` (one calm representative per family
first) and the card says so rather than dressing a tie up as a ranking.

**The monitor's early switching does not pay, and has been removed.**
Divergence *detects* well — instrumented over 6 nights, of 234 divergences 156
fired on a genuinely worse baby and only **7** cut a working motion, and the
mean *true hidden* personality gain of the motions it cut was 0.68 (1.0 =
neutral). But letting that drive an early switch measured **+2.9 % upset /
+8.4 % crying** over 60 paired nights — slightly *worse*. Twelve-night batches
swing −15 % to +15 %; our first batch said −15.6 % crying and did not
replicate. The tube, the `--dream` flag and `CradleMachine.switch_hook` were
**deleted** rather than left as a switched-off option — a measured-harmful path
through the safety machine is not worth its upkeep. `WorldModel` is the same
class minus the tube: the *model* is what the matcher ranks on, and it stays.

### What the world model was actually doing (three bugs and a surprise)

The first version of the planner beat the ladder by −23.7 % upset, and it was
easy to believe that was personalisation working. It was not. Auditing the
learned model against the infant's *hidden* taste — which the simulator knows
and the policy never sees — turned up three defects and one uncomfortable
explanation.

**1. The credit was asymmetric.** A play scored `(y0 − y_end) / y0`.
Improvement caps at +1; worsening is unbounded — and the machine only asks for
a motion while the infant is fussing, so `y0` is small and any upward drift
scored enormous negative. Over an hour the mean residual was −0.09 against a
median of +0.04. Scoring against a fixed scale (the fuss→cry span) is
symmetric; a ratio is not.

**2. Habituation was recorded as dislike.** A motion played until it stopped
working had that written into its *taste*, which is permanent, instead of its
*wear*, which recovers. `WorldModel` keeps the two apart and models wear
from its own play history — it knows what it played and for how long, so it
never has to learn the transient.

**3. The veto was an absorbing state.** At −0.15 over three samples, an hour
vetoed **21 of 26 motions, 19 of which the infant genuinely liked** (true gain
+1.3 to +2.5, learned −0.15 to −0.35). A single play is a coin flip — the
infant's own dwell process moves the level as much as any motion — so the bar
is now −0.40 over five.

**And the estimator was measuring nothing.** Rank correlation between learned
taste and real taste was **+0.03**. Five estimator variants all scored between
+0.01 and +0.07: not a formula bug but a sample-size wall, ~7 noisy plays per
motion per session. The same plays give each *feature value* ten times that,
and features are what the taste is made of:

| estimate | correlation with real taste |
|---|---|
| per-motion id | +0.03 |
| per-feature pooling | +0.44 |
| hierarchical, id shrunk by n/(n+40) | **+0.56** (+0.60 in the live loop) |

**The surprise:** fixing all of that made the cradle *worse* — −23.7 % against
the ladder became −6.6 %. The old model's advantage was never taste. Its
estimates were noisy and its veto trigger-happy, so it **churned through
motions constantly**, and against an infant that habituates, forced variety is
what soothes. Give it accurate estimates and it settles onto its best-known
motion, the infant wears out, and the advantage evaporates. `ReflexBrain`,
untouched by any of this, still gets −21 % — and it rotates by explicit rule
(`MAX_RUN = 2`). Repeat-rate tracks performance across every arm measured.

So the planner keeps the honest model and gets an explicit reason to move on:
a confidence bound, `c·sqrt(ln N / n)`, wide while a motion's evidence is thin
and fading as it accumulates. That is the standard answer to exploiting a
low-contrast ranking as though it were certain, and unlike a flat novelty
bonus it keeps working after every motion has been tried once.

That closes the loop. Corrected estimator alone measured −6.6 % against the
ladder — worse than the −23.7 % the broken one got by accident. Corrected
estimator *plus* the confidence bound measures **−26.4 %**: better than the
original, and this time the mechanism is the one on the tin. The lesson worth
keeping is that the audit was only possible because the simulator knows the
infant's hidden taste and the policy never sees it — without that ground
truth, the first number would still look like success.

### Why the plant decides whether dreaming is worth anything

The planner was worth nothing until the infant model changed, and that is the
real lesson rather than a tuning anecdote.

`perception/baby.py` originally stepped a soothable baby down as a **memoryless
Poisson process** (`SOOTHE_RATE = 0.08`/s at full sway, mean ≈ 12 s). Against
that plant, prediction is worthless by construction — P(no step-down in 7 s)
≈ 0.57 even for a motion that is working — and the numbers say exactly that:
on the memoryless baby the planner ties the reflex heuristic (**−0.0 %** upset,
t=−0.01, n=40). A world model cannot beat a heuristic in a world with no
structure to model.

Real infants settle *cumulatively*: rocking that is working shows progressive
calming, and interrupting it loses the progress gradually. `VirtualBaby`
(default `cumulative=True`) now integrates soothing into a 0..1 settling score
— one full unit steps the state down — and shows partial progress in the
distress level. Same mean settle time, far less variance, and a trajectory
that *is* predictable. The old model stays reachable as
`VirtualBaby(cumulative=False)` precisely so this comparison can be re-run.

The change was made on realism grounds and it did **not** simply make the
world easier: the fixed-ladder baseline got *worse* under it (18.6 → 20.4
upset min/night) and the reflex heuristic's margin over the ladder barely
moved (−17.8 % → −18.2 %). Only the *planner* specifically improved, which is
the mechanistically predicted result and not a general inflation. Say that
plainly if anyone asks whether the model was tuned until the method won.

## ASAP, in reduced form

`ChunkMatcher.note_residual()` records measured-minus-dreamed after each chunk.
ASAP adds its residual to the *actuator command*; we have no such channel, so we
fold it into the *world model* — the dream is corrected, the command is not.
That is the part of ASAP that actually runs here, and it is worth one slide.

## Files

| File | Role |
|---|---|
| `core/pvector.py` | P-Vector parsing (`MotionMap.csv`) + the quintic world model |
| `core/policy.py` | `ChunkMatcher` + `DreamBrain` (the ranking half, `--policy dream`) and `WorldModel` / `DreamConfig` (taste, wear, transitions) |
| `perception/baby.py` | the plant: `cumulative` settling, the thing that decides whether a world model can pay at all |
| `serve.py` | `--policy dream` selects the planner; the dashboard switches brains live |

The arm-side files below are from the pre-2026-08 stack (`dream.py`,
`decider.py`, `main.py`) and were removed with the DREAM/tag cleanup; the
sections that describe them are kept as the record of that implementation.

Bucketing is **unchanged**: direction/amplitude still choose *which* chunks are
candidates. The dream chooses *between* them. That keeps the existing behaviour
as a baseline you can A/B against.

## Chunk dictionary: two sources

1. **`MotionMap.csv`** from the robot's SD card — the real thing. Pass
   `--motion-map path/to/MotionMap.csv`.
2. **`slots.json`** — synthesised, one segment per axis, `start_pose → end_pose`.
   Coarser. Used automatically when no MotionMap is given, and every chunk is
   flagged `synthesised=True` so the log says so rather than implying we read
   the SD card.

This is why the whole stack runs today with no robot, and upgrades the moment a
real MotionMap appears.

---

# How to test it

Five layers, cheapest first. Layers 1–4 need **no robot and no ROS**.

### 1. World model — is the polynomial right?

```bash
python3 pvector.py --selftest
```

Checks the boundary conditions the format claims: `y(0)=y0`, `y(1)=yd`,
`y'(0)=y'(1)=0`; that `s0=sd=0` is monotone and never overshoots; that
`L_traj=1000 → 1.000 s`; that two segments join without a jump; that an axis
holds its final value past the end of its program; and the `MotionMap.csv` cell
parser (trailing zeros, `-`, blanks) plus `MD5`/`0x06` → axis-index mapping.

Then eyeball the dictionary:

```bash
python3 pvector.py                              # synthesised from slots.json
python3 pvector.py --motion-map MotionMap.csv   # the real one
```

### 2. Matcher and monitor — does the logic hold?

```bash
python3 dream.py --selftest
```

Nine assertions, each one a behaviour we actually depend on:

* with no external force, the gentlest chunk wins on continuity
* with `dob_a = 2.0 A` on axis 0, the big swing **loses** to the small one
* task fit can outvote continuity when nothing is pushing back
* the excursion veto fires, and `best()` never returns a vetoed chunk
* with no joint data, pose terms drop out (no crash, no faked pose)
* perfect tracking → **never** diverges (rms ≈ 0)
* a jammed axis → diverges, and early (< 1 s)
* a **single-frame spike** → ignored (this is what `diverge_hold_s` is for)
* the ASAP residual biases the dream's end, not its start

Rank the dictionary by hand from any pose:

```bash
python3 dream.py --demo --pose 0,0,0,0 --force 2.5,0,0,0
python3 dream.py --demo --pose 0,0,0,0 --recent-error 0.3
```

### 3. Offline replay — deterministic, instant, no clock

```bash
python3 stimulus.py --scenario demo --record demo.mp4 --ground-truth demo.csv
python3 decider.py --replay demo.csv            # baseline
python3 decider.py --replay demo.csv --dream    # DREAM-Chunk
```

Replay passes no joint state, so this specifically proves the **graceful
degradation** path: pose-dependent terms drop out and ranking falls back to task
fit. Both runs should choose the same slots; only the cost breakdown changes.

### 4. End-to-end on the mock — the real test

The mock is a **digital twin**: with `--dream` it drives its synthetic joints
along the *same* P-Vectors we dream, so a clean run should be silent and only an
injected fault breaks the agreement.

```bash
# A. clean — expect "0 dream divergences"
python3 main.py --mock --video demo.mp4 --once --dream

# B. jam axis 0 twelve seconds in — expect divergences ONLY after 12 s,
#    dob_a ≈ 2.5 A, and "DWELL PRE-EMPTED: dream diverged" on the next decision
python3 main.py --mock --video demo.mp4 --once --dream --mock-jam-after 12

# C. jam a different axis, earlier — expect only ~1 divergence, and that is CORRECT
python3 main.py --mock --video demo.mp4 --once --dream --mock-jam-after 6 --mock-jam-axis 2

# D. regression: the baseline path must still work untouched
python3 main.py --mock --video demo.mp4 --once
```

**What to look for**

| Signal | Meaning |
|---|---|
| `0 dream divergences` in A | world model and playback agree — plumbing correct |
| `DREAM DIVERGED slot N …% err=… dob=2.5A` | monitor caught the injected obstacle |
| `DWELL PRE-EMPTED: dream diverged` | the reactive switch fired at the boundary |
| `pose 0.4–0.5` in the cost breakdown under jam | resistance term is live (vs ≈0.01 clean) |
| `vetoed [...]` | excursion guard rejected a candidate |

Measured on the current `slots.json`: **A → 0 divergences / 9 plays**,
**B → 9 divergences / 12 plays / 7 BUSY**, **C → 1 divergence / 9 plays**,
**D → 9 plays, baseline unchanged**.

Case C looks weak but is the monitor behaving *proportionately*, and it is worth
understanding before you tune anything. Axis 2 barely moves in most slots
(0.02–0.07 rad of travel), all well under `diverge_rad = 0.12`, so jamming it
genuinely should not raise an alarm. The one divergence is slot 10 (settle),
whose axis-2 travel is 0.20 rad — the only chunk that actually commits enough
on that axis to notice being blocked. A monitor that fired on all of them would
be a monitor keyed to noise.

Because the twin uses the same model on both sides, a clean run proves the
**plumbing**, not the physics. Real sim-to-real error only appears against the
actual robot — which is exactly what `note_residual()` is for.

With a webcam and a tablet instead of the video file:

```bash
python3 stimulus.py --scenario demo --loop      # on the tablet
python3 main.py --mock --dream --show-debug     # webcam pointed at it
```

### 5. On the real robot

Prerequisites, in order — **robot power first, then the stack**:

```bash
cat /sys/class/net/eno1/operstate     # must be "up"
# terminal 1
ros2 run agx_phorce_bridge phorce_monitor --ros-args \
    -p nic:=eno1 -p mode:=command -p axes:=2 -p mbx_enabled:=true
# terminal 2
ros2 run agx_motion_slot motion_action_server --ros-args -p backend:=ecat
# terminal 3
phorce doctor && phorce list          # list must be non-empty
```

`phorce list` returning nothing means no motions are loaded on the pcm — teach
them in phorce Studio first (①설정 영점 **before** ②교시), or DREAM-Chunk has an
empty dictionary and nothing to match. Then hold the 영점 button (button 1) for
0.6 s and wait ~3 s, or every play is rejected with code 12.

```bash
# watch the ranking without moving anything
python3 main.py --dream --motion-map /path/to/MotionMap.csv --dry-run --show-debug

# for real — check nobody is near the robot
python3 main.py --dream --motion-map /path/to/MotionMap.csv --show-debug
```

Start with `--dry-run`. To provoke a real divergence, gently obstruct the arm
mid-motion and watch for `DREAM DIVERGED` with a rising `dob_a`. Physical
E-Stop is the only emergency stop — `cancel()` is not one.

## Tuning

Everything lives in `DreamConfig` (`dream.py`):

| Knob | Default | Raise it if… |
|---|---|---|
| `diverge_rad` | 0.12 rad (~7°) | false divergences on a healthy robot |
| `diverge_hold_s` | 0.15 s | noise trips the monitor |
| `w_resistance` | 2.0 | you want contact to dominate the choice more |
| `max_excursion_rad` | 1.2 rad | a legitimate large motion is being vetoed |
| `UNITS_PER_DEG` (`pvector.py`) | 1.0 | **calibrate this against real feedback** |

`UNITS_PER_DEG` is the one genuinely unpinned number: the P-Vector deck states
positions are int16 degrees, but its worked example plots `yd=200` as `1.0`, so
there is a scale the document does not state outright. It is isolated in one
named constant. Fix it by playing a known slot and comparing `position_rad`
against the dreamed trajectory.

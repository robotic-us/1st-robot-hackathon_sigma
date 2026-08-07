#!/usr/bin/env python3
"""A virtual infant: a random state process the cradle can try to soothe.

No camera, no models, no tags.  States wander SLEEP <-> CALM <-> FUSS <-> CRY
on randomized dwell times -- weighted so the infant is restless rather than
settled: a soothed baby drifts back into fussing within a minute or so, which
is what keeps the cradle (and the policy) working.  The loop closes: the
engine's live sway is fed back in as ``soothing``, and a soothable fuss or
cry *settles cumulatively* under it -- progress builds while good motion
plays, shows in the distress level on the way, and drains at rest, so the
machine's trials genuinely work,
sometimes.  About a third of cries are unsoothable (hunger, diaper): motion
never helps, which is exactly the path that must end in a caregiver alert.
Rarely the face hides for a moment, tripping the safety gate.

With a ``Personality`` (docs/IDEA.md) the motion identity matters too: a
loved motion soothes 3x, hated ones agitate, a liked transition doubles up.
Emotions are also time-related: habituation wears every motion out with use
(and lets it recover at rest), and a slow mood cycle makes some stretches
fussier -- so "play the favourite forever" stops being a winning strategy.

The baby draws itself as a circle: colour = state, radius = distress, and its
position rides the cradle's actual plate offset -- you can watch the sway
rock it.  serve.py --baby streams that as the camera; RViz shows the cradle
answering (./cad/view.sh).

Deterministic under a seed; tested by ``python3 tests.py baby``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# state -> (base distress, wander); thresholds live in core.cradle:
# CALM_LEVEL=0.12 separates quiet from fuss, CRY_LEVEL=0.45 fuss from cry.
STATES = {
    "SLEEP": (0.02, 0.01),
    "CALM":  (0.06, 0.03),
    "FUSS":  (0.30, 0.05),
    "CRY":   (0.62, 0.08),
}
DWELL_S = {           # how long a state lingers before rolling the dice again
    "SLEEP": (25.0, 60.0),
    "CALM":  (12.0, 30.0),
    "FUSS":  (15.0, 40.0),
    "CRY":   (15.0, 35.0),
}
# Weighted transitions at the end of a dwell.  This infant is deliberately
# *unsettled*: content stretches are short and rarely renew themselves, so
# calm is something the cradle keeps earning rather than a resting state it
# falls into.  A baby that mostly sits happy makes a demo where nothing
# happens -- and gives the policy almost no trials to learn from.
NEXT = {
    "SLEEP": (("SLEEP", 0.35), ("CALM", 0.65)),
    "CALM":  (("CALM", 0.25), ("SLEEP", 0.15), ("FUSS", 0.6)),
    "FUSS":  (("CALM", 0.3), ("FUSS", 0.3), ("CRY", 0.4)),
    "CRY":   (("CRY", 0.45), ("FUSS", 0.55)),
}
SOOTHE_RATE = 0.08    # per second at full sway: ~12 s of good motion to settle
SOOTHABLE_P = 0.7     # the rest are hunger/diaper -- caregiver work
AGITATE_RATE = 0.05   # per second under a *hated* motion: fussing worsens
# Settling is *cumulative*, not a per-tick coin flip.  A real infant winds
# down: rocking that is working shows progressive calming, and interrupting it
# loses the progress gradually rather than instantly.  So soothing integrates
# into a 0..1 "settling" score -- one full unit steps the state down -- and the
# distress level shows the partial progress on the way.  Rest lets it drain.
# (The memoryless model this replaces is still reachable as
# ``VirtualBaby(cumulative=False)``; it is what docs/dream-chunk.md's
# measurement was taken against, and the difference is the whole point there.)
SETTLE_DRAIN_S = 45.0     # progress half-lives away over ~30 s of no motion
SETTLE_SHOW = 0.6         # how much of the band the partial progress moves
# Time-related emotion (docs/IDEA.md follow-up): no motion works forever.
# Habituation builds while a motion is engaged and decays while it rests,
# so even the loved motion wears out and the policy must rotate; a slow
# mood cycle makes some stretches of the night fussier than others.
FATIGUE_S = 75.0          # this long at full strength ~= fully worn out
FATIGUE_RECOVER_S = 300.0
FATIGUE_FLOOR = 0.15      # a worn-out motion keeps 15% of its effect
MOOD_PERIOD_S = 540.0     # one good-to-grumpy cycle every 9 minutes
MOOD_DEPTH = 0.35         # grumpy half: soothing 35% weaker, calm shorter
HIDE_MEAN_S = 600.0   # a face-lost blip roughly every 10 min
HIDE_FOR_S = 1.5

COLORS_BGR = {        # matches the dashboard's palette
    "SLEEP": (255, 154, 76), "CALM": (80, 185, 63),
    "FUSS": (65, 179, 227), "CRY": (77, 72, 229),
}


@dataclass(frozen=True)
class Personality:
    """docs/IDEA.md: a fixed, hidden motion temperament (the '성격').

    Two layers, multiplied together by ``gain``:

    * **Specific motions** -- a loved id (3x), hated ids (agitate), a liked
      transition (combo[0] then combo[1], 2x).
    * **Features** (the N-system vocabulary, docs/motion-system.png) -- a
      favourite and a hated *shape*, small-vs-large, fast-vs-slow, and a
      feeling about the tremble.  Preferences therefore generalise: a baby
      that loves wide slow circles also likes wide slow ovals, a little.

    Time leaks in too: during a grumpy mood stretch, fast motions lose most
    of their charm.  The policy never sees any of this -- only the ranks
    it produces.
    """

    love: str = ""
    hate: frozenset = frozenset()
    combo: tuple = ()          # (a, b): b soothes 2x right after a
    shape_love: str = ""       # this shape soothes 1.9x
    shape_hate: str = ""       # this shape agitates
    size_pref: str = ""        # "small" | "large" | ""
    speed_pref: str = ""       # "fast" | "slow" | ""
    vibe_pref: int = 0         # +1 loves the tremble, -1 hates it

    # Felt-motion thresholds: how IMU features (web/baby.js -> serve.py's
    # IPadMotion) map into the taste vocabulary.  Demo-scale, like the
    # 0.12 m/s^2 strength reference -- never a physical safety measurement.
    FELT_FAST_HZ = 0.45      # the N system's own fast/slow boundary
    FELT_LARGE_MS2 = 0.09    # accel RMS above this reads as a wide motion
    FELT_VIBE_JERK = 2.5     # jerk RMS above this reads as a tremble

    @staticmethod
    def _features(motion):
        from core.cradle import LIBRARY_BY_ID
        return LIBRARY_BY_ID.get(motion)

    @classmethod
    def felt(cls, sensed) -> Optional[dict]:
        """IMU features -> the taste vocabulary, or None without a signal.

        What the tablet *measures* outranks what the engine *commanded*:
        speed from the dominant frequency, size from acceleration RMS,
        tremble from jerk.  Shape cannot be told from one IMU, so shape
        keeps coming from the motion id.
        """
        if not sensed or sensed.get("samples", 0) < 8:
            return None
        out: dict = {}
        hz = float(sensed.get("dominant_hz", 0.0) or 0.0)
        if hz > 0.05:
            out["speed"] = "fast" if hz >= cls.FELT_FAST_HZ else "slow"
        accel = float(sensed.get("accel_rms", 0.0) or 0.0)
        if accel > 0.02:
            out["size"] = "large" if accel >= cls.FELT_LARGE_MS2 else "small"
        out["vibe"] = float(sensed.get("jerk_rms", 0.0) or 0.0) >= cls.FELT_VIBE_JERK
        return out

    def gain(self, motion, prev, mood: float = 1.0,
             felt: Optional[dict] = None) -> float:
        if not motion and felt is None:
            return 0.0
        if motion and motion in self.hate:
            return -1.0
        if len(self.combo) == 2 and (prev, motion) == tuple(self.combo):
            base = 2.0
        elif motion and motion == self.love:
            base = 3.0
        else:
            base = 1.0
        # What the motion is like: declared by its library entry, then
        # overridden by whatever the tablet actually measured.
        m = self._features(motion) if motion else None
        shape = m.shape if m is not None else ""
        size = m.size if m is not None else ""
        speed = m.speed if m is not None else ""
        vibe = bool(m.vibe) if m is not None else False
        if felt:
            size = felt.get("size", size) or size
            speed = felt.get("speed", speed) or speed
            vibe = bool(felt.get("vibe", vibe))
        if shape:
            if self.shape_hate and shape == self.shape_hate:
                return -0.8
            if self.shape_love and shape == self.shape_love:
                base *= 1.9
        if self.size_pref and size:
            base *= 1.3 if size == self.size_pref else 0.8
        if self.speed_pref and speed:
            base *= 1.3 if speed == self.speed_pref else 0.8
        if self.vibe_pref:
            if vibe:
                base *= 1.5 if self.vibe_pref > 0 else 0.35
            elif self.vibe_pref > 0:
                base *= 0.9
        if mood < 0.8 and speed == "fast":       # grumpy: only slow works
            base *= 0.6
        return base

    @classmethod
    def random(cls, rng: random.Random, pool=None) -> "Personality":
        if pool is None:
            from core.cradle import N_CANDIDATES
            pool = N_CANDIDATES
        pool = list(pool)
        rng.shuffle(pool)
        shapes = ["horiz", "vert", "v", "parab", "circle", "ellipse",
                  "inf", "arc"]
        rng.shuffle(shapes)
        return cls(love=pool[0], hate=frozenset(pool[1:3]),
                   combo=(pool[3], pool[4]),
                   shape_love=shapes[0], shape_hate=shapes[1],
                   size_pref=rng.choice(("small", "large", "")),
                   speed_pref=rng.choice(("fast", "slow", "")),
                   vibe_pref=rng.choice((-1, 0, 1)))

    def describe(self) -> str:
        bits = []
        if self.love:
            bits.append(f"loves {self.love}")
        if self.hate:
            bits.append("hates " + "/".join(sorted(self.hate)))
        if len(self.combo) == 2:
            bits.append(f"likes {self.combo[0]}->{self.combo[1]}")
        if self.shape_love:
            bits.append(f"{self.shape_love} shapes soothe")
        if self.shape_hate:
            bits.append(f"{self.shape_hate} shapes upset")
        if self.size_pref:
            bits.append(f"prefers {self.size_pref}")
        if self.speed_pref:
            bits.append(f"prefers {self.speed_pref}")
        if self.vibe_pref:
            bits.append("loves the tremble" if self.vibe_pref > 0
                        else "hates the tremble")
        return ", ".join(bits) or "easygoing"


@dataclass(frozen=True)
class BabyReading:
    """Shaped like the other sensing modes: serve.py reads distress + present."""

    present: bool
    x: float
    y: float
    distance: float
    distress: float
    emotion: str          # the state name -- shown in the dashboard
    name: str = "virtual"
    ts: float = 0.0


class VirtualBaby:
    """Time is injected; ``soothing`` is the engine's live amplitude 0..1."""

    def __init__(self, seed: int | None = None,
                 personality: Personality | None = None,
                 cumulative: bool = True) -> None:
        self.rng = random.Random(seed)
        self.personality = personality
        # True: soothing accumulates and shows (the model of a real settle).
        # False: the original memoryless Poisson step-down, kept so the
        # docs/dream-chunk.md measurement can be reproduced against both.
        self.cumulative = cumulative
        self.settling = 0.0        # 0..1 progress towards the next step down
        self.state = "CALM"
        self.soothable = True
        self.level = STATES["CALM"][0]
        self._until = 0.0
        self._hidden_until = 0.0
        self._motion: str | None = None
        self._prev_motion: str | None = None
        self._fatigue: dict[str, float] = {}   # habituation per motion, 0..1
        self._mood = 1.0
        self._mood_phase = self.rng.uniform(0.0, 2.0 * math.pi)
        self._t: float | None = None
        # The first act is scripted, not diced: a few calm seconds, then a
        # fuss.  Left to the dice, CALM dwells 20-60 s with a 40% exit to
        # FUSS -- an opening that can sit quiet for many minutes, which reads
        # as "nothing is running".  Only the opening is special-cased; every
        # transition after it is the normal process.
        self._opening = True

    def _dwell(self) -> float:
        lo, hi = DWELL_S[self.state]
        d = self.rng.uniform(lo, hi)
        # a grumpy stretch cuts the calm short; upset dwell is unaffected
        return d * self._mood if self.state in ("CALM", "SLEEP") else d

    def _transition(self, now: float) -> None:
        if self._opening:
            self._opening = False
            self.state = "FUSS"
        else:
            roll, acc = self.rng.random(), 0.0
            for state, p in NEXT[self.state]:
                acc += p
                if roll <= acc:
                    self.state = state
                    break
        if self.state == "CRY":
            self.soothable = self.rng.random() < SOOTHABLE_P
        self._until = now + self._dwell()

    def _step_down(self, now: float) -> None:
        self.state = {"CRY": "FUSS", "FUSS": "CALM"}[self.state]
        self._until = now + self._dwell()

    def _step_up(self, now: float) -> None:
        worse = {"CALM": "FUSS", "FUSS": "CRY"}.get(self.state)
        if worse is not None:          # a CRY has nowhere worse to go
            self.state = worse
            self._until = now + self._dwell()

    def update(self, now: float, soothing: float = 0.0,
               motion: str | None = None,
               sensed: dict | None = None) -> BabyReading:
        dt = 0.0 if self._t is None else max(0.0, min(0.2, now - self._t))
        self._t = now
        if motion != self._motion:
            if self._motion is not None:
                self._prev_motion = self._motion
            self._motion = motion
        # Time-related emotion: the mood cycle, and habituation -- exposure
        # builds while a motion is engaged, every motion recovers at rest.
        self._mood = 1.0 - MOOD_DEPTH * (0.5 + 0.5 * math.sin(
            2.0 * math.pi * now / MOOD_PERIOD_S + self._mood_phase))
        if dt > 0.0:
            decay = math.exp(-dt / FATIGUE_RECOVER_S)
            for k in list(self._fatigue):
                self._fatigue[k] *= decay
                # cleanup threshold well under one tick's build increment,
                # or accumulation dies at birth (build adds ~4e-4 per frame)
                if self._fatigue[k] < 1e-6:
                    del self._fatigue[k]
        if motion and soothing > 0.2:
            self._fatigue[motion] = min(1.0, self._fatigue.get(motion, 0.0)
                                        + soothing * dt / FATIGUE_S)

        if self._until == 0.0:
            self._until = now + (self.rng.uniform(4.0, 8.0) if self._opening
                                 else self._dwell())
        if now >= self._until:
            self._transition(now)
        # The closed loop: a soothable fuss/cry yields to sway, hunger does
        # not.  With a personality the motion identity matters: the loved
        # motion soothes faster, a hated one agitates instead -- and any
        # motion, loved included, fades with heavy use.  When a tablet in
        # the cradle measures the *actual* motion, its felt character
        # (tempo, intensity, tremble) is what the taste judges.
        gain = 1.0
        if self.personality is not None and soothing > 0.2:
            gain = self.personality.gain(motion, self._prev_motion, self._mood,
                                         felt=Personality.felt(sensed))
        if gain > 0.0 and motion:
            gain *= max(FATIGUE_FLOOR, 1.0 - self._fatigue.get(motion, 0.0))
        upset = self.state in ("FUSS", "CRY")
        if upset and soothing > 0.2:
            if gain < 0.0:
                if self.rng.random() < 1.0 - math.exp(AGITATE_RATE * gain
                                                      * soothing * dt):
                    self._step_up(now)
            elif self.soothable and gain > 0.0:
                rate = SOOTHE_RATE * gain * self._mood * soothing
                if not self.cumulative:
                    if self.rng.random() < 1.0 - math.exp(-rate * dt):
                        self._step_down(now)
                else:
                    # the settle builds; one whole unit is a step down
                    self.settling += rate * dt
                    if self.settling >= 1.0:
                        self.settling = 0.0
                        self._step_down(now)
        elif dt > 0.0:
            # nothing helping right now: the progress drains away, it is not
            # banked for later
            self.settling *= math.exp(-dt / SETTLE_DRAIN_S)
        if not upset:
            self.settling = 0.0
        base, wander = STATES[self.state]
        target = base + wander * math.sin(now * 0.9 + sum(map(ord, self.state)) % 7)
        if self.cumulative and self.settling > 0.0:
            # partial progress is visible: a baby half-way to settling is
            # already quieter than one that has not started.  Without this the
            # level teleports at the step and no observer -- policy, dream
            # model or caregiver -- can tell "working" from "not yet".
            below = STATES[{"CRY": "FUSS", "FUSS": "CALM"}[self.state]][0]
            target -= (target - below) * SETTLE_SHOW * min(1.0, self.settling)
        self.level += (target - self.level) * min(1.0, dt / 2.0)

        if now >= self._hidden_until and dt > 0.0 \
                and self.rng.random() < dt / HIDE_MEAN_S:
            self._hidden_until = now + HIDE_FOR_S
        hidden = now < self._hidden_until
        x = 0.25 * math.sin(now / 19.0)
        return BabyReading(present=not hidden, x=x, y=0.0, distance=0.5,
                           distress=max(0.0, min(1.0, self.level)),
                           emotion=self.state, ts=now)


def baby_frame(reading: BabyReading, offsets_mm=(0.0, 0.0, 0.0)) -> np.ndarray:
    """The circle IS the baby: a plain ring -- state colour on the border,
    radius = distress -- riding the cradle's real plate offset."""
    frame = np.full((480, 640, 3), 26, np.uint8)
    if not reading.present:
        cv2.putText(frame, "FACE HIDDEN", (200, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (77, 72, 229), 2)
        return frame
    ap, ml, _ = offsets_mm
    cx = int(320 + reading.x * 120 + (ap + ml) * 6.0)   # sway, amplified 6x
    cy = 240
    radius = int(50 + 90 * reading.distress)
    color = COLORS_BGR.get(reading.emotion, (160, 160, 160))
    cv2.circle(frame, (cx, cy), radius, color, 6, cv2.LINE_AA)   # ring only
    cv2.putText(frame, f"{reading.emotion}  distress {reading.distress:.2f}",
                (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    return frame

#!/usr/bin/env python3
"""A virtual infant: a random state process the cradle can try to soothe.

States wander SLEEP <-> CALM <-> FUSS <-> CRY on seeded random dwells, biased
restless, under the engine's live sway (``soothing``); ~a third of cries are
unsoothable.  ``Personality`` (docs/IDEA.md) adds taste, habituation and mood.

    python3 serve.py --baby [--baby-seed N]   # repeatable; tests.py baby
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# state -> (base distress, wander), 0..1; CALM_LEVEL/CRY_LEVEL live in core.cradle.
STATES = {
    "SLEEP": (0.02, 0.01),
    "CALM":  (0.06, 0.03),
    "FUSS":  (0.30, 0.05),
    "CRY":   (0.62, 0.08),
}
DWELL_S = {           # seconds a state lingers before the dice roll again
    "SLEEP": (25.0, 60.0),
    "CALM":  (12.0, 30.0),
    "FUSS":  (15.0, 40.0),
    "CRY":   (15.0, 35.0),
}
# Weighted transitions at the end of a dwell, biased *unsettled*: calm is earned.
NEXT = {
    "SLEEP": (("SLEEP", 0.35), ("CALM", 0.65)),
    "CALM":  (("CALM", 0.25), ("SLEEP", 0.15), ("FUSS", 0.6)),
    "FUSS":  (("CALM", 0.3), ("FUSS", 0.3), ("CRY", 0.4)),
    "CRY":   (("CRY", 0.45), ("FUSS", 0.55)),
}
SOOTHE_RATE = 0.08    # per second at full sway: ~12 s of good motion to settle
SOOTHABLE_P = 0.7     # the rest are hunger/diaper -- caregiver work
AGITATE_RATE = 0.05   # per second under a *hated* motion: fussing worsens
# Rough handling (m/s^2 accel RMS), as in web/baby.js; only a tablet measures it.
SHAKE_FROM_MS2 = 2.0
SHAKE_FULL_MS2 = 7.0
# cumulative=True: soothing integrates into a 0..1 settling score, one unit = a
# step down.  False is the old memoryless model, kept so the comparison repeats.
SETTLE_DRAIN_S = 45.0     # progress half-lives away over ~30 s of no motion
SETTLE_SHOW = 0.6         # how much of the band the partial progress moves
# Habituation wears out even a loved motion; a mood cycle adds fussy stretches.
FATIGUE_S = 75.0          # this long at full strength ~= fully worn out
FATIGUE_RECOVER_S = 300.0
FATIGUE_FLOOR = 0.15      # a worn-out motion keeps this much of its effect
MOOD_PERIOD_S = 540.0
MOOD_DEPTH = 0.35         # grumpy half: soothing 35% weaker, calm shorter
HIDE_MEAN_S = 600.0   # a face-lost blip roughly every 10 min
HIDE_FOR_S = 1.5

COLORS_BGR = {        # the dashboard's palette
    "SLEEP": (255, 154, 76), "CALM": (80, 185, 63),
    "FUSS": (65, 179, 227), "CRY": (77, 72, 229),
}


@dataclass(frozen=True)
class Personality:
    """docs/IDEA.md: a fixed, hidden motion temperament (the '성격') -- ``gain``
    multiplies id tastes (love 3x, hate agitates, combo 2x) by N-system features
    (shape/size/speed/vibe).  The policy never sees it, only ranks."""

    love: str = ""
    hate: frozenset = frozenset()
    combo: tuple = ()          # (a, b): b soothes 2x right after a
    shape_love: str = ""
    shape_hate: str = ""
    size_pref: str = ""        # "small" | "large" | ""
    speed_pref: str = ""       # "fast" | "slow" | ""
    vibe_pref: int = 0         # +1 loves the tremble, -1 hates it

    # Felt-motion thresholds: IMU features -> the taste vocabulary (demo-scale).
    FELT_FAST_HZ = 0.45
    FELT_LARGE_MS2 = 0.09    # accel RMS above this reads as a wide motion
    FELT_VIBE_JERK = 2.5     # jerk RMS above this reads as a tremble

    @staticmethod
    def _features(motion):
        from core.cradle import LIBRARY_BY_ID
        return LIBRARY_BY_ID.get(motion)

    @classmethod
    def felt(cls, sensed) -> Optional[dict]:
        """IMU features -> the taste vocabulary (shape excepted), or None."""
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
        # Library-declared character, overridden by what the tablet measured.
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
    emotion: str
    name: str = "virtual"
    ts: float = 0.0


class VirtualBaby:
    """Time is injected; ``soothing`` is the engine's live amplitude 0..1."""

    def __init__(self, seed: int | None = None,
                 personality: Personality | None = None,
                 cumulative: bool = True, tempo: float = 1.0) -> None:
        self.rng = random.Random(seed)
        # Dwell divisor; --sense demos run tempo > 1 for more states per minute.
        self.tempo = max(0.1, tempo)
        self.personality = personality
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
        # The opening act alone is scripted: a few calm seconds, then a fuss.
        self._opening = True

    def _dwell(self) -> float:
        lo, hi = DWELL_S[self.state]
        d = self.rng.uniform(lo, hi) / self.tempo
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
        # Mood cycle, then habituation: builds while engaged, fades at rest.
        self._mood = 1.0 - MOOD_DEPTH * (0.5 + 0.5 * math.sin(
            2.0 * math.pi * now / MOOD_PERIOD_S + self._mood_phase))
        if dt > 0.0:
            decay = math.exp(-dt / FATIGUE_RECOVER_S)
            for k in list(self._fatigue):
                self._fatigue[k] *= decay
                # well under one tick's build (~4e-4/frame), or it dies at birth
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
        # The closed loop: a soothable fuss/cry yields to sway, hunger does not.
        # A loved motion soothes faster, a hated one agitates, use fades both.
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
                    self.settling += rate * dt
                    if self.settling >= 1.0:
                        self.settling = 0.0
                        self._step_down(now)
        elif dt > 0.0:
            self.settling *= math.exp(-dt / SETTLE_DRAIN_S)
        if not upset:
            self.settling = 0.0
        base, wander = STATES[self.state]
        target = base + wander * math.sin(now * 0.9 + sum(map(ord, self.state)) % 7)
        if self.cumulative and self.settling > 0.0:
            # partial progress shows, so an observer can tell working from not-yet
            below = STATES[{"CRY": "FUSS", "FUSS": "CALM"}[self.state]][0]
            target -= (target - below) * SETTLE_SHOW * min(1.0, self.settling)
        # A measured violent shake outranks every state's band (top rank, >= 0.62).
        if sensed:
            shake = ((float(sensed.get("accel_rms", 0.0) or 0.0) - SHAKE_FROM_MS2)
                     / (SHAKE_FULL_MS2 - SHAKE_FROM_MS2))
            if shake > 0.0:
                target = max(target, 0.62 + 0.38 * min(1.0, shake))
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
    """The circle IS the baby: ring colour = state, radius = distress."""
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
    cv2.circle(frame, (cx, cy), radius, color, 6, cv2.LINE_AA)
    cv2.putText(frame, f"{reading.emotion}  distress {reading.distress:.2f}",
                (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    return frame

#!/usr/bin/env python3
"""A virtual infant: a random state process the cradle can try to soothe.

No camera, no models, no tags.  States wander SLEEP <-> CALM <-> FUSS <-> CRY
on randomized dwell times, and the loop closes: the engine's live sway is fed
back in as ``soothing``, and a *soothable* fuss/cry steps down faster under
it -- so the machine's 30 s trials genuinely work, sometimes.  About a third
of cries are unsoothable (hunger, diaper): motion never helps, which is
exactly the path that must end in a caregiver alert.  Rarely the face hides
for a moment, tripping the safety gate.

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
    "SLEEP": (40.0, 120.0),
    "CALM":  (20.0, 60.0),
    "FUSS":  (15.0, 40.0),
    "CRY":   (20.0, 50.0),
}
NEXT = {              # weighted transitions at the end of a dwell
    "SLEEP": (("SLEEP", 0.5), ("CALM", 0.5)),
    "CALM":  (("CALM", 0.35), ("SLEEP", 0.25), ("FUSS", 0.4)),
    "FUSS":  (("CALM", 0.3), ("FUSS", 0.3), ("CRY", 0.4)),
    "CRY":   (("CRY", 0.6), ("FUSS", 0.4)),
}
SOOTHE_RATE = 0.08    # per second at full sway: mean ~12 s to step down
SOOTHABLE_P = 0.7     # the rest are hunger/diaper -- caregiver work
HIDE_MEAN_S = 600.0   # a face-lost blip roughly every 10 min
HIDE_FOR_S = 1.5

COLORS_BGR = {        # matches the dashboard's palette
    "SLEEP": (255, 154, 76), "CALM": (80, 185, 63),
    "FUSS": (65, 179, 227), "CRY": (77, 72, 229),
}


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

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.state = "CALM"
        self.soothable = True
        self.level = STATES["CALM"][0]
        self._until = 0.0
        self._hidden_until = 0.0
        self._t: float | None = None

    def _dwell(self) -> float:
        lo, hi = DWELL_S[self.state]
        return self.rng.uniform(lo, hi)

    def _transition(self, now: float) -> None:
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

    def update(self, now: float, soothing: float = 0.0) -> BabyReading:
        dt = 0.0 if self._t is None else max(0.0, min(0.2, now - self._t))
        self._t = now
        if self._until == 0.0:
            self._until = now + self._dwell()
        if now >= self._until:
            self._transition(now)
        # The closed loop: a soothable fuss/cry yields to sway, hunger does not.
        if (self.state in ("FUSS", "CRY") and self.soothable and soothing > 0.2
                and self.rng.random() < 1.0 - math.exp(-SOOTHE_RATE * soothing * dt)):
            self._step_down(now)
        base, wander = STATES[self.state]
        target = base + wander * math.sin(now * 0.9 + sum(map(ord, self.state)) % 7)
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

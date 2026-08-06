#!/usr/bin/env python3
"""The five-state infant watcher -- the team's baby recognition and motion plan.

A transcription of docs/example.py (the teammate's camera-only, rule-based
classifier) into this codebase: MediaPipe Face Mesh geometry -> five broad
visual states, each carrying a motion hint:

    UNKNOWN            face missing/too small/unreliable  -> STOP_AND_CHECK
    AWAKE              eyes open, nothing alarming        -> HOLD
    EYES_CLOSED        eyes closed, sleep not yet claimed -> HOLD_OR_TAPER
    SLEEP_CANDIDATE    relaxed closure, little motion     -> TAPER_TO_STOP
    DISTRESS_FACE      eye squeeze + wide mouth           -> GENTLE_TEST_ONLY

The classifier detects patterns, not causes: DISTRESS_FACE does not prove
crying and SLEEP_CANDIDATE does not prove sleep.  ``distress_of`` encodes that
honestly -- a visual-only DISTRESS_FACE maps into the fuss band (gentle M10
trial), and only audio fusion (perception/sense.py's noisy-OR) can push the
level over CRY_LEVEL where the machine's escalation ladder lives.  That is the
source file's own rule -- "audio or a caregiver check is required before
escalating cradle motion" -- expressed in the report's thresholds.

The state logic is pure and tested headlessly (tests.py, suite ``watch``).
Only :class:`Watcher` touches MediaPipe, imported lazily -- the module loads
fine on a machine without it.

Run standalone (needs mediapipe)::

    python3 perception/watch.py               # live window, USB camera 0
    python3 perception/watch.py --csi         # Jetson CSI camera
    python3 perception/watch.py --no-display --jsonl out.jsonl   # headless
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, Optional, Sequence, Tuple

import cv2

if __package__ in (None, ""):   # direct run: put the repo root on sys.path
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UNKNOWN = "UNKNOWN"
AWAKE = "AWAKE"
EYES_CLOSED = "EYES_CLOSED"
SLEEP_CANDIDATE = "SLEEP_CANDIDATE"
DISTRESS_FACE = "DISTRESS_FACE"
STATES = (UNKNOWN, AWAKE, EYES_CLOSED, SLEEP_CANDIDATE, DISTRESS_FACE)

MOTION_HINTS = {
    UNKNOWN: "STOP_AND_CHECK",
    AWAKE: "HOLD",
    EYES_CLOSED: "HOLD_OR_TAPER",
    SLEEP_CANDIDATE: "TAPER_TO_STOP",
    DISTRESS_FACE: "GENTLE_TEST_ONLY",
}

# The bridge to CradleMachine: each state as a 0..1 distress level, chosen
# against the report's thresholds (CALM_LEVEL 0.12, CRY_LEVEL 0.45).  AWAKE
# sits under the fuss line so the machine stays quiet; DISTRESS_FACE lands in
# the fuss band and *cannot* reach the cry ladder on its own -- strength only
# moves it inside [0.30, 0.42].  UNKNOWN is not a level at all: it drops
# ``present`` and lets the safety gate do the stopping-and-checking.
_STATE_LEVEL = {
    UNKNOWN: 0.0,
    AWAKE: 0.05,
    EYES_CLOSED: 0.0,
    SLEEP_CANDIDATE: 0.0,
    DISTRESS_FACE: 0.30,
}


def distress_of(state: str, strength: float = 0.0) -> float:
    """One state -> the machine's 0..1 input.  Visual-only stays sub-cry."""
    level = _STATE_LEVEL[state]
    if state == DISTRESS_FACE:
        level = min(0.42, level + 0.12 * clamp(strength))
    return level


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile() requires at least one value")
    ordered = sorted(values)
    return ordered[int(round((len(ordered) - 1) * clamp(fraction)))]


@dataclass
class VisualObservation:
    valid_face: bool
    eye_mode: str = "unknown"  # open, half, closed, tight, unknown
    eye_ratio: float = 0.0
    mouth_ratio: float = 0.0
    motion: float = 0.0
    sudden_motion: bool = False
    face_area_ratio: float = 0.0
    open_eye_baseline: float = 0.26
    bbox: Optional[Tuple[int, int, int, int]] = None
    roll_deg: Optional[float] = None   # eye-line head roll; posture watches it


@dataclass
class StateResult:
    timestamp: float
    state: str
    motion_hint: str
    eye_mode: str
    eye_ratio: float
    mouth_ratio: float
    face_motion: float
    sudden_motion: bool
    face_area_ratio: float
    evidence_strength: float
    closed_seconds: float


class TemporalStateClassifier:
    """Turns frame observations into a stable, conservative state.

    Hysteresis keeps one-frame flickers from changing the label; only UNKNOWN
    commits immediately, because a lost face must never wait out a dwell.
    """

    def __init__(self, sleep_seconds: float = 10.0) -> None:
        self.sleep_seconds = max(3.0, sleep_seconds)
        self.closed_since: Optional[float] = None
        self.distress_since: Optional[float] = None
        self.pending_state = UNKNOWN
        self.pending_since = 0.0
        self.committed_state = UNKNOWN

    def _raw_state(self, obs: VisualObservation, now: float) -> Tuple[str, float, float]:
        if not obs.valid_face:
            self.closed_since = None
            self.distress_since = None
            return UNKNOWN, 0.0, 0.0

        # A deliberately broad visual distress rule.  It detects a pattern,
        # not its cause -- a yawn or momentary grimace can look similar.
        strong_pattern = obs.eye_mode == "tight" and obs.mouth_ratio >= 0.14
        moving_pattern = (obs.eye_mode in ("closed", "tight")
                          and obs.mouth_ratio >= 0.22
                          and obs.motion >= 0.010)
        if strong_pattern or moving_pattern:
            if self.distress_since is None:
                self.distress_since = now
        else:
            self.distress_since = None

        if self.distress_since is not None and now - self.distress_since >= 0.60:
            strength = clamp(0.45
                             + max(0.0, obs.mouth_ratio - 0.14) * 1.4
                             + (0.15 if obs.eye_mode == "tight" else 0.0),
                             0.0, 0.90)
            self.closed_since = None
            return DISTRESS_FACE, strength, 0.0

        if obs.eye_mode in ("closed", "tight"):
            if self.closed_since is None:
                self.closed_since = now
            closed_seconds = now - self.closed_since
            relaxed_closure = (obs.eye_mode == "closed"
                               and obs.mouth_ratio < 0.14
                               and obs.motion < 0.012
                               and not obs.sudden_motion)
            if closed_seconds >= self.sleep_seconds and relaxed_closure:
                return SLEEP_CANDIDATE, 0.75, closed_seconds
            return EYES_CLOSED, 0.65, closed_seconds

        if obs.eye_mode == "half":
            # Half-open eyes stay AWAKE: no drowsiness claim off a single cue.
            self.closed_since = None
            return AWAKE, 0.55, 0.0

        self.closed_since = None
        return AWAKE, 0.72 if obs.eye_mode == "open" else 0.50, 0.0

    def update(self, obs: VisualObservation, now: Optional[float] = None) -> StateResult:
        now = time.monotonic() if now is None else now
        raw_state, strength, closed_seconds = self._raw_state(obs, now)

        dwell = {UNKNOWN: 0.0, AWAKE: 0.50, EYES_CLOSED: 0.50,
                 SLEEP_CANDIDATE: 0.80, DISTRESS_FACE: 0.60}[raw_state]
        if raw_state != self.pending_state:
            self.pending_state = raw_state
            self.pending_since = now
        if raw_state == UNKNOWN or now - self.pending_since >= dwell:
            self.committed_state = raw_state

        return StateResult(
            timestamp=time.time(),
            state=self.committed_state,
            motion_hint=MOTION_HINTS[self.committed_state],
            eye_mode=obs.eye_mode,
            eye_ratio=round(obs.eye_ratio, 4),
            mouth_ratio=round(obs.mouth_ratio, 4),
            face_motion=round(obs.motion, 4),
            sudden_motion=obs.sudden_motion,
            face_area_ratio=round(obs.face_area_ratio, 4),
            evidence_strength=round(strength, 3),
            closed_seconds=round(closed_seconds, 2),
        )


class FaceFeatureExtractor:
    """Simple geometry from MediaPipe Face Mesh landmarks."""

    LEFT_EYE = (33, 160, 158, 133, 153, 144)
    RIGHT_EYE = (362, 385, 387, 263, 373, 380)
    MOUTH_LEFT = 78
    MOUTH_RIGHT = 308
    LIP_TOP = 13
    LIP_BOTTOM = 14

    def __init__(self, min_face_area: float = 0.035) -> None:
        self.min_face_area = min_face_area
        self.open_eye_samples: Deque[float] = deque(maxlen=300)
        self.open_eye_baseline = 0.26
        self.previous_center: Optional[Tuple[float, float]] = None
        self.motion_ema = 0.0

    @staticmethod
    def _ear(points: Sequence[Tuple[float, float]], ids: Sequence[int]) -> float:
        p1, p2, p3, p4, p5, p6 = (points[i] for i in ids)
        width = max(distance(p1, p4), 1e-6)
        return (distance(p2, p6) + distance(p3, p5)) / (2.0 * width)

    def extract(self, landmarks, width: int, height: int) -> VisualObservation:
        points = [(lm.x * width, lm.y * height) for lm in landmarks]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        face_width = max(x2 - x1, 1.0)
        face_area_ratio = (face_width * max(y2 - y1, 1.0)) / float(width * height)
        bbox = (max(0, int(x1)), max(0, int(y1)),
                min(width - 1, int(x2)), min(height - 1, int(y2)))

        if face_area_ratio < self.min_face_area:
            self.previous_center = None
            return VisualObservation(valid_face=False,
                                     face_area_ratio=face_area_ratio, bbox=bbox)

        eye_ratio = (self._ear(points, self.LEFT_EYE)
                     + self._ear(points, self.RIGHT_EYE)) / 2.0

        # Update only from plausibly open eyes.  The 80th percentile adapts
        # the thresholds to face size, anatomy and camera angle.
        if 0.20 <= eye_ratio <= 0.48:
            self.open_eye_samples.append(eye_ratio)
        if len(self.open_eye_samples) >= 20:
            measured = percentile(tuple(self.open_eye_samples), 0.80)
            self.open_eye_baseline = clamp(measured, 0.22, 0.36)

        baseline = self.open_eye_baseline
        if eye_ratio < 0.48 * baseline:
            eye_mode = "tight"
        elif eye_ratio < 0.62 * baseline:
            eye_mode = "closed"
        elif eye_ratio < 0.80 * baseline:
            eye_mode = "half"
        else:
            eye_mode = "open"

        mouth_width = max(distance(points[self.MOUTH_LEFT],
                                   points[self.MOUTH_RIGHT]), 1e-6)
        mouth_ratio = distance(points[self.LIP_TOP],
                               points[self.LIP_BOTTOM]) / mouth_width

        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        raw_motion = 0.0 if self.previous_center is None \
            else distance(center, self.previous_center) / face_width
        self.previous_center = center
        self.motion_ema = 0.75 * self.motion_ema + 0.25 * raw_motion

        r_eye = points[self.RIGHT_EYE[0]]
        l_eye = points[self.LEFT_EYE[0]]
        return VisualObservation(
            valid_face=True, eye_mode=eye_mode, eye_ratio=eye_ratio,
            mouth_ratio=mouth_ratio, motion=self.motion_ema,
            sudden_motion=raw_motion >= 0.050,
            face_area_ratio=face_area_ratio,
            open_eye_baseline=baseline, bbox=bbox,
            roll_deg=face_roll_deg(r_eye, l_eye),
        )


class Watcher:
    """Camera frame -> (observation, committed state).  Owns MediaPipe."""

    def __init__(self, sleep_seconds: float = 10.0,
                 min_face_area: float = 0.035) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:   # keep the module importable without it
            raise ImportError(
                "the five-state watcher needs mediapipe -- install a wheel "
                "matching this JetPack/Python/aarch64, or use the FER+ "
                "fallback (serve.py --sense without mediapipe)") from exc
        self.extractor = FaceFeatureExtractor(min_face_area=min_face_area)
        self.classifier = TemporalStateClassifier(sleep_seconds=sleep_seconds)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=0.60, min_tracking_confidence=0.60)
        self.last: Tuple[VisualObservation, Optional[StateResult]] = (
            VisualObservation(valid_face=False), None)

    def update(self, frame_bgr, ts: Optional[float] = None) -> StateResult:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detected = self.face_mesh.process(rgb)
        if not detected.multi_face_landmarks:
            obs = VisualObservation(valid_face=False)
        else:
            obs = self.extractor.extract(detected.multi_face_landmarks[0].landmark,
                                         frame_bgr.shape[1], frame_bgr.shape[0])
        result = self.classifier.update(obs, ts)
        self.last = (obs, result)
        return result

    def close(self) -> None:
        self.face_mesh.close()


# --------------------------------------------------------------------------- #
# The report-spec recognizer (evidence report sections 4-5)
#
# The five-state MVP above detects patterns; the report demands *values*:
# vocal duty cycle and dB-above-background (4.2), blink-filtered eye closure
# and body optical flow for the sleep ladder (4.3), the five FER labels as the
# learned expression source (4.1 -- the repo's FER+ fold is exactly that
# label set), and a pain/posture alarm that interrupts instead of soothing
# (5.1 rules 1-2).  Everything here is pure and fed by streams, so the whole
# judgment layer tests headlessly; a signal a sensor cannot supply is None and
# the judgments that need it simply never fire -- nothing is fabricated.
# --------------------------------------------------------------------------- #
# Estimated states, straight from the report's section-5 table rows.
QUIET_AWAKE = "QUIET_AWAKE"          # happy/neutral, quiet          -> M01
STARTLE = "STARTLE"                  # surprise/desync noise         -> observe
FUSS_WEAK = "FUSS_WEAK"              # sad + short units, duty<20%   -> M10 band
CRY = "CRY"                          # sustained units, duty>=30%    -> M12 band
STRONG_DISTRESS = "STRONG_DISTRESS"  # cry + large body movement     -> ladder
PAIN_SUSPECT = "PAIN_SUSPECT"        # pain face / arching           -> ALARM
DROWSY = "DROWSY"                    # falling vocalisation, lids    -> hold
SLEEP_TENTATIVE = "SLEEP_TENTATIVE"  # eyes closed 10 s, no vocal    -> taper
SLEEP_STABLE = "SLEEP_STABLE"        # + low body flow 60 s          -> M01
STATE_UNCLEAR = "STATE_UNCLEAR"      # face hidden / low confidence  -> gate

# state -> the machine's 0..1 input, placed against CALM_LEVEL/CRY_LEVEL.
# CRY sits over 0.45 *because the audio confirmed it* -- the report's own
# order: the fuss band is reachable on sight, the cry ladder only on sound.
_JUDGE_LEVEL = {
    QUIET_AWAKE: 0.05, STARTLE: 0.05, FUSS_WEAK: 0.30, CRY: 0.55,
    STRONG_DISTRESS: 0.70, PAIN_SUSPECT: 0.0, DROWSY: 0.0,
    SLEEP_TENTATIVE: 0.0, SLEEP_STABLE: 0.0, STATE_UNCLEAR: 0.0,
}


@dataclass
class AudioFeatures:
    """Section 4.2: the values, not just loudness."""
    voiced: bool = False        # a negative-vocalisation chunk, right now
    duty_10s: float = 0.0       # fraction of the last 10 s that was voiced
    delta_l_db: float = 0.0     # dB of the last second above background L_bg
    unit_s: float = 0.0         # length of the current/last vocal unit
    chained: bool = False       # units >= 0.5 s with < 1 s of silence between
    label: str = "quiet"        # quiet | fuss | cry, by the 4.2 thresholds


class AudioTrack:
    """Feed (level, cry, ts) chunks from listen.py; read section-4.2 features.

    Background L_bg is the 10th percentile of the last five minutes, so a fan
    or the cradle's own motor lifts the *baseline* instead of counting as a
    voice.  The 4.2 rules verbatim: quiet = duty<5% and dL<6 dB; fuss = units
    under 0.5 s and duty<20%; cry = units >= 0.5 s (or chains with <1 s gaps)
    and duty >= 30%.  Between the bands the previous label holds -- a
    hysteresis the machine's own EMA then smooths further.
    """

    VOICED_CRY = 0.45           # band ratio that reads as a voice, not a fan
    VOICED_LEVEL = 0.06         # ...at more than a whisper

    def __init__(self) -> None:
        self.window: Deque[Tuple[float, bool]] = deque()      # (ts, voiced) 10 s
        self.history: Deque[Tuple[float, float]] = deque()    # (ts, dB) 5 min
        self.unit_start: Optional[float] = None
        self.last_unit: Tuple[float, float] = (0.0, 0.0)      # (end_ts, length)
        self.chain = False
        self.label = "quiet"

    @staticmethod
    def _db(level: float) -> float:
        return 20.0 * math.log10(max(level, 1e-4))

    def feed(self, level: float, cry: float, ts: float) -> AudioFeatures:
        voiced = cry >= self.VOICED_CRY and level >= self.VOICED_LEVEL
        self.window.append((ts, voiced))
        while self.window and ts - self.window[0][0] > 10.0:
            self.window.popleft()
        self.history.append((ts, self._db(level)))
        while self.history and ts - self.history[0][0] > 300.0:
            self.history.popleft()

        if voiced and self.unit_start is None:
            gap = ts - self.last_unit[0]
            self.chain = self.last_unit[1] >= 0.5 and gap < 1.0
            self.unit_start = ts
        elif not voiced and self.unit_start is not None:
            self.last_unit = (ts, ts - self.unit_start)
            self.unit_start = None

        unit_s = (ts - self.unit_start) if self.unit_start is not None \
            else self.last_unit[1]
        duty = (sum(1 for _, v in self.window if v)
                / max(1, len(self.window)))
        l_bg = percentile([db for _, db in self.history], 0.10)
        recent = [db for t, db in self.history if ts - t <= 1.0]
        delta = (max(recent) if recent else l_bg) - l_bg

        if duty < 0.05 and delta < 6.0:
            self.label = "quiet"
        elif (unit_s >= 0.5 or self.chain) and duty >= 0.30:
            self.label = "cry"
        elif 0.0 < unit_s < 0.5 and 0.0 < duty < 0.20:
            self.label = "fuss"
        return AudioFeatures(voiced=voiced, duty_10s=duty, delta_l_db=delta,
                             unit_s=unit_s, chained=self.chain, label=self.label)


class BodyMotion:
    """Frame differencing on a small grayscale -- the 4.3 'body optical flow'.

    Whole-frame, so it sees limbs and rolling, which the face box never did.
    No model: mean absolute difference on a 64-wide grayscale is robust, free,
    and monotone in how much the body is actually moving.
    """

    def __init__(self) -> None:
        self.prev = None
        self.flow = 0.0

    def feed(self, frame_bgr) -> float:
        small = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY),
                           (64, 48))
        if self.prev is not None:
            raw = float(cv2.absdiff(small, self.prev).mean()) / 255.0
            self.flow = 0.7 * self.flow + 0.3 * min(1.0, raw * 8.0)
        self.prev = small
        return self.flow


@dataclass
class Signals:
    """One fused tick.  None = that sensor cannot see this; never guessed."""
    ts: float
    face_conf: float = 0.0              # 0 = no face; the gate watches this
    emotion: Optional[str] = None       # FER+ label: the learned 4.1 source
    eyes_closed: Optional[bool] = None  # needs landmarks (mesh); FER+ can't
    pain_face: Optional[bool] = None    # squeeze+brow+furrow AUs (mesh)
    body_arch: Optional[bool] = None    # arching/gagging posture cue
    posture_risk: Optional[bool] = None # side/prone/rolling suspicion
    body_flow: float = 0.0              # BodyMotion, 0..1
    audio: AudioFeatures = None         # AudioTrack.feed of the same moment


@dataclass
class InfantReading:
    """What the recognizer owes the rest of the system, per the report."""
    ts: float
    state: str
    present: bool          # False -> the safety gate, not a level
    level: float           # the machine's 0..1 distress input
    alarm: bool            # pain/posture: interrupt + caregiver, never soothe
    improving: bool        # 4.3: 2 of the observable signals fell over 30 s
    worsening: bool
    closed_s: float        # blink-filtered eye closure, for the sleep ladder
    detail: str            # one line for the event log


class InfantJudge:
    """Sections 4.3 and 5, as a temporal judge over Signals.

    Sleep discipline is verbatim: a closure under one second is a blink and
    counts for nothing; closure within two seconds of a negative vocalisation
    is not labelled sleep; SLEEP_TENTATIVE needs ten silent closed seconds and
    SLEEP_STABLE adds sixty seconds of low body flow.  Pain or posture risk
    raises ``alarm`` -- the caller feeds that to the machine's gate, because
    rule one of 5.1 is that those interrupt rather than soothen.
    """

    BLINK_S = 1.0
    VOCAL_GUARD_S = 2.0
    TENTATIVE_S = 10.0
    STABLE_S = 60.0
    FLOW_SLEEP = 0.08          # "low body optical flow"
    FLOW_BIG = 0.35            # "large body movement" (section-5 row)

    def __init__(self) -> None:
        self.closed_since: Optional[float] = None
        self.last_vocal_ts = -1e9
        self.low_flow_since: Optional[float] = None
        self.trend: Deque[Tuple[float, float, float]] = deque()  # ts,duty,flow
        self.state = STATE_UNCLEAR

    def _trend(self, ts: float, audio: AudioFeatures, flow: float):
        self.trend.append((ts, audio.duty_10s, flow))
        while self.trend and ts - self.trend[0][0] > 60.0:
            self.trend.popleft()
        old = [(d, f) for t, d, f in self.trend if ts - t >= 30.0]
        if not old:
            return False, False
        d0 = sum(d for d, _ in old) / len(old)
        f0 = sum(f for _, f in old) / len(old)
        better = sum((audio.duty_10s <= d0 * 0.7,
                      audio.delta_l_db <= -3.0,
                      flow <= f0 * 0.7))
        worse = sum((audio.duty_10s >= d0 * 1.2 + 0.02,
                     audio.delta_l_db >= 6.0,
                     flow >= f0 * 1.3 + 0.02))
        # The report asks for 2 of 4; face tension is not yet observable, so
        # this is 2 of the 3 signals that are.  Same bar, honest denominator.
        return better >= 2, worse >= 2

    def update(self, sig: Signals) -> InfantReading:
        ts = ts_now = sig.ts
        audio = sig.audio or AudioFeatures()
        if audio.voiced:
            self.last_vocal_ts = ts

        # -- blink-filtered closure clock (4.3) ------------------------------ #
        if sig.eyes_closed:
            if self.closed_since is None:
                self.closed_since = ts
        else:
            self.closed_since = None
        closed_s = 0.0
        if self.closed_since is not None:
            closed_s = ts - self.closed_since
            if closed_s < self.BLINK_S:
                closed_s = 0.0          # a blink is not closure
        vocal_guard = ts - self.last_vocal_ts <= self.VOCAL_GUARD_S

        if sig.body_flow <= self.FLOW_SLEEP:
            if self.low_flow_since is None:
                self.low_flow_since = ts
        else:
            self.low_flow_since = None
        low_flow_s = 0.0 if self.low_flow_since is None \
            else ts - self.low_flow_since

        improving, worsening = self._trend(ts, audio, sig.body_flow)

        # -- the section-5 rows, gate first ---------------------------------- #
        alarm = bool(sig.pain_face) or bool(sig.body_arch) \
            or bool(sig.posture_risk)
        present = sig.face_conf >= 0.5
        if alarm:
            state, detail = PAIN_SUSPECT, "pain/posture signature -- interrupting"
        elif not present:
            state, detail = STATE_UNCLEAR, "face hidden or unreliable"
        elif closed_s >= self.TENTATIVE_S and not vocal_guard:
            if low_flow_s >= self.STABLE_S:
                state, detail = SLEEP_STABLE, f"stable sleep ({low_flow_s:.0f}s still)"
            else:
                state, detail = SLEEP_TENTATIVE, f"eyes closed {closed_s:.0f}s, quiet"
        elif audio.label == "cry":
            if sig.body_flow >= self.FLOW_BIG:
                state = STRONG_DISTRESS
                detail = f"crying + large movement (duty {audio.duty_10s:.0%})"
            else:
                state, detail = CRY, f"crying (duty {audio.duty_10s:.0%})"
        elif audio.label == "fuss" or sig.emotion in ("SAD", "ANGRY"):
            state, detail = FUSS_WEAK, "weak fussing"
        elif sig.emotion == "SURPRISE":
            state, detail = STARTLE, "startle -- observing"
        elif closed_s > 0.0:
            state, detail = DROWSY, "eyes closing"
        else:
            state, detail = QUIET_AWAKE, "quiet and awake"
        self.state = state

        return InfantReading(
            ts=ts_now, state=state,
            present=present and state != STATE_UNCLEAR,
            level=_JUDGE_LEVEL[state], alarm=alarm,
            improving=improving, worsening=worsening,
            closed_s=round(closed_s, 2), detail=detail,
        )


# --------------------------------------------------------------------------- #
# Posture (report 1.1: supine only -- side/prone/rolling suspicion gates)
# --------------------------------------------------------------------------- #
class PostureNet:
    """BlazePose landmarks through cv2.dnn -- shoulders, for posture only.

    Pinned by tools/fetch_models.py and verified to parse AND run under this
    box's cv2 4.5.4 before pinning (the YuNet-2023 lesson).  The face box
    seeds the body window: a crib camera always has the face as its anchor,
    and the landmark model wants a person-centred crop, not a whole room.

    Honesty note: BlazePose is trained on adults.  On an infant the absolute
    keypoints are unvalidated, which is why the posture rule below consumes
    only *coarse, relative* cues -- shoulder visibility asymmetry and
    foreshortening -- and why it must be bench-verified per docs/VERIFY.md
    before anyone trusts it near a real baby.
    """

    L_SHOULDER, R_SHOULDER = 11, 12

    def __init__(self) -> None:
        from perception.face import config
        if not config.BLAZEPOSE.exists():
            raise FileNotFoundError(
                f"{config.BLAZEPOSE} missing -- run: python3 tools/fetch_models.py")
        self.net = cv2.dnn.readNetFromONNX(str(config.BLAZEPOSE))

    def infer(self, frame_bgr, face_bbox=None):
        """-> ((lx,ly,lvis), (rx,ry,rvis)) in frame pixels, or None."""
        h, w = frame_bgr.shape[:2]
        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            fh = max(y2 - y1, 10)
            cx = (x1 + x2) / 2.0
            half = max(x2 - x1, fh) * 2.2       # face anchors the body window
            rx1 = int(max(0, cx - half)); rx2 = int(min(w, cx + half))
            ry1 = int(max(0, y1 - fh)); ry2 = int(min(h, y1 + 4.5 * fh))
        else:
            rx1, ry1, rx2, ry2 = 0, 0, w, h
        crop = frame_bgr[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            return None
        blob = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (256, 256))
        self.net.setInput(blob[None].astype("float32") / 255.0)
        out = self.net.forward("Identity").reshape(-1, 5)   # 39 x (x,y,z,vis,pres)
        sx = (rx2 - rx1) / 256.0
        sy = (ry2 - ry1) / 256.0
        pts = []
        for idx in (self.L_SHOULDER, self.R_SHOULDER):
            x, y, _, vis, pres = out[idx]
            conf = 1.0 / (1.0 + math.exp(-min(50.0, max(-50.0, float(min(vis, pres))))))
            pts.append((rx1 + float(x) * sx, ry1 + float(y) * sy, conf))
        return tuple(pts)


class PostureTrack:
    """Coarse cues -> a sustained posture-risk flag.  Pure and testable.

    Three cues, any of which counts, each held for ``sustain_s`` before the
    flag raises (a squirm is not a rollover):

    * face roll: the eye line past 60 deg from level -- side-lying head
    * shoulder visibility asymmetry: one shoulder gone while the other is
      confident -- the body has turned
    * foreshortening: shoulder width collapsing under 60% of face width --
      the shoulder line has rotated out of the image plane
    """

    ROLL_DEG = 60.0
    VIS_GAP = 0.45
    WIDTH_RATIO = 0.60

    def __init__(self, sustain_s: float = 2.0) -> None:
        self.sustain_s = sustain_s
        self.since: Optional[float] = None

    def feed(self, ts: float, face_roll_deg: Optional[float],
             shoulders=None, face_width_px: float = 0.0) -> bool:
        cue = False
        if face_roll_deg is not None and abs(face_roll_deg) >= self.ROLL_DEG:
            cue = True
        if shoulders is not None:
            (lx, ly, lv), (rx, ry, rv) = shoulders
            if abs(lv - rv) >= self.VIS_GAP and max(lv, rv) >= 0.6:
                cue = True
            width = math.hypot(lx - rx, ly - ry)
            if (min(lv, rv) >= 0.5 and face_width_px > 0
                    and width < self.WIDTH_RATIO * face_width_px):
                cue = True
        if cue:
            if self.since is None:
                self.since = ts
        else:
            self.since = None
        return self.since is not None and ts - self.since >= self.sustain_s


def face_roll_deg(right_eye, left_eye) -> float:
    """Eye line -> head roll in degrees; 0 = level, +-90 = fully sideways."""
    dx = left_eye[0] - right_eye[0]
    dy = left_eye[1] - right_eye[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    ang = math.degrees(math.atan2(dy, dx))
    while ang > 90.0:
        ang -= 180.0
    while ang < -90.0:
        ang += 180.0
    return ang


# --------------------------------------------------------------------------- #
# Overlay
# --------------------------------------------------------------------------- #
STATE_BGR = {
    UNKNOWN: (120, 120, 120),
    AWAKE: (80, 210, 120),
    EYES_CLOSED: (230, 190, 60),
    SLEEP_CANDIDATE: (230, 140, 60),
    DISTRESS_FACE: (70, 70, 240),
}


def draw_overlay(frame, obs: VisualObservation, result: StateResult) -> None:
    color = STATE_BGR[result.state]
    if obs.bbox is not None:
        x1, y1, x2, y2 = obs.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(frame, (12, 12), (590, 126), (16, 22, 30), -1)
    cv2.putText(frame, f"STATE: {result.state}", (26, 43),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    cv2.putText(frame,
                f"eye={result.eye_mode}  EAR={result.eye_ratio:.3f}  "
                f"mouth={result.mouth_ratio:.3f}", (26, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(frame,
                f"motion={result.face_motion:.3f}  sudden={result.sudden_motion}  "
                f"hint={result.motion_hint}", (26, 99),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(frame, "Visual estimate only - not cry/pain/sleep confirmation",
                (26, 119), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 185, 195), 1,
                cv2.LINE_AA)


def jetson_csi_pipeline(width: int, height: int, fps: int, sensor_id: int) -> str:
    return (f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM),width=1280,height=720,"
            f"framerate={fps}/1,format=NV12 ! "
            f"nvvidconv ! video/x-raw,width={width},height={height},format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Five-state infant face classifier (camera only).")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="USB camera index")
    source.add_argument("--csi", action="store_true", help="Jetson CSI camera")
    source.add_argument("--video", type=Path,
                        help="replay a recorded clip instead of a camera -- "
                             "the bench-verification path (docs/VERIFY.md)")
    parser.add_argument("--sensor-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sleep-seconds", type=float, default=10.0,
                        help="relaxed closure before SLEEP_CANDIDATE (min 3 s)")
    parser.add_argument("--min-face-area", type=float, default=0.035)
    parser.add_argument("--publish-hz", type=float, default=2.0)
    parser.add_argument("--jsonl", type=Path, help="JSON Lines output file")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args(argv)

    watcher = Watcher(sleep_seconds=args.sleep_seconds,
                      min_face_area=args.min_face_area)
    if args.video:
        capture = cv2.VideoCapture(str(args.video))
    elif args.csi:
        capture = cv2.VideoCapture(
            jetson_csi_pipeline(args.width, args.height, args.fps,
                                args.sensor_id), cv2.CAP_GSTREAMER)
    else:
        capture = cv2.VideoCapture(args.camera)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, args.fps)
    if not capture.isOpened():
        print("ERROR: camera could not be opened", file=sys.stderr)
        return 3

    publish_interval = 1.0 / max(args.publish_hz, 0.2)
    last_publish = 0.0
    json_file = None
    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        json_file = args.jsonl.open("a", encoding="utf-8")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                if args.video:
                    break           # end of the clip: exit, don't spin
                print("WARN: failed to read camera frame", file=sys.stderr)
                time.sleep(0.05)
                continue
            if args.mirror:
                frame = cv2.flip(frame, 1)
            result = watcher.update(frame)
            now = time.monotonic()
            if now - last_publish >= publish_interval:
                line = json.dumps(asdict(result), ensure_ascii=False)
                print(line, flush=True)
                if json_file is not None:
                    json_file.write(line + "\n")
                    json_file.flush()
                last_publish = now
            if not args.no_display:
                draw_overlay(frame, watcher.last[0], result)
                cv2.imshow("Infant Camera State", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        watcher.close()
        capture.release()
        if json_file is not None:
            json_file.close()
        if not args.no_display:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

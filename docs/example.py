#!/usr/bin/env python3
"""
Jetson infant camera state classifier (camera only, rule-based MVP)

Outputs only five broad visual states:
    UNKNOWN            Face is missing, too small, or landmarks are unreliable.
    AWAKE              Eyes are open and no strong distress-like pattern is seen.
    EYES_CLOSED        Eyes are closed, but sleep cannot yet be inferred.
    SLEEP_CANDIDATE    Relaxed eye closure continues with little face motion.
    DISTRESS_FACE      Eye squeeze + wide mouth/tension-like visual pattern.

Important:
    - DISTRESS_FACE does not prove crying, pain, hunger, or illness.
    - SLEEP_CANDIDATE does not prove physiological sleep.
    - Audio or a caregiver check is required before escalating cradle motion.
    - If the face is not reliably visible, the state is always UNKNOWN.

Dependencies:
    Python 3, OpenCV, MediaPipe

USB camera example:
    python3 jetson_infant_camera_state.py --camera 0

Jetson CSI camera example:
    python3 jetson_infant_camera_state.py --csi

Headless example (JSON output only):
    python3 jetson_infant_camera_state.py --csi --no-display \
        --jsonl infant_state.jsonl

The program writes one JSON object per update to stdout. A motion selector can
read the `state`, `motion_hint`, and `sudden_motion` fields.
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


UNKNOWN = "UNKNOWN"
AWAKE = "AWAKE"
EYES_CLOSED = "EYES_CLOSED"
SLEEP_CANDIDATE = "SLEEP_CANDIDATE"
DISTRESS_FACE = "DISTRESS_FACE"

MOTION_HINTS = {
    UNKNOWN: "STOP_AND_CHECK",
    AWAKE: "HOLD",
    EYES_CLOSED: "HOLD_OR_TAPER",
    SLEEP_CANDIDATE: "TAPER_TO_STOP",
    DISTRESS_FACE: "GENTLE_TEST_ONLY",
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile() requires at least one value")
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * clamp(fraction)))
    return ordered[index]


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
    """Turns frame observations into a stable, conservative state."""

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

        # A deliberately broad visual distress rule. It detects a pattern,
        # not its cause. A yawn or momentary grimace can look similar.
        strong_pattern = obs.eye_mode == "tight" and obs.mouth_ratio >= 0.14
        moving_pattern = (
            obs.eye_mode in ("closed", "tight")
            and obs.mouth_ratio >= 0.22
            and obs.motion >= 0.010
        )
        distress_pattern = strong_pattern or moving_pattern

        if distress_pattern:
            if self.distress_since is None:
                self.distress_since = now
        else:
            self.distress_since = None

        if self.distress_since is not None and now - self.distress_since >= 0.60:
            strength = clamp(
                0.45
                + max(0.0, obs.mouth_ratio - 0.14) * 1.4
                + (0.15 if obs.eye_mode == "tight" else 0.0),
                0.0,
                0.90,
            )
            self.closed_since = None
            return DISTRESS_FACE, strength, 0.0

        if obs.eye_mode in ("closed", "tight"):
            if self.closed_since is None:
                self.closed_since = now
            closed_seconds = now - self.closed_since

            relaxed_closure = (
                obs.eye_mode == "closed"
                and obs.mouth_ratio < 0.14
                and obs.motion < 0.012
                and not obs.sudden_motion
            )
            if closed_seconds >= self.sleep_seconds and relaxed_closure:
                return SLEEP_CANDIDATE, 0.75, closed_seconds
            return EYES_CLOSED, 0.65, closed_seconds

        if obs.eye_mode == "half":
            # Half-open eyes are kept in AWAKE. The classifier does not claim
            # drowsiness from a single visual cue.
            self.closed_since = None
            return AWAKE, 0.55, 0.0

        self.closed_since = None
        awake_strength = 0.72 if obs.eye_mode == "open" else 0.50
        return AWAKE, awake_strength, 0.0

    def update(self, obs: VisualObservation, now: Optional[float] = None) -> StateResult:
        now = time.monotonic() if now is None else now
        raw_state, strength, closed_seconds = self._raw_state(obs, now)

        # Hysteresis prevents one-frame label changes. UNKNOWN is immediate.
        dwell = {
            UNKNOWN: 0.0,
            AWAKE: 0.50,
            EYES_CLOSED: 0.50,
            SLEEP_CANDIDATE: 0.80,
            DISTRESS_FACE: 0.60,
        }[raw_state]

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
    """Extracts simple geometry from MediaPipe Face Mesh landmarks."""

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
        p1, p2, p3, p4, p5, p6 = (points[index] for index in ids)
        width = max(distance(p1, p4), 1e-6)
        return (distance(p2, p6) + distance(p3, p5)) / (2.0 * width)

    def extract(self, landmarks, width: int, height: int) -> VisualObservation:
        points = [(lm.x * width, lm.y * height) for lm in landmarks]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        face_width = max(x2 - x1, 1.0)
        face_height = max(y2 - y1, 1.0)
        face_area_ratio = (face_width * face_height) / float(width * height)
        bbox = (
            max(0, int(x1)),
            max(0, int(y1)),
            min(width - 1, int(x2)),
            min(height - 1, int(y2)),
        )

        if face_area_ratio < self.min_face_area:
            self.previous_center = None
            return VisualObservation(
                valid_face=False,
                face_area_ratio=face_area_ratio,
                bbox=bbox,
            )

        left_ear = self._ear(points, self.LEFT_EYE)
        right_ear = self._ear(points, self.RIGHT_EYE)
        eye_ratio = (left_ear + right_ear) / 2.0

        # Update only from plausibly open eyes. The 80th percentile makes the
        # thresholds adapt to face size, anatomy, and camera angle.
        if 0.20 <= eye_ratio <= 0.48:
            self.open_eye_samples.append(eye_ratio)
        if len(self.open_eye_samples) >= 20:
            measured = percentile(tuple(self.open_eye_samples), 0.80)
            self.open_eye_baseline = clamp(measured, 0.22, 0.36)

        baseline = self.open_eye_baseline
        tight_threshold = 0.48 * baseline
        closed_threshold = 0.62 * baseline
        open_threshold = 0.80 * baseline
        if eye_ratio < tight_threshold:
            eye_mode = "tight"
        elif eye_ratio < closed_threshold:
            eye_mode = "closed"
        elif eye_ratio < open_threshold:
            eye_mode = "half"
        else:
            eye_mode = "open"

        mouth_width = max(
            distance(points[self.MOUTH_LEFT], points[self.MOUTH_RIGHT]), 1e-6
        )
        mouth_ratio = distance(points[self.LIP_TOP], points[self.LIP_BOTTOM]) / mouth_width

        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        if self.previous_center is None:
            raw_motion = 0.0
        else:
            raw_motion = distance(center, self.previous_center) / face_width
        self.previous_center = center
        self.motion_ema = 0.75 * self.motion_ema + 0.25 * raw_motion
        sudden_motion = raw_motion >= 0.050

        return VisualObservation(
            valid_face=True,
            eye_mode=eye_mode,
            eye_ratio=eye_ratio,
            mouth_ratio=mouth_ratio,
            motion=self.motion_ema,
            sudden_motion=sudden_motion,
            face_area_ratio=face_area_ratio,
            open_eye_baseline=baseline,
            bbox=bbox,
        )


def jetson_csi_pipeline(width: int, height: int, fps: int, sensor_id: int) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width=1280,height=720,framerate={fps}/1,format=NV12 ! "
        f"nvvidconv ! video/x-raw,width={width},height={height},format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def draw_overlay(cv2, frame, obs: VisualObservation, result: StateResult) -> None:
    color = {
        UNKNOWN: (120, 120, 120),
        AWAKE: (80, 210, 120),
        EYES_CLOSED: (230, 190, 60),
        SLEEP_CANDIDATE: (230, 140, 60),
        DISTRESS_FACE: (70, 70, 240),
    }[result.state]

    if obs.bbox is not None:
        x1, y1, x2, y2 = obs.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.rectangle(frame, (12, 12), (590, 126), (16, 22, 30), -1)
    cv2.putText(
        frame,
        f"STATE: {result.state}",
        (26, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"eye={result.eye_mode}  EAR={result.eye_ratio:.3f}  mouth={result.mouth_ratio:.3f}",
        (26, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"motion={result.face_motion:.3f}  sudden={result.sudden_motion}  hint={result.motion_hint}",
        (26, 99),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Visual estimate only - not cry/pain/sleep confirmation",
        (26, 119),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (180, 185, 195),
        1,
        cv2.LINE_AA,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Five-state infant face classifier for Jetson (camera only)."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="USB camera index")
    source.add_argument("--csi", action="store_true", help="Use Jetson CSI camera")
    parser.add_argument("--sensor-id", type=int, default=0, help="CSI sensor id")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=10.0,
        help="Relaxed closure duration before SLEEP_CANDIDATE (minimum 3 s)",
    )
    parser.add_argument(
        "--min-face-area",
        type=float,
        default=0.035,
        help="Minimum face area as fraction of image area",
    )
    parser.add_argument("--publish-hz", type=float, default=2.0)
    parser.add_argument("--jsonl", type=Path, help="Optional JSON Lines output file")
    parser.add_argument("--mirror", action="store_true", help="Mirror preview and input")
    parser.add_argument("--no-display", action="store_true", help="Headless mode")
    parser.add_argument("--self-test", action="store_true", help="Run logic test and exit")
    return parser.parse_args()


def run_self_test() -> None:
    classifier = TemporalStateClassifier(sleep_seconds=3.0)
    t = 100.0
    unknown = VisualObservation(valid_face=False)
    assert classifier.update(unknown, t).state == UNKNOWN

    awake = VisualObservation(valid_face=True, eye_mode="open", eye_ratio=0.27)
    classifier.update(awake, t + 0.1)
    assert classifier.update(awake, t + 0.7).state == AWAKE

    closed = VisualObservation(
        valid_face=True,
        eye_mode="closed",
        eye_ratio=0.13,
        mouth_ratio=0.04,
        motion=0.002,
    )
    classifier.update(closed, t + 1.0)
    assert classifier.update(closed, t + 1.6).state == EYES_CLOSED
    classifier.update(closed, t + 4.1)
    assert classifier.update(closed, t + 5.0).state == SLEEP_CANDIDATE

    distress = VisualObservation(
        valid_face=True,
        eye_mode="tight",
        eye_ratio=0.08,
        mouth_ratio=0.25,
        motion=0.02,
    )
    classifier.update(distress, t + 6.0)
    classifier.update(distress, t + 6.7)
    assert classifier.update(distress, t + 7.4).state == DISTRESS_FACE
    print("SELF_TEST_OK")


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0

    try:
        import cv2
    except ImportError:
        print("ERROR: OpenCV is not installed (module: cv2).", file=sys.stderr)
        return 2
    try:
        import mediapipe as mp
    except ImportError:
        print(
            "ERROR: MediaPipe is not installed. Install a wheel compatible with "
            "your JetPack, Python, and aarch64 version.",
            file=sys.stderr,
        )
        return 2

    if args.csi:
        pipeline = jetson_csi_pipeline(args.width, args.height, args.fps, args.sensor_id)
        capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        capture = cv2.VideoCapture(args.camera)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, args.fps)

    if not capture.isOpened():
        print("ERROR: Camera could not be opened.", file=sys.stderr)
        return 3

    extractor = FaceFeatureExtractor(min_face_area=args.min_face_area)
    classifier = TemporalStateClassifier(sleep_seconds=args.sleep_seconds)
    publish_interval = 1.0 / max(args.publish_hz, 0.2)
    last_publish = 0.0
    json_file = None
    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        json_file = args.jsonl.open("a", encoding="utf-8")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.60,
        min_tracking_confidence=0.60,
    )

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("WARN: Failed to read camera frame.", file=sys.stderr)
                time.sleep(0.05)
                continue
            if args.mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected = face_mesh.process(rgb)
            if not detected.multi_face_landmarks:
                obs = VisualObservation(valid_face=False)
            else:
                obs = extractor.extract(
                    detected.multi_face_landmarks[0].landmark,
                    frame.shape[1],
                    frame.shape[0],
                )

            result = classifier.update(obs)
            now = time.monotonic()
            if now - last_publish >= publish_interval:
                payload = asdict(result)
                line = json.dumps(payload, ensure_ascii=False)
                print(line, flush=True)
                if json_file is not None:
                    json_file.write(line + "\n")
                    json_file.flush()
                last_publish = now

            if not args.no_display:
                draw_overlay(cv2, frame, obs, result)
                cv2.imshow("Infant Camera State", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        face_mesh.close()
        capture.release()
        if json_file is not None:
            json_file.close()
        if not args.no_display:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
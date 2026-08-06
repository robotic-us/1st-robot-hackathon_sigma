#!/usr/bin/env python3
"""Webcam + microphone -> one small reading the robot can act on.

This is the perception front end.  It replaces the coloured-circle tracker we
prototyped with: a real face, its emotion, and the sound in the room.

    frame ──► YuNet ──► FER+ ──► emotion probabilities ─┐
                                                        ├──► distress 0..1
    mic   ──► RMS + voice-band ratio ───────────────────┘

Everything downstream sees only :class:`Reading` -- five numbers and two
labels.  No pixels, no audio buffers.  That is what keeps the decision layer
testable and this file replaceable.

Two design choices worth knowing:

* **Distress uses the whole probability vector, not the argmax label.**  A face
  that is 45% ANGRY / 40% SAD is clearly upset, but its argmax flickers between
  two labels frame to frame.  Weighting all five probabilities gives a signal
  that moves smoothly, which matters because it drives how big a motion we play.

* **Face and sound combine as a "noisy OR"**: ``1 - (1-face)*(1-sound)``.
  In plain terms -- either channel alone can raise the alarm, and both together
  raise it further, but neither can ever pull it back down.  A crying person who
  turns away from the camera still reads as distressed.

Run standalone::

    python3 sense.py                  # live window, face + sound + distress
    python3 sense.py --no-window      # print the numbers instead
    python3 sense.py --selftest       # no camera, no microphone
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from listen import Microphone, Sound
from sigma import config as face_config
from sigma.draw import draw_result
from sigma.pipeline import SigmaPipeline

LOGGER = logging.getLogger("sense")

# How much each emotion counts as "this person needs attention", 0..1.
# ANGRY and SAD are the states a caregiver should respond to; HAPPY explicitly
# scores zero so a smiling person is left alone.
DISTRESS_WEIGHT = {
    "ANGRY":    1.00,
    "SAD":      0.85,
    "SURPRISE": 0.50,
    "NEUTRAL":  0.10,
    "HAPPY":    0.00,
}


@dataclass(frozen=True)
class Reading:
    """What we know about the person in front of the robot, right now."""

    present: bool     # is there a face?
    x: float          # [-1, 1] left..right, face centre
    y: float          # [-1, 1] top..bottom
    distance: float   # [0, 1] apparent size; bigger = nearer
    distress: float   # [0, 1] fused face + sound
    emotion: str      # the argmax label, for the log and the overlay
    name: str         # who it is, if enrolled
    face_distress: float = 0.0   # the two halves, kept separate so the log can
    sound_distress: float = 0.0  # say *why* distress is high
    ts: float = 0.0

    def describe(self) -> str:
        if not self.present:
            return (f"no face | sound {self.sound_distress:.2f} "
                    f"-> distress {self.distress:.2f}")
        return (f"{self.name} {self.emotion} @ x={self.x:+.2f} d={self.distance:.2f} "
                f"| face {self.face_distress:.2f} + sound {self.sound_distress:.2f} "
                f"-> distress {self.distress:.2f}")


def emotion_distress(probs: np.ndarray) -> float:
    """Expected distress under the emotion distribution.  See the module note."""
    if probs is None or len(probs) != len(face_config.EMOTIONS):
        return 0.0
    weights = np.array(
        [DISTRESS_WEIGHT.get(e, 0.0) for e in face_config.EMOTIONS], np.float32
    )
    return float(np.clip(np.dot(probs, weights), 0.0, 1.0))


def fuse(face: float, sound: float) -> float:
    """Noisy-OR: either channel can raise distress, neither can lower it."""
    return float(np.clip(1.0 - (1.0 - face) * (1.0 - sound), 0.0, 1.0))


class Sense:
    """Turns frames + sound into :class:`Reading`.  Owns no camera."""

    def __init__(
        self,
        pipeline: Optional[SigmaPipeline] = None,
        microphone: Optional[Microphone] = None,
        near_px: float = 260.0,   # face height that reads as distance = 1.0
        far_px: float = 40.0,     # ...and as 0.0
    ) -> None:
        self.pipeline = pipeline or SigmaPipeline()
        self.microphone = microphone
        self.near_px = near_px
        self.far_px = far_px
        self.results: list = []   # last frame's faces, for the overlay

    def update(self, frame: np.ndarray, ts: Optional[float] = None) -> Reading:
        ts = time.monotonic() if ts is None else ts
        height, width = frame.shape[:2]

        self.results = self.pipeline.process(frame)
        sound = self.microphone.latest() if self.microphone is not None \
            else Sound(0.0, 0.0, 0.0, ts)

        if not self.results:
            # Nobody in view, but a cry still counts -- that is the whole point
            # of having a second channel.
            return Reading(
                present=False, x=0.0, y=0.0, distance=0.0,
                distress=sound.distress, emotion="-", name="-",
                face_distress=0.0, sound_distress=sound.distress, ts=ts,
            )

        # The nearest face is the one being cared for. Largest box wins.
        primary = max(self.results, key=lambda r: r.box[2] * r.box[3])
        x_px, y_px, w_px, h_px = primary.box

        x = (2.0 * (x_px + w_px * 0.5) / width) - 1.0
        y = (2.0 * (y_px + h_px * 0.5) / height) - 1.0
        span = max(1.0, self.near_px - self.far_px)
        distance = float(np.clip((h_px - self.far_px) / span, 0.0, 1.0))

        face = emotion_distress(primary.probs)
        return Reading(
            present=True,
            x=float(np.clip(x, -1.0, 1.0)),
            y=float(np.clip(y, -1.0, 1.0)),
            distance=distance,
            distress=fuse(face, sound.distress),
            emotion=primary.emotion,
            name=primary.name,
            face_distress=face,
            sound_distress=sound.distress,
            ts=ts,
        )


# --------------------------------------------------------------------------- #
# Overlay
# --------------------------------------------------------------------------- #
def draw(frame: np.ndarray, reading: Reading, sense: Sense) -> np.ndarray:
    """Boxes and labels from sigma.draw, plus the fused distress bar."""
    canvas = frame.copy()
    for r in sense.results:
        draw_result(canvas, r)

    height, width = canvas.shape[:2]
    bar_x, bar_w, bar_h = 10, width - 20, 16
    bar_y = height - bar_h - 10
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), 1)
    filled = int(bar_w * reading.distress)
    if filled > 0:
        # green -> red as distress climbs
        color = (0, int(255 * (1.0 - reading.distress)), int(255 * reading.distress))
        cv2.rectangle(canvas, (bar_x + 1, bar_y + 1),
                      (bar_x + filled, bar_y + bar_h - 1), color, -1)
    cv2.putText(canvas, f"distress {reading.distress:.2f}  "
                        f"(face {reading.face_distress:.2f} + "
                        f"sound {reading.sound_distress:.2f})",
                (bar_x + 4, bar_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (230, 230, 230), 1, cv2.LINE_AA)
    return canvas


# --------------------------------------------------------------------------- #
# Selftest
# --------------------------------------------------------------------------- #
def run_selftest() -> int:
    """Check the two bits of maths that decide how the robot behaves."""
    print("emotion -> distress")
    n = len(face_config.EMOTIONS)

    def one_hot(label: str) -> np.ndarray:
        v = np.zeros(n, np.float32)
        v[face_config.EMOTIONS.index(label)] = 1.0
        return v

    for label in face_config.EMOTIONS:
        d = emotion_distress(one_hot(label))
        assert abs(d - DISTRESS_WEIGHT[label]) < 1e-6, f"{label} weight wrong"
        print(f"  {label:<9} -> {d:.2f}")

    assert emotion_distress(one_hot("HAPPY")) == 0.0, "a smile must not summon the robot"
    assert emotion_distress(one_hot("ANGRY")) > emotion_distress(one_hot("NEUTRAL"))

    # The mixed case that motivated using probabilities instead of the label.
    mixed = np.zeros(n, np.float32)
    mixed[face_config.EMOTIONS.index("ANGRY")] = 0.45
    mixed[face_config.EMOTIONS.index("SAD")] = 0.40
    mixed[face_config.EMOTIONS.index("NEUTRAL")] = 0.15
    d = emotion_distress(mixed)
    assert 0.7 < d < 0.9, f"45% angry + 40% sad should read clearly upset, got {d:.3f}"
    print(f"  45% ANGRY + 40% SAD + 15% NEUTRAL -> {d:.2f} (stable across a label flip)")

    print("\nnoisy-OR fusion")
    assert fuse(0.0, 0.0) == 0.0
    assert abs(fuse(0.8, 0.0) - 0.8) < 1e-6, "sound silent -> face alone decides"
    assert abs(fuse(0.0, 0.8) - 0.8) < 1e-6, "face absent -> sound alone decides"
    assert fuse(0.5, 0.5) > 0.5, "both channels must reinforce"
    assert fuse(0.9, 0.9) <= 1.0, "fusion must stay bounded"
    for a, b in ((0.3, 0.4), (0.9, 0.1), (0.0, 1.0)):
        assert fuse(a, b) >= max(a, b) - 1e-9, "fusion must never lower distress"
    print(f"  face only  0.80 + 0.00 -> {fuse(0.8, 0.0):.2f}")
    print(f"  sound only 0.00 + 0.80 -> {fuse(0.0, 0.8):.2f}")
    print(f"  both       0.50 + 0.50 -> {fuse(0.5, 0.5):.2f}")

    print("\nselftest PASSED")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--audio-device", default="plughw:WEBCAM,0")
    parser.add_argument("--no-sound", action="store_true")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)-7s %(name)s: %(message)s")
    if args.selftest:
        return run_selftest()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        LOGGER.error("could not open camera %d", args.camera_index)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, face_config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, face_config.FRAME_H)

    mic = None if args.no_sound else Microphone(args.audio_device)
    if mic is not None:
        mic.start()
    sense = Sense(microphone=mic)
    window = "sense: face + sound -> distress"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            reading = sense.update(frame)
            if args.no_window:
                print(reading.describe(), flush=True)
            else:
                cv2.imshow(window, draw(frame, reading, sense))
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if mic is not None:
            mic.close()
        if not args.no_window:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

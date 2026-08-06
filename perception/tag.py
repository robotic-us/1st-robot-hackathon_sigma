#!/usr/bin/env python3
"""AprilTag tracking: a controlled, hand-held stand-in for the person.

Why a tag, when sense.py already reads faces?  Testability.  A face's
"distress" depends on lighting, the model and your acting skills; a tag in
your hand is ground truth you control completely:

    WHERE you hold it   -> x        -> which way the cradle leans
    HOW HARD you shake  -> motion   -> how big a motion it answers with

Same contract the circle prototype used, so everything downstream is already
built for it.  Detection is cv2.aruco with the AprilTag 36h11 family -- no new
dependencies, and one tag on a phone screen is enough.

Run::

    python3 perception/tag.py --make    # write tags/tag_0.png -- open it on a phone
    python3 perception/tag.py           # live viewer
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)


@dataclass(frozen=True)
class TagReading:
    """Where the tag is, WHICH tag it is, and how much it moves.  Unit-free.

    ``tag_id`` is the 36h11 marker id -- serve.py reads it as a state card
    (tag_0 calm, tag_1 fussing, tag_2 crying), so which tag you show matters
    and how you wave it does not.
    """

    present: bool
    x: float          # [-1, 1] left .. right
    y: float          # [-1, 1] top .. bottom
    distance: float   # [0, 1] apparent size; bigger = nearer
    motion: float     # [0, 1] EMA-smoothed shake intensity
    ts: float
    tag_id: int = -1  # -1 = no tag


class TagTracker:
    """Frames in, TagReading out.  Owns no camera (that keeps it testable)."""

    def __init__(
        self,
        motion_smoothing_s: float = 0.6,
        motion_full_scale: float = 2.0,   # normalised units/s that reads as 1.0
        near_frac: float = 0.45,          # tag height/frame height at distance=1
        hold_frames: int = 5,
    ) -> None:
        self.motion_smoothing_s = motion_smoothing_s
        self.motion_full_scale = motion_full_scale
        self.near_frac = near_frac
        self.hold_frames = hold_frames
        self._last_center: Optional[tuple[float, float]] = None
        self._last_ts: Optional[float] = None
        self._motion = 0.0
        self._miss = 0
        self._last_good: Optional[TagReading] = None
        self.corners: Optional[np.ndarray] = None   # for the overlay

    def update(self, frame: np.ndarray, ts: Optional[float] = None) -> TagReading:
        ts = time.monotonic() if ts is None else ts
        height, width = frame.shape[:2]
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, DICT)

        dt = (ts - self._last_ts) if self._last_ts is not None else 0.0
        self._last_ts = ts
        alpha = 1.0 - math.exp(-dt / self.motion_smoothing_s) if dt > 0 else 0.0

        if ids is None or len(ids) == 0:
            # Same dropout policy as the circle tracker: hold briefly, decay
            # motion (a gap tells us nothing about speed), then report absent.
            self._miss += 1
            self._last_center = None
            self._motion += alpha * (0.0 - self._motion)
            self.corners = None
            if self._last_good is not None and self._miss <= self.hold_frames:
                return self._record(TagReading(
                    True, self._last_good.x, self._last_good.y,
                    self._last_good.distance, self._motion, ts,
                    self._last_good.tag_id))
            last = self._last_good
            return TagReading(False, last.x if last else 0.0, last.y if last else 0.0,
                              0.0, self._motion, ts)

        self._miss = 0
        quad = corners[0][0]                    # (4,2) px, first tag wins
        tag_id = int(ids.flatten()[0])
        self.corners = quad
        cx, cy = float(quad[:, 0].mean()), float(quad[:, 1].mean())
        side = float(np.linalg.norm(quad[0] - quad[3]))   # tag height, px

        x = (2.0 * cx / width) - 1.0
        y = (2.0 * cy / height) - 1.0
        distance = min(1.0, (side / height) / self.near_frac)

        measured = 0.0
        if self._last_center is not None and dt > 1e-6:
            travelled = math.hypot(x - self._last_center[0], y - self._last_center[1])
            measured = min(1.0, (travelled / dt) / self.motion_full_scale)
        self._motion += alpha * (measured - self._motion)
        self._last_center = (x, y)

        return self._record(TagReading(True, max(-1, min(1, x)), max(-1, min(1, y)),
                                       distance, self._motion, ts, tag_id))

    def _record(self, reading: TagReading) -> TagReading:
        self._last_good = reading
        return reading


def draw_overlay(frame: np.ndarray, reading: TagReading, tracker: TagTracker) -> np.ndarray:
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    if tracker.corners is not None:
        cv2.polylines(canvas, [tracker.corners.astype(int)], True, (0, 255, 0), 2)
    if reading.present:
        px = int((reading.x + 1) * 0.5 * width)
        cv2.line(canvas, (px, 0), (px, height), (0, 200, 255), 1)
    bar = int((width - 20) * reading.motion)
    cv2.rectangle(canvas, (10, height - 26), (10 + width - 20, height - 10), (60, 60, 60), 1)
    if bar > 0:
        color = (0, int(255 * (1 - reading.motion)), int(255 * reading.motion))
        cv2.rectangle(canvas, (11, height - 25), (10 + bar, height - 11), color, -1)
    cv2.putText(canvas, f"tag {reading.tag_id if reading.present else '-'}  "
                        f"x {reading.x:+.2f}  motion {reading.motion:.2f}"
                        f"{'' if reading.present else '   NO TAG'}",
                (10, height - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return canvas


def render_tag(tag_id: int = 0, size: int = 200) -> np.ndarray:
    """A tag with the white quiet zone detection requires -- do not crop it."""
    marker = cv2.aruco.drawMarker(DICT, tag_id, size)
    return cv2.copyMakeBorder(marker, size // 4, size // 4, size // 4, size // 4,
                              cv2.BORDER_CONSTANT, value=255)


def synthetic_frame(width: int, height: int, x: float, y: float, side: int,
                    tag_id: int = 0) -> np.ndarray:
    """A tag at normalised (x, y) on a grey background -- selftest fuel."""
    frame = np.full((height, width), 110, np.uint8)
    tag = render_tag(tag_id, side)
    half = tag.shape[0] // 2
    cx = int((x + 1) * 0.5 * width)
    cy = int((y + 1) * 0.5 * height)
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(width, x0 + tag.shape[1]), min(height, y0 + tag.shape[0])
    frame[y0:y1, x0:x1] = tag[:y1 - y0, :x1 - x0]
    return frame

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--make", action="store_true", help="write tags/tag_0.png")
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args(argv)

    if args.make:
        out = Path("tags"); out.mkdir(exist_ok=True)
        for tag_id in range(3):
            cv2.imwrite(str(out / f"tag_{tag_id}.png"), render_tag(tag_id, 400))
        print(f"wrote tags/tag_0..2.png -- open one on a phone (keep the white "
              f"border) or print it")
        return 0

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"could not open camera {args.camera_index}")
        return 1
    tracker = TagTracker()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            reading = tracker.update(frame)
            cv2.imshow("tag", draw_overlay(frame, reading, tracker))
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

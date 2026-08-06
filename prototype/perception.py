#!/usr/bin/env python3
"""Perception: watch a tablet screen and describe the colored circle on it.

This is the "eyes" of the prototype.  A USB webcam (Logitech C270) is pointed at
a tablet that plays a video of a moving colored circle.  The circle is a
*controlled stand-in* for an infant: its position stands in for "where the baby
is", its movement intensity for "how much the baby is stirring".

The detector is deliberately boring -- HSV color masking plus the largest
contour -- because the whole point of the tablet stimulus is that detection is
100% reliable, so we can validate the *decision algorithm* instead of fighting a
flaky detector.

Design notes worth knowing before you edit this file:

* ``CircleTracker`` never touches a camera.  It takes frames and returns
  ``Perception``.  That is what makes ``--selftest`` (synthetic frames) and
  ``main.py --video`` (a recorded file) possible with no hardware at all.
* ``motion`` is the speed of the *circle centroid*, not a whole-frame diff.  The
  circle is the only thing moving on the tablet, so tracking its centroid is both
  cheaper and far less sensitive to screen flicker, glare and moire.
* The capture backend is chosen per platform: V4L2 on Linux (the Jetson target),
  whatever OpenCV picks elsewhere (so this runs on a Mac today).

Run standalone::

    python3 perception.py                       # live viewer on camera 0
    python3 perception.py --calibrate           # click the circle, get an HSV range
    python3 perception.py --video demo.mp4      # no camera needed
    python3 perception.py --selftest            # no camera, no window, asserts
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass, field, replace
from typing import Optional, Protocol

import cv2
import numpy as np

LOGGER = logging.getLogger("perception")

# OpenCV moved the fourcc helper around between releases; support both spellings.
_FOURCC = getattr(cv2, "VideoWriter_fourcc", None) or cv2.VideoWriter.fourcc


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` into ``[lo, hi]``."""
    return lo if value < lo else (hi if value > hi else value)


def clamp01(value: float) -> float:
    """Clamp ``value`` into ``[0, 1]``."""
    return clamp(value, 0.0, 1.0)


def ema_alpha(dt: float, smoothing_s: float, fixed_alpha: Optional[float] = None) -> float:
    """EMA weight for a step of ``dt`` seconds, given a time constant.

    Deriving alpha from elapsed time rather than hard-coding it per frame keeps
    the smoothing identical at 15, 30 or 60 FPS.  ``fixed_alpha`` overrides it
    for callers who really do want a per-frame constant.
    """
    if fixed_alpha is not None:
        return clamp01(fixed_alpha)
    if dt <= 0.0:
        return 0.0
    return 1.0 - math.exp(-dt / max(1e-3, smoothing_s))


# --------------------------------------------------------------------------- #
# Public data types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Perception:
    """One frame's worth of "what the camera sees", normalised and unit-free.

    Everything downstream (``decider.py``) consumes only this -- it never sees a
    pixel.  That keeps the decision logic testable without a camera.
    """

    present: bool     # was the circle found (or recently held)?
    x: float          # [-1, 1] left .. right, circle centre
    y: float          # [-1, 1] top .. bottom, circle centre
    distance: float   # [0, 1] size proxy from radius; bigger = "nearer"
    motion: float     # [0, 1] EMA-smoothed movement intensity
    ts: float         # time.monotonic() when the frame was processed


@dataclass(frozen=True)
class HSVRange:
    """An inclusive HSV window, in OpenCV's ranges (H 0-179, S/V 0-255).

    The default is a bright green circle on a dark background.  Bright-on-dark is
    the recommended stimulus: it survives screen glare and moire far better than
    a dark circle on white, because the mask keys on high V *and* high S, and
    glare is high-V but low-S.
    """

    lo: tuple[int, int, int] = (40, 120, 120)
    hi: tuple[int, int, int] = (85, 255, 255)

    @classmethod
    def parse(cls, text: str) -> "HSVRange":
        """Parse ``"lo_h,lo_s,lo_v,hi_h,hi_s,hi_v"`` (the ``--hsv-range`` form)."""
        parts = [p.strip() for p in text.split(",")]
        if len(parts) != 6:
            raise argparse.ArgumentTypeError(
                "--hsv-range needs 6 comma-separated ints: "
                "lo_h,lo_s,lo_v,hi_h,hi_s,hi_v"
            )
        try:
            values = [int(p) for p in parts]
        except ValueError as exc:  # pragma: no cover - argparse surfaces this
            raise argparse.ArgumentTypeError(f"--hsv-range: {exc}") from exc
        return cls(lo=tuple(values[:3]), hi=tuple(values[3:]))  # type: ignore[arg-type]

    def as_cli(self) -> str:
        """Render back into the ``--hsv-range`` form (used by --calibrate)."""
        return ",".join(str(v) for v in (*self.lo, *self.hi))

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        """Binary mask of pixels inside this window.

        Hue is circular, so a range like red (``lo_h=170, hi_h=10``) wraps past
        179.  We detect that case and OR two slices together.  The default green
        range does not wrap, but retuning to red must not silently produce an
        empty mask.
        """
        lo_h, lo_s, lo_v = self.lo
        hi_h, hi_s, hi_v = self.hi
        if lo_h <= hi_h:
            return cv2.inRange(
                hsv,
                np.array((lo_h, lo_s, lo_v), np.uint8),
                np.array((hi_h, hi_s, hi_v), np.uint8),
            )
        upper = cv2.inRange(
            hsv,
            np.array((lo_h, lo_s, lo_v), np.uint8),
            np.array((179, hi_s, hi_v), np.uint8),
        )
        lower = cv2.inRange(
            hsv,
            np.array((0, lo_s, lo_v), np.uint8),
            np.array((hi_h, hi_s, hi_v), np.uint8),
        )
        return cv2.bitwise_or(upper, lower)


@dataclass
class PerceptionConfig:
    """Everything tunable about seeing the circle.

    The four knobs that actually matter in the field are ``hsv``,
    ``motion_full_scale_px_s``, ``motion_alpha`` and ``min_radius_px``.
    """

    # --- capture ---------------------------------------------------------- #
    camera_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30

    # --- detection -------------------------------------------------------- #
    hsv: HSVRange = field(default_factory=HSVRange)
    blur_ksize: int = 5            # Gaussian blur, kills sensor noise + moire
    morph_ksize: int = 5           # open+close kernel, kills speckle
    min_radius_px: float = 6.0     # anything smaller is noise, not the circle
    min_fill_ratio: float = 0.45   # contour_area / enclosing_circle_area.
    #                                A real circle is ~1.0; glare streaks and
    #                                specular smears are long and thin and score
    #                                low, so this cheaply rejects them.

    # --- normalisation ---------------------------------------------------- #
    radius_far_px: float = 8.0     # radius that maps to distance = 0.0
    radius_near_px: float = 110.0  # radius that maps to distance = 1.0

    # --- motion ----------------------------------------------------------- #
    motion_smoothing_s: float = 0.8   # EMA time constant.  Must be comfortably
    #                                   longer than one cycle of the movement we
    #                                   are measuring: a circle oscillating at
    #                                   2 Hz has an instantaneous speed that goes
    #                                   from zero to peak four times a second, and
    #                                   what we want is the *envelope* of that --
    #                                   "how much is it stirring" -- not a value
    #                                   that depends on when we happened to look.
    motion_alpha: Optional[float] = None  # fixed per-frame alpha; overrides the
    #                                   time constant if set.  Frame-rate
    #                                   dependent, so prefer motion_smoothing_s.
    motion_full_scale: float = 2.0    # centroid speed that reads as motion=1.0,
    #                                   in NORMALISED units per second (x spans
    #                                   [-1, 1], so 2.0 = crossing the frame in
    #                                   one second).  Deliberately not pixels/s:
    #                                   a 1280x800 video file and a 640x480
    #                                   webcam must report the same motion for
    #                                   the same movement.

    # --- robustness ------------------------------------------------------- #
    hold_frames: int = 5           # keep reporting the last position for this
    #                                many consecutive misses before giving up


# --------------------------------------------------------------------------- #
# The tracker (no camera -- feed it frames)
# --------------------------------------------------------------------------- #
class CircleTracker:
    """Turns frames into :class:`Perception` values.

    Stateful across frames (motion EMA, dropout hold), but completely unaware of
    where the frames came from.
    """

    def __init__(self, config: Optional[PerceptionConfig] = None) -> None:
        self.cfg = config or PerceptionConfig()

        self._last_center_norm: Optional[tuple[float, float]] = None
        self._last_ts: Optional[float] = None
        self._motion: float = 0.0
        self._miss_streak: int = 0
        self._last_good: Optional[Perception] = None

        # Kept for the debug overlay only.
        self._last_circle_px: Optional[tuple[float, float, float]] = None
        self._frame_size: tuple[int, int] = (0, 0)
        self.fps: float = 0.0

    # -- public ------------------------------------------------------------ #
    def update(self, frame_bgr: np.ndarray, ts: Optional[float] = None) -> Perception:
        """Process one BGR frame and return the current :class:`Perception`."""
        ts = time.monotonic() if ts is None else ts
        height, width = frame_bgr.shape[:2]
        self._frame_size = (width, height)

        dt = (ts - self._last_ts) if self._last_ts is not None else 0.0
        self._last_ts = ts
        if dt > 1e-6:
            # Smoothed frame rate, for the overlay / the >=20 FPS target.
            self.fps += 0.1 * ((1.0 / dt) - self.fps)

        circle = self._detect(frame_bgr)
        if circle is not None:
            return self._on_hit(circle, width, height, dt, ts)
        return self._on_miss(dt, ts)

    @property
    def debug_info(self) -> dict[str, object]:
        """Extra detail for the overlay; deliberately not part of Perception."""
        return {
            "circle_px": self._last_circle_px,
            "miss_streak": self._miss_streak,
            "held": 0 < self._miss_streak <= self.cfg.hold_frames,
            "fps": self.fps,
            "frame_size": self._frame_size,
        }

    def reset(self) -> None:
        """Forget all history (used when a looping video file wraps around)."""
        self._last_center_norm = None
        self._last_ts = None
        self._motion = 0.0
        self._miss_streak = 0
        self._last_good = None
        self._last_circle_px = None

    # -- internals --------------------------------------------------------- #
    def _detect(self, frame_bgr: np.ndarray) -> Optional[tuple[float, float, float]]:
        """Find the circle. Returns ``(cx_px, cy_px, radius_px)`` or ``None``.

        Pipeline: blur -> HSV -> color mask -> morphological open+close ->
        largest contour -> minimum enclosing circle -> sanity checks.
        """
        cfg = self.cfg

        blurred = frame_bgr
        if cfg.blur_ksize >= 3:
            k = cfg.blur_ksize | 1  # OpenCV demands an odd kernel
            blurred = cv2.GaussianBlur(frame_bgr, (k, k), 0)

        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cfg.hsv.mask(hsv)

        if cfg.morph_ksize >= 3:
            k = cfg.morph_ksize | 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # drop speckle
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill pinholes

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(largest)
        if radius < cfg.min_radius_px:
            return None

        # Circularity check: a filled circle fills its enclosing circle; a glare
        # streak does not.
        enclosing_area = math.pi * radius * radius
        if enclosing_area > 0.0:
            fill = cv2.contourArea(largest) / enclosing_area
            if fill < cfg.min_fill_ratio:
                return None

        return float(cx), float(cy), float(radius)

    def _on_hit(
        self,
        circle: tuple[float, float, float],
        width: int,
        height: int,
        dt: float,
        ts: float,
    ) -> Perception:
        cfg = self.cfg
        cx, cy, radius = circle

        # --- normalise ------------------------------------------------------ #
        x = (2.0 * cx / width) - 1.0 if width else 0.0
        y = (2.0 * cy / height) - 1.0 if height else 0.0
        span = max(1e-6, cfg.radius_near_px - cfg.radius_far_px)
        distance = clamp01((radius - cfg.radius_far_px) / span)

        # --- motion: centroid speed, in normalised units per second --------- #
        # Frame-rate independent (we divide by dt) and resolution independent
        # (we measure in normalised units, not pixels).
        measured = 0.0
        if self._last_center_norm is not None and dt > 1e-6:
            travelled = math.hypot(x - self._last_center_norm[0], y - self._last_center_norm[1])
            measured = clamp01((travelled / dt) / cfg.motion_full_scale)
        alpha = ema_alpha(dt, cfg.motion_smoothing_s, cfg.motion_alpha)
        self._motion += alpha * (measured - self._motion)

        self._last_center_norm = (x, y)
        self._last_circle_px = circle
        self._miss_streak = 0

        perc = Perception(
            present=True,
            x=clamp(x, -1.0, 1.0),
            y=clamp(y, -1.0, 1.0),
            distance=distance,
            motion=self._motion,
            ts=ts,
        )
        self._last_good = perc
        return perc

    def _on_miss(self, dt: float, ts: float) -> Perception:
        """Detection failed this frame: hold briefly, then declare absence."""
        cfg = self.cfg
        self._miss_streak += 1

        # Velocity across a detection gap is meaningless (we do not know how far
        # the circle travelled while we could not see it), so drop the anchor and
        # feed the EMA a zero.  Motion therefore decays during a dropout rather
        # than spiking when detection resumes.
        self._last_center_norm = None
        self._motion += ema_alpha(dt, cfg.motion_smoothing_s, cfg.motion_alpha) * (0.0 - self._motion)

        if self._last_good is not None and self._miss_streak <= cfg.hold_frames:
            # Brief dropout: keep the last known position, present stays True.
            return replace(self._last_good, motion=self._motion, ts=ts)

        self._last_circle_px = None
        last = self._last_good
        return Perception(
            present=False,
            # Last known values are kept for the overlay's benefit; consumers
            # must ignore them when present is False.
            x=last.x if last else 0.0,
            y=last.y if last else 0.0,
            distance=last.distance if last else 0.0,
            motion=self._motion,
            ts=ts,
        )


# --------------------------------------------------------------------------- #
# Frame sources
# --------------------------------------------------------------------------- #
class FrameSource(Protocol):
    """Anything that can hand us BGR frames."""

    def read(self) -> tuple[bool, Optional[np.ndarray]]: ...
    def close(self) -> None: ...


def capture_backend() -> int:
    """V4L2 on Linux (the Jetson), OpenCV's default elsewhere.

    Hard-coding ``CAP_V4L2`` would refuse to open the webcam on a Mac, and the
    whole point of this prototype is that it runs on a laptop today.
    """
    return cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY


class CameraSource:
    """A live webcam, configured for low latency rather than image quality."""

    def __init__(self, config: PerceptionConfig) -> None:
        self.cfg = config
        self.cap = cv2.VideoCapture(config.camera_index, capture_backend())
        if not self.cap.isOpened():
            raise RuntimeError(
                f"could not open camera index {config.camera_index} "
                f"(backend={capture_backend()}). Try a different --camera-index."
            )

        # MJPG lets the C270 deliver 30 FPS at 640x480; raw YUYV cannot.
        self.cap.set(cv2.CAP_PROP_FOURCC, _FOURCC(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        self.cap.set(cv2.CAP_PROP_FPS, config.fps)
        # Keep only the newest frame: we want fresh data, not a backlog.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        LOGGER.info("camera %d opened at %.0fx%.0f", config.camera_index, actual_w, actual_h)

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        return self.cap.read()

    def close(self) -> None:
        self.cap.release()


class VideoFileSource:
    """A video file, optionally looping.

    Lets the whole pipeline run with no camera at all -- record a scenario with
    ``stimulus.py --record`` and replay it here for bit-identical test runs.
    """

    def __init__(self, path: str, loop: bool = True) -> None:
        self.path = path
        self.loop = loop
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open video file: {path}")
        self.wrapped = False  # set on each loop-around so callers can reset state
        # Callers must pace playback to this: decoding as fast as possible would
        # compress the timeline, and motion is a speed -- it would come out wrong.
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps: float = fps if fps and fps > 1.0 else 30.0

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        self.wrapped = False
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.wrapped = True
            ok, frame = self.cap.read()
        return ok, frame

    def close(self) -> None:
        self.cap.release()


def open_source(
    config: PerceptionConfig,
    video: Optional[str] = None,
    loop: bool = True,
) -> FrameSource:
    """Open a video file if given, otherwise the configured camera."""
    if video:
        return VideoFileSource(video, loop=loop)
    return CameraSource(config)


# --------------------------------------------------------------------------- #
# Debug overlay
# --------------------------------------------------------------------------- #
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_overlay(
    frame: np.ndarray,
    perc: Perception,
    tracker: Optional[CircleTracker] = None,
) -> np.ndarray:
    """Draw the detected circle, an x/y crosshair, a motion bar and the numbers.

    Returns a new image; the input frame is not modified.
    """
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    info = tracker.debug_info if tracker is not None else {}
    circle_px = info.get("circle_px")
    held = bool(info.get("held"))

    # --- the detection itself --------------------------------------------- #
    if circle_px is not None:
        cx, cy, radius = circle_px  # type: ignore[misc]
        cv2.circle(canvas, (int(cx), int(cy)), int(radius), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (int(cx), int(cy)), 3, (0, 0, 255), -1, cv2.LINE_AA)

    # --- x / y crosshair, drawn from the normalised values ----------------- #
    if perc.present:
        px = int((perc.x + 1.0) * 0.5 * width)
        py = int((perc.y + 1.0) * 0.5 * height)
        color = (0, 200, 255) if held else (0, 255, 0)
        cv2.line(canvas, (px, 0), (px, height), color, 1, cv2.LINE_AA)
        cv2.line(canvas, (0, py), (width, py), color, 1, cv2.LINE_AA)

    # --- motion bar (green -> red) ----------------------------------------- #
    bar_x, bar_w, bar_h = 10, width - 20, 14
    bar_y = height - bar_h - 10
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), 1)
    filled = int(bar_w * clamp01(perc.motion))
    bar_color = (0, int(255 * (1.0 - perc.motion)), int(255 * perc.motion))
    if filled > 0:
        cv2.rectangle(
            canvas, (bar_x + 1, bar_y + 1), (bar_x + filled, bar_y + bar_h - 1),
            bar_color, -1,
        )
    cv2.putText(
        canvas, f"motion {perc.motion:.2f}", (bar_x + 4, bar_y - 6),
        _FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
    )

    # --- numbers ----------------------------------------------------------- #
    if perc.present:
        status = "HELD" if held else "TRACKING"
    else:
        status = "NO TARGET"
    lines = [
        f"{status}   fps {float(info.get('fps', 0.0)):5.1f}",
        f"x {perc.x:+.3f}  y {perc.y:+.3f}",
        f"distance {perc.distance:.3f}",
        f"motion   {perc.motion:.3f}",
    ]
    box_h = 18 * len(lines) + 10
    cv2.rectangle(canvas, (5, 5), (215, 5 + box_h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(
            canvas, line, (12, 24 + 18 * i),
            _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return canvas


# --------------------------------------------------------------------------- #
# Synthetic frames (used by --selftest, and handy for debugging)
# --------------------------------------------------------------------------- #
def synthetic_frame(
    width: int,
    height: int,
    x: float,
    y: float,
    radius_px: int = 40,
    color: tuple[int, int, int] = (0, 255, 0),
    background: tuple[int, int, int] = (12, 12, 12),
) -> np.ndarray:
    """Render a circle at normalised ``(x, y)`` -- the same convention as Perception."""
    frame = np.full((height, width, 3), background, dtype=np.uint8)
    cx = int((x + 1.0) * 0.5 * width)
    cy = int((y + 1.0) * 0.5 * height)
    cv2.circle(frame, (cx, cy), radius_px, color, -1, cv2.LINE_AA)
    return frame


# --------------------------------------------------------------------------- #
# Standalone modes
# --------------------------------------------------------------------------- #
def run_selftest() -> int:
    """Push a known trajectory through the tracker and check it comes back out.

    No camera, no window -- a cheap regression guard you can run anywhere.
    """
    width, height, fps = 640, 480, 30.0
    cfg = PerceptionConfig(motion_alpha=1.0)  # no smoothing, so we test raw math
    tracker = CircleTracker(cfg)

    errors: list[float] = []
    for i in range(90):
        t = i / fps
        true_x = 0.6 * math.sin(2.0 * math.pi * 0.5 * t)  # 0.5 Hz sweep
        frame = synthetic_frame(width, height, true_x, 0.0)
        perc = tracker.update(frame, ts=t)
        if i > 2:  # let the tracker see one frame before judging it
            assert perc.present, f"lost a plainly visible circle at frame {i}"
            errors.append(abs(perc.x - true_x))

    worst = max(errors)
    mean = sum(errors) / len(errors)
    print(f"tracking error: mean {mean:.4f}  worst {worst:.4f}  (normalised units)")
    assert worst < 0.02, f"tracker drifted too far from ground truth: {worst:.4f}"

    # A still circle must read as (nearly) no motion.
    still = CircleTracker(PerceptionConfig())
    for i in range(30):
        still.update(synthetic_frame(width, height, 0.0, 0.0), ts=i / fps)
    print(f"still-circle motion: {still.update(synthetic_frame(width, height, 0.0, 0.0), ts=1.0).motion:.4f}")
    assert still._motion < 0.02, "a stationary circle should read as no motion"

    # Dropout hold: hide the circle and check present stays True for a few frames.
    holder = CircleTracker(PerceptionConfig(hold_frames=5))
    holder.update(synthetic_frame(width, height, 0.4, 0.0), ts=0.0)
    blank = np.full((height, width, 3), (12, 12, 12), dtype=np.uint8)
    for i in range(5):
        held = holder.update(blank, ts=0.1 + i * 0.033)
        assert held.present, f"hold failed on miss {i + 1}"
        assert abs(held.x - 0.4) < 0.02, "held position drifted"
    gone = holder.update(blank, ts=0.4)
    assert not gone.present, "should have given up after hold_frames misses"
    print("dropout hold: held 5 frames, then reported absent -- ok")

    print("\nselftest PASSED")
    return 0


def run_calibrate(source: FrameSource, config: PerceptionConfig) -> int:
    """Click the circle in the live feed; print a ready-to-paste --hsv-range."""
    window = "calibrate - click the circle, q to quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    state: dict[str, object] = {"hsv": None, "suggestion": None}

    def on_mouse(event: int, mx: int, my: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or state["hsv"] is None:
            return
        hsv = state["hsv"]
        h, w = hsv.shape[:2]  # type: ignore[union-attr]
        half = 3
        patch = hsv[  # type: ignore[index]
            max(0, my - half):min(h, my + half + 1),
            max(0, mx - half):min(w, mx + half + 1),
        ]
        if patch.size == 0:
            return
        median = np.median(patch.reshape(-1, 3), axis=0).astype(int)
        hue, sat, val = (int(v) for v in median)
        # Generous S/V floors, tight-ish hue: hue is the stable signal, while
        # brightness varies a lot across a tablet screen and with camera gain.
        suggestion = HSVRange(
            lo=((hue - 15) % 180, max(60, sat - 90), max(60, val - 90)),
            hi=((hue + 15) % 180, 255, 255),
        )
        state["suggestion"] = suggestion
        print(f"sampled HSV: ({hue}, {sat}, {val})")
        print(f"suggested:   --hsv-range {suggestion.as_cli()}")

    cv2.setMouseCallback(window, on_mouse)

    try:
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                LOGGER.warning("no frame from source")
                break
            hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
            state["hsv"] = hsv

            view = frame.copy()
            suggestion = state["suggestion"]
            if suggestion is not None:
                mask = suggestion.mask(hsv)  # type: ignore[union-attr]
                view[mask > 0] = (0, 0, 255)  # paint what the range would select
                cv2.putText(
                    view, f"--hsv-range {suggestion.as_cli()}",  # type: ignore[union-attr]
                    (10, view.shape[0] - 15), _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
                )
            cv2.putText(
                view, "click the circle", (10, 25),
                _FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.imshow(window, view)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        cv2.destroyAllWindows()

    final = state["suggestion"]
    if final is not None:
        print(f"\nuse:  --hsv-range {final.as_cli()}")  # type: ignore[union-attr]
        return 0
    print("\nnothing sampled (you never clicked the circle)")
    return 1


def run_viewer(source: FrameSource, config: PerceptionConfig, show: bool) -> int:
    """Live tracker with the debug overlay -- the standalone perception demo."""
    tracker = CircleTracker(config)
    window = "perception"
    if show:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = source.read()
            if not ok or frame is None:
                LOGGER.warning("frame source exhausted")
                break
            if getattr(source, "wrapped", False):
                tracker.reset()  # a looping file jumps in time; forget history

            perc = tracker.update(frame)
            if show:
                cv2.imshow(window, draw_overlay(frame, perc, tracker))
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
            else:
                print(
                    f"present={perc.present} x={perc.x:+.3f} y={perc.y:+.3f} "
                    f"distance={perc.distance:.3f} motion={perc.motion:.3f} "
                    f"fps={tracker.fps:.1f}"
                )
    except KeyboardInterrupt:
        pass
    finally:
        if show:
            cv2.destroyAllWindows()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--video", help="read frames from a file instead of a camera")
    parser.add_argument("--hsv-range", type=HSVRange.parse, default=None,
                        help="lo_h,lo_s,lo_v,hi_h,hi_s,hi_v (see --calibrate)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--no-window", action="store_true",
                        help="print values instead of showing a window")
    parser.add_argument("--calibrate", action="store_true",
                        help="click the circle to derive an --hsv-range")
    parser.add_argument("--selftest", action="store_true",
                        help="run the synthetic-frame checks and exit")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.selftest:
        return run_selftest()

    config = PerceptionConfig(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
    )
    if args.hsv_range is not None:
        config.hsv = args.hsv_range

    source = open_source(config, video=args.video)
    try:
        if args.calibrate:
            return run_calibrate(source, config)
        return run_viewer(source, config, show=not args.no_window)
    finally:
        source.close()


if __name__ == "__main__":
    raise SystemExit(main())

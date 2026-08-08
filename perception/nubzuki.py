#!/usr/bin/env python3
"""Reading Nubzuki's expression back out of a camera frame.

Vision front end for the mascot, never an infant: finds every Nubzuki figure,
names its pose (nearest reference crop, data/nubzuki_templates.npz), and places
it on the sheet's circumplex.  Closes the iPad loop: the state the machine
commanded should be the state the camera recovers.

    python3 perception/nubzuki.py docs/Nubzuki.jpg --grade   # grade vs the sheet
    python3 perception/nubzuki.py --camera 0 --server http://localhost:8080
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):                       # run me from the repo root
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# The five-state vocabulary
# --------------------------------------------------------------------------- #
# Kept from the removed infant-face stack (2026-08-08): the vocabulary the
# decision layer is built on.  Only who fills it in changed.
UNKNOWN = "UNKNOWN"                  # nothing recognisable -> STOP_AND_CHECK
AWAKE = "AWAKE"                      # eyes open              -> HOLD
EYES_CLOSED = "EYES_CLOSED"          # eyes not visible       -> HOLD_OR_TAPER
SLEEP_CANDIDATE = "SLEEP_CANDIDATE"  # shut and settled       -> TAPER_TO_STOP
DISTRESS_FACE = "DISTRESS_FACE"      # upset                  -> GENTLE_TEST_ONLY
STATES = (UNKNOWN, AWAKE, EYES_CLOSED, SLEEP_CANDIDATE, DISTRESS_FACE)

MOTION_HINTS = {
    UNKNOWN: "STOP_AND_CHECK",
    AWAKE: "HOLD",
    EYES_CLOSED: "HOLD_OR_TAPER",
    SLEEP_CANDIDATE: "TAPER_TO_STOP",
    DISTRESS_FACE: "GENTLE_TEST_ONLY",
}

# Each state as a 0..1 distress level.  DISTRESS_FACE is capped in the fuss
# band [0.30, 0.42] < CRY_LEVEL 0.45: vision alone never opens the cry ladder.
# UNKNOWN is not a level: it drops ``present`` to the safety gate.
_STATE_LEVEL = {
    UNKNOWN: 0.0, AWAKE: 0.05, EYES_CLOSED: 0.0,
    SLEEP_CANDIDATE: 0.0, DISTRESS_FACE: 0.30,
}


def distress_of(state: str, strength: float = 0.0) -> float:
    """One state -> the machine's 0..1 input.  Visual-only stays sub-cry."""
    level = _STATE_LEVEL[state]
    if state == DISTRESS_FACE:
        level = min(0.42, level + 0.12 * max(0.0, min(1.0, strength)))
    return level


# --------------------------------------------------------------------------- #
# The sheet's own frame of reference
# --------------------------------------------------------------------------- #
# docs/Nubzuki.jpg's printed circumplex, fitted to the ring itself (2.3 px
# rms).  Grader only -- the classifier never sees where a figure sits.
SHEET_CENTRE = (493.56, 525.31)
SHEET_RADIUS = 412.59

# x valence (POSITIVE right), y screen-down so ACTIVE is negative -- the frame
# web/baby.js blends in; tests.py::nubzuki asserts the two tables agree.
POSES: dict[str, tuple[float, float]] = {
    "rage":     (-.73, -.68),
    "angry":    (-.50, -.58),
    "shocked":  (-.05, -.78),
    "kiss":     (+.29, -.58),
    "dancing":  (+.65, -.78),
    "star":     (+.88, -.46),
    "bashful":  (+.55, -.33),
    "cool":     (-.43, -.08),
    "neutral":  (-.03, -.14),
    "crying":   (-.90, +.06),
    "nerdy":    (+.08, +.25),
    "lounging": (+.73, +.13),
    "sitHeart": (+.45, +.45),
    "gloomy":   (-.71, +.45),
    "sick":     (-.33, +.70),
    "sleeping": (+.20, +.83),
    "dreaming": (+.70, +.71),
}

LABELS: dict[str, str] = {
    "rage": "Raging", "angry": "Angry", "shocked": "Startled",
    "kiss": "Blowing a kiss", "dancing": "Dancing", "star": "Celebrating",
    "bashful": "Bashful", "cool": "Playing it cool", "neutral": "Neutral",
    "crying": "Crying", "nerdy": "Thinking", "lounging": "Lounging",
    "sitHeart": "Content", "gloomy": "Gloomy", "sick": "Poorly",
    "sleeping": "Sleeping", "dreaming": "Dreaming",
}

# web/baby.js::LIVE_LADDER inverted: a named pose recovers the level that drew
# it.  A pose off the ladder means a person is driving the wheel by hand.
LIVE_LEVEL: dict[str, float] = {
    "sitHeart": 0.0, "neutral": .22, "crying": .38, "angry": .78, "rage": 1.0,
}
SLEEP_POSES = ("sleeping", "dreaming")
DISTRESS_POSES = ("rage", "angry", "crying")

# Where each distress pose sits within the allowed 0.30..0.42 fuss band.
POSE_STRENGTH: dict[str, float] = {"crying": 0.0, "angry": .5, "rage": 1.0}


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
# The figure mask is "saturated colour, or ink" -- never "darker than white",
# which would keep the grey drop-shadows and bridge neighbouring figures.
INK_V = 90          # anything this dark is outline ink
COLOUR_S = 55       # ...and anything this saturated is the character's paint
COLOUR_V = 60
# How far above the background's own saturation the paint cut must sit.
BACKGROUND_MARGIN = 14
# 55 not 70: the angry pose's flush is a gradient that drops out at 70; the
# drop-shadows this rejects peak at S=4, so the margin below is wide.
MIN_FIGURE_PX = 6000
# Thresholds were written against renders (pure white page, saturated paint).
# A photographed panel is neither, so the frame is mapped back to those
# statistics -- white-balance on the page, then restore saturation -- rather
# than loosening every threshold until the room qualifies too.
NORMALISE_WHITE_PCT = 97.0      # the page is the brightest thing in frame
NORMALISE_WHITE = 245.0
NORMALISE_SAT_PCT = 92.0        # the paint is the most saturated thing on it
NORMALISE_SAT = 175.0
# Smallest blob worth calling a figure: a fraction of the frame, never a
# fixed pixel count.
MIN_FIGURE_FRAC = 0.004
# Smallest page, also a frame fraction -- a distant iPad is the normal case.
PAGE_MIN_FRAC = 0.004
# "Page" brightness anchors near the maximum, not the 90th percentile: with a
# distant panel most of the frame is room, and the furniture would qualify.
PAGE_BRIGHT_PCT = 99.5


def normalise(bgr: np.ndarray) -> np.ndarray:
    """White-balance the page, restore saturation -- percentile-anchored, so a
    reflection or a dark bezel moves neither."""
    out = bgr.astype(np.float32)
    for channel in range(3):
        level = np.percentile(out[:, :, channel], NORMALISE_WHITE_PCT)
        if level > 1.0:
            out[:, :, channel] *= NORMALISE_WHITE / level
    hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8),
                       cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[:, :, 1]
    lively = sat > 12                      # ignore the page itself
    if lively.any():
        level = float(np.percentile(sat[lively], NORMALISE_SAT_PCT))
        if level > 4.0:
            # Gain clamped >= 1.0: rescue lost saturation only, never cut it.
            gain = min(4.0, max(1.0, NORMALISE_SAT / level))
            hsv[:, :, 1] = np.clip(sat * gain, 0, 255)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def find_page(bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """The bright, near-neutral region the mascot is drawn on, or None.

    Order is page -> normalise -> mask: normalising the whole room lifts grey
    walls into the mask.  Found on the *raw* frame, before any gain.  None ->
    the caller treats the whole frame as the page (a render or the sheet).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    bright = float(np.percentile(val, PAGE_BRIGHT_PCT))
    page = ((val >= bright * .80) & (sat <= 70)).astype(np.uint8)
    page = cv2.morphologyEx(page, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(page, 8)
    if count <= 1:
        return None
    best = max(range(1, count), key=lambda i: stats[i][cv2.CC_STAT_AREA])
    if stats[best][cv2.CC_STAT_AREA] < PAGE_MIN_FRAC * bgr.shape[0] * bgr.shape[1]:
        return None
    return tuple(int(v) for v in stats[best][:4])


def figure_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary mask of Nubzuki-coloured pixels: paint or ink, never shadow.

    The saturation cut is COLOUR_S *or* clear of the measured background,
    whichever is higher: ``normalise`` amplifies the page's tint too, and a
    fixed cut let the page join the mask on camera frames.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    # 60th percentile: on any crop the mascot does not fill, this is background.
    floor = float(np.percentile(s, 60)) + BACKGROUND_MARGIN
    cut = max(COLOUR_S, floor)
    mask = (((s > cut) & (v > COLOUR_V)) | (v < INK_V)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def find_figures(bgr: np.ndarray,
                 min_area: int = MIN_FIGURE_PX) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of every figure, in reading order (top row, then left)."""
    mask = figure_mask(bgr)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = [tuple(int(t) for t in stats[i][:4])
             for i in range(1, count) if stats[i][cv2.CC_STAT_AREA] >= min_area]
    # Row band tolerates a few pixels of vertical offset within a row.
    return sorted(boxes, key=lambda b: (b[1] // 60, b[0]))


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
@dataclass
class Features:
    """Everything the rule ladder may look at.  Colour terms are fractions of
    the silhouette, the eye term / head_w**2 -- scale-invariant throughout."""
    box: tuple[int, int, int, int]
    aspect: float               # width / height of the silhouette
    head_w: int                 # width of the widest row -- the head ellipse
    wide_at: float              # where that row sits, 0 top .. 1 bottom
    eye_rel: float              # largest eye disc / head_w**2
    eyes: int                   # eye discs found (0, 1 or 2)
    blue: float = 0.0           # the character's own paint
    red: float = 0.0
    pink: float = 0.0
    pink_at: float = float("nan")   # pink prop's height in the box, 0 top
    pink_high: float = 0.0          # pink in the top quarter -- worn, not held
    purple: float = 0.0
    yellow: float = 0.0
    brown: float = 0.0
    black: float = 0.0

    flush: float = 0.0          # pink or purple *worn on the head*, not held

    @property
    def lying(self) -> bool:
        """Widest row >= 35% down: the cut that separates lying from standing
        on both the sheet's and the rig's (much less prone) drawings."""
        return self.wide_at >= .35


# Hue windows (cv2's 0..179 scale), named for what they pick out.
def _bands(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "blue":   (h >= 95) & (h <= 115) & (s > 90) & (v > 90),
        "red":    ((h <= 8) | (h >= 172)) & (s > 120) & (v > 80),
        "pink":   (h >= 140) & (h <= 175) & (s > 90) & (v > 90),
        "purple": (h >= 118) & (h <= 140) & (s > 60),
        "yellow": (h >= 20) & (h <= 35) & (s > 110) & (v > 140),
        "brown":  (h >= 8) & (h <= 22) & (s > 80) & (v < 190),
        "black":  v < 70,
    }


def _silhouette(mask: np.ndarray) -> np.ndarray:
    """Fill the figure in: the eyes and the KAIST wordmark are holes in it."""
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(closed)
    cv2.drawContours(filled, contours, -1, 1, -1)
    return filled


def _eye_pair(white: np.ndarray, head_w: int) -> tuple[int, int]:
    """Largest eye disc and how many of the pair we can see.

    Wants a *pair* -- similar size, level with each other -- not the two
    biggest white blobs (held props and the wordmark are white too).  The
    size floor is a fraction of the head, never a fixed pixel count.
    """
    floor = max(12.0, .0025 * head_w * head_w)
    count, _, stats, cent = cv2.connectedComponentsWithStats(white, 8)
    blobs = [(int(stats[i][cv2.CC_STAT_AREA]), float(cent[i][0]), float(cent[i][1]))
             for i in range(1, count) if stats[i][cv2.CC_STAT_AREA] >= floor]
    if not blobs:
        return 0, 0
    blobs.sort(key=lambda b: -b[0])
    for i, (ai, axi, ayi) in enumerate(blobs):
        for aj, axj, ayj in blobs[i + 1:]:
            level = abs(ayi - ayj) <= .45 * abs(axi - axj) + 4
            alike = aj >= .45 * ai
            if level and alike:
                return max(ai, aj), 2
    return blobs[0][0], 1


def features(bgr: np.ndarray, box: tuple[int, int, int, int],
             mask: np.ndarray | None = None) -> Features:
    """Measure one figure. ``box`` comes from :func:`find_figures`."""
    if mask is None:
        mask = figure_mask(bgr)
    x, y, w, h = box
    pad = 14
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(bgr.shape[1], x + w + pad), min(bgr.shape[0], y + h + pad)
    # Isolate *this* figure so a neighbour's props stay out of its vector.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask[y0:y1, x0:x1], 8)
    mine = max(range(1, count), key=lambda i: stats[i][cv2.CC_STAT_AREA])
    sil = _silhouette((labels == mine).astype(np.uint8))

    hsv = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hh, ss, vv = (hsv[:, :, i].astype(int) for i in range(3))
    inside = sil > 0
    total = float(inside.sum()) or 1.0

    rows = sil.sum(1)
    head_w = int(rows.max())
    wide_at = float(np.argmax(rows)) / max(1, sil.shape[0] - 1)

    eye_area, eyes = _eye_pair(((ss < 45) & (vv > 195) & inside).astype(np.uint8),
                               head_w)

    out = Features(
        box=box, aspect=w / h, head_w=head_w, wide_at=wide_at,
        eye_rel=eye_area / float(head_w * head_w or 1), eyes=eyes,
    )
    for name, band in _bands(hh, ss, vv).items():
        setattr(out, name, float((band & inside).sum()) / total)
    pink_mask = _bands(hh, ss, vv)["pink"] & inside
    pink_y, _ = np.nonzero(pink_mask)
    if len(pink_y):
        out.pink_at = float(pink_y.mean()) / max(1, sil.shape[0] - 1)
    crown = int(.25 * sil.shape[0])
    out.pink_high = float(pink_mask[:crown].sum()) / total
    # Flush = pink *or* purple worn on the head: the sheet flushes magenta,
    # the rig violet.
    bands = _bands(hh, ss, vv)
    face = np.zeros_like(inside)
    face[:int(.45 * sil.shape[0])] = True
    out.flush = float(((bands["pink"] | bands["purple"]) & inside & face).sum()) / total
    return out


# --------------------------------------------------------------------------- #
# Naming a pose: nearest reference crop
# --------------------------------------------------------------------------- #
# Naming is nearest-reference-crop, not colour rules: the separating marks
# are positions on a face that whole-figure fractions discard (the rule
# ladder's failure).  If a pose misreads, add or re-bake its reference crop.
TEMPLATE_FILE = Path(__file__).resolve().parent.parent / "data" / "nubzuki_templates.npz"
CROP_N = 48                # every figure is squared and scaled to this
# Measured over the rig's renderings under camera-like distortion: refuse
# past 32, and refuse when the runner-up is within 8% -- abstain, don't guess.
MAX_DISTANCE = 32.0        # beyond this, nothing is recognised at all
MIN_MARGIN = 1.08          # winner must beat the runner-up by this ratio


def square_crop(bgr: np.ndarray, box: tuple[int, int, int, int],
                mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One figure, isolated on a square canvas at CROP_N.

    Squared by padding, not stretching -- aspect is part of what is compared.
    """
    x, y, w, h = box
    pad = 6
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(bgr.shape[1], x + w + pad), min(bgr.shape[0], y + h + pad)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask[y0:y1, x0:x1], 8)
    mine = max(range(1, count), key=lambda i: stats[i][cv2.CC_STAT_AREA])
    sil = _silhouette((labels == mine).astype(np.uint8))
    sub = bgr[y0:y1, x0:x1]
    hh, ww = sil.shape
    side = max(hh, ww)
    canvas = np.zeros((side, side, 3), np.uint8)
    alpha = np.zeros((side, side), np.uint8)
    oy, ox = (side - hh) // 2, (side - ww) // 2
    canvas[oy:oy + hh, ox:ox + ww] = sub
    alpha[oy:oy + hh, ox:ox + ww] = sil * 255
    return (cv2.resize(canvas, (CROP_N, CROP_N), interpolation=cv2.INTER_AREA),
            cv2.resize(alpha, (CROP_N, CROP_N), interpolation=cv2.INTER_AREA))


def descriptor(bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """A crop as a vector that survives being photographed off a screen.

    Hue enters as a unit vector scaled by saturation (hue wraps at 180),
    brightness standardised over the figure, silhouette as its own channel.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32) * (2 * np.pi / 180.0)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0
    a = alpha.astype(np.float32) / 255.0
    inside = a > .5
    if inside.any():
        val = (val - val[inside].mean()) / (val[inside].std() + 1e-6)
    return np.stack([np.cos(hue) * sat * a, np.sin(hue) * sat * a,
                     val * a * .5, a], -1).reshape(-1)


_TEMPLATES: tuple[list[str], np.ndarray] | None = None


def templates() -> tuple[list[str], np.ndarray]:
    """Labels and descriptors, loaded once.  Baked by tools/make_nubzuki_templates.py."""
    global _TEMPLATES
    if _TEMPLATES is None:
        if not TEMPLATE_FILE.exists():
            raise FileNotFoundError(
                f"{TEMPLATE_FILE} is missing -- run "
                f"python3 tools/make_nubzuki_templates.py")
        data = np.load(TEMPLATE_FILE)
        labels = [str(k) for k in data["labels"]]
        vectors = np.stack([descriptor(c, a)
                            for c, a in zip(data["crops"], data["alphas"])])
        _TEMPLATES = (labels, vectors)
    return _TEMPLATES


def classify(bgr: np.ndarray, box: tuple[int, int, int, int],
             mask: np.ndarray) -> tuple[str | None, str]:
    """Name the pose by its nearest reference.  Returns ``(pose_key, why)``.

    ``None`` when nothing is close enough or the best two are too near to
    choose between -- abstain rather than guess.
    """
    labels, vectors = templates()
    d = np.linalg.norm(vectors - descriptor(*square_crop(bgr, box, mask)), axis=1)
    order = np.argsort(d)
    best = labels[order[0]]
    runner = next((labels[i] for i in order[1:] if labels[i] != best), None)
    runner_d = next((d[i] for i in order[1:] if labels[i] != best), float("inf"))
    if d[order[0]] > MAX_DISTANCE:
        return None, f"nothing within reach (nearest {best} at {d[order[0]]:.2f})"
    if runner_d < d[order[0]] * MIN_MARGIN:
        return None, f"too close to call: {best} {d[order[0]]:.2f} vs {runner} {runner_d:.2f}"
    return best, (f"nearest the {best} reference ({d[order[0]]:.2f}; "
                  f"next {runner} at {runner_d:.2f})")


# --------------------------------------------------------------------------- #
# Putting it together
# --------------------------------------------------------------------------- #
@dataclass
class Sighting:
    """One figure, named."""
    pose: str
    why: str
    box: tuple[int, int, int, int]
    valence: float              # the pose's canonical anchor, from POSES
    arousal: float              # negative is ACTIVE, matching the sheet
    feat: Features = field(repr=False)

    @property
    def label(self) -> str:
        return LABELS[self.pose]

    @property
    def recovered_level(self) -> float | None:
        """The distress number that would have drawn this, or None.

        An *echo* of what the machine sent -- right for checking the loop,
        wrong as an input to the machine (see :meth:`asserted_level`).  None
        means the pose is off the live ladder: a person took the wheel.
        """
        return LIVE_LEVEL.get(self.pose)

    @property
    def state(self) -> str:
        """This sighting in the cradle's five-state vocabulary."""
        return five_state(self)

    @property
    def asserted_level(self) -> float:
        """What this reading is *allowed* to claim: under 0.42 by construction.

        The gap vs :attr:`recovered_level` is the point -- §5 says vision
        alone cannot cross CRY_LEVEL, however high the echoed level is.
        """
        return distress_of(self.state, POSE_STRENGTH.get(self.pose, 0.0))

    @property
    def asleep(self) -> bool:
        return self.pose in SLEEP_POSES


def is_nubzuki(f: Features) -> bool:
    """Is this blob the character at all, or just something else on screen?

    The test is the paint: mostly its own blue, or one of the recoloured
    heads.  Keeps dark UI (buttons, text) from being named as figures.
    """
    return (f.blue >= .15 or f.red >= .15 or f.purple >= .10 or f.yellow >= .05)


# --------------------------------------------------------------------------- #
# Framing the camera: which pixels get read at all
# --------------------------------------------------------------------------- #
# One home for the framing rules: both readers must frame identically, so
# serve.py imports these names rather than copying them.
def parse_crop(spec: str) -> tuple[float, float, float, float]:
    """``"x,y,w,h"`` -> a rectangle: fractions when all <= 1, else pixels
    (resolved against the real frame in :func:`crop_frame`)."""
    parts = [p.strip() for p in spec.replace(" ", ",").split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError(f"crop wants x,y,w,h -- got {spec!r}")
    try:
        x, y, w, h = (float(p) for p in parts)
    except ValueError:
        raise ValueError(f"crop wants four numbers -- got {spec!r}") from None
    if w <= 0 or h <= 0:
        raise ValueError(f"crop width and height must be positive: {spec!r}")
    return (x, y, w, h)


def crop_frame(bgr: np.ndarray,
               crop: tuple[float, float, float, float] | None) -> np.ndarray:
    """The frame cut to the region of interest: settles the page competition
    (any lit region bigger than the iPad would otherwise win find_page)."""
    if crop is None:
        return bgr
    fh, fw = bgr.shape[:2]
    x, y, w, h = crop
    if max(crop) <= 1.0:                    # fractions of this frame
        x, y, w, h = x * fw, y * fh, w * fw, h * fh
    x0 = max(0, min(fw - 1, int(round(x))))
    y0 = max(0, min(fh - 1, int(round(y))))
    x1 = max(x0 + 1, min(fw, int(round(x + w))))
    y1 = max(y0 + 1, min(fh, int(round(y + h))))
    return bgr[y0:y1, x0:x1]


def centre_region(zoom: float) -> tuple[float, float, float, float]:
    """The middle ``1/zoom`` of a frame, as fractions -- what a bare zoom reads."""
    side = 1.0 / max(1.0, zoom)
    return ((1.0 - side) / 2, (1.0 - side) / 2, side, side)


def magnify(bgr: np.ndarray, zoom: float) -> np.ndarray:
    """Digital zoom: creates no detail, but the fixed-pixel steps before the
    classifier starve a small figure -- a distant-panel rescue, not free."""
    if zoom is None or zoom <= 1.0:
        return bgr
    return cv2.resize(bgr, None, fx=zoom, fy=zoom,
                      interpolation=cv2.INTER_CUBIC)


def read(bgr: np.ndarray, min_area: int | None = None,
         balance: bool = True) -> list[Sighting]:
    """Find and name every Nubzuki in the frame.

    ``balance`` runs :func:`normalise` first; leave it on for anything that
    came through a lens.  ``min_area`` defaults to MIN_FIGURE_FRAC of the frame.
    """
    ox = oy = 0
    if balance:
        page = find_page(bgr)
        if page is not None:
            x, y, w, h = page
            pad = int(.02 * max(w, h))
            ox, oy = max(0, x - pad), max(0, y - pad)
            bgr = bgr[oy:min(bgr.shape[0], y + h + pad),
                      ox:min(bgr.shape[1], x + w + pad)]
        bgr = normalise(bgr)
    if min_area is None:
        min_area = max(400, int(MIN_FIGURE_FRAC * bgr.shape[0] * bgr.shape[1]))
    mask = figure_mask(bgr)
    out = []
    for box in find_figures(bgr, min_area):
        feat = features(bgr, box, mask)
        if not is_nubzuki(feat):
            continue
        pose, why = classify(bgr, box, mask)
        if pose is None:               # seen, but not confidently named
            continue
        v, a = POSES[pose]
        # Boxes go back into the caller's frame; the page crop is internal.
        out.append(Sighting(pose, why,
                            (box[0] + ox, box[1] + oy, box[2], box[3]),
                            v, a, feat))
    return out


def five_state(sighting: "Sighting | None") -> str:
    """A sighting in the five-state vocabulary, decided on visible evidence.

    UNKNOWN (nothing recognisable) drops ``present`` to the safety gate;
    EYES_CLOSED means no open eye pair while upright -- deliberately not a
    claim about sleep; DISTRESS_FACE stays capped in the fuss band.
    """
    if sighting is None:
        return UNKNOWN
    if sighting.pose in SLEEP_POSES:
        return SLEEP_CANDIDATE
    if sighting.pose in DISTRESS_POSES:
        return DISTRESS_FACE
    if sighting.feat.eyes < 2:
        return EYES_CLOSED
    return AWAKE


# --------------------------------------------------------------------------- #
# The wheel
# --------------------------------------------------------------------------- #
# Drawn in the sheet's own frame: +x POSITIVE, y screen-down, ACTIVE on top.
STATE_COLOUR: dict[str, tuple[int, int, int]] = {     # BGR
    AWAKE: (120, 200, 90),
    EYES_CLOSED: (200, 170, 80),
    SLEEP_CANDIDATE: (200, 130, 120),
    DISTRESS_FACE: (90, 90, 235),
    UNKNOWN: (140, 140, 140),
}


def wheel_image(seen: list["Sighting"], size: int = 300,
                expected: str | None = None,
                dark: bool = True) -> np.ndarray:
    """The circumplex with every sighting plotted against the 17 anchors."""
    ink = (235, 235, 235) if dark else (40, 40, 40)
    faint = (90, 90, 90) if dark else (190, 190, 190)
    img = np.full((size, size, 3),
                  (28, 26, 24) if dark else (252, 250, 248), np.uint8)
    c = size // 2
    r = int(size * .38)

    def at(v: float, a: float) -> tuple[int, int]:
        return int(c + v * r), int(c + a * r)

    cv2.circle(img, (c, c), r, faint, 1, cv2.LINE_AA)
    cv2.line(img, (c, c - r), (c, c + r), faint, 1, cv2.LINE_AA)
    cv2.line(img, (c - r, c), (c + r, c), faint, 1, cv2.LINE_AA)
    font, fs = cv2.FONT_HERSHEY_SIMPLEX, size / 900
    for text, org in (("ACTIVE", (c - int(size * .06), c - r - 6)),
                      ("CALM", (c - int(size * .05), c + r + 16)),
                      ("NEG", (c - r - int(size * .11), c + 4)),
                      ("POS", (c + r + 4, c + 4))):
        cv2.putText(img, text, org, font, fs, faint, 1, cv2.LINE_AA)
    for key, (v, a) in POSES.items():
        cv2.circle(img, at(v, a), 2, faint, -1, cv2.LINE_AA)
    if expected is not None and expected in POSES:
        cv2.circle(img, at(*POSES[expected]), int(size * .038),
                   (120, 210, 255), 1, cv2.LINE_AA)
    for s in seen:
        p = at(s.valence, s.arousal)
        cv2.circle(img, p, int(size * .022), STATE_COLOUR[s.state], -1, cv2.LINE_AA)
        cv2.circle(img, p, int(size * .022), ink, 1, cv2.LINE_AA)
    if len(seen) == 1:
        cv2.putText(img, seen[0].label, (8, size - 10), font,
                    fs * 1.5, ink, 1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------- #
# Live: the loop this exists to close
# --------------------------------------------------------------------------- #
def nearest_rung(level: float, asleep: bool = False) -> str:
    """The pose a level should have drawn.  web/baby.js blends the two rungs
    either side, so recovery is only ever to the nearest rung."""
    if asleep:
        return "sleeping"
    return min(LIVE_LEVEL, key=lambda k: abs(LIVE_LEVEL[k] - level))


class Tracker:
    """Modal pose over a short window, so the live readout does not flicker.

    A frame is a vote, not an answer: at a blend midpoint consecutive frames
    genuinely round either way.
    """

    def __init__(self, window: float = 1.0) -> None:
        self.window = window
        self._votes: list[tuple[float, str]] = []

    def update(self, now: float, pose: str | None) -> str | None:
        self._votes.append((now, pose))
        self._votes = [(t, p) for t, p in self._votes if t >= now - self.window]
        seen = [p for _, p in self._votes if p is not None]
        if not seen:
            return None
        return max(set(seen), key=seen.count)


class ServerLink:
    """Reads serve.py's /events in the background: what the machine *sent*.

    Best-effort: a dropped connection dims the comparison line and leaves the
    recogniser running.
    """

    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")
        self.level: float | None = None
        self.emotion = ""
        self.connected = False

    def start(self) -> "ServerLink":
        import threading
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def _run(self) -> None:
        import json
        import time as _time
        from urllib.request import urlopen
        while True:
            try:
                with urlopen(self.base + "/events", timeout=10) as stream:
                    self.connected = True
                    for raw in stream:
                        if not raw.startswith(b"data: "):
                            continue
                        tag = json.loads(raw[6:]).get("tag", {})
                        self.level = float(tag.get("level", 0.0))
                        self.emotion = str(tag.get("emotion", "") or "")
            except Exception:
                self.connected = False
                self.level = None
                _time.sleep(1.0)


def _panel(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    """Draw the readout, dark on a translucent slab so it survives any wallpaper."""
    pad, lh = 10, 24
    box = frame[0:pad * 2 + lh * len(lines), 0:430]
    frame[0:box.shape[0], 0:430] = cv2.addWeighted(
        box, .35, np.zeros_like(box), .65, 0)
    for i, (text, colour) in enumerate(lines):
        cv2.putText(frame, text, (pad, pad + lh * i + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, colour, 1, cv2.LINE_AA)


def _live(args) -> int:
    """Camera -> recogniser -> (optionally) compare against what was sent."""
    import json
    import time

    if args.video:
        capture = cv2.VideoCapture(str(args.video))
    else:
        capture = cv2.VideoCapture(args.camera)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not capture.isOpened():
        print(f"ERROR: could not open {'video' if args.video else 'camera'}",
              file=sys.stderr)
        return 3

    link = ServerLink(args.server).start() if args.server else None
    tracker = Tracker(args.hold)
    dumped = 0
    if args.dump:
        args.dump.mkdir(parents=True, exist_ok=True)
    json_file = open(args.jsonl, "a", encoding="utf-8") if args.jsonl else None
    agree = total = frames = 0
    started = time.monotonic()
    print("reading the iPad" + (f", comparing against {args.server}" if link else "")
          + ".  q or esc to stop.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                if args.video:
                    break
                time.sleep(.05)
                continue
            if args.mirror:
                frame = cv2.flip(frame, 1)
            # Framed before anything reads it -- overlay, votes and --dump see
            # the same pixels.  Same flags and code as serve.py --sense.
            frame = magnify(crop_frame(frame, args.crop), args.zoom)
            frames += 1
            now = time.monotonic()
            seen = read(frame, args.min_area)
            # The iPad shows one face; the biggest figure wins.
            best = max(seen, key=lambda s: s.box[2] * s.box[3], default=None)
            held = tracker.update(now, best.pose if best else None)

            lines = []
            if held:
                rec = LIVE_LEVEL.get(held)
                state = five_state(best) if best and best.pose == held else UNKNOWN
                lines.append((f"camera sees   {LABELS[held]}"
                              + (f"   level {rec:.2f}" if rec is not None else
                                 "   (off the live ladder)"), (120, 255, 120)))
                if best and best.pose == held:
                    lines.append((f"              {best.why}", (200, 200, 200)))
                    lines.append((f"five-state    {state}"
                                  f"   may assert {best.asserted_level:.2f}",
                                  STATE_COLOUR[state]))
            else:
                lines.append(("camera sees   no Nubzuki in frame", (120, 200, 255)))
                lines.append((f"five-state    {UNKNOWN}   gate stops and checks",
                              STATE_COLOUR[UNKNOWN]))

            want = None
            if link is not None:
                if link.connected and link.level is not None:
                    want = nearest_rung(link.level, link.emotion == "SLEEP")
                    lines.append((f"jetson sent   level {link.level:.2f}"
                                  f"  -> {LABELS[want]}", (255, 210, 120)))
                    if held is not None:
                        total += 1
                        agree += (held == want)
                        pct = 100 * agree / max(1, total)
                        good = held == want
                        lines.append((
                            f"loop          {'MATCH' if good else 'MISMATCH'}"
                            f"   {pct:.0f}% of {total}",
                            (120, 255, 120) if good else (110, 110, 255)))
                else:
                    lines.append((f"jetson        not connected ({args.server})",
                                  (110, 110, 255)))

            # Saved with the classifier's verdict, so a retune can be measured
            # against the camera it will actually face.
            if args.dump and frames % max(1, args.every) == 0:
                stem = args.dump / f"{args.label}_{dumped:04d}"
                cv2.imwrite(str(stem) + ".png", frame)
                stem.with_suffix(".json").write_text(json.dumps({
                    "label": args.label, "got": best.pose if best else None,
                    "correct": bool(best and best.pose == args.label),
                    "features": ({k: (round(v, 5) if isinstance(v, float) else v)
                                  for k, v in vars(best.feat).items()
                                  if k != "box"} if best else None),
                }, indent=1))
                dumped += 1

            if json_file is not None:
                json_file.write(json.dumps({
                    "t": round(now - started, 2), "pose": held,
                    "state": five_state(best) if best else UNKNOWN,
                    "recovered": LIVE_LEVEL.get(held) if held else None,
                    "asserted": best.asserted_level if best else 0.0,
                    "commanded": link.level if link else None,
                    "expected": want, "match": (held == want) if want else None,
                }) + "\n")
                json_file.flush()

            if args.debug:
                # Every stage separately: "it sees nothing" is three bugs.
                page = find_page(frame)
                work = frame if page is None else frame[
                    max(0, page[1]):page[1] + page[3],
                    max(0, page[0]):page[0] + page[2]]
                work = normalise(work)
                dm = figure_mask(work)
                floor = max(400, int(MIN_FIGURE_FRAC * work.shape[0] * work.shape[1]))
                blobs = find_figures(work, floor)
                kept = [b for b in blobs if is_nubzuki(features(work, b, dm))]
                lines.append((f"debug         page {'yes' if page else 'NO'}"
                              f"  blobs {len(blobs)}  passed gate {len(kept)}"
                              f"  floor {floor}", (255, 255, 140)))
                # Per-blob verdicts: "passed the gate but named nothing" needs
                # its own line.
                for b in sorted(kept, key=lambda b: -b[2] * b[3])[:3]:
                    names, vectors = templates()
                    d = np.linalg.norm(
                        vectors - descriptor(*square_crop(work, b, dm)), axis=1)
                    order = np.argsort(d)
                    top = names[order[0]]
                    second = next((names[i] for i in order[1:] if names[i] != top), "-")
                    sd = next((d[i] for i in order[1:] if names[i] != top), float("inf"))
                    verdict = ("ok" if d[order[0]] <= MAX_DISTANCE
                               and sd >= d[order[0]] * MIN_MARGIN
                               else "too far" if d[order[0]] > MAX_DISTANCE
                               else "too close to call")
                    lines.append((f"  {b[2]:3d}x{b[3]:<3d} {top:<9}{d[order[0]]:6.1f}"
                                  f"  vs {second:<9}{sd:6.1f}   {verdict}"
                                  f"  (max {MAX_DISTANCE:.0f})", (255, 220, 120)))
                panel = cv2.cvtColor(dm * 255, cv2.COLOR_GRAY2BGR)
                side = min(frame.shape[1] // 3, 320)
                tall = max(1, int(side * panel.shape[0] / panel.shape[1]))
                frame[:min(tall, frame.shape[0]), :side] = cv2.resize(
                    panel, (side, tall))[:min(tall, frame.shape[0])]
                if page is not None:
                    cv2.rectangle(frame, (page[0], page[1]),
                                  (page[0] + page[2], page[1] + page[3]),
                                  (255, 200, 60), 2)

            if not args.no_display:
                shown = annotate(frame, seen)
                # The wheel shows where the call sits among the alternatives.
                side = min(300, shown.shape[0] // 2, shown.shape[1] // 2)
                wheel = wheel_image(seen[:1] if best is None else [best], side,
                                    expected=want)
                shown[shown.shape[0] - side:, shown.shape[1] - side:] = wheel
                _panel(shown, lines)
                cv2.imshow("Nubzuki loopback", shown)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            elif frames % 30 == 0:
                print(" | ".join(t for t, _ in lines), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if json_file is not None:
            json_file.close()
        if not args.no_display:
            cv2.destroyAllWindows()
    took = time.monotonic() - started
    print(f"\n  {frames} frames in {took:.0f} s ({frames / max(took, 1e-9):.1f} fps)")
    if total:
        print(f"  loop closed on {agree}/{total} readings "
              f"({100 * agree / total:.0f}%)")
    if dumped:
        print(f"  wrote {dumped} labelled frames to {args.dump}/ "
              f"as '{args.label}'")
    return 0


def sheet_position(box: tuple[int, int, int, int]) -> tuple[float, float]:
    """Where a box sits on docs/Nubzuki.jpg's own circumplex.

    Grading only: this is the ground truth, so no rule may consult it.
    """
    x, y, w, h = box
    return ((x + w / 2 - SHEET_CENTRE[0]) / SHEET_RADIUS,
            (y + h / 2 - SHEET_CENTRE[1]) / SHEET_RADIUS)


def overlay_scale(bgr: np.ndarray) -> float:
    """How big to draw on this frame, relative to the 640-wide it was tuned
    for.  A crop makes the read frame small; clamped so 4K stays sane too."""
    return max(0.35, min(1.6, bgr.shape[1] / 640.0))


def annotate(bgr: np.ndarray, seen: list[Sighting]) -> np.ndarray:
    """Draw the boxes and names onto a copy of the frame, sized to it."""
    out = bgr.copy()
    k = overlay_scale(bgr)
    box_w = max(1, round(2 * k))
    for s in seen:
        x, y, w, h = s.box
        cv2.rectangle(out, (x, y), (x + w, y + h), (60, 200, 60), box_w)
        origin = (x, max(round(14 * k), y - round(6 * k)))
        cv2.putText(out, s.label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    .45 * k, (20, 20, 20), max(2, round(3 * k)), cv2.LINE_AA)
        cv2.putText(out, s.label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    .45 * k, (60, 200, 60), max(1, round(1 * k)), cv2.LINE_AA)
    return out


def _still(args) -> int:
    """Name every figure in one image, and optionally grade it on the sheet."""
    bgr = cv2.imread(args.image)
    if bgr is None:
        print(f"cannot read {args.image}", file=sys.stderr)
        return 2
    seen = read(bgr, args.min_area)
    print(f"{args.image}: {len(seen)} figure(s)")
    worst = 0.0
    for s in seen:
        print(f"  {s.label:<16} {s.why}")
        if args.grade:
            sv, sa = sheet_position(s.box)
            err = float(np.hypot(sv - s.valence, sa - s.arousal))
            worst = max(worst, err)
            print(f"    sheet ({sv:+.2f},{sa:+.2f}) "
                  f"anchor ({s.valence:+.2f},{s.arousal:+.2f}) err {err:.2f}")
    if args.grade:
        named = {s.pose for s in seen}
        missing = sorted(set(POSES) - named)
        print(f"\n  {len(named)}/{len(POSES)} poses named"
              + (f", missing {missing}" if missing else "")
              + f", worst placement error {worst:.2f}")
    if args.annotate:
        cv2.imwrite(args.annotate, annotate(bgr, seen))
        print(f"  wrote {args.annotate}")
    return 0


def _main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Read Nubzuki's pose out of an image or a live camera.")
    p.add_argument("image", nargs="?", default="docs/Nubzuki.jpg",
                   help="still to read (default: the reference sheet)")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, metavar="N",
                     help="read a live camera instead of a still")
    src.add_argument("--video", type=Path, help="replay a recorded clip")
    p.add_argument("--server", metavar="URL",
                   help="serve.py base URL; compares what the camera recovers "
                        "against the level the machine sent")
    p.add_argument("--grade", action="store_true",
                   help="score the still against the reference sheet's own chart")
    p.add_argument("--annotate", metavar="OUT", help="write a labelled copy")
    p.add_argument("--min-area", type=int, default=None, dest="min_area",
                   help="smallest blob treated as a figure; default scales with "
                        "the frame (MIN_FIGURE_FRAC)")
    p.add_argument("--hold", type=float, default=1.0,
                   help="seconds of frames the live readout votes over")
    # Same two flags and code as serve.py --sense; aim_camera.py prints both.
    p.add_argument("--crop", metavar="X,Y,W,H",
                   help="read only this region of each frame (fractions when "
                        "all <= 1, else pixels).  Live modes only")
    p.add_argument("--zoom", type=float, default=1.0, metavar="N",
                   help="magnify what is read by N; with no --crop it reads "
                        "the middle 1/N, so the cost stays flat")
    # 1280x720: the mascot is a small object inside a screen inside the frame,
    # so pixels on the figure are the cheapest accuracy available.
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--mirror", action="store_true")
    p.add_argument("--jsonl", type=Path, help="JSON Lines log of the live loop")
    p.add_argument("--dump", type=Path, metavar="DIR",
                   help="save frames and their feature vectors here -- the "
                        "bench corpus for retuning against a real camera")
    p.add_argument("--label", default="unknown",
                   help="ground-truth pose for --dump (open the iPad at "
                        "/baby?pose=<label> and they agree by construction)")
    p.add_argument("--every", type=int, default=10,
                   help="with --dump, save one frame in this many")
    p.add_argument("--debug", action="store_true",
                   help="show the page, the mask and the rejected blobs -- what "
                        "to look at when it sees nothing")
    p.add_argument("--no-display", action="store_true")
    args = p.parse_args(argv)
    if args.crop:
        try:
            args.crop = parse_crop(args.crop)
        except ValueError as exc:
            p.error(str(exc))
    else:
        args.crop = None
    if args.zoom < 1.0:
        p.error(f"--zoom magnifies, so it starts at 1 -- got {args.zoom:g}")
    if args.zoom > 1.0 and args.crop is None:
        args.crop = centre_region(args.zoom)
    if args.camera is not None or args.video:
        return _live(args)
    if args.server:
        p.error("--server needs a live source: pass --camera N or --video FILE")
    return _still(args)


if __name__ == "__main__":
    raise SystemExit(_main())

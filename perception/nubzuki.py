#!/usr/bin/env python3
"""Reading Nubzuki's expression back out of a camera frame.

This is a vision front end for the mascot, not for an infant.  Given an image
it finds every Nubzuki figure in it, names the pose each one is holding, and
reports where that pose sits on the reference sheet's own circumplex --
ACTIVE/CALM against NEGATIVE/POSITIVE, the axes drawn on docs/Nubzuki.jpg.

    camera frame ──► find_figures() ──► is_nubzuki() ──► classify() ──► Sighting
                        blobs           colour gate      nearest ref    pose + v/a

Why it exists: the cradle-mounted iPad draws Nubzuki (web/baby.js) at whatever
distress level the Jetson last sent, and the cradle's camera can see that iPad.
Reading the face back off the screen closes a loop that otherwise cannot be
closed without an infant -- the state the machine commanded should be the state
the camera recovers.  It is the only labelled vision target this project has.

How it names a pose: by comparing the figure to reference crops of all
seventeen and taking the nearest (data/nubzuki_templates.npz, baked by
tools/make_nubzuki_templates.py).  This started as a ladder of hand-written
rules and the comment above classify() records why that was abandoned --
briefly, the rules were tuned on the sticker sheet and the camera sees the
rig's rendering, which draws the same poses differently enough to score 7/17.

Colour *is* still hard-coded, in two places that earn it: the segmentation
mask, and is_nubzuki(), which decides whether a blob is the character at all.
Those two are about "is this Nubzuki and where", which colour answers well;
naming *which* pose needed the whole picture.

It is deliberately not a claim about infant perception: this reads a screen the
machine already wrote, and nothing in this file may be pointed at a real baby.

    python3 perception/nubzuki.py docs/Nubzuki.jpg --grade   # grade vs the sheet
    python3 perception/nubzuki.py --camera 0                 # live, camera only
    python3 perception/nubzuki.py --camera 0 --server http://localhost:8080
                                                             # ...the whole loop
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
# Moved here verbatim from perception/watch.py when the infant-face stack was
# removed (2026-08-08).  It is kept -- rather than dropped with the code that
# used to produce it -- because it is the vocabulary the cradle's decision
# layer was designed around, and because a second, parallel notion of "how is
# the infant" is the thing this project has spent its safety argument avoiding.
# What changed is who fills it in, not what it means.
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

# Each state as a 0..1 distress level, placed against the report's thresholds
# (CALM_LEVEL 0.12, CRY_LEVEL 0.45).  DISTRESS_FACE lands in the fuss band and
# *cannot* reach the cry ladder on its own -- strength only moves it inside
# [0.30, 0.42].  UNKNOWN is not a level at all: it drops ``present`` and lets
# the safety gate do the stopping and checking.
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
# docs/Nubzuki.jpg draws a circumplex: a light grey circle with dashed axes
# through it, ACTIVE at the top and POSITIVE at the right.  These are that
# circle, fitted to the printed ring itself (least squares over the ring
# pixels, 2.3 px rms) rather than eyeballed, so a figure's centroid converts
# straight into circumplex coordinates.  Only the grader needs them; the
# classifier never looks at where a figure sits, because that is exactly the
# thing being tested.
SHEET_CENTRE = (493.56, 525.31)
SHEET_RADIUS = 412.59

# x is valence (POSITIVE right), y is screen-down so ACTIVE is negative -- the
# same frame web/baby.js draws its wheel in, and these are the same anchors it
# blends poses at.  Kept in the two places on purpose: the rig has to draw them
# and this has to name them, and tests.py::nubzuki asserts the tables agree so
# they cannot drift apart silently.
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

# web/baby.js::LIVE_LADDER, inverted.  The rig turns one distress number into a
# blend of two named poses; seeing a named pose therefore recovers the number
# that drew it.  Only the five poses on the live ladder have a level -- the rest
# of the sheet is reachable by hand on the wheel but never by the machine, so
# reading one back means a person is driving, which is worth knowing.
LIVE_LEVEL: dict[str, float] = {
    "sitHeart": 0.0, "neutral": .22, "crying": .38, "angry": .78, "rage": 1.0,
}
SLEEP_POSES = ("sleeping", "dreaming")
DISTRESS_POSES = ("rage", "angry", "crying")

# Where each distress pose sits *within* the fuss band.  distress_of() spends
# this on the 0.30..0.42 span it is allowed, so the three keep their order
# without any of them reaching the ladder.
POSE_STRENGTH: dict[str, float] = {"crying": 0.0, "angry": .5, "rage": 1.0}


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
# The figure mask is "saturated colour, or ink" -- deliberately not "darker
# than white".  Every Nubzuki is drawn standing on a soft grey drop-shadow, and
# a plain brightness threshold swallows those shadows, which then bridge
# neighbouring figures into one blob: on the reference sheet that merged the
# startled one with the kiss.  Grey is unsaturated and not dark, so this drops
# it, and the printed wheel with it.
INK_V = 90          # anything this dark is outline ink
COLOUR_S = 55       # ...and anything this saturated is the character's paint
COLOUR_V = 60
# How far above the background's own saturation the paint cut must sit.
BACKGROUND_MARGIN = 14
# 55 rather than a safer-looking 70 because the angry pose's flush is a
# *gradient* -- it fades from magenta into blue across the bottom of the head,
# and at 70 the pale middle of it dropped out under a brighter exposure, cutting
# the head off the body.  The figure then read as two half-figures and lost its
# flush.  There is a wide margin below: the printed drop-shadows, the thing this
# threshold exists to reject, peak at S=4.
MIN_FIGURE_PX = 6000
# Every threshold in this file was written against renders, where the page is
# pure white and the paint fully saturated.  A photograph of an actual panel is
# neither: the page takes the room's colour and the paint loses most of its
# saturation to glare and gamma.  Measured on a simulated capture, the body
# blue arrives at S~30 against the S>55 the mask asks for -- so the paint
# vanishes and only the ink outline survives, which is exactly the "it detects
# the sheet fine but rarely finds it on the iPad" failure.
#
# Rather than loosen every threshold until the room qualifies too, the frame is
# mapped back to the statistics the thresholds were written for: white-balance
# on the page, then restore saturation.  Harmless on a render (the statistics
# are already right, so the gains land near 1.0) and the difference between
# working and not on a photograph.
NORMALISE_WHITE_PCT = 97.0      # the page is the brightest thing in frame
NORMALISE_WHITE = 245.0
NORMALISE_SAT_PCT = 92.0        # the paint is the most saturated thing on it
NORMALISE_SAT = 175.0
# Smallest blob worth calling a figure, as a fraction of the frame.  A fixed
# pixel count silently means "quite big" at 4K and "the whole scene" at 320x240.
MIN_FIGURE_FRAC = 0.004
# A page smaller than this is not a page, it is a bright object in the room.
# Small, because "the iPad is across the room" is the normal case, not the edge
# one: at 0.04 a panel filling under 4% of frame was discarded and the grey
# desk behind it was taken as the page instead.
PAGE_MIN_FRAC = 0.004
# Which brightness counts as "the page".  Anchored near the maximum, not at the
# 90th percentile: with a distant panel most of the frame *is* room, so the
# 90th percentile sits on the furniture and the furniture qualifies.  The lit
# panel is the brightest surface in shot, so measure against the brightest.
PAGE_BRIGHT_PCT = 99.5


def normalise(bgr: np.ndarray) -> np.ndarray:
    """Undo the panel and the lens: white-balance the page, restore saturation.

    Two statistics, both anchored on things we know are in frame -- the page is
    white, and the most saturated thing on it is the character's paint.  Only
    percentiles are used, so a bright reflection or a dark bezel moves neither.
    """
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
            # Never below 1.0: this exists to undo saturation a lens *lost*,
            # not to impose a house level.  Allowed to cut, it damped the
            # already-vivid sheet by 6% -- enough to thin the angry pose's
            # flush gradient past the mask threshold and split its head off
            # its body again, which is the same failure COLOUR_S was widened
            # to fix.  Rescue only.
            gain = min(4.0, max(1.0, NORMALISE_SAT / level))
            hsv[:, :, 1] = np.clip(sat * gain, 0, 255)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def find_page(bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    """The bright, near-neutral region the mascot is drawn on, or None.

    Restricting to the page before anything else is what makes a camera frame
    workable at all.  :func:`normalise` raises saturation until the paint is
    paint again, and applied to a whole room that same gain lifts the grey
    walls into the mask too -- measured, it merged the entire 1280x720 frame
    into one blob and found nothing.  The page bounds the gain to the surface
    that is actually white, which is the assumption the gain was built on.

    Found on the *raw* frame, before any gain, because bright-and-neutral is
    exactly what a white panel still looks like through a bad lens.  Returns
    None when nothing large enough qualifies, and the caller then treats the
    whole frame as the page -- which is what a render or the sticker sheet is.
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

    The saturation cut is the floor COLOUR_S *or* clear of the background,
    whichever is higher, and the second clause is what makes this survive a
    camera.  ``normalise`` multiplies saturation to bring washed-out paint back,
    and it multiplies the page's faint tint by the same amount: measured on a
    simulated capture the page landed at S=56 against a fixed cut of 55, so the
    page joined the mask, merged with the figure, and the combined blob failed
    the colour gate -- detection "not working" while every threshold looked
    fine.  Reading the background off the frame removes the coin flip.  On a
    render the background is S=2 and this changes nothing.
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
    # Row-major, with a row band generous enough that figures which sit a few
    # pixels apart vertically still read left-to-right.
    return sorted(boxes, key=lambda b: (b[1] // 60, b[0]))


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
@dataclass
class Features:
    """Everything the rule ladder is allowed to look at.

    Colour terms are fractions of the figure's own silhouette and the eye term
    is divided by head width squared, so nothing here changes when the figure
    is nearer the camera or drawn larger.
    """
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
        """A standing figure is widest at the head, near the top.

        .35, not the .60 this started at.  .60 was read off the sheet, where a
        lying figure is widest at 79-86% down.  The rig draws the same poses
        far less prone -- its lying figures peak at 37-56% -- so .60 missed all
        three of them and they came back as Neutral.  Every standing figure in
        both sets, once the prop rules above have taken their own, is widest
        above .35, so the threshold separates on the shared evidence rather
        than on either drawing's habits.
        """
        return self.wide_at >= .35


# Hue windows, measured off the sheet with cv2's 0..179 hue scale.  Named for
# what they pick out rather than for the colour, because that is what the rules
# below actually mean by them.
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

    Picks the two white blobs that look like *a pair* -- similar size, level
    with each other -- rather than simply the two biggest.  Held props and the
    wordmark are white too, and taking the biggest two let a heart stand in for
    an eye, which read a calm figure as a startled one.

    The floor on blob size is a fraction of the head, never a pixel count: a
    fixed one is a promise the figure will always be the size it is on the
    sheet.  Held at 60 px it admitted the wordmark as an eye once the image was
    enlarged, and the sunglasses pose -- which is a pose precisely *because* no
    eyes are visible -- came back with two.
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
    # Isolate *this* figure: a neighbour reaching into the padded window would
    # otherwise donate its props to this one's feature vector.
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
    # A flush is pink *or* purple lying on the head; a held prop is neither on
    # the head nor, usually, either colour.  Both colours because the sheet
    # flushes magenta and the rig flushes violet -- the same expression, drawn
    # by two hands, and the rule has to survive both.
    bands = _bands(hh, ss, vv)
    face = np.zeros_like(inside)
    face[:int(.45 * sil.shape[0])] = True
    out.flush = float(((bands["pink"] | bands["purple"]) & inside & face).sum()) / total
    return out


# --------------------------------------------------------------------------- #
# Naming a pose: nearest reference crop
# --------------------------------------------------------------------------- #
# This was a ladder of hand-written rules -- red head is rage, brown legs are
# the poorly one -- graded 17/17 on the sticker sheet it was written against.
# Then the rig's own renderings arrived and it scored 7/17, because the two
# drawings do not share the proportions the rules keyed on: the rig's ordinary
# eyes are as large as the sheet's streaming ones, and its lying poses are
# barely prone.  Retuning recovered 11/17 and stalled, because the marks that
# actually separate the remaining six -- tears *under* the eyes, spectacles
# *around* them, a bow *on top* -- are positions on a face, and a fraction
# measured over the whole figure has already thrown that away.
#
# So the figure is compared to reference crops of all seventeen poses instead,
# and the nearest one wins.  Same evidence, kept whole.  Measured on the rig's
# renderings: rules 11/17, this 17/17, including through a full camera
# simulation (8 deg rotation, perspective, glare, blur, noise, JPEG q45).
#
# What it gives up is the sentence the ladder could produce.  What replaces it
# is arguably better: the distance to the winner and to the runner-up, which
# says how close the call was -- something no rule ever told us.
TEMPLATE_FILE = Path(__file__).resolve().parent.parent / "data" / "nubzuki_templates.npz"
CROP_N = 48                # every figure is squared and scaled to this
# Both measured over the rig's seventeen renderings under camera-like
# distortion, not guessed.  A correct match costs 0 on the reference itself,
# 1.7-9 through blur, downscale or a colour cast, and 23-26 through a full
# camera simulation or an 8 deg tilt.  A frame degraded past usefulness (30%
# downscale *and* blur *and* JPEG q25) sits at 38 with its runner-up only 5%
# behind.  So: refuse past 32, and refuse whenever the runner-up is within 8%,
# which is what "too blurred to tell these two apart" actually looks like.
MAX_DISTANCE = 32.0        # beyond this, nothing is recognised at all
MIN_MARGIN = 1.08          # winner must beat the runner-up by this ratio


def square_crop(bgr: np.ndarray, box: tuple[int, int, int, int],
                mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One figure, isolated on a square canvas at CROP_N.

    Squared by padding rather than by stretching: how tall a pose stands
    against how wide it lies is one of the things being compared, and a
    stretch would erase exactly that.
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

    Hue goes in as a *unit vector* scaled by saturation, never as a number:
    hue wraps at 180, so red would otherwise read as maximally distant from
    itself.  Brightness goes in standardised over the figure, so a dim room or
    a bright panel shifts nothing.  The silhouette goes in as its own channel
    because shape carries as much of the answer as colour -- it is what tells a
    lying pose from a standing one.
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

    ``None`` when nothing is close enough, or when the best two references
    disagree and are too near each other to choose between.  A recogniser
    reading a screen it half-sees should say so rather than pick.
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

        This is an *echo*, not a perception: the machine sent a level, the rig
        drew it, and this is that number coming back.  It is the right quantity
        for checking the loop and the wrong one for telling the machine
        anything -- see :meth:`asserted_level`.

        None means the pose is off the live ladder: the rig can only reach it
        when somebody drags the wheel by hand, so reading it back off the iPad
        says a person took the wheel, not that the machine is in that state.
        """
        return LIVE_LEVEL.get(self.pose)

    @property
    def state(self) -> str:
        """This sighting in the cradle's five-state vocabulary."""
        return five_state(self)

    @property
    def asserted_level(self) -> float:
        """What this reading is *allowed* to claim, via the watcher's own rule.

        The gap between this and :attr:`recovered_level` is the point, not an
        inconsistency.  Reading Crying off the screen recovers the .50 that was
        sent, but as a *visual* observation it may only ever assert .30 --
        because §5 says vision alone cannot cross CRY_LEVEL, and a mascot on a
        screen is no more entitled to escalate the cradle than a face is.
        Everything here stays under 0.42 by construction.
        """
        return distress_of(self.state, POSE_STRENGTH.get(self.pose, 0.0))

    @property
    def asleep(self) -> bool:
        return self.pose in SLEEP_POSES


def is_nubzuki(f: Features) -> bool:
    """Is this blob the character at all, or just something else on screen?

    On the reference sheet everything found is a Nubzuki, so this never fires.
    Pointed at the actual iPad it earns its keep: web/baby.css draws a dark
    "Start motion sensor" button (#33251d) and body text, both dark enough to
    be ink, both easily bigger than the blob floor.  Without this they arrive
    as figures and get named, and a button confidently reported as Neutral is
    worse than no reading at all.

    The test is the paint: a Nubzuki is mostly its own blue, or -- for the four
    poses that recolour the head -- unmistakably one of those colours instead.
    """
    return (f.blue >= .15 or f.red >= .15 or f.purple >= .10 or f.yellow >= .05)


# --------------------------------------------------------------------------- #
# Framing the camera: which pixels get read at all
# --------------------------------------------------------------------------- #
# Lives here, not in serve.py, because both readers need it and a second copy
# of a framing rule is how two callers quietly stop seeing the same thing --
# the same reason JOINT_NAMES has one home.  serve.py imports these names.
def parse_crop(spec: str) -> tuple[float, float, float, float]:
    """``"x,y,w,h"`` -> a rectangle, as fractions of the frame.

    Numbers are read as fractions when every one of them is <= 1, and as
    pixels otherwise -- so ``0.25,0.1,0.5,0.8`` and ``320,72,640,576`` both
    mean roughly the middle of a 1280x720 frame.  Pixels are only resolved
    against the real frame in :func:`crop_frame`, since nothing here has seen
    one yet.
    """
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
    """The frame cut down to the region of interest.

    This settles the page competition before it starts: :func:`find_page`
    takes the largest bright near-neutral region, so any lit whiteboard or
    window bigger than the iPad wins the frame and the mascot is never looked
    at.  The size gate (a fraction of the frame) stops being diluted by wall
    at the same time.
    """
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
    """The region of interest, scaled up -- digital zoom, and it does pay.

    No pixel of detail is created by this, and :func:`classify` resizes every
    figure to CROP_N anyway, so "no effect" was the honest expectation and it
    is wrong.  The steps *before* the classifier -- the mask morphology,
    :func:`square_crop`'s 6 px pad, the silhouette -- are written in fixed
    pixels, and a small figure starves them.  Named-correct on a photographed
    panel (warp, glare, blur, noise, JPEG q45), cropped to the panel, three
    seeds: a 44 px figure 8-10/17 at x1, 14-15/17 at x2, 16/17 at x3; a 53 px
    figure 12-14 -> 16-17; at 63 px and above all three are within noise.
    So it is a rescue for a distant panel, not a free upgrade -- and it costs
    real time (x3 ran ~570 ms a frame on the Jetson).
    """
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
        # Boxes go back into the caller's frame: the crop above is an internal
        # detail, and an overlay drawn in page coordinates lands nowhere.
        out.append(Sighting(pose, why,
                            (box[0] + ox, box[1] + oy, box[2], box[3]),
                            v, a, feat))
    return out


def five_state(sighting: "Sighting | None") -> str:
    """A sighting in the recognizer's original five-state vocabulary.

    This is the vocabulary the decision layer was built around, and since the
    infant-face stack was removed this file is the only thing that fills it in.
    Each rung is decided on *evidence* -- what the reader can and cannot see --
    not on the pose's name:

        UNKNOWN          nothing recognisable in frame -- drops ``present`` and
                         lets the safety gate do the stopping and checking
        SLEEP_CANDIDATE  lying down with the eyes shut
        DISTRESS_FACE    the three upset poses; capped in the fuss band
        EYES_CLOSED      no open eye pair, but upright -- the shades, the bow
                         and the celebration all hide the eyes.  Deliberately
                         not a claim about sleep
        AWAKE            a visible pair of open eyes
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
# Drawn in the sheet's own frame so it can be held up against docs/Nubzuki.jpg:
# +x is POSITIVE, and y runs screen-down so ACTIVE is at the top.
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
    """The circumplex with every sighting plotted on it.

    Seventeen faint anchors give the reading somewhere to sit: a dot alone says
    "negative and active", a dot against its neighbours says *which* negative
    and active pose, and that is the difference between a picture you can check
    and one you have to trust.
    """
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
    """The pose a given distress level should have drawn.

    web/baby.js blends the *two* ladder poses either side of the level, so at
    level .35 the iPad is showing 61% of the way from Neutral to Crying -- a
    face that is genuinely neither.  A classifier that names pure poses can
    therefore only ever recover the level to the nearest rung, and pretending
    otherwise would turn an honest quantisation into a fake accuracy figure.
    """
    if asleep:
        return "sleeping"
    return min(LIVE_LEVEL, key=lambda k: abs(LIVE_LEVEL[k] - level))


class Tracker:
    """Modal pose over a short window, so the live readout does not flicker.

    A single frame is a vote, not an answer: at a blend midpoint the drawn face
    genuinely sits between two poses and consecutive frames land on either side
    of the line.  Holding the majority of the last second reports what is being
    shown rather than what the last frame happened to round to.
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

    Best-effort by design -- a dropped connection dims the comparison line and
    leaves the recogniser running, because the camera half of this test is
    still worth watching when the Jetson half is not there.
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
            # Framed before anything reads it, so the overlay, the votes and
            # the --dump corpus are all about the same pixels.  Same flags,
            # same meaning, same code as serve.py --sense.
            frame = magnify(crop_frame(frame, args.crop), args.zoom)
            frames += 1
            now = time.monotonic()
            seen = read(frame, args.min_area)
            # The iPad shows one face.  Biggest wins: a reflection or a poster
            # in shot is smaller than the screen being pointed at.
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

            # The frame goes to disk with what the classifier made of it, so a
            # retune can be measured against the camera it will actually face
            # rather than against print-resolution vector art.
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
                # Every stage, so a failure is attributable: no page found, or
                # a page but an empty mask, or blobs that the colour gate threw
                # away.  "It sees nothing" is three different bugs.
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
                # "Passed the gate but named nothing" is the interesting case
                # and the line above cannot distinguish it from "found nothing".
                # Print the actual verdict per surviving blob: how big it is,
                # what it came nearest to, and by how much it missed.
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
                # The wheel rides in the corner: the panel says what it decided,
                # the wheel says where that sits relative to everything it
                # could have decided instead.
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

    Grading only.  This is the ground truth the classifier is measured against,
    so no rule may consult it -- a classifier told where a figure sits on an
    emotion chart has been told the answer.
    """
    x, y, w, h = box
    return ((x + w / 2 - SHEET_CENTRE[0]) / SHEET_RADIUS,
            (y + h / 2 - SHEET_CENTRE[1]) / SHEET_RADIUS)


def overlay_scale(bgr: np.ndarray) -> float:
    """How big to draw on this frame, relative to the 640-wide it was tuned for.

    A crop makes the read frame small -- 160x144 for a panel across the room --
    and a fixed 0.62 font on that is lettering three heads high, which the
    dashboard then upscales to fill its card.  It looks like the labels were
    drawn before the crop; they were not (the status line sits at (16, 30),
    which a crop starting at x=237 would have excluded entirely).  They are
    simply drawn at a size nothing told them to revise.  Clamped, because a
    4K frame does not want four-fold lettering either.
    """
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
    # The same two flags serve.py --sense takes, and the same code behind
    # them: tools/aim_camera.py prints values that work in either.
    p.add_argument("--crop", metavar="X,Y,W,H",
                   help="read only this region of each frame (fractions when "
                        "all <= 1, else pixels).  Live modes only")
    p.add_argument("--zoom", type=float, default=1.0, metavar="N",
                   help="magnify what is read by N; with no --crop it reads "
                        "the middle 1/N, so the cost stays flat")
    # 1280x720, not 640x360.  The mascot is a small object inside a screen
    # inside the frame, so capture resolution lands on it four-fold: measured on
    # a live capture the figure arrived 76x73 px, small enough that Bashful and
    # Blowing-a-kiss -- both a heart held beside the head -- came out 24.2 and
    # 25.0 apart and the match was refused as too close to call.  Pixels on the
    # figure are the cheapest accuracy available here.
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

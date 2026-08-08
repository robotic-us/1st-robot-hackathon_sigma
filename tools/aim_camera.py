#!/usr/bin/env python3
"""Point the camera at the cradle iPad, and print the flags that read it.

Finds the panel, measures the figure, prints the serve.py --crop/--zoom line
to run, then re-reads through its own recommendation and prints named-before
vs named-after.  Zoom bands and per-crop costs are measured, not guessed.

    python3 tools/aim_camera.py                       # camera 0, print flags
    python3 tools/aim_camera.py --camera-size 1920x1080
    python3 tools/aim_camera.py --save data/aim.png   # write what it saw
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from perception import nubzuki as nz
from perception.nubzuki import magnify

# Figure height (px) -> best-measuring zoom (photographed-panel sweep in
# CLAUDE.md); above BIG_PX all three zooms were within noise.
TINY_PX = 48
SMALL_PX = 62
BIG_PX = 62


def suggest_zoom(figure_h: int) -> tuple[float, str]:
    """The zoom for a figure this tall, and why."""
    if figure_h <= 0:
        return 2.0, "no figure measured: x2 is the safe default for a demo"
    if figure_h < TINY_PX:
        return 3.0, (f"figure is {figure_h} px, under {TINY_PX}: x3 named 16/17 "
                     f"where x1 named 8-10/17.  A figure this small comes with "
                     f"a small crop, so x3 is cheap here -- see the table")
    if figure_h < SMALL_PX:
        return 2.0, (f"figure is {figure_h} px, under {SMALL_PX}: x2 named "
                     f"16-17/17 where x1 named 12-14/17")
    return 1.0, (f"figure is {figure_h} px, over {BIG_PX}: every zoom measured "
                 f"the same, so spend the frame time elsewhere")


def suggest_crop(page: tuple[int, int, int, int],
                 shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
    """The page rectangle, padded and clamped -- None if it already fills the frame."""
    fh, fw = shape[:2]
    x, y, w, h = page
    pad = int(0.06 * max(w, h))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(fw, x + w + pad), min(fh, y + h + pad)
    if (x1 - x0) * (y1 - y0) >= 0.92 * fw * fh:
        return None                       # the panel already fills the view
    return (x0, y0, x1 - x0, y1 - y0)


def page_candidates(bgr) -> list[tuple[int, int, int, int]]:
    """Every bright near-neutral region large enough to be a panel, biggest first.

    ``nz.find_page`` returns only the largest, which is exactly wrong for
    aiming; collect the candidates and let :func:`pick_page` choose.
    """
    import numpy as np

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    bright = float(np.percentile(val, nz.PAGE_BRIGHT_PCT))
    page = ((val >= bright * .80) & (sat <= 70)).astype("uint8")
    page = cv2.morphologyEx(page, cv2.MORPH_CLOSE, np.ones((31, 31), "uint8"))
    count, _, stats, _ = cv2.connectedComponentsWithStats(page, 8)
    floor = nz.PAGE_MIN_FRAC * bgr.shape[0] * bgr.shape[1]
    keep = [stats[i] for i in range(1, count)
            if stats[i][cv2.CC_STAT_AREA] >= floor]
    keep.sort(key=lambda s: -s[cv2.CC_STAT_AREA])
    return [tuple(int(v) for v in s[:4]) for s in keep]


def pick_page(frame, zooms=(1.0, 2.0, 3.0)):
    """The candidate the mascot is on: (page, zoom, poses) or None.

    Reads each candidate at each zoom and keeps the one a figure is named in.
    """
    for page in page_candidates(frame):
        box = suggest_crop(page, frame.shape)
        view = frame if box is None else frame[box[1]:box[1] + box[3],
                                               box[0]:box[0] + box[2]]
        for z in zooms:
            got = _named(view if z == 1.0 else
                         cv2.resize(view, None, fx=z, fy=z,
                                    interpolation=cv2.INTER_CUBIC))
            if got:
                return page, z, got
    return None


def _named(frame) -> list[str]:
    """Every pose named in a frame, biggest figure first (the reading's own
    rule), skipping any box that encloses another -- a bezel is not a mascot."""
    seen = nz.read(frame, min_area=int(0.002 * frame.shape[0] * frame.shape[1]))

    def encloses(a, b) -> bool:
        ax, ay, aw, ah = a.box
        bx, by, bw, bh = b.box
        return (a is not b and ax <= bx and ay <= by
                and ax + aw >= bx + bw and ay + ah >= by + bh
                and aw * ah > bw * bh)

    keep = [q for q in seen if not any(encloses(q, o) for o in seen)] or seen
    return [q.pose for q in sorted(keep, key=lambda q: -q.box[2] * q.box[3])]


def grab(index: int, size: tuple[int, int] | None, warmup: int):
    """One settled frame, plus the size the camera really gave us."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"camera {index} would not open "
                         f"(in use? try --camera-index 1; ls /dev/video*)")
    if size is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    frame = None
    for _ in range(max(1, warmup)):       # auto-exposure needs a few frames
        ok, got = cap.read()
        if ok:
            frame = got
    cap.release()
    if frame is None:
        raise SystemExit(f"camera {index} opened but returned no frame")
    return frame


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--image", metavar="PATH",
                   help="read this still instead of the camera -- the same "
                        "advice for a photo taken with a phone")
    p.add_argument("--camera-size", metavar="WxH",
                   help="ask the camera for this capture size first")
    p.add_argument("--warmup", type=int, default=15,
                   help="frames to discard while auto-exposure settles")
    p.add_argument("--save", metavar="PNG",
                   help="write the frame with the proposed crop drawn on it")
    args = p.parse_args(argv)

    size = None
    if args.camera_size:
        try:
            size = tuple(int(v) for v in args.camera_size.lower().split("x"))
            if len(size) != 2 or min(size) <= 0:
                raise ValueError
        except ValueError:
            p.error(f"--camera-size wants WxH -- got {args.camera_size!r}")

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"--image: cannot read {args.image}")
    else:
        frame = grab(args.camera_index, size, args.warmup)
    fh, fw = frame.shape[:2]
    print(f"{args.image or f'camera {args.camera_index}'}: {fw}x{fh}"
          + (f"  (asked {size[0]}x{size[1]}, refused)"
             if size is not None and not args.image and (fw, fh) != size
             else ""))

    candidates = page_candidates(frame)
    if not candidates:
        print("\nno page found: nothing bright and near-neutral is large "
              "enough to be the iPad.\n"
              "  -- is the panel on, showing /baby, and facing the camera?")
        return 1
    print(f"bright regions: {len(candidates)} big enough to be a panel"
          + ("" if len(candidates) == 1 else
             "  (a lit wall or window competes with the iPad)"))

    found = pick_page(frame)
    if found is None:
        print("\nNo Nubzuki in any of them -- so there is no crop to "
              "recommend.\n"
              "  -- the panel must be showing /baby, unblocked and roughly "
              "square-on\n"
              "  -- if it is, the figure may be too small even to detect: "
              "raise --camera-size,\n"
              "     or move the camera closer, and run this again")
        return 1
    page, found_at, poses = found
    print(f"panel:       x={page[0]} y={page[1]} w={page[2]} h={page[3]}"
          f"  ({100 * page[2] * page[3] / (fw * fh):.0f}% of the frame)"
          + ("" if found_at == 1.0 else f"  -- only readable at x{found_at:g}"))

    before = _named(frame)
    crop = suggest_crop(page, frame.shape)
    view = frame if crop is None else frame[crop[1]:crop[1] + crop[3],
                                            crop[0]:crop[0] + crop[2]]
    figure_h = 0
    seen_here = nz.read(view, min_area=int(0.002 * view.shape[0] * view.shape[1]))
    if seen_here:
        figure_h = max(q.box[3] for q in seen_here)
    zoom, why = suggest_zoom(figure_h)
    zoom = max(zoom, found_at)      # never advise less than what worked above
    print(f"figure:      {figure_h} px tall in that crop"
          f"   named {poses[0]}")
    print(f"\nzoom:        x{zoom:g} -- {why}")

    # Cost per zoom timed on *this* crop: the read scales with magnified area.
    print(f"\n{'zoom':>6} {'named':>22} {'ms/frame':>9} {'fps':>6}")
    for z in (1.0, 2.0, 3.0):
        v = magnify(view, z)
        got = _named(v)
        t = time.monotonic()
        for _ in range(3):
            nz.read(v, min_area=int(0.002 * v.shape[0] * v.shape[1]))
        ms = (time.monotonic() - t) / 3 * 1000
        print(f"{'x' + f'{z:g}':>6} {', '.join(got)[:22] or '(nothing)':>22} "
              f"{ms:9.0f} {1000 / ms:6.1f}"
              + ("   <- recommended" if z == zoom else ""))
    if zoom > 1.0:
        view = magnify(view, zoom)
    after = _named(view)
    print(f"\nnamed now:        {', '.join(before) or '(nothing)'}")
    print(f"named with these: {', '.join(after) or '(nothing)'}")
    if after and not before:
        print("  -> the crop is what makes the panel readable at all")
    elif before and not after:
        print("  -> WARNING: the crop loses the figure.  Is the panel moving "
              "outside it?  Widen it by hand before using it.")

    flags = [f"--camera-index {args.camera_index}"]
    if args.camera_size:
        flags.append(f"--camera-size {args.camera_size}")
    if crop is not None:
        flags.append("--crop " + ",".join(str(v) for v in crop))
    if zoom > 1.0:
        flags.append(f"--zoom {zoom:g}")
    print("\nrun:\n  python3 serve.py --sense " + " ".join(flags))
    if crop is None:
        print("  (no --crop: the panel already fills the view)")

    if args.save:
        shot = frame.copy()
        if crop is not None:
            cv2.rectangle(shot, (crop[0], crop[1]),
                          (crop[0] + crop[2], crop[1] + crop[3]),
                          (60, 220, 60), 3)
        cv2.imwrite(args.save, nz.annotate(shot, nz.read(
            frame, min_area=int(0.002 * fh * fw))))
        print(f"\nwrote {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

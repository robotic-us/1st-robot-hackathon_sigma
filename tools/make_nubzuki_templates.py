#!/usr/bin/env python3
"""Bake the mascot reference crops into one small file the recognizer loads.

`perception/nubzuki.py` names a pose by finding the nearest of these, so this
is where its knowledge actually lives.  Two sources, because there are two
drawings of the same seventeen poses and the camera may be pointed at either:

    docs/poses/nubzuki-NN-*.png   what web/baby.js renders -- the live target
    docs/Nubzuki.jpg              the original sticker sheet

The sheet's labels are not taken on trust: the sheet *is* a circumplex, so each
figure is labelled by which pose anchor its printed position is nearest to.
That is ground truth from the artwork's own chart, independent of any
classifier, which is the only reason it is safe to train the classifier on.

Each figure is baked at several sizes (SCALES), because a distant figure is not
just a smaller one -- its mask is coarser and its edges softer, and a
full-resolution template does not match it.  Crops are stored rather than
descriptors, so the matching function can change without re-baking.

    python3 tools/make_nubzuki_templates.py        # data/nubzuki_templates.npz
    python3 tools/make_nubzuki_templates.py --check  # report, write nothing
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from perception import nubzuki as nz

# The rig's filenames are lowercased; the pose keys are not.
ALIAS = {"sitheart": "sitHeart"}
# Each reference is baked at several sizes, because a figure 60 px tall is not
# a shrunken version of one 600 px tall -- its mask is coarser, its edges are
# soft, and its descriptor lands too far from a full-resolution template to be
# recognised.  Measured against a simulated panel at increasing distance, a
# single scale named 8/17 at 300 px and 0/17 at 120 px; these five name 17/17
# and 11/17.  This is the whole reason "it does not catch when it is too far".
SCALES = (1.0, .45, .25, .15, .10)
POSE_DIR = Path("docs/poses")
SHEET = Path("docs/Nubzuki.jpg")
OUT = Path("data/nubzuki_templates.npz")


def crops_at_scales(img, scales=SCALES):
    """The biggest figure's crop, re-extracted after shrinking the source.

    Shrinking the *source* and re-running the real pipeline -- rather than
    scaling a finished crop -- is the point: it reproduces the coarser mask and
    softer edges a distant figure actually arrives with.
    """
    out = []
    for scale in scales:
        small = (img if scale == 1.0 else
                 cv2.resize(img, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA))
        small = nz.normalise(small)
        mask = nz.figure_mask(small)
        floor = max(120, int(nz.MIN_FIGURE_FRAC * small.shape[0] * small.shape[1]))
        found = [b for b in nz.find_figures(small, floor)
                 if nz.is_nubzuki(nz.features(small, b, mask))]
        if not found:
            continue
        out.append(nz.square_crop(small, max(found, key=lambda b: b[2] * b[3]), mask))
    return out


def biggest(path: Path):
    """The largest Nubzuki in an image, as (box, mask, image).

    Normalised exactly as ``nz.read`` normalises a camera frame: a template
    measured on raw pixels and compared against a balanced one is comparing two
    different colour spaces, which quietly costs accuracy on every query.
    """
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"cannot read {path}")
    img = nz.normalise(img)
    mask = nz.figure_mask(img)
    found = [b for b in nz.find_figures(img)
             if nz.is_nubzuki(nz.features(img, b, mask))]
    if not found:
        raise SystemExit(f"no Nubzuki found in {path}")
    return max(found, key=lambda b: b[2] * b[3]), mask, img


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="report what would be baked, write nothing")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)

    labels, sources, crops, alphas = [], [], [], []

    # 1. The rig's own renderings.
    for path in sorted(POSE_DIR.glob("nubzuki-[0-9]*.png")):
        match = re.match(r"nubzuki-\d+-(\w+)\.png", path.name)
        if not match:
            continue
        key = ALIAS.get(match.group(1), match.group(1))
        if key not in nz.POSES:
            raise SystemExit(f"{path.name}: '{key}' is not a pose key")
        for bgr, alpha in crops_at_scales(cv2.imread(str(path))):
            labels.append(key); sources.append("rig")
            crops.append(bgr); alphas.append(alpha)

    # 2. The sticker sheet, each figure labelled by where it is printed.
    sheet = nz.normalise(cv2.imread(str(SHEET)))
    mask = nz.figure_mask(sheet)
    boxes = [b for b in nz.find_figures(sheet)
             if nz.is_nubzuki(nz.features(sheet, b, mask))]
    # Nearest anchor per figure is not enough: the celebrating pose is printed
    # *outside* the circle (radius 1.37) while its anchor was pulled inside so
    # the rig's wheel could still reach it, which leaves it nearer to dancing
    # than to itself.  Assigning globally-smallest distances first, one figure
    # per pose, uses the one fact we do know -- each pose is drawn exactly once
    # -- and lets the well-placed figures claim their anchors before the
    # awkward one has to choose.
    pairs = sorted(
        (((nz.POSES[k][0] - nz.sheet_position(b)[0]) ** 2
          + (nz.POSES[k][1] - nz.sheet_position(b)[1]) ** 2) ** .5, i, k)
        for i, b in enumerate(boxes) for k in nz.POSES)
    taken_box: dict[int, str] = {}
    taken_key: set[str] = set()
    for dist, i, key in pairs:
        if i in taken_box or key in taken_key:
            continue
        taken_box[i], _ = key, taken_key.add(key)
    for i, box in enumerate(boxes):
        bgr, alpha = nz.square_crop(sheet, box, mask)
        labels.append(taken_box[i]); sources.append("sheet")
        crops.append(bgr); alphas.append(alpha)

    for source in ("rig", "sheet"):
        keys = [k for k, s in zip(labels, sources) if s == source]
        missing = sorted(set(nz.POSES) - set(keys))
        # Duplicates are expected now (one per scale) -- what must hold is that
        # every pose is present, and that none of them lost every scale.
        print(f"  {source:<6} {len(keys):3d} crops, {len(set(keys))} distinct poses"
              + (f", MISSING {missing}" if missing else ""))
        if missing:
            raise SystemExit(f"{source}: every pose must be represented")

    if args.check:
        print("  --check: nothing written")
        return 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, labels=np.array(labels), sources=np.array(sources),
                        crops=np.array(crops, np.uint8),
                        alphas=np.array(alphas, np.uint8))
    print(f"  {out}  {len(labels)} templates, {out.stat().st_size / 1024:.0f} kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

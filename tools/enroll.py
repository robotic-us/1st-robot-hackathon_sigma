#!/usr/bin/env python3
"""Manage the enrolled-face database.

    python3 tools/enroll.py --name alice          # capture from the webcam
    python3 tools/enroll.py --name bob --images ./bob_photos
    python3 tools/enroll.py --list
    python3 tools/enroll.py --delete alice
"""
import argparse
import sys
import time

import cv2
import numpy as np

if __package__ in (None, ""):   # direct run: put the repo root on sys.path
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.face import config, draw
from perception.face.detect import FaceDetector
from perception.face.recognize import FaceDB, FaceRecognizer


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, config.CAM_FPS)
    return cap


def capture_from_camera(name, n_samples, cam_index, show=True):
    """Grab n_samples embeddings of the largest face, spaced a few frames apart."""
    det, rec = FaceDetector(), FaceRecognizer()
    cap = open_camera(cam_index)
    if cap is None:
        print(f"Cannot open camera {cam_index}", file=sys.stderr)
        return None

    embs, last, frames = [], -999, 0
    print(f"Enrolling '{name}': look at the camera and vary your pose slightly.")
    print("  q / Esc to abort.")
    try:
        while len(embs) < n_samples:
            ok, frame = cap.read()
            if not ok:
                continue
            frames += 1
            faces = det.detect(frame)

            if len(faces) and frames - last >= config.ENROLL_MIN_GAP:
                embs.append(rec.embed(frame, faces[0]))
                last = frames

            if show:
                for f in faces[:1]:
                    x, y, w, h = (int(v) for v in f[:4])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 220, 80), 2)
                draw.draw_banner(frame, [
                    f"Enrolling: {name}",
                    f"captured {len(embs)}/{n_samples}",
                    "turn your head slowly" if len(faces) else "no face detected",
                ], (80, 220, 80))
                cv2.imshow("SIGMA - enroll", frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    print("Aborted.")
                    return None
            else:
                print(f"\r  captured {len(embs)}/{n_samples}", end="", flush=True)
                time.sleep(0.01)
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()

    if not show:
        print()
    return np.array(embs, np.float32)


def embeddings_from_images(folder):
    from pathlib import Path
    det, rec = FaceDetector(), FaceRecognizer()
    embs = []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() not in exts:
            continue
        img = cv2.imread(str(p))
        if img is None:
            print(f"  skip (unreadable): {p.name}")
            continue
        faces = det.detect(img)
        if not len(faces):
            print(f"  skip (no face):    {p.name}")
            continue
        embs.append(rec.embed(img, faces[0]))
        print(f"  ok:                {p.name}")
    return np.array(embs, np.float32) if embs else None


def main():
    ap = argparse.ArgumentParser(description="Manage SIGMA's enrolled faces.")
    ap.add_argument("--name", help="person to enroll")
    ap.add_argument("--images", help="enroll from a folder of photos instead of the webcam")
    ap.add_argument("--samples", type=int, default=config.ENROLL_SAMPLES)
    ap.add_argument("--camera", type=int, default=config.CAM_INDEX)
    ap.add_argument("--no-window", action="store_true", help="headless capture")
    ap.add_argument("--list", action="store_true", help="show enrolled people")
    ap.add_argument("--delete", metavar="NAME", help="remove a person")
    args = ap.parse_args()

    db = FaceDB.load()

    if args.list:
        people = db.people()
        if not people:
            print("No faces enrolled yet.  Try: python3 tools/enroll.py --name <you>")
        else:
            print(f"{len(people)} enrolled ({len(db.names)} embeddings):")
            for n, c in people.items():
                print(f"  {n:<24} {c} samples")
        return 0

    if args.delete:
        removed = db.remove(args.delete)
        if removed:
            db.save()
            print(f"Removed '{args.delete}' ({removed} embeddings).")
        else:
            print(f"No such person: '{args.delete}'")
        return 0 if removed else 1

    if not args.name:
        ap.error("--name is required (or use --list / --delete)")

    if args.images:
        print(f"Reading images from {args.images} ...")
        embs = embeddings_from_images(args.images)
    else:
        embs = capture_from_camera(args.name, args.samples, args.camera,
                                   show=not args.no_window)

    if embs is None or len(embs) == 0:
        print("Nothing enrolled.", file=sys.stderr)
        return 1

    existing = db.people().get(args.name, 0)
    db.add(args.name, embs)
    db.save()
    print(f"Enrolled '{args.name}': +{len(embs)} embeddings "
          f"(now {existing + len(embs)}).  DB: {config.FACES_DB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

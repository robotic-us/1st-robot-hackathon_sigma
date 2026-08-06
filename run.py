#!/usr/bin/env python3
"""SIGMA - live face recognition + 5-class emotion classification.

    python3 run.py                 # live window
    python3 run.py --headless      # terminal only (no X needed)
    python3 run.py --source clip.mp4 --save out.mp4

Keys (windowed):  e enroll   r reset tracks   b toggle bars   space pause   q quit
"""
import argparse
import collections
import sys
import time

import cv2
import numpy as np

from sigma import config, draw
from sigma.pipeline import SigmaPipeline
from sigma.recognize import FaceDB


def open_source(source, camera):
    if source:
        cap = cv2.VideoCapture(source)
        return cap, f"file:{source}"
    cap = cv2.VideoCapture(camera, cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
        cap.set(cv2.CAP_PROP_FPS, config.CAM_FPS)
    return cap, f"camera:{camera}"


def enroll_interactive(cap, pipeline, n_samples=config.ENROLL_SAMPLES):
    """Blocking in-app enrollment; the name is typed in the terminal."""
    cv2.destroyWindow("SIGMA")
    print("\n--- enroll ---")
    try:
        name = input("Name (blank to cancel): ").strip()
    except (EOFError, KeyboardInterrupt):
        name = ""
    if not name:
        print("Cancelled.\n")
        return

    det, rec = pipeline.detector, pipeline.recognizer
    embs, last, frames = [], -999, 0
    while len(embs) < n_samples:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        faces = det.detect(frame)
        if len(faces) and frames - last >= config.ENROLL_MIN_GAP:
            embs.append(rec.embed(frame, faces[0]))
            last = frames
        if len(faces):
            x, y, w, h = (int(v) for v in faces[0][:4])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 220, 80), 2)
        draw.draw_banner(frame, [f"Enrolling: {name}",
                                 f"captured {len(embs)}/{n_samples}",
                                 "turn your head slowly" if len(faces) else "no face detected"],
                         (80, 220, 80))
        cv2.imshow("SIGMA - enroll", frame)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break
    cv2.destroyWindow("SIGMA - enroll")

    if embs:
        rec.db.add(name, np.array(embs, np.float32))
        rec.db.save()
        pipeline.reset_tracks()
        print(f"Enrolled '{name}' (+{len(embs)} embeddings).\n")
    else:
        print("No samples captured.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=config.CAM_INDEX)
    ap.add_argument("--source", help="video file instead of the webcam")
    ap.add_argument("--headless", action="store_true", help="no GUI window")
    ap.add_argument("--save", metavar="OUT.mp4", help="write the annotated video")
    ap.add_argument("--no-bars", action="store_true", help="hide per-emotion bars")
    ap.add_argument("--no-recognize", action="store_true", help="emotion only")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames")
    args = ap.parse_args()

    for path in (config.YUNET, config.SFACE, config.FERPLUS):
        if not path.exists():
            print(f"Missing model: {path}\nRun: python3 fetch_models.py", file=sys.stderr)
            return 1

    pipeline = SigmaPipeline(recognize=not args.no_recognize)
    n_people = len(pipeline.db.people()) if pipeline.db else 0
    if not args.no_recognize and n_people == 0:
        print("Note: no faces enrolled yet - everyone will show as 'unknown'.")
        print("      Enroll with:  python3 enroll.py --name <you>   (or press 'e')")

    cap, label = open_source(args.source, args.camera)
    if not cap.isOpened():
        print(f"Cannot open {label}", file=sys.stderr)
        return 1
    print(f"Source: {label}   emotions: {', '.join(config.EMOTIONS)}")

    writer = None
    show_bars = not args.no_bars
    paused = False
    frames = 0
    recent = collections.deque(maxlen=30)
    last_print = 0.0
    t_prev = time.perf_counter()

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break
                frames += 1
                results = pipeline.process(frame)

                now = time.perf_counter()
                recent.append(now - t_prev)
                t_prev = now
                fps = len(recent) / max(sum(recent), 1e-6)

                for r in results:
                    draw.draw_result(frame, r, show_bars)
                draw.draw_hud(frame, fps, pipeline.timings, len(results), n_people, paused)

                if args.save:
                    if writer is None:
                        h, w = frame.shape[:2]
                        writer = cv2.VideoWriter(args.save,
                                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                                 min(30.0, max(5.0, fps)), (w, h))
                    writer.write(frame)

                if args.headless and now - last_print > 0.5:
                    last_print = now
                    if results:
                        desc = "  ".join(
                            f"[{r.name}/{r.emotion} {r.confidence * 100:.0f}%]" for r in results)
                    else:
                        desc = "(no face)"
                    print(f"\rframe {frames:6d}  {fps:5.1f} FPS  {desc}    ",
                          end="", flush=True)

                if args.max_frames and frames >= args.max_frames:
                    break

            if not args.headless:
                cv2.imshow("SIGMA", frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                elif k == ord("b"):
                    show_bars = not show_bars
                elif k == ord("r"):
                    pipeline.reset_tracks()
                elif k == ord(" "):
                    paused = not paused
                elif k == ord("e") and pipeline.recognizer is not None:
                    enroll_interactive(cap, pipeline)
                    n_people = len(pipeline.db.people())
                    t_prev = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"\nSaved {args.save}")
        cv2.destroyAllWindows()

    print(f"\nProcessed {frames} frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

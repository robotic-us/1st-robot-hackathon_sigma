#!/usr/bin/env python3
"""Download the three ONNX models SIGMA needs.

YuNet is pinned to the 2022mar revision: the 2023mar rewrite changed the output
head and will not parse under OpenCV < 4.8 (this box runs 4.5.4).
"""
import sys
import urllib.request

if __package__ in (None, ""):   # direct run: put the repo root on sys.path
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sigma import config

ZOO = "https://github.com/opencv/opencv_zoo/raw"
YUNET_COMMIT = "2121e57073a0e2b640a644f99f58dc4b4941724b"

MODELS = [
    (config.YUNET,
     f"{ZOO}/{YUNET_COMMIT}/models/face_detection_yunet/face_detection_yunet_2022mar.onnx",
     300_000),
    (config.SFACE,
     f"{ZOO}/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
     30_000_000),
    (config.FERPLUS,
     "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/"
     "emotion_ferplus/model/emotion-ferplus-8.onnx",
     30_000_000),
]


def main():
    config.MODELS.mkdir(parents=True, exist_ok=True)
    failed = []
    for path, url, min_size in MODELS:
        if path.exists() and path.stat().st_size >= min_size:
            print(f"  ok       {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
            continue
        print(f"  fetching {path.name} ...", flush=True)
        try:
            urllib.request.urlretrieve(url, path)
            size = path.stat().st_size
            if size < min_size:
                # GitHub serves a 404 HTML page with a 200-ish body; catch that.
                path.unlink(missing_ok=True)
                raise RuntimeError(f"got {size} bytes, expected >= {min_size}")
            print(f"  ok       {path.name} ({size / 1e6:.1f} MB)")
        except Exception as e:
            print(f"  FAILED   {path.name}: {e}")
            failed.append(path.name)

    if failed:
        print("\nCould not fetch:", ", ".join(failed))
        return 1
    print("\nAll models present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

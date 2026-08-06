#!/usr/bin/env python3
"""Download and verify the three ONNX models SIGMA needs.

YuNet is pinned to the 2022mar revision: the 2023mar rewrite changed the output
head and will not parse under OpenCV < 4.8 (this box runs 4.5.4).  Every model
carries a pinned SHA-256 -- a wrong hash means a moved/replaced upstream file
or a truncated download, and either way the pipeline must not run on it.

    python3 tools/fetch_models.py            # fetch what is missing, verify all
    python3 tools/fetch_models.py --force    # re-download everything
"""
import argparse
import hashlib
import sys
import urllib.request

if __package__ in (None, ""):   # direct run: put the repo root on sys.path
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.face import config

ZOO = "https://github.com/opencv/opencv_zoo/raw"
YUNET_COMMIT = "2121e57073a0e2b640a644f99f58dc4b4941724b"

MODELS = [
    (config.YUNET,
     f"{ZOO}/{YUNET_COMMIT}/models/face_detection_yunet/face_detection_yunet_2022mar.onnx",
     "50ef07f702a31741ca46a4c0d947773b64143b9362780237bf0d427d6c79bab7"),
    (config.SFACE,
     f"{ZOO}/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
     "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
    (config.FERPLUS,
     "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/"
     "emotion_ferplus/model/emotion-ferplus-8.onnx",
     "a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c"),
]


def sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download everything")
    args = parser.parse_args(argv)

    config.MODELS.mkdir(parents=True, exist_ok=True)
    failed = []
    for path, url, want in MODELS:
        if path.exists() and not args.force:
            got = sha256(path)
            if got == want:
                print(f"  ok       {path.name} ({path.stat().st_size / 1e6:.1f} MB, sha256 match)")
                continue
            print(f"  BAD HASH {path.name}: {got[:12]}... != {want[:12]}... -- refetching")
        print(f"  fetching {path.name} ...", flush=True)
        try:
            urllib.request.urlretrieve(url, path)
            got = sha256(path)
            if got != want:
                # GitHub serves 404 HTML with a 200-ish body; a moved upstream
                # file hashes differently.  Neither is a model we should trust.
                path.unlink(missing_ok=True)
                raise RuntimeError(f"sha256 {got[:12]}... != pinned {want[:12]}...")
            print(f"  ok       {path.name} ({path.stat().st_size / 1e6:.1f} MB, sha256 match)")
        except Exception as e:
            print(f"  FAILED   {path.name}: {e}")
            failed.append(path.name)

    if failed:
        print("\nCould not fetch:", ", ".join(failed))
        return 1
    print("\nAll models present and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

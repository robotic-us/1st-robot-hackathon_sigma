"""Face detection with YuNet.

YuNet returns one row per face with 15 values:
    [x, y, w, h,
     right_eye_x, right_eye_y, left_eye_x, left_eye_y,
     nose_x, nose_y,
     right_mouth_x, right_mouth_y, left_mouth_x, left_mouth_y,
     score]
The rest of SIGMA passes these rows around unchanged, because both
FaceRecognizerSF.alignCrop() and our emotion alignment need the landmarks.
"""
import cv2
import numpy as np

from . import config


class FaceDetector:
    def __init__(self, size=(config.FRAME_W, config.FRAME_H)):
        if not config.YUNET.exists():
            raise FileNotFoundError(
                f"YuNet model missing: {config.YUNET}\nRun: python3 fetch_models.py"
            )
        self.net = cv2.FaceDetectorYN.create(
            str(config.YUNET), "", size,
            config.DET_SCORE_THRESH, config.DET_NMS_THRESH, config.DET_TOPK,
        )
        self._size = size

    def detect(self, frame):
        """Return an (N, 15) float32 array of faces, largest first."""
        h, w = frame.shape[:2]
        if (w, h) != self._size:
            self.net.setInputSize((w, h))
            self._size = (w, h)

        _, faces = self.net.detect(frame)
        if faces is None or len(faces) == 0:
            return np.empty((0, 15), np.float32)

        faces = faces.astype(np.float32)
        # Drop faces too small for the 64x64 emotion crop to carry real signal.
        keep = np.maximum(faces[:, 2], faces[:, 3]) >= config.MIN_FACE_PX
        faces = faces[keep]
        if len(faces) == 0:
            return np.empty((0, 15), np.float32)

        order = np.argsort(-(faces[:, 2] * faces[:, 3]))
        return faces[order]


def bbox(face):
    """Integer (x, y, w, h) from a YuNet row."""
    x, y, w, h = face[:4]
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    return inter / float(aw * ah + bw * bh - inter)

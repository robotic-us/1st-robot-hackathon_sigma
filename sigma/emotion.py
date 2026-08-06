"""The 5 emotion classifiers, backed by FER+.

FER+ (emotion-ferplus-8.onnx) is a VGG13 trained on FER2013 images relabelled
by 10 annotators. It emits 8 logits; SIGMA collapses them onto 5 classes by
summing probabilities according to config.FER_TO_EMOTION.

Preprocessing follows the ONNX model zoo reference: 64x64 grayscale, raw 0-255
values, no mean subtraction. We additionally roll the face upright using the
eye landmarks, which FER2013's crops are implicitly normalised for.
"""
import cv2
import numpy as np

from . import config


def _softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def align_face_gray(frame, face, out=config.FER_INPUT, margin=1.0):
    """Rotate the face upright by its eye line and return an out*out gray crop."""
    x, y, w, h = face[:4]
    cx, cy = x + w * 0.5, y + h * 0.5

    reye = face[4:6]
    leye = face[6:8]
    dx, dy = float(leye[0] - reye[0]), float(leye[1] - reye[1])
    angle = np.degrees(np.arctan2(dy, dx))

    side = max(float(w), float(h)) * margin
    if side <= 1:
        return np.zeros((out, out), np.uint8)

    # Rotate about the face centre and scale the box down to `out` pixels...
    M = cv2.getRotationMatrix2D((cx, cy), angle, out / side)
    # ...then shift the (still fixed) centre into the middle of the output.
    M[0, 2] += out * 0.5 - cx
    M[1, 2] += out * 0.5 - cy

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.warpAffine(gray, M, (out, out), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


class EmotionClassifier:
    """Wraps FER+ and exposes a 5-way probability vector over config.EMOTIONS."""

    def __init__(self):
        if not config.FERPLUS.exists():
            raise FileNotFoundError(
                f"FER+ model missing: {config.FERPLUS}\nRun: python3 fetch_models.py"
            )
        self.net = cv2.dnn.readNetFromONNX(str(config.FERPLUS))

        # (8, 5) 0/1 matrix that folds raw FER+ probabilities into our 5 classes.
        M = np.zeros((len(config.FER_CLASSES), len(config.EMOTIONS)), np.float32)
        for i, raw in enumerate(config.FER_CLASSES):
            M[i, config.EMOTIONS.index(config.FER_TO_EMOTION[raw])] = 1.0
        self.fold = M

    def predict(self, frame, face):
        """Return (probs5, raw_probs8) for one detected face."""
        crop = align_face_gray(frame, face)
        blob = crop.astype(np.float32).reshape(1, 1, config.FER_INPUT, config.FER_INPUT)
        self.net.setInput(blob)
        raw = _softmax(self.net.forward().ravel())
        return raw @ self.fold, raw

    @staticmethod
    def label(probs5):
        """Argmax label plus confidence, or ('?', conf) when the model is unsure."""
        i = int(np.argmax(probs5))
        conf = float(probs5[i])
        if conf < config.EMOTION_MIN_CONF:
            return "?", conf
        return config.EMOTIONS[i], conf

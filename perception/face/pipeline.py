"""Detect -> track -> (identify | classify emotion) for one video frame."""
import time

import numpy as np

from . import config
from .detect import FaceDetector
from .emotion import EmotionClassifier
from .recognize import FaceRecognizer
from .track import Tracker


class Result:
    """One face, this frame."""
    __slots__ = ("track_id", "box", "name", "sim", "emotion", "confidence",
                 "probs", "eyes")

    def __init__(self, track_id, box, name, sim, emotion, confidence, probs,
                 eyes=None):
        self.track_id = track_id
        self.box = box
        self.name = name
        self.sim = sim
        self.emotion = emotion
        self.confidence = confidence
        self.probs = probs
        self.eyes = eyes    # ((rx,ry),(lx,ly)) from the YuNet row, for head roll

    def __repr__(self):
        return (f"<{self.name} ({self.sim:.2f}) "
                f"{self.emotion} {self.confidence:.2f} id={self.track_id}>")


class SigmaPipeline:
    def __init__(self, recognize=True, emotion=True):
        self.detector = FaceDetector()
        self.tracker = Tracker()
        self.recognizer = FaceRecognizer() if recognize else None
        self.classifier = EmotionClassifier() if emotion else None
        self.timings = {"detect": 0.0, "recognize": 0.0, "emotion": 0.0}

    @property
    def db(self):
        return self.recognizer.db if self.recognizer else None

    def process(self, frame):
        t0 = time.perf_counter()
        faces = self.detector.detect(frame)
        t1 = time.perf_counter()

        tracks = self.tracker.update(faces)

        t_rec = 0.0
        if self.recognizer is not None:
            for t in tracks:
                # Identity is stable, so only re-run SFace periodically per track.
                if t.needs_recognition():
                    s = time.perf_counter()
                    name, sim = self.recognizer.identify(frame, t.face)
                    t_rec += time.perf_counter() - s
                    t.vote_identity(name, sim)

        t_emo = 0.0
        if self.classifier is not None:
            for t in tracks:
                s = time.perf_counter()
                probs, _ = self.classifier.predict(frame, t.face)
                t_emo += time.perf_counter() - s
                t.update_emotion(probs)

        self.timings = {
            "detect": (t1 - t0) * 1000,
            "recognize": t_rec * 1000,
            "emotion": t_emo * 1000,
        }

        results = []
        for t in tracks:
            if t.probs is not None:
                label, conf = EmotionClassifier.label(t.probs)
                probs = t.probs
            else:
                label, conf = "?", 0.0
                probs = np.zeros(len(config.EMOTIONS), np.float32)
            results.append(Result(t.id, t.box, t.name, t.sim, label, conf, probs,
                                  eyes=(tuple(t.face[4:6]), tuple(t.face[6:8]))))
        return results

    def reset_tracks(self):
        self.tracker.reset()

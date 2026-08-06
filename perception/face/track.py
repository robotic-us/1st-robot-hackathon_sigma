"""Greedy IoU tracker.

Tracking exists for two reasons: it lets identity be computed once every
RECOG_EVERY frames instead of every frame (SFace costs ~13 ms), and it gives
each face a stable slot to smooth emotion probabilities over time.
"""
import numpy as np

from . import config
from .detect import bbox, iou


class Track:
    __slots__ = ("id", "box", "face", "misses", "age", "probs",
                 "name", "sim", "votes", "frames_since_recog")

    def __init__(self, tid, face):
        self.id = tid
        self.face = face
        self.box = bbox(face)
        self.misses = 0
        self.age = 0
        self.probs = None                    # smoothed 5-way emotion vector
        self.name = config.UNKNOWN_NAME
        self.sim = 0.0
        self.votes = []                      # recent identity votes
        self.frames_since_recog = 10 ** 6     # force recognition on frame 1

    def update_box(self, face):
        self.face = face
        self.box = bbox(face)
        self.misses = 0
        self.age += 1

    def update_emotion(self, probs):
        a = config.EMOTION_EMA
        self.probs = probs if self.probs is None else a * self.probs + (1 - a) * probs

    def vote_identity(self, name, sim):
        self.votes.append((name, sim))
        if len(self.votes) > config.IDENTITY_VOTES:
            self.votes.pop(0)
        self.frames_since_recog = 0

        # Majority vote over the window; ties broken by best similarity.
        tally = {}
        for n, s in self.votes:
            cnt, best = tally.get(n, (0, 0.0))
            tally[n] = (cnt + 1, max(best, s))
        self.name, (_, self.sim) = max(tally.items(), key=lambda kv: (kv[1][0], kv[1][1]))

    def needs_recognition(self):
        return self.frames_since_recog >= config.RECOG_EVERY


class Tracker:
    def __init__(self):
        self.tracks = []
        self._next_id = 1

    def update(self, faces):
        """Match detections to tracks; returns the live tracks for this frame."""
        for t in self.tracks:
            t.misses += 1
            t.frames_since_recog += 1

        unmatched = list(range(len(faces)))
        pairs = []
        for ti, t in enumerate(self.tracks):
            for di in unmatched:
                score = iou(t.box, bbox(faces[di]))
                if score >= config.IOU_MATCH:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)

        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            self.tracks[ti].update_box(faces[di])
            used_t.add(ti)
            used_d.add(di)

        for di in unmatched:
            if di not in used_d:
                t = Track(self._next_id, faces[di])
                self._next_id += 1
                self.tracks.append(t)

        self.tracks = [t for t in self.tracks if t.misses <= config.TRACK_MAX_MISS]
        return [t for t in self.tracks if t.misses == 0]

    def reset(self):
        self.tracks.clear()


def probs_or_zeros(track):
    return track.probs if track.probs is not None else np.zeros(len(config.EMOTIONS), np.float32)

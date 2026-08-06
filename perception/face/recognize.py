"""Face identification with SFace + a tiny on-disk embedding database.

Each enrolled person keeps several 128-d embeddings rather than one average;
matching takes the best cosine similarity across that person's samples, which
handles pose/lighting variation far better than a centroid.
"""
import cv2
import numpy as np

from . import config


class FaceDB:
    """names: (N,) str  |  embs: (N, 128) float32, L2-normalised."""

    def __init__(self, names=None, embs=None):
        self.names = list(names or [])
        self.embs = (embs if embs is not None
                     else np.zeros((0, 128), np.float32)).astype(np.float32)

    @classmethod
    def load(cls, path=config.FACES_DB):
        if not path.exists():
            return cls()
        z = np.load(path, allow_pickle=False)
        return cls(list(z["names"]), z["embs"])

    def save(self, path=config.FACES_DB):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path,
                            names=np.array(self.names, dtype="U64"),
                            embs=self.embs)

    def add(self, name, embeddings):
        embeddings = np.atleast_2d(np.asarray(embeddings, np.float32))
        self.names.extend([name] * len(embeddings))
        self.embs = np.vstack([self.embs, embeddings]) if len(self.embs) else embeddings

    def remove(self, name):
        keep = [i for i, n in enumerate(self.names) if n != name]
        removed = len(self.names) - len(keep)
        self.names = [self.names[i] for i in keep]
        self.embs = self.embs[keep] if keep else np.zeros((0, 128), np.float32)
        return removed

    def people(self):
        """{name: sample_count} in insertion order."""
        out = {}
        for n in self.names:
            out[n] = out.get(n, 0) + 1
        return out

    def match(self, emb):
        """Best (name, cosine_similarity). Returns (UNKNOWN_NAME, sim) below threshold."""
        if len(self.embs) == 0:
            return config.UNKNOWN_NAME, 0.0
        sims = self.embs @ emb
        i = int(np.argmax(sims))
        best = float(sims[i])
        if best < config.COSINE_THRESH:
            return config.UNKNOWN_NAME, best
        return self.names[i], best


class FaceRecognizer:
    def __init__(self, db=None):
        if not config.SFACE.exists():
            raise FileNotFoundError(
                f"SFace model missing: {config.SFACE}\nRun: python3 tools/fetch_models.py"
            )
        self.net = cv2.FaceRecognizerSF.create(str(config.SFACE), "")
        self.db = db if db is not None else FaceDB.load()

    def embed(self, frame, face):
        """128-d L2-normalised embedding for one YuNet detection row."""
        aligned = self.net.alignCrop(frame, face)
        emb = self.net.feature(aligned).ravel().astype(np.float32)
        n = np.linalg.norm(emb)
        return emb / n if n > 0 else emb

    def identify(self, frame, face):
        return self.db.match(self.embed(frame, face))

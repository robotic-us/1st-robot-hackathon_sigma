"""Central configuration for SIGMA."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
FACES_DIR = ROOT / "faces"
FACES_DB = FACES_DIR / "faces.npz"

YUNET = MODELS / "face_detection_yunet_2022mar.onnx"
SFACE = MODELS / "face_recognition_sface_2021dec.onnx"
FERPLUS = MODELS / "emotion-ferplus-8.onnx"

# --- camera -----------------------------------------------------------------
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480
CAM_FPS = 30

# --- detection (YuNet) ------------------------------------------------------
DET_SCORE_THRESH = 0.80
DET_NMS_THRESH = 0.30
DET_TOPK = 500
MIN_FACE_PX = 40           # ignore faces smaller than this (too small to classify)

# --- recognition (SFace) ----------------------------------------------------
COSINE_THRESH = 0.363      # OpenCV's recommended SFace cosine threshold
RECOG_EVERY = 8            # frames between re-identification attempts per track
ENROLL_SAMPLES = 12        # embeddings captured per person during enrollment
ENROLL_MIN_GAP = 3         # min frames between captured enrollment samples

# --- emotion (FER+ 8 classes collapsed to SIGMA's 5) ------------------------
# FER+ output order is fixed by the model.
FER_CLASSES = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt",
]

# The 5 classifiers SIGMA reports.
EMOTIONS = ["NEUTRAL", "HAPPY", "SAD", "ANGRY", "SURPRISE"]

# How the 8 raw FER+ classes fold into the 5. Probabilities are summed, so
# ANGRY absorbs the other negative-approach affects and SURPRISE absorbs fear
# (both are high-arousal startle responses and FER+ confuses them heavily).
FER_TO_EMOTION = {
    "neutral":   "NEUTRAL",
    "happiness": "HAPPY",
    "sadness":   "SAD",
    "anger":     "ANGRY",
    "disgust":   "ANGRY",
    "contempt":  "ANGRY",
    "surprise":  "SURPRISE",
    "fear":      "SURPRISE",
}

EMOTION_EMA = 0.65         # temporal smoothing: p = a*p_prev + (1-a)*p_obs
EMOTION_MIN_CONF = 0.34    # below this the label is reported as "?"
FER_INPUT = 64             # FER+ expects 64x64 grayscale

# --- tracking ---------------------------------------------------------------
IOU_MATCH = 0.30
TRACK_MAX_MISS = 15        # frames a track survives without a detection
IDENTITY_VOTES = 5         # rolling votes used to stabilise a track's identity

# --- display ----------------------------------------------------------------
EMOTION_COLORS = {         # BGR
    "NEUTRAL":  (200, 200, 200),
    "HAPPY":    (80, 220, 80),
    "SAD":      (220, 150, 60),
    "ANGRY":    (60, 60, 235),
    "SURPRISE": (40, 210, 240),
    "?":        (130, 130, 130),
}
UNKNOWN_NAME = "unknown"

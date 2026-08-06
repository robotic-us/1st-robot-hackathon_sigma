# SIGMA

Real-time face recognition + 5-class emotion classification on the Jetson AGX Orin,
running off a Logitech C270 webcam.

Everything runs through OpenCV's DNN module on CPU — no PyTorch, no TensorRT, no
extra pip installs beyond what the board already has.

```
frame ──► YuNet (detect) ──► IoU tracker ──┬──► SFace  (who is it)   [every 8th frame]
                                           └──► FER+   (how do they feel)
```

## The 5 emotion classes

`NEUTRAL` `HAPPY` `SAD` `ANGRY` `SURPRISE`

The backing model (FER+) emits 8 classes. SIGMA sums their probabilities into 5:

| FER+ raw | SIGMA |
|---|---|
| neutral | NEUTRAL |
| happiness | HAPPY |
| sadness | SAD |
| anger, disgust, contempt | ANGRY |
| surprise, fear | SURPRISE |

Fear folds into SURPRISE because both are high-arousal startle responses that FER+
confuses heavily; disgust and contempt fold into ANGRY as negative-approach affects.
Edit `FER_TO_EMOTION` in `perception/face/config.py` to change the grouping — the fold matrix
is rebuilt from that dict, so nothing else needs touching.

## Setup

Models are already downloaded in `models/`. To re-fetch:

```bash
python3 tools/fetch_models.py
```

## Use

```bash
python3 tools/enroll.py --name yourname     # capture ~12 embeddings from the webcam
python3 apps/run.py                        # live window
```

| | |
|---|---|
| `python3 apps/run.py --headless` | terminal only, no X server needed |
| `python3 apps/run.py --source clip.mp4 --save out.mp4` | run on a file, write annotated video |
| `python3 apps/run.py --no-recognize` | emotion only, skip identity |
| `python3 tools/enroll.py --list` | show enrolled people |
| `python3 tools/enroll.py --delete NAME` | remove someone |
| `python3 tools/enroll.py --name bob --images ./photos` | enroll from a folder instead of the webcam |

Keys in the live window: `e` enroll · `r` reset tracks · `b` toggle bars · `space` pause · `q` quit

## Measured on this box

Per-call, 640×480, OpenCV 4.5.4 CPU (12 threads):

| stage | cost |
|---|---|
| YuNet detect | 13.3 ms/frame |
| FER+ emotion | 15.7 ms/face |
| SFace identity | 12.8 ms/face |
| **end-to-end, 1 face** | **~30 FPS (camera-bound)** |

Identity only re-runs every `RECOG_EVERY` (8) frames per track, so SFace's cost is
amortised; emotion runs every frame and is smoothed with an EMA over each track.

## Accuracy — what's actually been verified

Honest status, because this matters more than a number:

- **Model wiring is exact.** FER+ reproduces the ONNX zoo's reference output tensor
  to within 3e-6. (Note that reference input is random noise, not a face — it validates
  numerics only.)
- **HAPPY 6/6 and NEUTRAL 7/7** on real Commons portraits that were checked by eye.
- **SAD / ANGRY / SURPRISE are unverified.** An attempt to auto-build a labelled set
  from Wikimedia search terms produced unusable ground truth — the "angry" query
  returned mostly smiling men and one "sad" result was a dog. Those labels were
  discarded rather than reported as accuracy.

To validate the remaining three, pose for them in the live window and watch the
per-class bars. FER+ is trained on FER2013, which is frontal, tightly cropped and
mostly acted — expect it to be strongest on HAPPY/SURPRISE, weakest on SAD, and to
fall back to NEUTRAL when unsure (it returns NEUTRAL ~96% on pure noise).

Recognition uses SFace's standard cosine threshold of 0.363. Enrolling in the
lighting you'll actually run in matters much more than enrolling many samples.

## Layout

```
perception/face/config.py      thresholds, class mapping, paths — most tuning lives here
perception/face/detect.py      YuNet wrapper, bbox/IoU helpers
perception/face/emotion.py     eye-line alignment + FER+ 8→5 fold
perception/face/recognize.py   SFace embeddings + the enrolled-face database
perception/face/track.py       IoU tracker, identity voting, emotion smoothing
perception/face/pipeline.py    per-frame orchestration
perception/face/draw.py        overlay rendering
faces/faces.npz      enrolled embeddings (created on first enrol)
```

## Notes

- YuNet is pinned to the **2022mar** revision. The 2023mar rewrite changed the output
  head and will not parse under OpenCV < 4.8; this board has 4.5.4.
- This OpenCV build has no CUDA backend (`cudaarithm` etc. are listed unavailable), so
  everything is CPU. The Orin's 12 A78AE cores keep it camera-bound anyway.
- `faces/faces.npz` stores raw face embeddings. They're biometric data — treat the file
  accordingly if this leaves the bench.

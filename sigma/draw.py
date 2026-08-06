"""Overlay rendering for the live window."""
import cv2
import numpy as np

from . import config

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale=0.5, color=(255, 255, 255), thick=1):
    """Text with a dark outline so it stays readable over any background."""
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_result(frame, r, show_bars=True):
    x, y, w, h = r.box
    color = config.EMOTION_COLORS.get(r.emotion, config.EMOTION_COLORS["?"])
    known = r.name != config.UNKNOWN_NAME

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    name = f"{r.name} {r.sim:.2f}" if known else config.UNKNOWN_NAME
    _text(frame, name, (x, max(14, y - 24)), 0.55,
          (255, 255, 255) if known else (150, 150, 150))
    _text(frame, f"{r.emotion} {r.confidence * 100:.0f}%", (x, max(28, y - 6)), 0.6, color)

    if show_bars and r.probs is not None and float(np.sum(r.probs)) > 0:
        bx, by, bw = x + w + 8, y, 78
        if bx + bw > frame.shape[1]:            # flip to the left if off-screen
            bx = max(0, x - bw - 8)
        for i, name_ in enumerate(config.EMOTIONS):
            p = float(r.probs[i])
            top = by + i * 16
            c = config.EMOTION_COLORS[name_]
            cv2.rectangle(frame, (bx, top), (bx + bw, top + 11), (35, 35, 35), -1)
            cv2.rectangle(frame, (bx, top), (bx + int(bw * p), top + 11), c, -1)
            _text(frame, f"{name_[:4]} {p * 100:2.0f}", (bx + 2, top + 9), 0.32, (255, 255, 255))


def draw_hud(frame, fps, timings, n_faces, n_people, paused=False):
    h = frame.shape[0]
    cv2.rectangle(frame, (0, h - 46), (frame.shape[1], h), (25, 25, 25), -1)
    _text(frame, f"{fps:5.1f} FPS   det {timings['detect']:.0f}ms  "
                 f"rec {timings['recognize']:.0f}ms  emo {timings['emotion']:.0f}ms",
          (8, h - 28), 0.45, (200, 255, 200))
    _text(frame, f"faces {n_faces}   enrolled {n_people}   "
                 f"[e]nroll  [r]eset  [b]ars  [q]uit" + ("   PAUSED" if paused else ""),
          (8, h - 10), 0.45, (180, 180, 180))


def draw_banner(frame, lines, color=(255, 255, 255)):
    """Centred message box, used by the enrollment overlay."""
    if not lines:
        return
    fh, fw = frame.shape[:2]
    box_h = 22 * len(lines) + 16
    y0 = fh // 2 - box_h // 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (30, y0), (fw - 30, y0 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    for i, line in enumerate(lines):
        size = cv2.getTextSize(line, FONT, 0.6, 1)[0]
        _text(frame, line, ((fw - size[0]) // 2, y0 + 30 + i * 22), 0.6, color)

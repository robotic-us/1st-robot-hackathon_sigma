#!/usr/bin/env python3
"""Stimulus: the controlled "baby" -- a colored circle whose motion we author.

This runs on the tablet (or a second laptop window) and is what the webcam
watches.  Because we write the trajectory ourselves, every run is identical and
we know exactly what the circle did -- so any disagreement with what
``perception.py`` reported is a bug in *our* algorithm, not in the world.

Four phase kinds, composed into scenarios:

    calm         circle nearly still, centred
    stir_left    drifts left with a small oscillation
    stir_strong  large fast oscillation (high restlessness)
    settle       oscillation decays back to still

The trajectory is a pure function of time built from sums of sines -- no random
numbers anywhere -- so ``--record`` twice produces byte-identical video, and the
ground-truth CSV describes the video exactly.

Typical uses::

    # live, on the tablet or a second window (press f for fullscreen, q to quit)
    python3 stimulus.py --scenario demo --loop

    # produce the demo video + ground truth, no window needed
    python3 stimulus.py --scenario demo --record demo.mp4 --ground-truth demo.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from perception import PerceptionConfig, clamp, clamp01, ema_alpha

LOGGER = logging.getLogger("stimulus")

_FOURCC = getattr(cv2, "VideoWriter_fourcc", None) or cv2.VideoWriter.fourcc
_FONT = cv2.FONT_HERSHEY_SIMPLEX

PHASE_KINDS = ("calm", "stir_left", "stir_strong", "settle")

# Per-kind defaults.  ``amp`` and the drift target are in normalised screen units
# ([-1, 1], the same convention perception.py reports), ``freq`` is in Hz.
PHASE_DEFAULTS: dict[str, dict[str, float]] = {
    "calm":        {"amp": 0.010, "freq": 0.25, "drift_to": 0.00},
    "stir_left":   {"amp": 0.100, "freq": 1.20, "drift_to": -0.50},
    "stir_strong": {"amp": 0.350, "freq": 2.50, "drift_to": 0.00},
    "settle":      {"amp": 0.350, "freq": 1.20, "drift_to": 0.00},
}


# --------------------------------------------------------------------------- #
# Scenario description
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Phase:
    """One segment of a scenario.  ``None`` fields fall back to PHASE_DEFAULTS."""

    kind: str
    duration_s: float
    amp: Optional[float] = None       # oscillation amplitude, normalised units
    freq: Optional[float] = None      # oscillation frequency, Hz
    drift_to: Optional[float] = None  # x the centre eases toward by phase end

    def __post_init__(self) -> None:
        if self.kind not in PHASE_KINDS:
            raise ValueError(f"unknown phase kind {self.kind!r}; expected one of {PHASE_KINDS}")
        if self.duration_s <= 0.0:
            raise ValueError(f"phase {self.kind!r} needs a positive duration")


@dataclass(frozen=True)
class Segment:
    """A phase resolved onto the scenario timeline, with its defaults applied."""

    phase: Phase
    start_t: float
    end_t: float
    center_from: float
    center_to: float
    amp: float
    freq: float


@dataclass(frozen=True)
class Sample:
    """Where the circle is at some instant, plus what we know is true about it."""

    t: float
    phase: str
    x: float       # [-1, 1]
    y: float       # [-1, 1]
    amp: float     # the envelope in force at this instant (ground-truth "stirring")


# Built-in scenarios.  ``demo`` is the one to show; the single-phase ones are for
# tuning one behaviour at a time; ``boundary`` and ``sweep`` are test fixtures.
SCENARIOS: dict[str, list[Phase]] = {
    "demo": [
        Phase("calm", 6.0),
        Phase("stir_left", 8.0),
        Phase("stir_strong", 8.0),
        # Long enough for the whole wind-down ramp to play out: the decider needs
        # settle_hold_s to notice, then one dwell per step of the ramp.
        Phase("settle", 18.0),
    ],
    "calm": [Phase("calm", 12.0)],
    "stir_left": [Phase("stir_left", 12.0)],
    "stir_strong": [Phase("stir_strong", 12.0)],
    "settle": [Phase("stir_strong", 4.0), Phase("settle", 12.0)],
    # Parks the circle right on the LEFT/CENTER direction threshold, moving
    # enough to be actionable (so the "too calm to bother" gate does not mask the
    # test). The decider must latch one direction and stay there rather than
    # flip-flopping between slot 1 and slot 4.
    "boundary": [
        Phase("calm", 2.0, amp=0.06, freq=1.6, drift_to=-0.22),
        Phase("calm", 24.0, amp=0.06, freq=1.6, drift_to=-0.22),
    ],
    # Slow left -> centre -> right pass, for checking direction bucketing.
    "sweep": [
        Phase("calm", 3.0, drift_to=-0.6),
        Phase("calm", 6.0, drift_to=0.0),
        Phase("calm", 6.0, drift_to=0.6),
        Phase("calm", 6.0, drift_to=0.0),
    ],
}


def _smoothstep(u: float) -> float:
    """Ease-in/ease-out on [0, 1] -- keeps centre drifts from starting with a jerk."""
    u = clamp01(u)
    return u * u * (3.0 - 2.0 * u)


class ScenarioPlayer:
    """Turns a list of phases into a position lookup that is pure in ``t``.

    Being a pure function of absolute time (rather than an incremental
    simulation) is what makes the output deterministic and seekable: offline
    rendering and live playback produce the same trajectory.
    """

    def __init__(self, phases: Sequence[Phase]) -> None:
        if not phases:
            raise ValueError("a scenario needs at least one phase")
        self.phases = list(phases)
        self.segments: list[Segment] = []

        t = 0.0
        center = 0.0   # the circle starts centred
        amp = PHASE_DEFAULTS["calm"]["amp"]
        for phase in self.phases:
            defaults = PHASE_DEFAULTS[phase.kind]
            # ``settle`` with no explicit amplitude decays from whatever the
            # previous phase was doing -- that is what makes it read as a
            # wind-down rather than a fresh burst.
            if phase.amp is not None:
                seg_amp = phase.amp
            elif phase.kind == "settle":
                seg_amp = amp
            else:
                seg_amp = defaults["amp"]
            seg_freq = phase.freq if phase.freq is not None else defaults["freq"]
            center_to = phase.drift_to if phase.drift_to is not None else defaults["drift_to"]

            self.segments.append(
                Segment(
                    phase=phase,
                    start_t=t,
                    end_t=t + phase.duration_s,
                    center_from=center,
                    center_to=center_to,
                    amp=seg_amp,
                    freq=seg_freq,
                )
            )
            t += phase.duration_s
            center = center_to
            amp = seg_amp

        self.duration_s = t

    # -- sampling ---------------------------------------------------------- #
    def segment_at(self, t: float) -> Segment:
        for segment in self.segments:
            if t < segment.end_t:
                return segment
        return self.segments[-1]

    def sample(self, t: float) -> Sample:
        """Position at time ``t`` (seconds from the start of the scenario)."""
        t = clamp(t, 0.0, self.duration_s)
        seg = self.segment_at(t)
        local = t - seg.start_t
        u = clamp01(local / max(1e-6, seg.end_t - seg.start_t))

        center = seg.center_from + (seg.center_to - seg.center_from) * _smoothstep(u)
        env = self._envelope(seg, u)
        omega = 2.0 * math.pi * seg.freq

        x = center + env * math.sin(omega * local)
        # A slightly different vertical frequency turns a metronome into
        # something that reads as restless.
        y = env * 0.55 * math.sin(omega * 0.73 * local + 0.7)
        if seg.phase.kind == "calm":
            # A slow second sine: gentle "breathing" rather than dead-still.
            x += 0.4 * env * math.sin(2.0 * math.pi * 0.11 * local)

        return Sample(t=t, phase=seg.phase.kind, x=x, y=y, amp=env)

    @staticmethod
    def _envelope(seg: Segment, u: float) -> float:
        """Amplitude in force at progress ``u`` through the segment."""
        if seg.phase.kind == "settle":
            return seg.amp * math.exp(-5.0 * u)      # exponential wind-down
        if seg.phase.kind == "stir_strong":
            return seg.amp * min(1.0, u / 0.15)      # short ramp-in, no jump-scare
        return seg.amp


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
@dataclass
class RenderConfig:
    """How the circle is drawn.  Colors are BGR, matching OpenCV."""

    width: int = 1280
    height: int = 800
    fps: int = 30
    circle_bgr: tuple[int, int, int] = (0, 255, 0)   # bright green
    background_bgr: tuple[int, int, int] = (12, 12, 12)  # near-black
    radius_frac: float = 0.06   # circle radius as a fraction of frame height
    show_label: bool = False    # phase name in a corner (off by default: keep
    #                             the demo screen clean)

    @property
    def radius_px(self) -> int:
        return max(4, int(self.radius_frac * self.height))

    def to_normalised(self, x: float, y: float) -> tuple[float, float]:
        """Clamp so the circle never clips off the edge.

        A clipped circle would corrupt both its radius (and therefore
        ``distance``) and its centroid, so the drawn position -- not the
        requested one -- is what the ground truth must record.
        """
        mx = self.radius_px / (self.width * 0.5)
        my = self.radius_px / (self.height * 0.5)
        return clamp(x, -1.0 + mx, 1.0 - mx), clamp(y, -1.0 + my, 1.0 - my)

    def to_pixels(self, x: float, y: float) -> tuple[int, int]:
        """Normalised [-1, 1] -> pixel centre, kept fully on screen."""
        x, y = self.to_normalised(x, y)
        return (
            int(round((x + 1.0) * 0.5 * self.width)),
            int(round((y + 1.0) * 0.5 * self.height)),
        )


def render_frame(cfg: RenderConfig, sample: Sample) -> np.ndarray:
    """Draw one frame of the stimulus."""
    frame = np.full((cfg.height, cfg.width, 3), cfg.background_bgr, dtype=np.uint8)
    cx, cy = cfg.to_pixels(sample.x, sample.y)
    cv2.circle(frame, (cx, cy), cfg.radius_px, cfg.circle_bgr, -1, cv2.LINE_AA)
    if cfg.show_label:
        # Dim grey, bottom-left, far from the circle and well outside any
        # sensible HSV mask -- it cannot perturb detection.
        cv2.putText(
            frame, f"{sample.phase}  t={sample.t:5.1f}s",
            (20, cfg.height - 20), _FONT, 0.6, (90, 90, 90), 1, cv2.LINE_AA,
        )
    return frame


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #
class GroundTruthWriter:
    """Writes per-frame truth alongside the video.

    ``motion_true`` is computed with the *same* normalisation and smoothing
    ``perception.py`` uses, so the two columns are directly comparable: line up
    this CSV with what main.py logged and you can score the perception layer.
    """

    FIELDS = ("t", "phase", "x_true", "y_true", "motion_true", "amp_true")

    def __init__(self, path: str, render: RenderConfig,
                 perception: Optional[PerceptionConfig] = None) -> None:
        pcfg = perception or PerceptionConfig()
        self.render = render
        self.smoothing_s = pcfg.motion_smoothing_s
        self.fixed_alpha = pcfg.motion_alpha
        self.full_scale = pcfg.motion_full_scale
        self._motion = 0.0
        self._prev: Optional[tuple[float, float]] = None
        self._handle = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle)
        self._writer.writerow(self.FIELDS)

    def write(self, sample: Sample, dt: float) -> float:
        # Record the position as *drawn* (edge-clamped), which is what a camera
        # would see, and measure speed the same way CircleTracker does.
        x, y = self.render.to_normalised(sample.x, sample.y)
        measured = 0.0
        if self._prev is not None and dt > 1e-6:
            travelled = math.hypot(x - self._prev[0], y - self._prev[1])
            measured = clamp01((travelled / dt) / self.full_scale)
        self._prev = (x, y)
        self._motion += ema_alpha(dt, self.smoothing_s, self.fixed_alpha) * (measured - self._motion)
        self._writer.writerow([
            f"{sample.t:.4f}", sample.phase,
            f"{x:.5f}", f"{y:.5f}",
            f"{self._motion:.5f}", f"{sample.amp:.5f}",
        ])
        return self._motion

    def close(self) -> None:
        self._handle.close()


# --------------------------------------------------------------------------- #
# Playback modes
# --------------------------------------------------------------------------- #
def run_offline(
    player: ScenarioPlayer,
    cfg: RenderConfig,
    record_path: Optional[str],
    truth_path: Optional[str],
    preview: bool,
) -> int:
    """Deterministic, frame-indexed render: the ``--record`` / ``--ground-truth`` path.

    Steps time by exactly ``1/fps`` instead of following the wall clock, so the
    output is identical on any machine, however slow.
    """
    writer = None
    if record_path:
        writer = cv2.VideoWriter(record_path, _FOURCC(*"mp4v"), cfg.fps, (cfg.width, cfg.height))
        if not writer.isOpened():
            raise RuntimeError(f"could not open VideoWriter for {record_path!r}")

    truth = GroundTruthWriter(truth_path, cfg) if truth_path else None
    dt = 1.0 / cfg.fps
    total = int(round(player.duration_s * cfg.fps))

    if preview:
        cv2.namedWindow("stimulus (offline preview)", cv2.WINDOW_NORMAL)
    try:
        for i in range(total):
            sample = player.sample(i * dt)
            frame = render_frame(cfg, sample)
            if writer is not None:
                writer.write(frame)
            if truth is not None:
                truth.write(sample, dt)
            if preview:
                cv2.imshow("stimulus (offline preview)", frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
    finally:
        if writer is not None:
            writer.release()
        if truth is not None:
            truth.close()
        if preview:
            cv2.destroyAllWindows()

    LOGGER.info("rendered %d frames (%.1fs at %d fps)", total, player.duration_s, cfg.fps)
    if record_path:
        LOGGER.info("video:        %s", record_path)
    if truth_path:
        LOGGER.info("ground truth: %s", truth_path)
    return 0


def run_opencv(player: ScenarioPlayer, cfg: RenderConfig, loop: bool, fullscreen: bool) -> int:
    """Live playback in an OpenCV window.  ``f`` toggles fullscreen, ``q`` quits."""
    window = "stimulus"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    is_full = fullscreen

    frame_interval = 1.0 / cfg.fps
    start = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            elapsed = now - start
            if elapsed >= player.duration_s:
                if not loop:
                    break
                start = now
                elapsed = 0.0

            frame = render_frame(cfg, player.sample(elapsed))
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("f"):
                is_full = not is_full
                cv2.setWindowProperty(
                    window, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if is_full else cv2.WINDOW_NORMAL,
                )

            # Pace to the requested frame rate.
            slack = frame_interval - (time.monotonic() - now)
            if slack > 0:
                time.sleep(slack)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
    return 0


def run_pygame(player: ScenarioPlayer, cfg: RenderConfig, loop: bool, fullscreen: bool) -> int:
    """Live playback via pygame -- nicer fullscreen behaviour on a tablet."""
    try:
        import pygame  # imported lazily: pygame is an optional dependency
    except ImportError:  # pragma: no cover - depends on the environment
        LOGGER.error("pygame is not installed; use the default --backend opencv "
                     "or `pip install pygame`")
        return 2

    pygame.init()
    flags = pygame.FULLSCREEN if fullscreen else 0
    screen = pygame.display.set_mode((cfg.width, cfg.height), flags)
    pygame.display.set_caption("stimulus")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24) if cfg.show_label else None

    # pygame speaks RGB; our config is BGR to match OpenCV.
    circle_rgb = tuple(reversed(cfg.circle_bgr))
    background_rgb = tuple(reversed(cfg.background_bgr))
    is_full = fullscreen

    start = time.monotonic()
    running = True
    try:
        while running:
            elapsed = time.monotonic() - start
            if elapsed >= player.duration_s:
                if not loop:
                    break
                start = time.monotonic()
                elapsed = 0.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_f:
                        is_full = not is_full
                        screen = pygame.display.set_mode(
                            (cfg.width, cfg.height),
                            pygame.FULLSCREEN if is_full else 0,
                        )

            sample = player.sample(elapsed)
            screen.fill(background_rgb)
            pygame.draw.circle(screen, circle_rgb, cfg.to_pixels(sample.x, sample.y), cfg.radius_px)
            if font is not None:
                label = font.render(f"{sample.phase}  t={sample.t:5.1f}s", True, (90, 90, 90))
                screen.blit(label, (20, cfg.height - 40))
            pygame.display.flip()
            clock.tick(cfg.fps)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()
    return 0


# --------------------------------------------------------------------------- #
# Scenario files
# --------------------------------------------------------------------------- #
def load_scenarios(path: str) -> dict[str, list[Phase]]:
    """Load scenarios from JSON.

    Either a dict of ``{"name": [phase, ...]}`` or a bare list of phases (which
    becomes a scenario named ``"custom"``).  A phase is
    ``{"kind": ..., "duration_s": ..., "amp": ..., "freq": ..., "drift_to": ...}``.
    """
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)

    def to_phases(items: Sequence[dict]) -> list[Phase]:
        return [
            Phase(
                kind=item["kind"],
                duration_s=float(item["duration_s"]),
                amp=item.get("amp"),
                freq=item.get("freq"),
                drift_to=item.get("drift_to"),
            )
            for item in items
        ]

    if isinstance(raw, list):
        return {"custom": to_phases(raw)}
    return {name: to_phases(items) for name, items in raw.items() if not name.startswith("_")}


def parse_bgr(text: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected B,G,R (three 0-255 ints)")
    try:
        values = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"colour: {exc}") from exc
    return values  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", default="demo")
    parser.add_argument("--scenario-file", help="JSON file of extra/overriding scenarios")
    parser.add_argument("--list-scenarios", action="store_true")

    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--color", type=parse_bgr, default=(0, 255, 0),
                        help="circle colour as B,G,R (default bright green)")
    parser.add_argument("--bg", type=parse_bgr, default=(12, 12, 12),
                        help="background colour as B,G,R (default near-black)")
    parser.add_argument("--radius-frac", type=float, default=0.06,
                        help="circle radius as a fraction of frame height")
    parser.add_argument("--label", action="store_true", help="show the phase name on screen")

    parser.add_argument("--loop", action="store_true", help="repeat the scenario forever")
    parser.add_argument("--fullscreen", action="store_true", help="start fullscreen")
    parser.add_argument("--backend", choices=("opencv", "pygame"), default="opencv")

    parser.add_argument("--record", metavar="OUT.MP4", help="render the scenario to a video file")
    parser.add_argument("--ground-truth", metavar="OUT.CSV", help="write per-frame truth")
    parser.add_argument("--preview", action="store_true",
                        help="also show a window while rendering offline")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    scenarios = dict(SCENARIOS)
    if args.scenario_file:
        scenarios.update(load_scenarios(args.scenario_file))

    if args.list_scenarios:
        for name, phases in sorted(scenarios.items()):
            total = sum(p.duration_s for p in phases)
            kinds = " -> ".join(f"{p.kind}({p.duration_s:g}s)" for p in phases)
            print(f"{name:<12} {total:5.1f}s   {kinds}")
        return 0

    if args.scenario not in scenarios:
        LOGGER.error("unknown scenario %r; known: %s",
                     args.scenario, ", ".join(sorted(scenarios)))
        return 2

    player = ScenarioPlayer(scenarios[args.scenario])
    cfg = RenderConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        circle_bgr=args.color,
        background_bgr=args.bg,
        radius_frac=args.radius_frac,
        show_label=args.label,
    )
    LOGGER.info("scenario %r: %.1fs, %d phases", args.scenario, player.duration_s,
                len(player.segments))

    if args.record or args.ground_truth:
        return run_offline(player, cfg, args.record, args.ground_truth, args.preview)
    if args.backend == "pygame":
        return run_pygame(player, cfg, args.loop, args.fullscreen)
    return run_opencv(player, cfg, args.loop, args.fullscreen)


if __name__ == "__main__":
    raise SystemExit(main())

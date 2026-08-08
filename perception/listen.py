#!/usr/bin/env python3
"""Sound detection from the webcam's own microphone (ALSA card "WEBCAM").

Shells out to ``arecord`` for raw PCM (no sounddevice/pyaudio here).  ``level``
(0..1 RMS loudness) x ``cry`` (0..1 energy, 300-1500 Hz voice band) = distress;
report §4.2 decides a cry by duty cycle + vocal-unit length, never loudness.

    python3 perception/listen.py [--list]   # live meter / ALSA capture devices
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

LOGGER = logging.getLogger("listen")

SAMPLE_RATE = 16000
CHUNK = 1024               # 64 ms per read at 16 kHz
CRY_BAND_HZ = (300.0, 1500.0)   # voice/cry fundamental + first harmonics

# RMS that reads as level=1.0; speech at arm's length lands 0.05-0.15 here.
FULL_SCALE_RMS = 0.30


@dataclass(frozen=True)
class Sound:
    """What the microphone hears, smoothed."""

    level: float      # 0..1 loudness
    cry: float        # 0..1 fraction of energy in the voice band
    distress: float
    ts: float

    @property
    def loud_voice(self) -> bool:
        return self.distress > 0.15


def _analyse(samples: np.ndarray) -> tuple[float, float]:
    """One chunk -> (level, cry); pure, so tests.py can drive it directly."""
    if samples.size == 0:
        return 0.0, 0.0

    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    level = min(1.0, rms / FULL_SCALE_RMS)

    spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size))) ** 2
    freqs = np.fft.rfftfreq(samples.size, 1.0 / SAMPLE_RATE)
    total = float(spectrum.sum())
    if total <= 1e-12:
        return level, 0.0

    band = (freqs >= CRY_BAND_HZ[0]) & (freqs <= CRY_BAND_HZ[1])
    cry = float(spectrum[band].sum() / total)
    return level, cry


class Microphone:
    """Background reader over ``arecord``.  Store only -- never decides."""

    def __init__(
        self,
        device: str = "plughw:WEBCAM,0",
        smoothing: float = 0.3,
        enabled: bool = True,
    ) -> None:
        self.device = device
        self.smoothing = smoothing   # EMA weight per chunk; lower = steadier
        self.enabled = enabled
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = Sound(0.0, 0.0, 0.0, time.monotonic())
        self._level = 0.0
        self._cry = 0.0
        self.failed = False

    # -- lifecycle --------------------------------------------------------- #
    def start(self) -> None:
        if not self.enabled:
            LOGGER.info("microphone disabled")
            return
        cmd = [
            "arecord", "-D", self.device, "-f", "S16_LE",
            "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw", "-q",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            LOGGER.warning("arecord not found -- running without sound")
            self.failed = True
            return
        self._thread = threading.Thread(target=self._loop, name="mic", daemon=True)
        self._thread.start()
        LOGGER.info("microphone open on %s (%d Hz)", self.device, SAMPLE_RATE)

    def close(self) -> None:
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:  # pragma: no cover
                self._proc.kill()
            self._proc = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def __enter__(self) -> "Microphone":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- the reader thread -------------------------------------------------- #
    def _loop(self) -> None:
        need = CHUNK * 2  # 2 bytes per sample, S16_LE
        assert self._proc is not None and self._proc.stdout is not None
        while not self._stop.is_set():
            raw = self._proc.stdout.read(need)
            if not raw or len(raw) < need:
                if not self._stop.is_set():
                    LOGGER.warning("microphone stream ended")
                    self.failed = True
                return
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            level, cry = _analyse(samples)

            a = self.smoothing
            self._level += a * (level - self._level)
            self._cry += a * (cry - self._cry)
            with self._lock:
                self._latest = Sound(
                    level=self._level, cry=self._cry,
                    distress=self._level * self._cry, ts=time.monotonic(),
                )

    # -- public ------------------------------------------------------------- #
    def latest(self) -> Sound:
        with self._lock:
            return self._latest

def run_meter(device: str, seconds: float) -> int:
    """Live bar meter -- sanity-checks the mic and the thresholds."""
    with Microphone(device) as mic:
        if mic.failed:
            return 1
        deadline = time.monotonic() + seconds
        print("speak, shout, stay quiet -- Ctrl-C to stop\n")
        print(f"{'level':>7} {'cry':>6} {'distress':>9}  bar")
        try:
            while time.monotonic() < deadline:
                s = mic.latest()
                bar = "#" * int(s.distress * 50)
                flag = "  <-- LOUD VOICE" if s.loud_voice else ""
                print(f"{s.level:>7.3f} {s.cry:>6.3f} {s.distress:>9.3f}  "
                      f"{bar}{flag}", flush=True)
                time.sleep(0.15)
        except KeyboardInterrupt:
            print()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="plughw:WEBCAM,0",
                        help="ALSA capture device (see --list)")
    parser.add_argument("--seconds", type=float, default=1e9)
    parser.add_argument("--list", action="store_true", help="show capture devices")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")

    if args.list:
        subprocess.run(["arecord", "-l"], check=False)
        return 0
    return run_meter(args.device, args.seconds)


if __name__ == "__main__":
    raise SystemExit(main())

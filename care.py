#!/usr/bin/env python3
"""SIGMA: webcam + microphone -> emotion -> the robot responds.

The whole product, in one readable loop::

    camera ─┐
            ├─► sense.py ─► Reading(x, distress) ─► pick a slot ─► robot.play()
    mic   ──┘

Read this file top to bottom and you have the system.  The heavy lifting lives
behind three small interfaces, each testable on its own:

    sense.Sense          frames + sound  -> Reading            (sense.py)
    ChunkMatcher         candidates      -> the best one       (dream.py)
    RobotInterface       slot id         -> the arm moves      (phorce_iface.py)

The mapping is deliberately simple, because a rule you can explain at a booth
beats a model you cannot:

    WHERE the person is  (x)         picks a direction  left / center / right
    HOW upset they are   (distress)  picks an amplitude small / medium / large

which lands on one cell of the 3x3 slot grid in slots.json.  Two guards keep it
from looking twitchy: a calm floor (a content person is left alone) and a dwell
timer (one motion at a time, no thrashing).

Run it::

    python3 care.py --mock                  # no robot, no ROS -- start here
    python3 care.py --mock --dream          # + DREAM-Chunk candidate matching
    python3 care.py                         # real robot; check nobody is near it
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Optional

import cv2

from slot_table import load_slot_table
from listen import Microphone
from phorce_iface import PlayOutcome, make_robot
from sense import Reading, Sense, draw
from sigma import config as face_config
from sigma.pipeline import SigmaPipeline

LOGGER = logging.getLogger("care")

# --- the mapping ----------------------------------------------------------- #
CALM_FLOOR = 0.20      # below this distress nobody needs handling
DEAD_BAND = 0.25       # |x| inside this counts as "centred"
MEDIUM_AT = 0.45       # distress thresholds for the amplitude bucket
LARGE_AT = 0.70
MIN_DWELL_S = 2.5      # gap between motions; halved when distress is high


def pick_direction(x: float) -> str:
    """Which way to lean, from where the face is."""
    if x < -DEAD_BAND:
        return "left"
    if x > DEAD_BAND:
        return "right"
    return "center"


def pick_amplitude(distress: float) -> str:
    """How big a motion, from how upset the person is."""
    if distress >= LARGE_AT:
        return "large"
    if distress >= MEDIUM_AT:
        return "medium"
    return "small"


class Care:
    """Decides, at about 2 Hz, which motion slot to play."""

    def __init__(self, table, robot, matcher=None) -> None:
        self.table = table
        self.robot = robot
        self.matcher = matcher          # None -> pick the lowest slot id
        self.last_play_ts = -1e9
        self.last_slot: Optional[int] = None
        self.reason = "starting up"

    def step(self, reading: Reading, now: float) -> Optional[int]:
        """Return a slot to play, or None.  Pure decision -- it sends nothing."""
        if reading.distress < CALM_FLOOR:
            self.reason = f"calm ({reading.distress:.2f} < {CALM_FLOOR})"
            return None

        # A face tells us where to lean; sound alone cannot, so we answer a
        # cry from an unseen person with a centred motion.
        direction = pick_direction(reading.x) if reading.present else "center"
        amplitude = pick_amplitude(reading.distress)

        candidates = [
            s.slot_id for s in self.table.slots.values()
            if s.direction.value == direction and s.amplitude.value == amplitude
        ]
        if not candidates:
            self.reason = f"no slot for {direction}/{amplitude}"
            return None

        # Urgency shortens the dwell: a very upset person should not wait.
        dwell = MIN_DWELL_S * (0.5 if reading.distress >= LARGE_AT else 1.0)
        waited = now - self.last_play_ts
        if waited < dwell:
            self.reason = f"dwell {dwell - waited:.1f}s to go"
            return None

        slot = self._choose(candidates, reading)
        self.reason = f"{direction}/{amplitude} -> slot {slot}"
        self.last_play_ts = now
        self.last_slot = slot
        return slot

    def _choose(self, candidates: list[int], reading: Reading) -> int:
        """One candidate cell may hold several slots.  DREAM-Chunk breaks the tie.

        Without a matcher we take the lowest id, which is deterministic and
        perfectly adequate while each cell holds exactly one slot.
        """
        if self.matcher is None or len(candidates) == 1:
            return min(candidates)

        state = self.robot.latest()
        joints = state.positions(self.table.axes) if state else None
        forces = state.external_force(self.table.axes) if state else None
        best = self.matcher.best(
            candidates, joints, external_force_a=forces,
            target_amplitude=reading.distress,
        )
        return best.slot_id if best is not None else min(candidates)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mock", action="store_true", help="no robot, no ROS")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--audio-device", default="plughw:WEBCAM,0")
    parser.add_argument("--no-sound", action="store_true", help="face only")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--slot-table", default="slots.json")
    parser.add_argument("--target", default="robot", help="real mode phorce target")
    parser.add_argument("--dry-run", action="store_true", help="decide but never play")
    parser.add_argument("--decide-hz", type=float, default=2.0)
    parser.add_argument("--dream", action="store_true",
                        help="use DREAM-Chunk to choose between candidate slots")
    parser.add_argument("--motion-map", help="MotionMap.csv (implies --dream)")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)-8s %(message)s", datefmt="%H:%M:%S",
    )

    table = load_slot_table(args.slot_table)

    matcher = None
    if args.dream or args.motion_map:
        from dream import ChunkMatcher, DreamConfig
        from pvector import load_chunk_dictionary
        matcher = ChunkMatcher(
            load_chunk_dictionary(motion_map=args.motion_map,
                                  slot_table=args.slot_table, axes=table.axes),
            DreamConfig(),
        )

    if not args.mock:
        LOGGER.warning("REAL ROBOT MODE (target=%s) -- check that nobody and nothing "
                       "is near the robot before it moves", args.target)
    robot = make_robot(mock=args.mock, target=args.target, axes=table.axes)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        LOGGER.error("could not open camera %d", args.camera_index)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, face_config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, face_config.FRAME_H)

    mic = None if args.no_sound else Microphone(args.audio_device)
    sense = Sense(SigmaPipeline(), mic)
    care = Care(table, robot, matcher)
    window = "SIGMA - care"

    plays = busy = frames = 0
    decide_period = 1.0 / max(0.1, args.decide_hz)
    next_decision = 0.0

    LOGGER.info("camera %d | %s | robot %s%s", args.camera_index,
                "no sound" if args.no_sound else args.audio_device,
                "MOCK" if args.mock else f"REAL({args.target})",
                " | DREAM-Chunk" if matcher else "")

    try:
        robot.start()
        if mic is not None:
            mic.start()

        while True:
            ok, frame = cap.read()
            if not ok:
                LOGGER.info("camera stopped")
                break
            frames += 1

            # Fast: look. Every frame.
            reading = sense.update(frame)

            # Slow: decide. play() is never called at camera rate -- the robot
            # takes one motion at a time and has no queue.
            now = time.monotonic()
            if now >= next_decision:
                next_decision = now + decide_period
                slot = care.step(reading, now)
                if slot is None:
                    LOGGER.debug("SKIP  %s | %s", reading.describe(), care.reason)
                elif args.dry_run:
                    LOGGER.info("WOULD PLAY %s | %s", reading.describe(), care.reason)
                else:
                    outcome = robot.play(slot)
                    if outcome is PlayOutcome.BUSY:
                        busy += 1
                        LOGGER.debug("BUSY  %s", care.reason)
                    else:
                        plays += 1
                        LOGGER.info("PLAY  %s | %s", reading.describe(), care.reason)

            if not args.no_window:
                cv2.imshow(window, draw(frame, reading, sense))
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        print()
    finally:
        cap.release()
        if mic is not None:
            mic.close()
        robot.close()
        if not args.no_window:
            cv2.destroyAllWindows()

    LOGGER.info("%d frames, %d motions played, %d deferred as BUSY", frames, plays, busy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

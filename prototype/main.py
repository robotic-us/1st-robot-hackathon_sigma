#!/usr/bin/env python3
"""Wire the pieces together: camera -> perception -> decider -> play(slot).

Two rates, deliberately separated:

    fast (camera rate, ~30 Hz)   grab a frame, track the circle, store the result.
                                 Also where the robot's 1 kHz feedback callback
                                 lands. Stores only -- never decides, never sends.

    slow (~2 Hz)                 read the latest perception + joint state, decide,
                                 and play a slot.

Mixing them is the classic mistake: the robot plays one motion at a time and has
no queue, so a play() call from the fast path would be rejected as BUSY roughly
a thousand times a second. **play() is never called from the fast loop.**

Threading note: the fast loop runs on the *main* thread because ``cv2.imshow``
must, on macOS. The decision loop is the side thread, and it only ever reads a
lock-protected snapshot.

Run it today, no robot and no ROS::

    # terminal A -- the stimulus on a tablet or a second window
    python3 stimulus.py --scenario demo --loop

    # terminal B -- the eyes and brain, webcam pointed at that screen
    python3 main.py --mock --show-debug

Or with no camera at all::

    python3 stimulus.py --scenario demo --record demo.mp4
    python3 main.py --mock --video demo.mp4 --show-debug
"""

from __future__ import annotations

import sys as _sys, pathlib as _pathlib
# This module lives in prototype/ but imports the shared robot layer from the
# project root (dream, pvector, phorce_iface, slot_table).
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import logging
import threading
import time
from typing import Optional

import cv2

from decider import Decider, DeciderConfig, load_slot_table
from dream import ChunkMatcher, ChunkMockModel, DreamConfig, DreamMonitor
from pvector import load_chunk_dictionary
from perception import (
    CircleTracker,
    HSVRange,
    Perception,
    PerceptionConfig,
    VideoFileSource,
    draw_overlay,
    open_source,
)
from phorce_iface import PlayOutcome, RobotInterface, make_robot

LOGGER = logging.getLogger("main")


class Pipeline:
    """Owns the shared state between the fast and slow loops."""

    def __init__(
        self,
        tracker: CircleTracker,
        decider: Decider,
        robot: RobotInterface,
        decide_hz: float,
        dry_run: bool,
        monitor: Optional[DreamMonitor] = None,
        matcher: Optional[ChunkMatcher] = None,
        axes: tuple[int, ...] = (),
    ) -> None:
        self.tracker = tracker
        self.decider = decider
        self.robot = robot
        self.decide_period = 1.0 / max(0.1, decide_hz)
        self.dry_run = dry_run
        # DREAM-Chunk. The monitor is fed from the fast loop (it only stores);
        # the matcher is consulted by the decider on the slow tick.
        self.monitor = monitor
        self.matcher = matcher
        self.axes = axes
        self._pending_slot: Optional[int] = None

        self._lock = threading.Lock()
        self._latest: Optional[Perception] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Counters for the closing summary.
        self.frames = 0
        self.decisions = 0
        self.plays_sent = 0
        self.plays_busy = 0
        self.divergences = 0

    # -- fast side --------------------------------------------------------- #
    def publish(self, perc: Perception) -> None:
        """Called from the fast loop.  Store the latest value and return."""
        with self._lock:
            self._latest = perc
        self.frames += 1

    def poll_dream(self) -> None:
        """Compare live feedback against the executing chunk's dream.

        Called from the fast loop.  Stores a verdict and returns -- it never
        plays anything, which is what keeps the two-rate rule intact.
        """
        if self.monitor is None:
            return
        state = self.robot.latest()
        if state is None:
            return
        running = self.robot.is_motion_active()

        # Anchor the dream on the first frame where the motion is actually
        # running, not on the pose we sampled when we decided to send it.
        # play() is asynchronous, so those are different moments -- and a dream
        # anchored half a decision-period early diverges from its own start.
        if self._pending_slot is not None and running:
            chunk = self.matcher.chunks.get(self._pending_slot) if self.matcher else None
            start = state.positions(self.axes)
            if chunk is not None and start is not None:
                self.monitor.begin(chunk, start, time.monotonic())
            self._pending_slot = None

        if not self.monitor.active:
            return
        if not running:
            # The motion finished. Keeping the comparison alive past the end
            # would score the dream against whatever the arm does next, which
            # is not the dream's fault.
            report = self.monitor.latest()
            if self.matcher is not None and report.samples > 0:
                self.matcher.note_residual([report.rms_rad] * len(self.axes))
            LOGGER.debug("chunk finished: %s", report.describe())
            self.monitor.end()
            return

        before = self.monitor.latest().diverged
        report = self.monitor.update(
            state.positions(self.axes),
            time.monotonic(),
            external_force_a=state.external_force(self.axes),
        )
        if report.diverged and not before:
            self.divergences += 1
            LOGGER.warning("DREAM DIVERGED  %s", report.describe())

    def _snapshot(self) -> Optional[Perception]:
        with self._lock:
            return self._latest

    # -- slow side --------------------------------------------------------- #
    def start(self) -> None:
        self._thread = threading.Thread(target=self._decision_loop, name="decide", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _decision_loop(self) -> None:
        """~2 Hz: look at the latest state, decide, maybe play."""
        while not self._stop.is_set():
            time.sleep(self.decide_period)

            perc = self._snapshot()
            if perc is None:
                continue  # no frame yet

            joint_state = self.robot.latest()
            divergence = self.monitor.latest() if self.monitor is not None else None
            slot = self.decider.decide(perc, joint_state, divergence=divergence)
            decision = self.decider.explain()

            if slot is None:
                # Skips are the common case (dwell, dead-band, calm) -- keep them
                # at DEBUG so the INFO log is a clean record of what was played.
                LOGGER.debug("SKIP   %s", decision.describe())
                continue

            self.decisions += 1
            if self.dry_run:
                LOGGER.info("DECIDE %s  [dry-run, not sent]", decision.describe())
                continue

            outcome = self.robot.play(slot)
            if outcome is PlayOutcome.BUSY:
                self.plays_busy += 1
                # No queue on the robot, so there is nothing to do but try again
                # on the next tick. Expected and harmless.
                LOGGER.debug("DECIDE %s -> BUSY, will retry", decision.describe())
            else:
                self.plays_sent += 1
                LOGGER.info("DECIDE %s -> play(%d) %s",
                            decision.describe(), slot, outcome.value.upper())
                self._begin_dream(slot, joint_state)

    def _begin_dream(self, slot: int, joint_state) -> None:
        """Queue a chunk for dreaming; the fast loop anchors it once it starts."""
        if self.monitor is None or self.matcher is None:
            return
        self.monitor.end()
        self._pending_slot = slot


def on_play_result(slot_id: int, outcome: PlayOutcome) -> None:
    """Terminal result of a motion, delivered from the robot's worker thread."""
    if outcome is PlayOutcome.OK:
        LOGGER.debug("motion %d completed", slot_id)
    elif outcome is not PlayOutcome.NEEDS_OPERATOR:
        # NEEDS_OPERATOR already logs a loud warning inside phorce_iface.
        LOGGER.warning("motion %d ended: %s", slot_id, outcome.value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --- the specced flags ------------------------------------------------- #
    parser.add_argument("--mock", action="store_true",
                        help="use the mock robot (no ROS, no phorce, no hardware)")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--show-debug", action="store_true",
                        help="show the perception overlay window")
    parser.add_argument("--slot-table", default="slots.json")
    parser.add_argument("--hsv-range", type=HSVRange.parse, default=None,
                        help="lo_h,lo_s,lo_v,hi_h,hi_s,hi_v "
                             "(get one from `perception.py --calibrate`)")

    # --- extras ------------------------------------------------------------ #
    parser.add_argument("--video", help="read frames from a file instead of the camera")
    parser.add_argument("--once", action="store_true",
                        help="with --video: play it through once and exit "
                             "(default is to loop, for demos)")
    parser.add_argument("--decide-hz", type=float, default=2.0)
    parser.add_argument("--target", default="robot",
                        help="real-mode phorce target: 'robot' or e.g. 'sim:demo'")
    parser.add_argument("--dry-run", action="store_true",
                        help="decide and log, but never send play()")
    parser.add_argument("--mock-reject-rate", type=float, default=0.0,
                        help="mock only: fraction of play() calls that are rejected")
    parser.add_argument("--mock-jam-after", type=float, default=0.0, metavar="SECONDS",
                        help="mock only: jam an axis after N seconds and report a large "
                             "dob_a, to exercise DREAM-Chunk divergence with no robot")
    parser.add_argument("--mock-jam-axis", type=int, default=0)

    # --- DREAM-Chunk -------------------------------------------------------- #
    parser.add_argument("--dream", action="store_true",
                        help="enable DREAM-Chunk: dream each candidate slot from the "
                             "measured pose and match against live feedback")
    parser.add_argument("--motion-map",
                        help="the robot's MotionMap.csv (implies --dream). Without it "
                             "the chunk dictionary is synthesised from the slot table")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)-12s %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- perception -------------------------------------------------------- #
    perception_config = PerceptionConfig(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
    )
    if args.hsv_range is not None:
        perception_config.hsv = args.hsv_range
    tracker = CircleTracker(perception_config)

    # --- decision ---------------------------------------------------------- #
    table = load_slot_table(args.slot_table)

    matcher: Optional[ChunkMatcher] = None
    monitor: Optional[DreamMonitor] = None
    mock_model: Optional[ChunkMockModel] = None
    if args.dream or args.motion_map:
        chunks = load_chunk_dictionary(
            motion_map=args.motion_map, slot_table=args.slot_table, axes=table.axes,
        )
        dream_config = DreamConfig()
        matcher = ChunkMatcher(chunks, dream_config)
        monitor = DreamMonitor(dream_config)
        # Only the mock needs this; the real robot follows its own SD-card
        # trajectories and needs no help pretending.
        mock_model = ChunkMockModel(chunks) if args.mock else None
        LOGGER.info("DREAM-Chunk enabled: %d chunks, dt=%.3fs, horizon=%.1fs, "
                    "diverge at %.3f rad held %.2fs",
                    len(chunks), dream_config.dt, dream_config.horizon_s,
                    dream_config.diverge_rad, dream_config.diverge_hold_s)

    decider = Decider(table, DeciderConfig(), matcher=matcher)

    # --- robot ------------------------------------------------------------- #
    if not args.mock:
        LOGGER.warning("REAL ROBOT MODE (target=%s) -- check that nobody and "
                       "nothing is near the robot before it moves", args.target)
    robot = make_robot(
        mock=args.mock,
        target=args.target,
        axes=table.axes,
        reject_rate=args.mock_reject_rate,
        jam_after_s=args.mock_jam_after,
        jam_axis=args.mock_jam_axis,
        motion_model=mock_model,
        on_play_result=on_play_result,
    )

    source = open_source(perception_config, video=args.video, loop=not args.once)
    # A video file must be played at its own rate: motion is a speed, so decoding
    # flat out would report movement several times faster than it really is.
    frame_interval = 1.0 / source.fps if isinstance(source, VideoFileSource) else 0.0

    LOGGER.info("source=%s  robot=%s  decide=%.1f Hz  hsv=%s%s",
                args.video or f"camera {args.camera_index}",
                "MOCK" if args.mock else f"REAL({args.target})",
                args.decide_hz, perception_config.hsv.as_cli(),
                "  [DRY RUN]" if args.dry_run else "")

    pipeline = Pipeline(tracker, decider, robot, args.decide_hz, args.dry_run,
                        monitor=monitor, matcher=matcher, axes=table.axes)
    window = "perception + decision"

    try:
        robot.start()
        pipeline.start()
        if args.show_debug:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        while True:
            loop_start = time.monotonic()

            # ---- fast loop: grab, track, store. Nothing else. -------------- #
            ok, frame = source.read()
            if not ok or frame is None:
                LOGGER.info("frame source exhausted")
                break
            if getattr(source, "wrapped", False):
                # A looping file jumps backwards in time; stale history would
                # produce one enormous bogus motion spike.
                tracker.reset()
                decider.reset()

            perc = tracker.update(frame)
            pipeline.publish(perc)
            pipeline.poll_dream()

            if args.show_debug:
                cv2.imshow(window, draw_overlay(frame, perc, tracker))
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break

            if frame_interval:
                slack = frame_interval - (time.monotonic() - loop_start)
                if slack > 0:
                    time.sleep(slack)

    except KeyboardInterrupt:
        print()  # keep the summary off the ^C line
        LOGGER.info("interrupted")
    finally:
        pipeline.stop()
        robot.close()
        source.close()
        if args.show_debug:
            cv2.destroyAllWindows()

    LOGGER.info(
        # "dispatched" not "succeeded": play() is asynchronous, so the terminal
        # outcome arrives later (and any rejection has already logged a WARNING).
        "%d frames, %.1f FPS, %d decisions, %d plays dispatched, %d deferred as BUSY, "
        "%d dream divergences",
        pipeline.frames, tracker.fps, pipeline.decisions,
        pipeline.plays_sent, pipeline.plays_busy, pipeline.divergences,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

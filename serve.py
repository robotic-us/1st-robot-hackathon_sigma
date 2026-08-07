#!/usr/bin/env python3
"""The evidence-report cradle machine, live in a web browser.

One stdlib HTTP server:

    /            the dashboard (web/index.html; /style.css and /app.js beside it)
    /events      Server-Sent Events: the whole state as JSON, ~20 Hz
    /history     the last ~15 min as 1 Hz samples (seeds the emotion timeline)
    /frame       MJPEG camera stream with the sensing overlay
    /baby        full-screen virtual infant + iPad motion capture
    /motion-sensor  iPad -> server motion feature packets (POST)
    /motions     the M01-M50 library from the evidence report, once
    /jam         simulate a mechanism fault (trips the safety gate)
    /motion      queue a library command by id (R grade needs --research)
    /auto        the report's state machine on/off (?set=on|off)
    /policy      switch the decision brain live (?set=off|reflex|ollama|claude)
    /taste       retune the virtual infant's hidden temperament (--baby only)

Motion behaviour follows docs/infant_robotic_cradle_evidence_report_ko.pdf
(see core/cradle.py): the default is *not moving*.  Two sensing modes feed
the CradleMachine's 0..1 distress input:

* **Verification scenario** (``--verify``, alias ``--fake``, and the default):
  no camera.  A scripted nursery episode asserts its state and level phase by
  phase and drives the real CradleMachine with them -- the ladder, the trials,
  the taper and the gate are all exercised.  The camera panel shows a drawn
  baby acting the script, clearly watermarked as synthetic.  Since the
  infant-face stack was removed (2026-08-08) the phases are scripted ground
  truth rather than judged signals; docs/VERIFY.md says what that costs.
* **Virtual infant** (``--baby``, closed-loop sim): a random state process
  (perception/baby.py) that the sway genuinely soothes -- or, for hunger
  cries, does not.

Either way the CradleMachine runs the report's ladder over that state: gate
first, sleep taper, quiet hold, 30 s sway trials with improvement checks and
caregiver alerts.  The engine's plate offset drives the same joint angles the
canvas and RViz already animate; serve.py itself stays 2D and leaves the 3D
view to the viz.

No new dependencies: SSE and MJPEG are both plain HTTP, which is why the
stdlib server is enough.  Open it from any laptop on the same network.

Run::

    python3 serve.py --verify      # no camera: the acted verification episode
    python3 serve.py --baby        # no camera: a virtual infant, closed loop
    python3 serve.py --research    # unlock the R-grade modes (sim only!)
    python3 tests.py               # all suites, incl. these endpoints
"""

from __future__ import annotations

import argparse
import json
import math
import random
import socket
import ssl
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from core.cradle import LIBRARY_BY_ID, CradleMachine, MotionEngine, catalog
from core.rig import cradle_angles, cradle_joint_state
from perception import nubzuki as nz
# The camera framing lives with the recognizer, so serve.py and
# `nubzuki.py --camera` cannot drift into reading different pixels.
from perception.nubzuki import centre_region, crop_frame, magnify, parse_crop

WEB_DIR = Path(__file__).resolve().parent / "web"

# The served files, as a fixed whitelist -- not a static directory, so there is
# nothing to traverse.  The font entries are built from what web/fonts/ holds
# at import (the IBM Plex subsets ship with the repo so the dashboard keeps its
# type on a nursery LAN with no route to a font CDN).
STATIC: dict[str, tuple[str, str]] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/baby": ("baby.html", "text/html; charset=utf-8"),
    "/baby.css": ("baby.css", "text/css; charset=utf-8"),
    "/baby.js": ("baby.js", "text/javascript; charset=utf-8"),
}
STATIC.update({f"/fonts/{p.name}": (f"fonts/{p.name}", "font/woff2")
               for p in sorted((WEB_DIR / "fonts").glob("*.woff2"))})


class IPadMotion:
    """Latest physical cradle motion reported by the tablet web page.

    Browser ``DeviceMotionEvent.acceleration`` is expressed in m/s^2 and its
    rotation rate in deg/s.  The tablet reduces its high-rate samples into a
    100 ms feature packet; the server deliberately accepts only a small,
    finite numeric surface.  A stale tablet can never keep soothing the
    virtual infant: ``fresh`` expires after one second.
    """

    FRESH_S = 1.0
    MAX_BODY = 4096

    def __init__(self) -> None:
        self.received_at = 0.0
        self.samples = 0
        self.accel_rms = 0.0
        self.rotation_rms = 0.0
        self.jerk_rms = 0.0
        self.dominant_hz = 0.0
        self.permission = "waiting"

    @staticmethod
    def _number(payload: dict, key: str, high: float) -> float:
        value = float(payload.get(key, 0.0))
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        return max(0.0, min(high, value))

    def update(self, payload: dict, now: float) -> None:
        if not isinstance(payload, dict):
            raise ValueError("JSON object required")
        self.samples = int(self._number(payload, "samples", 500.0))
        self.accel_rms = self._number(payload, "accelRms", 20.0)
        self.rotation_rms = self._number(payload, "rotationRms", 2000.0)
        self.jerk_rms = self._number(payload, "jerkRms", 500.0)
        self.dominant_hz = self._number(payload, "dominantHz", 20.0)
        permission = str(payload.get("permission", "granted"))[:24]
        self.permission = permission
        self.received_at = now

    def fresh(self, now: float) -> bool:
        return self.received_at > 0.0 and now - self.received_at <= self.FRESH_S

    def strength(self, now: float) -> float:
        """0..1 measured motion exposure for the virtual infant.

        0.12 m/s^2 RMS is a useful demo-scale reference for the gentle
        horizontal library; angular motion also counts for AP/tilt entries.
        This is a virtual-infant input, never a physical safety measurement.
        """
        if not self.fresh(now):
            return 0.0
        linear = self.accel_rms / 0.12
        angular = self.rotation_rms / 8.0
        return max(0.0, min(1.0, max(linear, angular)))

    def snapshot(self, now: float) -> dict:
        age = None if self.received_at == 0.0 else max(0.0, now - self.received_at)
        return {
            "connected": self.fresh(now),
            "age_s": None if age is None else round(age, 2),
            "samples": self.samples,
            "accel_rms": round(self.accel_rms, 4),
            "rotation_rms": round(self.rotation_rms, 3),
            "jerk_rms": round(self.jerk_rms, 3),
            "dominant_hz": round(self.dominant_hz, 3),
            "strength": round(self.strength(now), 3),
            "permission": self.permission,
            # Physically DERIVED figures for the demo: with the rig playing
            # SD-card slots the engine's envelope numbers describe a screen
            # motion, not the machine -- but this tablet rides the cradle, so
            # its accelerometer is ground truth.  Peak push is direct
            # (rms*sqrt2); travel assumes a rocking sinusoid A = a/(2*pi*f)^2
            # and is only offered when a dominant frequency exists -- shaken
            # by hand it would be a nonsense number, so it is None instead.
            "meas_peak_g": round(self.accel_rms * 1.414 / 9.80665, 4),
            "meas_travel_mm": (
                round(1000.0 * self.accel_rms * 1.414
                      / (2.0 * math.pi * self.dominant_hz) ** 2, 1)
                if self.dominant_hz >= 0.15 else None),
        }


# --------------------------------------------------------------------------- #
# Shared state between the sensor loop and the HTTP handlers
# --------------------------------------------------------------------------- #
class Shared:
    def __init__(self, allow_research: bool = False,
                 pace_s: float = 10.0, give_up: bool = False) -> None:
        self.engine = MotionEngine(allow_research=allow_research)
        # Demo rhythm: each motion gets ~10 s before the machine re-decides
        # (the report spec's own cadence is 30 -- serve.py --pace 30).
        # The report's §5 hand-over is off here by default: on the stand the
        # cradle should keep trying motions rather than park and call for a
        # human every time a stretch of soothing does not land.  --give-up
        # restores it.  The safety gate is unaffected either way.
        self.machine = CradleMachine(self.engine, check_every_s=pace_s,
                                     give_up=give_up)
        self.jam = False              # simulated mechanism fault (the gate)
        self.pose = [0.0, 0.0, 0.0, 0.0]   # crank angles the viz mirrors
        self.lock = threading.Lock()
        self.state_json = b"{}"
        self.jpeg: Optional[bytes] = None
        self.events: deque[str] = deque(maxlen=14)
        self.motion_request: Optional[str] = None
        self.policy = None      # core/policy.py SoothePolicy when a brain is on
        self.llm_model = None   # --llm-model override for ollama/claude
        self.baby = None        # the VirtualBaby, so /taste can retune it
        self.play_slot = None   # --play <number>: hold one raw SD slot
        self.play_once = False  # --once: clear the hold after one success
        self.face = None        # --sense: the plant's truth, drawn on the iPad
        self.ipad_motion = IPadMotion()
        # 1 Hz samples for the dashboard's emotion timeline: a fresh page
        # seeds the last ~15 min from /history instead of starting empty.
        self.history: deque[dict] = deque(maxlen=900)
        self.viz_gain = 1.0     # RViz display exaggeration; 1.0 = honest
        self.offsets = (0.0, 0.0, 0.0)   # last (ap, ml, z) the engine produced
        self.stop = threading.Event()

    def log(self, text: str) -> None:
        self.events.appendleft(f"{time.strftime('%H:%M:%S')}  {text}")


# --------------------------------------------------------------------------- #
# The verification scenario (docs/VERIFY.md layer 1, live)
# --------------------------------------------------------------------------- #
def infant_level(reading) -> float:
    """The machine's 0..1 distress input."""
    return float(reading.distress) if reading.present else 0.0


# All 17 poses, partitioned into the plant's four states by the reference
# sheet's own circumplex (valence x arousal).  The drawn face rotates inside
# a state's pool (web/baby.js STATE_POOL mirrors this); the reading maps any
# of them straight back to the state word, so the round-trip stays exact.
POSE_STATE = {
    "neutral": "CALM", "sitHeart": "CALM", "lounging": "CALM",
    "nerdy": "CALM", "bashful": "CALM", "kiss": "CALM",
    "dancing": "CALM", "star": "CALM",
    "gloomy": "FUSS", "sick": "FUSS", "cool": "FUSS", "shocked": "FUSS",
    "crying": "CRY", "angry": "CRY", "rage": "CRY",
    "sleeping": "SLEEP", "dreaming": "SLEEP",
}


def mascot_reading(seen: list) -> SimpleNamespace:
    """Camera sightings -> the machine's sensor contract.

    The vision link, in one place so tests can hit it without a camera.  The
    largest figure in frame is the reading (the iPad fills most of the view;
    anything smaller is clutter).  ``distress`` is the sighting's
    *asserted* level -- capped inside the fuss band by construction, because
    §5 says vision alone may never cross CRY_LEVEL -- and an empty frame maps
    to UNKNOWN with ``present=False``, which is the safety gate's input, not
    a distress claim.

    "Largest" skips any figure that *encloses* another, because that shape is
    a frame and not a mascot.  Measured, not guessed: whenever the surround is
    darker than the page -- an iPad bezel, a dim nursery -- the ink mask keeps
    the ring around the page, it passes ``is_nubzuki`` and gets confidently
    named, and at ~4x Nubzuki's area it would otherwise *be* the reading.
    """
    if not seen:
        return SimpleNamespace(present=False, distress=0.0,
                               emotion=nz.UNKNOWN, alarm=False,
                               name="nubzuki", x=0.0, phase="")

    def encloses(a, b) -> bool:
        ax, ay, aw, ah = a.box
        bx, by, bw, bh = b.box
        return (a is not b and ax <= bx and ay <= by
                and ax + aw >= bx + bw and ay + ah >= by + bh
                and aw * ah > bw * bh)

    figures = [q for q in seen if not any(encloses(q, o) for o in seen)] or seen
    s = max(figures, key=lambda q: q.box[2] * q.box[3])
    # Every pose maps back to a plant state (POSE_STATE partitions all 17 by
    # the sheet's own circumplex), so whatever the face wears, the reported
    # word matches the plant's -- an exact state round-trip.  The asserted
    # level follows the state's semantics and stays inside section 5's caps:
    # a CRY pose asserts the DISTRESS_FACE band, a FUSS pose mid-fuss, and
    # nothing from vision alone ever crosses CRY_LEVEL.
    emotion = POSE_STATE.get(s.pose, "CALM")
    distress = (s.asserted_level if emotion == "CRY"
                else 0.20 if emotion == "FUSS"
                else 0.02 if emotion == "SLEEP" else 0.05)
    return SimpleNamespace(present=True, distress=distress,
                           emotion=emotion, alarm=False, name="nubzuki",
                           x=0.0, phase=s.label, echo=s.recovered_level)


def desired_slot(engine: MotionEngine) -> Optional[int]:
    """The PCM slot the engine's mode maps to, or ``None`` when parked.

    ``None`` during a taper as well: the machine has decided to stop, and the
    bridge must not queue another replay behind the one still running.
    """
    m = engine.mode
    return None if m is None or engine.tapering else m.slot


class ScenarioPlayer:
    """A scripted nursery episode: the stand-in infant, driving the machine.

    **This used to be a recognizer test and is now a machine test.**  It fed
    raw signals through the AudioTrack -> InfantJudge chain, so what the
    dashboard showed was the report's section-4/5 logic actually running.  That
    chain was removed with the infant-face stack (2026-08-08), and nothing here
    pretends to replace it: each phase now *asserts* its state and level as
    scripted ground truth.  What is still verified end to end is everything
    downstream -- the escalation ladder, the trial logic, the taper, the gate.

    One consequence worth stating plainly: the scenario reaches the cry band
    because it declares it, not because anything perceived it.  Since the
    removal there is no live path that can cross CRY_LEVEL on its own --
    `perception/nubzuki.py` is vision, and vision is capped in the fuss band by
    the same §5 rule that always capped it.
    """

    # The quiet opener is deliberately short: it only has to establish the
    # awake baseline, and 20 s of nothing read as "is this even on?".  The
    # fussing phase absorbs the difference so every later phase keeps its
    # wall-clock position (tests.py::serve times its trial check against it).
    #        name                    dur  level  state              eyes   flow roll
    PHASES = (("quiet and awake",      8, 0.05, nz.AWAKE,           "open", .03,  0),
              ("starts fussing",      42, 0.34, nz.DISTRESS_FACE,   "open", .10,  0),
              ("soothed by M10",      20, 0.06, nz.AWAKE,           "open", .05,  0),
              ("crying hard",         35, 0.62, nz.DISTRESS_FACE,   "open", .30,  0),
              ("the trial works",     25, 0.08, nz.AWAKE,           "open", .05,  0),
              ("drifting off",        15, 0.02, nz.EYES_CLOSED,     "shut", .03,  0),
              ("stable sleep",        70, 0.00, nz.SLEEP_CANDIDATE, "shut", .02,  0),
              ("rolls onto the side",  6, 0.00, nz.SLEEP_CANDIDATE, "shut", .25, 75),
              ("caregiver resettles", 30, 0.06, nz.AWAKE,           "open", .05,  0))
    CYCLE_S = sum(d for _, d, *_ in PHASES)
    ROLL_ALARM_DEG = 60      # report §1.1: a sustained roll is not a squirm

    def __init__(self) -> None:
        self.reading = None
        # The parts of the frame that never change, drawn once: canvas, the
        # cradle ellipse and the watermark.  30 Hz redraws of static pixels
        # were a measurable slice of a Jetson core.
        bg = np.full((480, 640, 3), 24, np.uint8)
        cv2.ellipse(bg, (320, 300), (250, 150), 0, 10, 170, (60, 55, 50), 14)
        cv2.putText(bg, "SYNTHETIC VERIFICATION SCENARIO - not a camera",
                    (24, 462), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 130, 150), 1,
                    cv2.LINE_AA)
        self._bg = bg

    def _phase(self, t: float):
        into = t % self.CYCLE_S
        for row in self.PHASES:
            if into < row[1]:
                return row, into
            into -= row[1]
        return self.PHASES[-1], 0.0

    @staticmethod
    def _voiced(level: float, t: float) -> bool:
        """Is the mouth open this instant?  Drawing only -- nothing judges it.

        The rhythm still tells the two apart the way §4.2 did, because it is
        what the acted baby should *look* like: a fuss is short and sparse, a
        wail is long and chained.
        """
        if level >= .45:
            return (t % 2.2) < 1.5
        if level >= .12:
            return (t % 2.5) < 0.3
        return False

    def update(self, now: float):
        (name, _dur, level, state, eyes, flow, roll), into = self._phase(now)
        self.reading = SimpleNamespace(
            present=True, distress=level, emotion=state,
            alarm=roll >= self.ROLL_ALARM_DEG, name="scenario",
            x=0.0, y=0.0, phase=name, phase_t=into,
            voiced=self._voiced(level, now), eyes=eyes, roll=roll)
        return self.reading

    def frame(self, now: float):
        """The acted baby, drawn -- illustration only, and it says so."""
        r = self.reading
        img = self._bg.copy()
        cx, cy = 320, 250
        face = np.zeros_like(img)
        cv2.circle(face, (cx, cy), 90, (140, 170, 235), -1)          # skin
        if r.eyes == "shut":
            for ex in (cx - 35, cx + 35):
                cv2.ellipse(face, (ex, cy - 15), (18, 6), 0, 0, 180, (60, 60, 60), 4)
        else:
            for ex in (cx - 35, cx + 35):
                cv2.circle(face, (ex, cy - 15), 9, (50, 50, 50), -1)
        if r.voiced:
            cv2.ellipse(face, (cx, cy + 35), (22, 28), 0, 0, 360, (60, 60, 120), -1)
        else:
            cv2.ellipse(face, (cx, cy + 38), (18, 8), 0, 0, 180, (80, 80, 80), 4)
        if r.roll:   # only one 6 s phase actually rolls; skip the warp otherwise
            M = cv2.getRotationMatrix2D((cx, cy), -r.roll, 1.0)
            face = cv2.warpAffine(face, M, (640, 480))
        img = cv2.add(img, face)
        cv2.putText(img, f"scenario: {r.phase}", (24, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA)
        cv2.putText(img, f"judge: {r.emotion}", (24, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 200, 235), 1, cv2.LINE_AA)
        return img


def sensor_loop(shared: Shared, camera_index: int, fake: bool, ros,
                baby_seed: int | None = None,
                baby: bool = False, bridge=None,
                personality: bool = False, sense_cap=None,
                crop: Optional[tuple[float, float, float, float]] = None,
                zoom: float = 1.0, vote_s: float = 3.0) -> None:
    infant = scenario = None
    if sense_cap is not None:
        pass                       # the camera arrived open, from main()
    elif baby:
        from perception.baby import Personality, VirtualBaby
        quirks = (Personality.random(random.Random(baby_seed))
                  if personality else None)
        infant = VirtualBaby(seed=baby_seed, personality=quirks)
        shared.baby = infant          # /taste retunes it live
        shared.log("virtual infant awake"
                   + (f" (seed {baby_seed})" if baby_seed is not None else ""))
        if quirks is not None:
            shared.log("hidden temperament: " + quirks.describe())
    elif sense_cap is None:
        scenario = ScenarioPlayer()
        shared.log("verification scenario: docs/VERIFY.md layer 1, live")
    # The capture is opened in main() and passed in still open: a probe-and-
    # release preflight races anything else on the device, and a failed open
    # in this thread could only die quietly while the server kept serving a
    # frozen page.
    cap = sense_cap
    plant = None
    if cap is not None:
        # The plant behind the drawn face is the REAL VirtualBaby -- states,
        # dwell times, cumulative settling, and (with --personality) the
        # hidden temperament -- not a bespoke mascot approximation.  Its
        # truth drives what the iPad draws; the machine still only sees what
        # the camera reads back.  One algorithm, two windows onto it.
        from perception.baby import Personality, VirtualBaby
        quirks = (Personality.random(random.Random(baby_seed))
                  if personality else None)
        # tempo 2.5: a demo audience should see the state vocabulary, not
        # a realistic infant's twenty quiet minutes
        plant = VirtualBaby(seed=baby_seed, personality=quirks, tempo=2.5)
        plant_state_since = None      # (state, t0): when this state began
        shared.baby = plant           # /taste retunes it live, as in --baby
        shared.log(f"vision link: camera {camera_index} -> perception/nubzuki; "
                   "the virtual baby lives behind the drawn face")
        shared.log("sense mode: while the figure is unseen the machine "
                   "hears the plant's own state (not vision); jam and camera "
                   "failure still trip the gate")
        if quirks is not None:
            shared.log("hidden temperament: " + quirks.describe())
        if crop is not None:
            shared.log("camera crop: x,y,w,h = "
                       + ", ".join(f"{v:g}" for v in crop)
                       + ("  (fractions)" if max(crop) <= 1.0 else "  (px)"))
        if zoom > 1.0:
            shared.log(f"camera zoom: x{zoom:g} on the region above")
        shared.log(f"state vote: the most-shown pose of the last {vote_s:g}s "
                   "is what the machine is told (presence is not voted)"
                   if vote_s > 0 else
                   "state vote: OFF -- every frame's pose goes straight through")
    # The cradle swings the very screen the camera watches, so single frames
    # blur into nothing mid-play.  A miss inside the grace window keeps the
    # last sighting instead of reporting an absent baby -- genuine absence
    # still stops the cradle at grace + the machine's own 0.7 s gate.
    last_seen_t = None           # overlay bookkeeping: how stale is vision
    # The anti-flicker vote: modal pose over the last `vote_s` seconds.
    # `recent_pose` keeps the newest sighting of each pose so the winning vote
    # can be turned back into a reading with its own label and level.
    vote = nz.Tracker(vote_s)
    recent_pose: dict = {}

    if infant is not None:
        from perception.baby import baby_frame
    # The linkage solve is pure in (ap, ml, z, gain), and the cradle spends
    # most of its life parked at (0, 0, 0) -- cache the last solve instead of
    # running 8 leg IKs per tick to recompute an unchanged pose.
    ik_key = ik_pose = ik_joints = None

    t0 = time.monotonic()
    hist_t = 0.0                     # last 1 Hz timeline sample
    while not shared.stop.is_set():
        now = time.monotonic()
        if infant is not None:
            # Closed loop: the engine's live amplitude is the soothing input,
            # and the frame IS the baby -- a circle riding the plate offset.
            time.sleep(1.0 / 30.0)
            commanded = shared.engine.env * shared.engine.amp_scale
            # When the tablet is alive, its IMU is the physical truth: its
            # strength replaces the commanded envelope AND its felt character
            # (tempo/intensity/tremble) is what the taste judges -- so the
            # baby responds to what the cradle (or a human hand) actually
            # does.  With no tablet the desk simulation is unchanged.
            live_imu = shared.ipad_motion.fresh(now)
            soothing = (shared.ipad_motion.strength(now) if live_imu
                        else commanded)
            reading = infant.update(
                now, soothing=soothing,
                motion=shared.engine.mode.id if shared.engine.mode else None,
                sensed=shared.ipad_motion.snapshot(now) if live_imu else None)
            frame = baby_frame(reading, shared.engine.offsets_mm())
        elif scenario is not None:
            time.sleep(1.0 / 30.0)
            reading = scenario.update(now - t0)
            frame = scenario.frame(now - t0)
        else:
            # The vision link: the camera watches the iPad the machine itself
            # draws on, perception/nubzuki names the pose, and the reading
            # goes through the same contract as every other source.  A frame
            # with no figure maps to present=False and the 0.7 s safety gate
            # does the stopping -- never this loop.
            ok_cam, raw = cap.read()
            if not ok_cam:
                reading = mascot_reading([])
                frame = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(frame, "camera lost", (24, 46),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (60, 60, 220), 2, cv2.LINE_AA)
                time.sleep(0.2)
            else:
                # The region of interest, if one was given, before anything
                # looks at the frame -- the recognizer and the dashboard's
                # camera panel then share one view of the world.
                raw = magnify(crop_frame(raw, crop), zoom)
                # half the stock size gate: a swaying, motion-blurred figure
                # shrinks in the mask before it vanishes from it
                h_, w_ = raw.shape[:2]
                # 0.0012, down from 0.002: a swaying figure half out of
                # the crop is small before it is gone
                seen = nz.read(raw, min_area=int(0.0012 * h_ * w_))
                held = ""
                if seen:
                    # One frame is a vote, not an answer.  At a blend midpoint
                    # the drawn face genuinely sits between two poses and
                    # consecutive frames round either way, so the state sent
                    # to the machine is the *most shown* of the last few
                    # seconds -- what the panel is displaying, not what this
                    # frame happened to land on.  Presence is deliberately NOT
                    # voted: it keeps the grace window below, so a figure that
                    # truly goes away still reaches the safety gate on time.
                    big = max(seen, key=lambda q: q.box[2] * q.box[3])
                    recent_pose[big.pose] = big
                    modal = vote.update(now, big.pose) or big.pose
                    reading = mascot_reading([recent_pose.get(modal, big)])
                    if modal != big.pose:
                        held = f"  (vote {modal} over {big.pose})"
                    last_seen_t = now
                else:
                    reading = None      # no sighting: the plant answers below
                # The plant is the REAL VirtualBaby, ticking on the same
                # physical inputs as --baby: the tablet's measured shake when
                # streaming (the cradle rocks the very screen the face lives
                # on), else the commanded envelope.  Its truth goes OUT to
                # the iPad as the drawn face; the machine's input stays what
                # the camera read back.  One algorithm, two windows onto it.
                live_imu = shared.ipad_motion.fresh(now)
                soothing = (shared.ipad_motion.strength(now) if live_imu
                            else shared.engine.env * shared.engine.amp_scale)
                truth = plant.update(
                    now, soothing=soothing,
                    motion=shared.engine.mode.id if shared.engine.mode else None,
                    sensed=shared.ipad_motion.snapshot(now) if live_imu else None)
                # "deep" is decided by time-in-state, not by the level:
                # the level jitters across any threshold and would flap the
                # face between sibling poses every few frames.  15 s settled
                # earns the deeper pose (sitHeart / dreaming); the reported
                # state word is unchanged, so the exact match holds.
                if (plant_state_since is None
                        or plant_state_since[0] != truth.emotion):
                    plant_state_since = (truth.emotion, now)
                shared.face = {"level": round(truth.distress, 3),
                               "state": truth.emotion,
                               "deep": now - plant_state_since[1] > 15.0}
                if reading is None:
                    # No sighting -> the machine hears the plant itself, not
                    # a stale echo: the figure it cannot find is a drawing of
                    # a state the server already knows exactly.  Vision takes
                    # back over on the next sighting; the overlay names the
                    # source either way.  The gate is untouched -- the jam
                    # input and a dead camera (ok_cam above) still trip it.
                    gap = 0.0 if last_seen_t is None else now - last_seen_t
                    reading = SimpleNamespace(
                        present=True, distress=truth.distress,
                        emotion=truth.emotion, alarm=False, name="nubzuki",
                        x=0.0, phase="plant (unseen)")
                    held = f"  (unseen {gap:.0f}s -- plant state)"
                else:
                    # Vision names the state -- the proof the loop reads back
                    # what the screen drew -- but the LEVEL the machine acts
                    # on is always the plant's continuous truth.  A pose can
                    # only encode four flat bands, so acting on the band made
                    # the input sawtooth between band and truth every time
                    # the named state and the plant disagreed (a blend
                    # midpoint, a flickering detection).  The band was also
                    # what let a hand on the iPad's pose wheel steer the
                    # machine; that trick goes with it, stated here so its
                    # loss is a decision and not a surprise.
                    reading.distress = truth.distress
                frame = nz.annotate(raw, seen)
                # Sized to the frame, not to 640x480: a crop can leave this
                # only 160 px wide, and a fixed 0.62 font there covers the
                # figure it is labelling (the dashboard then upscales it).
                k = nz.overlay_scale(frame)
                cv2.putText(frame, f"{reading.emotion}  "
                            f"asserts {reading.distress:.2f}{held}",
                            (round(16 * k), round(30 * k)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62 * k,
                            (40, 200, 60), max(1, round(2 * k)), cv2.LINE_AA)

        # Manual requests from the browser win over any automatic decision.
        with shared.lock:
            motion_req, shared.motion_request = shared.motion_request, None
        if motion_req is not None:
            ok, msg = shared.engine.command(motion_req, now)
            shared.log(msg if ok else "refused: " + msg)

        # The soothing policy watches the same distress the machine gets, so
        # it can settle each advised motion's outcome (core/policy.py).
        if shared.policy is not None:
            shared.policy.observe(now, infant_level(reading), shared.engine)

        # The evidence-report ladder.  A pain/posture alarm rides the same
        # fault input as a jam: interrupt and call, never soothe (5.1).
        fault = shared.jam or getattr(reading, "alarm", False)
        shared.machine.tick(now, reading.present, infant_level(reading), fault)
        for line in shared.machine.events:
            shared.log(line)
        shared.machine.events.clear()
        shared.engine.tick(now)
        ap_mm, ml_mm, z_mm = shared.engine.offsets_mm()
        shared.offsets = (ap_mm, ml_mm, z_mm)
        if (ap_mm, ml_mm, z_mm, shared.viz_gain) != ik_key:
            ik_key = (ap_mm, ml_mm, z_mm, shared.viz_gain)
            ik_pose = cradle_angles(ap_mm, ml_mm, z_mm)
            # The full linkage: cranks, knees and the bearing.  Gain applies
            # to plate travel before solving, so the exaggerated pose is
            # still a real pose and the rods stay on their bearings.
            if ros is not None:
                ik_joints = cradle_joint_state(ap_mm, ml_mm, z_mm,
                                               gain=shared.viz_gain)
        if not shared.jam:
            shared.pose = ik_pose
        if ros is not None:
            ros.send_joints(ik_joints)

        if bridge is not None:
            # The physical cradle plays the same decision as slots.  The gate
            # needs no special case here: a fault makes the machine taper, so
            # desired_slot() goes None and no further slot is requested.  A
            # raw --play <slot> hold bypasses the engine, so IT checks the
            # gate itself -- same rule, stated once.
            if shared.play_slot is not None:
                gated = shared.jam or shared.machine.state == "gate_fail"
                bridge.tick(now, None if gated else shared.play_slot)
            else:
                bridge.tick(now, desired_slot(shared.engine))

        if now - hist_t >= 1.0:      # the timeline's 1 Hz sample
            hist_t = now
            shared.history.append({
                "t": round(now, 2),
                "level": round(infant_level(reading), 3),
                "ema": round(shared.machine.ema, 3),
                "state": getattr(reading, "emotion", "") or "",
                "motion": shared.engine.mode.id if shared.engine.mode else None,
                "env": round(shared.engine.env * shared.engine.amp_scale, 3),
                "alarm": bool(fault or shared.machine.state == "gate_fail"),
            })

        # Both remaining sources draw their own frame.
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        state = build_state(shared, reading, now)
        with shared.lock:
            if ok:
                shared.jpeg = encoded.tobytes()
            shared.state_json = json.dumps(state).encode()

    if cap is not None:
        cap.release()


def _ipad_state(shared: Shared, now: float) -> dict:
    """The IMU snapshot plus its *felt* reading in the taste vocabulary --
    classified once, server-side, so the dashboard and the virtual baby can
    never disagree about what a motion felt like."""
    snap = shared.ipad_motion.snapshot(now)
    if snap["connected"]:
        from perception.baby import Personality
        snap["felt"] = Personality.felt(snap)
    return snap


def build_state(shared: Shared, reading, now: float) -> dict:
    baby = shared.baby
    return {
        # the learning panel's feed; absent unless a brain is on
        **({"policy": shared.policy.snapshot()}
           if shared.policy is not None else {}),
        # the taste editor's ground truth; only a virtual infant has one
        **({"taste": ({"love": baby.personality.love,
                       "hate": sorted(baby.personality.hate),
                       "combo": list(baby.personality.combo)}
                      if baby.personality is not None else {})}
           if baby is not None else {}),
        "t": round(now, 3),
        "pose": [round(v, 4) for v in shared.pose],
        "tag": {"present": reading.present,
                "level": round(infant_level(reading), 3),
                "x": round(getattr(reading, "x", 0.0), 3),
                "emotion": getattr(reading, "emotion", ""),
                "name": getattr(reading, "name", ""),
                "alarm": bool(getattr(reading, "alarm", False)),
                "phase": getattr(reading, "phase", "")},
        # the drawn face's own channel (--sense): the plant's truth, which
        # may differ from tag.* -- tag is what the machine believes it saw
        **({"face": shared.face} if shared.face is not None else {}),
        "jam": shared.jam,
        "ipad": _ipad_state(shared, now),
        "cradle": {**shared.engine.snapshot(), **shared.machine.snapshot(now)},
        "events": list(shared.events),
    }


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
# A client that walks away mid-stream.  Over plain HTTP that surfaces as
# BrokenPipe/ConnectionReset; over TLS the same event arrives as SSLEOFError
# (the tablet closed the socket without a close_notify), which is an SSLError
# and *not* a ConnectionError -- so it has to be named or every closed tab
# prints a traceback.
GONE = (BrokenPipeError, ConnectionResetError, ssl.SSLError)


class DashboardServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that does TLS per connection, in the worker thread.

    Wrapping the *listening* socket (the obvious ssl recipe) runs every TLS
    handshake inside the single accept loop -- so one client stalling mid
    handshake (a phone parked on a certificate warning, a browser preconnect
    that never speaks) wedges the whole server, and every later connection
    times out, loopback included.  Wrapping here, in the per-connection
    worker thread with a handshake deadline, keeps a stalling client's
    damage to its own thread.
    """

    request_queue_size = 32   # stdlib default of 5 is too small for a
                              # dashboard tab (SSE+MJPEG+preconnects) + tablet
    ssl_context: Optional[ssl.SSLContext] = None

    def finish_request(self, request, client_address):
        if self.ssl_context is not None:
            request.settimeout(12.0)          # a handshake must not dawdle
            try:
                request = self.ssl_context.wrap_socket(request,
                                                       server_side=True)
                request.settimeout(None)
            except (ssl.SSLError, OSError):
                try:
                    request.close()
                except OSError:
                    pass
                return
        super().finish_request(request, client_address)

    def handle_error(self, request, client_address):
        """Only real faults reach the terminal; a departed client is not one."""
        if not isinstance(sys.exc_info()[1], GONE):
            super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    shared: Shared = None   # set before serving

    def log_message(self, *args) -> None:   # keep the terminal for real events
        pass

    def _json(self, payload) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        try:
            if url.path in STATIC:
                name, ctype = STATIC[url.path]
                body = (WEB_DIR / name).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                if ctype == "font/woff2":   # never changes under its own name
                    self.send_header("Cache-Control", "public, max-age=604800")
                else:
                    # The dashboard trio is edited constantly and carries no
                    # ETag or Last-Modified, so a browser is free to invent its
                    # own freshness window and serve a stale app.js -- which
                    # looks exactly like "my change did nothing".
                    self.send_header("Cache-Control", "no-cache, must-revalidate")
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/motions":
                self._json(catalog())
            elif url.path == "/history":
                # a snapshot copy: the sensor thread appends concurrently
                self._json(list(self.shared.history))
            elif url.path == "/motion":
                self._motion(url)
            elif url.path == "/policy":
                self._policy(url)
            elif url.path == "/taste":
                self._taste(url)
            elif url.path == "/auto":
                query = parse_qs(url.query).get("set", [""])[0]
                if query in ("on", "off"):
                    self.shared.machine.auto = query == "on"
                else:
                    self.shared.machine.auto = not self.shared.machine.auto
                state = "ON" if self.shared.machine.auto else "off"
                self.shared.log(f"cradle machine auto {state}")
                self._json({"auto": self.shared.machine.auto})
            elif url.path == "/jam":
                self.shared.jam = not self.shared.jam
                self.shared.log("JAM " + ("ON" if self.shared.jam else "off"))
                self._json({"jam": self.shared.jam})
            elif url.path == "/events":
                self._sse()
            elif url.path == "/frame":
                self._mjpeg()
            else:
                self.send_error(404)
        except GONE:
            pass   # a browser tab closed; entirely normal

    def do_POST(self) -> None:
        """Receive the iPad's reduced IMU features; no motion commands here."""
        url = urlparse(self.path)
        if url.path != "/motion-sensor":
            return self.send_error(404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self.send_error(400, "invalid Content-Length")
        if length <= 0 or length > IPadMotion.MAX_BODY:
            return self.send_error(413, "motion packet too large or empty")
        try:
            payload = json.loads(self.rfile.read(length))
            with self.shared.lock:
                self.shared.ipad_motion.update(payload, time.monotonic())
            self._json({"ok": True})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.send_error(400, str(exc))

    def _motion(self, url) -> None:
        """Validate here, so the browser hears why; execute on the sensor loop."""
        mid = parse_qs(url.query).get("id", [""])[0].upper()
        m = LIBRARY_BY_ID.get(mid)
        if m is None:
            self._json({"ok": False, "msg": f"unknown motion {mid or '?'}"})
        elif m.grade == "R" and not self.shared.engine.allow_research:
            self._json({"ok": False, "msg": f"{m.id} {m.name} is research-only "
                                            "(R) -- start with --research"})
        else:
            with self.shared.lock:
                self.shared.motion_request = mid
            self._json({"ok": True, "queued": mid, "name": m.name,
                        "grade": m.grade})

    def _policy(self, url) -> None:
        """Switch the decision brain live: ?set=off|reflex|ollama|claude.

        Re-selecting the current brain is the "reset memory" gesture --
        every switch starts a fresh SoothePolicy.  The machine keeps every
        safety decision either way; a brain only ever suggests P1 motions.
        """
        kind = parse_qs(url.query).get("set", [""])[0].lower()
        shared = self.shared
        if kind in ("", "off", "none", "ladder"):
            shared.policy = None
            shared.machine.advisor = None
            shared.log("decision brain off -- the report's fixed ladder")
            return self._json({"ok": True, "brain": None})
        try:
            from core.policy import SoothePolicy, load_scenarios, make_brain
            brain = make_brain(kind, model=shared.llm_model)
        except Exception as exc:
            shared.log(f"brain '{kind}' unavailable: {exc}")
            return self._json({"ok": False, "msg": f"{exc}"})
        shared.policy = SoothePolicy(brain, scenarios=load_scenarios())
        shared.machine.advisor = shared.policy.pick
        shared.log(f"decision brain: {kind} (fresh memory, "
                   f"{len(shared.policy.scenarios)} taught scenarios)")
        self._json({"ok": True, "brain": kind})

    def _taste(self, url) -> None:
        """Retune the virtual infant's hidden temperament live (--baby only).

        ?random=1, or ?love=M13&hate=M10,M12&combo=M09,M13 -- everything
        must be a P1 candidate id.  Habituation resets with the new taste.
        """
        baby = self.shared.baby
        if baby is None:
            return self._json({"ok": False,
                               "msg": "no virtual infant -- --baby mode only"})
        from core.policy import CANDIDATES
        from perception.baby import Personality
        q = parse_qs(url.query)
        if q.get("random"):
            quirks = Personality.random(random.Random())
        else:
            love = q.get("love", [""])[0].upper()
            hate = [h for h in q.get("hate", [""])[0].upper().split(",") if h]
            combo = [c for c in q.get("combo", [""])[0].upper().split(",") if c]
            if (love not in CANDIDATES
                    or any(h not in CANDIDATES for h in hate)
                    or any(c not in CANDIDATES for c in combo)
                    or love in hate or len(combo) not in (0, 2)):
                return self._json({"ok": False,
                                   "msg": "love/hate/combo must be M09..M18 "
                                          "(combo needs exactly two)"})
            quirks = Personality(love=love, hate=frozenset(hate),
                                 combo=tuple(combo))
        baby.personality = quirks
        baby._fatigue.clear()
        self.shared.log("temperament retuned: " + quirks.describe())
        self._json({"ok": True, "msg": quirks.describe()})

    def _sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        while not self.shared.stop.is_set():
            with self.shared.lock:
                payload = self.shared.state_json
            self.wfile.write(b"data: " + payload + b"\n\n")
            self.wfile.flush()
            time.sleep(0.05)

    def _mjpeg(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        while not self.shared.stop.is_set():
            with self.shared.lock:
                jpeg = self.shared.jpeg
            if jpeg is not None:
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                 + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                                 + jpeg + b"\r\n")
                self.wfile.flush()
            time.sleep(0.07)


def lan_ip() -> str:
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        ip = probe.getsockname()[0]
        probe.close()
        return ip
    except OSError:
        return "127.0.0.1"


def start(shared: Shared, port: int, camera_index: int, fake: bool, use_ros: bool,
          baby: bool = False,
          baby_seed: int | None = None, robot_target: str | None = None,
          personality: bool = False, sense_cap=None, slot_cap: int | None = None,
          crop: Optional[tuple[float, float, float, float]] = None,
          zoom: float = 1.0, vote_s: float = 3.0):
    # Bind FIRST, before any thread exists.  A busy port used to surface as
    # a raw traceback then a C++ abort ("terminate called without an active
    # exception"): the bind failed after worker/ROS threads were already
    # alive and the interpreter tore down under them.  Fail here, cleanly.
    Handler.shared = shared
    try:
        server = DashboardServer(("0.0.0.0", port), Handler)
    except OSError as exc:
        if exc.errno == 98:
            raise SystemExit(
                f"port {port} is already in use -- an old serve.py still "
                f"running?  (ss -ltnp | grep {port}; or pass --port)") from exc
        raise
    ros = None
    if use_ros:
        try:
            from core.rig import RosSide
            ros = RosSide()
        except Exception as exc:   # ROS absent or misconfigured: not fatal
            shared.log(f"RViz mirroring off ({type(exc).__name__})")
    # After RosSide on purpose: RosSide calls rclpy.init() unguarded, while
    # PhorceRobot's is guarded -- this order works in both directions.
    robot = bridge = None
    if robot_target is not None:
        from core.phorce_iface import PlayOutcome, SlotBridge, make_robot
        try:
            def _played(slot: int, outcome) -> None:
                # --once: the hold clears itself after the first *successful*
                # completion (rejects keep retrying under the bridge's own
                # hold-offs -- once means once played, not once attempted)
                if outcome is PlayOutcome.OK and shared.play_once \
                        and shared.play_slot is not None:
                    shared.play_slot = None
                    shared.log(f"played slot {slot} once -- parked (--once)")
            robot = make_robot(mock=False, target=robot_target,
                               on_play_result=_played)
            robot.start()
            # One physical episode per decision (the team's demo rule): a
            # completed slot is not replayed while the machine keeps naming
            # it -- the rig moves again when the decision changes.  The one
            # exception is --play WITHOUT --once, whose documented job is to
            # loop a raw slot.  1 s rest still spaces any two plays.
            looping = shared.play_slot is not None and not shared.play_once
            bridge = SlotBridge(robot, log=shared.log, rest_s=1.0,
                                max_slot=slot_cap, repeat=looping)
            shared.log(f"phorce: playing slots on {robot_target}")
        except Exception as exc:
            # Asked-for hardware that is absent is a real failure, not a
            # degraded mode: nobody should watch RViz rock while believing
            # the physical cradle is doing the same.
            raise SystemExit(
                f"--robot {robot_target}: {type(exc).__name__}: {exc}\n"
                "real robot: ./robot.sh first, and export ROS_DOMAIN_ID=21 "
                "in this terminal; simulator: ./sim.sh and --robot sim:demo"
            ) from exc
    worker = threading.Thread(
        target=sensor_loop,
        args=(shared, camera_index, fake, ros,
              baby_seed, baby, bridge, personality, sense_cap, crop, zoom,
              vote_s),
        daemon=True)
    worker.start()
    return server, worker, ros, robot

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--certfile", metavar="PEM",
                        help="TLS certificate (default: a local one is made "
                             "and reused from .local-certs/)")
    parser.add_argument("--keyfile", metavar="PEM",
                        help="TLS private key paired with --certfile")
    parser.add_argument("--http", action="store_true",
                        help="serve plain HTTP (no iPad motion permission)")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--crop", metavar="X,Y,W,H",
                        help="--sense: read only this region of the camera "
                             "frame (fractions when all <= 1, else pixels), "
                             "e.g. --crop 0.25,0.1,0.5,0.8.  The dashboard "
                             "panel shows the same crop")
    parser.add_argument("--zoom", type=float, default=1.0, metavar="N",
                        help="--sense: magnify what is read by N (digital, "
                             "e.g. --zoom 2).  With no --crop it reads the "
                             "middle 1/N of the frame, so the cost is flat.  "
                             "Worth it only for a distant panel")
    parser.add_argument("--vote", type=float, default=3.0, metavar="SECONDS",
                        help="--sense: report the most-shown pose of the last "
                             "N seconds instead of this frame's, so a blend "
                             "midpoint stops flickering.  0 disables it; "
                             "presence is never voted, so the safety gate is "
                             "unaffected")
    parser.add_argument("--camera-size", metavar="WxH",
                        help="--sense: ask the camera for this capture size "
                             "(e.g. 1920x1080).  Real detail, unlike --zoom -- "
                             "try this first when the panel reads small")
    parser.add_argument("--sense", action="store_true",
                        help="real camera: read the iPad's drawn face through "
                             "perception/nubzuki -- the vision link")
    parser.add_argument("--verify", "--fake", dest="fake", action="store_true",
                        help="no camera: the acted verification episode "
                             "through the real recognizer (docs/VERIFY.md)")
    parser.add_argument("--baby", action="store_true",
                        help="no camera: a virtual infant (random state "
                             "process) the sway can genuinely soothe")
    parser.add_argument("--baby-seed", type=int, default=None,
                        help="seed the virtual infant for a repeatable run")
    parser.add_argument("--personality", action="store_true",
                        help="with --baby: give the infant a hidden motion "
                             "temperament (docs/IDEA.md) the policy can learn")
    parser.add_argument("--policy",
                        choices=("reflex", "dream", "ollama", "claude"),
                        default=None,
                        help="start with a decision brain advising the trial "
                             "motion: 'reflex' = local algorithm, 'dream' = "
                             "the DREAM-Chunk planner (dreams every candidate "
                             "forward, docs/dream-chunk.md), 'ollama' = "
                             "local LLM (OLLAMA_URL/OLLAMA_MODEL), 'claude' = "
                             "Anthropic API (paid, ANTHROPIC_API_KEY).  The "
                             "dashboard can switch brains live either way.")
    parser.add_argument("--llm-model", default=None, metavar="NAME",
                        help="model override for the ollama/claude brains")
    parser.add_argument("--pace", type=float, default=10.0, metavar="SECONDS",
                        help="trial rhythm: each motion plays this long "
                             "before the machine re-decides (default 10, "
                             "the demo rig's per-motion length; the evidence "
                             "report's own cadence is 30)")
    parser.add_argument("--give-up", action="store_true",
                        help="restore the evidence report's §5 hand-over: a "
                             "trial that does not improve (or worsens) tapers "
                             "and alerts the caregiver.  Off by default -- the "
                             "cradle keeps trying other motions instead.  The "
                             "safety gate (jam / face lost / pain) always runs")
    parser.add_argument("--no-ros", action="store_true",
                        help="do not mirror joints to RViz")
    parser.add_argument("--robot", nargs="?", const="robot", default=None,
                        metavar="TARGET",
                        help="also play the machine's motion as PCM slots on "
                             "a phorce target.  --robot cli = the SD card's "
                             "slots through `phorce play N` (the proven path; "
                             "export ROS_DOMAIN_ID=21); bare --robot = our "
                             "own action client; --robot sim:demo / cli:sim:"
                             "demo exercise the same paths against ./sim.sh")
    parser.add_argument("--viz-gain", type=float, default=1.0,
                        help="exaggerate the RViz mirror by this factor "
                             "(display only; the real sway is ~2.5 deg)")
    parser.add_argument("--play", metavar="ID",
                        help="demo: hold ONE motion (e.g. N05) -- automatic "
                             "care starts off, the safety gate still runs")
    parser.add_argument("--once", action="store_true",
                        help="with --play <slot>: play it a single time, "
                             "then park")
    parser.add_argument("--slots", type=int, default=None, metavar="N",
                        help="only slots 1..N are loaded on the robot's SD "
                             "card (default 14 when --robot is on): brains "
                             "only decide motions the rig can actually play")
    parser.add_argument("--research", action="store_true",
                        help="unlock the R-grade library modes (sim only; "
                             "--fake already implies it)")
    args = parser.parse_args(argv)
    if bool(args.certfile) != bool(args.keyfile):
        parser.error("--certfile and --keyfile must be supplied together")
    if args.baby and args.fake:
        parser.error("--baby is its own world; drop --verify")
    if args.personality and not (args.baby or args.sense):
        parser.error("--personality needs an infant to wear it: --baby, or "
                     "--sense (the plant behind the drawn face)")

    # --fake is a desk demo: a synthetic tag, no camera and no rig, so there is
    # nothing an R-grade mode can hurt.  Unlock the whole library there so the
    # dashboard's selector can drive all 50 without a second flag.  This does
    # not put R into automatic behaviour -- TRIAL_LADDER is P1 only, and the
    # machine never commands anything outside it plus M05/M06/M08.  --baby
    # still needs --research explicitly.
    research = args.research or args.fake
    shared = Shared(allow_research=research, pace_s=args.pace,
                    give_up=args.give_up)
    slot_cap = args.slots if args.slots is not None else (
        14 if args.robot else None)
    if slot_cap is not None and args.robot:
        from core import policy as policy_mod
        keep = policy_mod.restrict_candidates(
            [c for c in policy_mod.N_CANDIDATES if int(c[1:]) <= slot_cap])
        shared.log(f"robot card holds slots 1..{slot_cap}: every brain now "
                   f"decides among {len(keep)} motions")
    if args.play:
        # One motion, held, automatic care off, safety gate still armed.
        # A bare NUMBER is the demo form: the SD card's slots are their own
        # motions, so `--play 5` asks the rig for slot 5 directly and the
        # screen stays parked (we cannot draw a motion we did not author).
        # A library id (N05/M12) still works and plays on screen too.
        shared.machine.auto = False
        if args.play.isdigit():
            slot = int(args.play)
            if args.robot is None:
                raise SystemExit("--play <number> holds a raw SD slot: "
                                 "it needs --robot cli")
            if slot_cap is not None and not (1 <= slot <= slot_cap):
                raise SystemExit(f"--play: slot {slot} is not on the card "
                                 f"(1..{slot_cap})")
            shared.play_slot = slot
            shared.play_once = args.once
            shared.log(f"demo hold: SD slot {slot}"
                       + (" once" if args.once else "")
                       + " -- raw card motion, screen parked, "
                         "safety gate still armed")
        else:
            if args.once:
                raise SystemExit("--once needs --play <slot number>: library "
                                 "motions are continuous, there is no 'once'")
            pid = args.play.upper()
            m = LIBRARY_BY_ID.get(pid)
            if m is None:
                raise SystemExit(f"--play: unknown motion {pid!r} "
                                 "(a slot number, N01-N34 or M01-M50)")
            if m.grade == "R" and not research:
                raise SystemExit(f"--play: {pid} is research-only; "
                                 "add --research")
            shared.motion_request = pid   # the sensor loop starts it softly
            shared.log(f"demo hold: {pid} ({m.name}) -- automatic care off, "
                       "safety gate still armed")
    shared.viz_gain = max(1.0, args.viz_gain)
    shared.llm_model = args.llm_model
    # never a silent deviation from the report: say which rule is running
    shared.log("hand-over rule: " + (
        "ON -- report 5 taper + caregiver alert when a trial does not improve"
        if args.give_up else
        "off -- the cradle keeps trying motions (safety gate still armed)"))
    if args.policy:
        from core.policy import SoothePolicy, load_scenarios, make_brain
        shared.policy = SoothePolicy(make_brain(args.policy, args.llm_model),
                                     scenarios=load_scenarios())
        shared.machine.advisor = shared.policy.pick
        shared.log(f"decision brain: {args.policy} "
                   f"({len(shared.policy.scenarios)} taught scenarios)")
    # A bare launch means the verification episode; the camera is asked for
    # explicitly (--sense), because a missing camera should be a loud failure
    # on the mode that needs it, not a silent fallback on the mode that don't.
    if args.sense and args.baby:
        parser.error("--sense already runs the virtual baby as the plant "
                     "behind the drawn face; --baby is the no-camera mode")
    if not (args.fake or args.baby or args.sense):
        args.fake = True
    crop = None
    if args.crop:
        try:
            crop = parse_crop(args.crop)
        except ValueError as exc:
            parser.error(str(exc))
    zoom = args.zoom
    if zoom < 1.0:
        parser.error(f"--zoom magnifies, so it starts at 1 -- got {zoom:g}")
    if zoom > 1.0 and crop is None:
        # "Zoom the webcam" with nothing else said means the middle of the
        # frame, which also keeps the cost flat: the region shrinks by N as
        # the scale grows by N, so the recognizer sees the same pixel count.
        crop = centre_region(zoom)
    cam_size = None
    if args.camera_size:
        try:
            cam_size = tuple(int(v) for v in args.camera_size.lower().split("x"))
            if len(cam_size) != 2 or min(cam_size) <= 0:
                raise ValueError
        except ValueError:
            parser.error(f"--camera-size wants WxH -- got {args.camera_size!r}")
    if (crop is not None or zoom > 1.0) and not args.sense:
        shared.log("--crop/--zoom have no camera to read (needs --sense) "
                   "-- ignored")
        crop, zoom = None, 1.0
    sense_cap = None
    if args.sense:
        # Opened HERE and handed to the loop still open: a probe-and-release
        # preflight races anything else on the device, and a failed open in
        # the worker could only kill that thread quietly while the server
        # kept serving a frozen page.  Open once, fail loudly.
        sense_cap = cv2.VideoCapture(args.camera_index)
        if not sense_cap.isOpened():
            raise SystemExit(
                f"--sense: camera {args.camera_index} would not open "
                f"(in use? try --camera-index 1; ls /dev/video*)")
        # One frame in the driver's queue, not the default four: nz.read on a
        # big crop can take ~0.7 s/frame, and a deeper queue makes read()
        # return the *oldest* buffered frame -- the machine then reads a
        # screen seconds in the past and every reading lags reality.
        sense_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cam_size is not None:
            # A request, not a setting: UVC cameras silently keep their own
            # size when asked for one they do not have, so report what the
            # device actually handed back rather than what we asked for.
            sense_cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_size[0])
            sense_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_size[1])
        got = (int(sense_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
               int(sense_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        shared.log(f"camera {args.camera_index} capturing {got[0]}x{got[1]}"
                   + (f" (asked {cam_size[0]}x{cam_size[1]}, refused)"
                      if cam_size is not None and got != cam_size else ""))
    server, worker, ros, robot = start(shared, args.port, args.camera_index,
                                       args.fake, use_ros=not args.no_ros,
                                       baby=args.baby,
                                       baby_seed=args.baby_seed,
                                       robot_target=args.robot,
                                       personality=args.personality,
                                       sense_cap=sense_cap,
                                       slot_cap=slot_cap, crop=crop, zoom=zoom,
                                       vote_s=args.vote)
    # HTTPS is the default, because the demo's most fragile feature -- the
    # iPad motion permission -- silently refuses on plain HTTP.  A local CA +
    # server cert pair is minted once (tools/make_https_cert.sh) into
    # .local-certs/ and reused; it is re-minted when the LAN IP changes,
    # since the cert names the IP the tablet will actually dial.
    if args.http:
        if args.certfile:
            parser.error("--http and --certfile contradict each other")
    elif not args.certfile:
        import subprocess
        cert_dir = Path(".local-certs")
        crt, key = cert_dir / "server.crt", cert_dir / "server.key"
        ip = lan_ip()
        good = False
        if crt.exists() and key.exists():
            text = subprocess.run(
                ["openssl", "x509", "-in", str(crt), "-noout", "-text"],
                capture_output=True, text=True).stdout
            good = f"IP Address:{ip}" in text
        if not good:
            made = subprocess.run(
                ["bash", "tools/make_https_cert.sh", ip],
                capture_output=True, text=True)
            if made.returncode != 0:
                raise SystemExit(
                    "could not mint a local HTTPS certificate:\n"
                    + made.stderr.strip()
                    + "\n(pass --http to serve without TLS)")
            print(f"minted HTTPS certificate for {ip} -> {cert_dir}/")
        args.certfile, args.keyfile = str(crt), str(key)
    if args.certfile:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.certfile, args.keyfile)
        server.ssl_context = context     # per-connection wrap: DashboardServer
    source = ("virtual infant" if args.baby
              else "verification scenario" if args.fake
              else f"camera {args.camera_index} -> nubzuki (vision link)")
    scheme = "https" if args.certfile else "http"
    host = lan_ip()
    print(f"SIGMA dashboard:  {scheme}://{host}:{args.port}   ({source})")
    print(f"iPad baby:       {scheme}://{host}:{args.port}/baby")
    if not args.certfile:
        print("iPad IMU note: motion permission normally requires HTTPS; "
              "the display still works over HTTP")
    else:
        print(f"iPad trust:      install .local-certs/sigma-ca.crt once "
              "(Settings > General > About > Certificate Trust)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        shared.stop.set()
        server.shutdown()
        if robot is not None:
            robot.close()
        if ros is not None:
            ros.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

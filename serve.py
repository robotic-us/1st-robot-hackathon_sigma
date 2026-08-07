#!/usr/bin/env python3
"""The evidence-report cradle machine, live in a web browser.

One stdlib HTTP server:

    /            the dashboard (web/index.html; /style.css and /app.js beside it)
    /events      Server-Sent Events: the whole state as JSON, ~20 Hz
    /history     the last ~15 min as 1 Hz samples (seeds the emotion timeline)
    /frame       MJPEG camera stream with the sensing overlay
    /motions     the M01-M50 library from the evidence report, once
    /jam         simulate a mechanism fault (trips the safety gate)
    /motion      queue a library command by id (R grade needs --research)
    /auto        the report's state machine on/off (?set=on|off)

Motion behaviour follows docs/infant_robotic_cradle_evidence_report_ko.pdf
(see core/cradle.py): the default is *not moving*.  Two sensing modes feed
the CradleMachine's 0..1 distress input:

* **Real sensing** (``--sense``): the five-state watcher / FER+ fused with
  the microphone drives the report-spec judge (perception/watch.py); a lost
  face or a posture alarm trips the safety gate.
* **Verification scenario** (``--verify``, alias ``--fake``): no camera.  A
  scripted nursery episode is rendered as the *sensor signals* a real baby
  would produce (whimpers, wails, closed eyes, a rollover) and pushed through
  the real AudioTrack -> InfantJudge -> CradleMachine chain -- the dashboard
  and the webapp visualise docs/VERIFY.md layer 1 live.  The camera panel
  shows a drawn baby acting the script, clearly watermarked as synthetic.
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

    python3 serve.py --sense       # webcam + mic: the real recognizer
    python3 serve.py --verify      # no camera: the acted verification episode
    python3 serve.py --baby        # no camera: a virtual infant, closed loop
    python3 serve.py --research    # unlock the R-grade modes (sim only!)
    python3 tests.py               # all suites, incl. these endpoints
"""

from __future__ import annotations

import argparse
import json
import random
import socket
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

WEB_DIR = Path(__file__).resolve().parent / "web"


# --------------------------------------------------------------------------- #
# Shared state between the sensor loop and the HTTP handlers
# --------------------------------------------------------------------------- #
class Shared:
    def __init__(self, allow_research: bool = False) -> None:
        self.engine = MotionEngine(allow_research=allow_research)
        self.machine = CradleMachine(self.engine)
        self.jam = False              # simulated mechanism fault (the gate)
        self.pose = [0.0, 0.0, 0.0, 0.0]   # crank angles the viz mirrors
        self.lock = threading.Lock()
        self.state_json = b"{}"
        self.jpeg: Optional[bytes] = None
        self.events: deque[str] = deque(maxlen=14)
        self.motion_request: Optional[str] = None
        self.policy = None      # core/policy.py SoothePolicy when --policy
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


def desired_slot(engine: MotionEngine) -> Optional[int]:
    """The PCM slot the engine's mode maps to, or ``None`` when parked.

    ``None`` during a taper as well: the machine has decided to stop, and the
    bridge must not queue another replay behind the one still running.
    """
    m = engine.mode
    return None if m is None or engine.tapering else m.slot


class ScenarioPlayer:
    """A scripted nursery episode, pushed through the real recognizer chain.

    Nothing here shortcuts to a distress number: each phase emits the raw
    signals (voiced chunks, eye state, body flow, head roll) and the same
    AudioTrack -> InfantJudge machinery used on a live camera turns them into
    states.  What the dashboard shows in this mode is therefore the actual
    section-4/5 logic running, not a canned animation of it.
    """

    # The quiet opener is deliberately short: it only has to establish the
    # QUIET_AWAKE baseline, and 20 s of nothing read as "is this even on?".
    # The fussing phase absorbs the difference so every later phase keeps its
    # wall-clock position (tests.py::serve times its trial check against it).
    #        name                    dur  audio      eyes    flow  roll
    PHASES = (("quiet and awake",      8, "quiet",   "open", 0.03,  0),
              ("starts fussing",      42, "whimper", "open", 0.10,  0),
              ("soothed by M10",      20, "quiet",   "open", 0.05,  0),
              ("crying hard",         35, "wail",    "open", 0.30,  0),
              ("the trial works",     25, "quiet",   "open", 0.05,  0),
              ("drifting off",        15, "quiet",   "shut", 0.03,  0),
              ("stable sleep",        70, "quiet",   "shut", 0.02,  0),
              ("rolls onto the side",  6, "quiet",   "shut", 0.25, 75),
              ("caregiver resettles", 30, "quiet",   "open", 0.05,  0))
    CYCLE_S = sum(d for _, d, *_ in PHASES)

    def __init__(self) -> None:
        from perception.watch import AudioTrack, InfantJudge, PostureTrack, Signals
        self.Signals = Signals
        self.audio = AudioTrack()
        self.judge = InfantJudge()
        self.posture = PostureTrack()
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
        for name, dur, audio, eyes, flow, roll in self.PHASES:
            if into < dur:
                return name, into, audio, eyes, flow, roll
            into -= dur
        return self.PHASES[-1][0], 0.0, "quiet", "open", 0.05, 0

    @staticmethod
    def _chunk(kind: str, t: float):
        """(level, cry) for this instant -- whimpers are short and sparse,
        wails long and chained, exactly the 4.2 distinction."""
        if kind == "whimper":
            voiced = (t % 2.5) < 0.3
        elif kind == "wail":
            voiced = (t % 2.2) < 1.5
        else:
            voiced = False
        return (0.35, 0.8) if voiced else (0.02, 0.1)

    def update(self, now: float):
        name, into, audio_kind, eyes, flow, roll = self._phase(now)
        level, cry = self._chunk(audio_kind, now)
        feats = self.audio.feed(level, cry, now)
        risk = self.posture.feed(now, float(roll) if roll else 0.0, None, 0.0)
        judged = self.judge.update(self.Signals(
            ts=now, face_conf=1.0, emotion=None,
            eyes_closed=(eyes == "shut"), pain_face=False, body_arch=None,
            posture_risk=risk, body_flow=flow, audio=feats))
        self.reading = SimpleNamespace(
            present=judged.present, distress=judged.level,
            emotion=judged.state, alarm=judged.alarm, name="scenario",
            x=0.0, y=0.0, phase=name, phase_t=into, voiced=feats.voiced,
            eyes=eyes, roll=roll)
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
                sense: bool = False, audio_device: str = "plughw:WEBCAM,0",
                no_sound: bool = False, baby_seed: int | None = None,
                baby: bool = False, bridge=None,
                personality: bool = False) -> None:
    sensor = mic = infant = scenario = None
    if baby:
        from perception.baby import Personality, VirtualBaby
        quirks = (Personality.random(random.Random(baby_seed))
                  if personality else None)
        infant = VirtualBaby(seed=baby_seed, personality=quirks)
        shared.log("virtual infant awake"
                   + (f" (seed {baby_seed})" if baby_seed is not None else ""))
        if quirks is not None:
            shared.log("hidden temperament: " + quirks.describe())
    elif sense:
        # Real sensing.  The five-state watcher (perception/watch.py) is the
        # primary visual channel; YuNet+FER+ is the fallback without mediapipe.
        from perception.listen import Microphone
        from perception.sense import Sense
        if not no_sound:
            mic = Microphone(audio_device)
            mic.start()
        try:
            from perception.watch import Watcher
            sensor = Sense(microphone=mic, watcher=Watcher())
            shared.log("real sensing: five-state watcher"
                       + ("" if no_sound else " + microphone"))
        except ImportError:
            from perception.face.pipeline import SigmaPipeline
            sensor = Sense(SigmaPipeline(), mic)
            shared.log("real sensing: face+emotion (no mediapipe -- "
                       "watcher unavailable)"
                       + ("" if no_sound else " + microphone"))
    else:
        scenario = ScenarioPlayer()
        shared.log("verification scenario: docs/VERIFY.md layer 1, live")
    cap = None
    if not fake and not baby and sense:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            shared.log(f"camera {camera_index} failed -- verification scenario "
                       "instead")
            sensor, cap = None, None
            scenario = ScenarioPlayer()

    if infant is not None:
        from perception.baby import baby_frame
    if sensor is not None:
        from perception.sense import draw as sense_draw
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
            reading = infant.update(
                now, soothing=shared.engine.env * shared.engine.amp_scale,
                motion=shared.engine.mode.id if shared.engine.mode else None)
            frame = baby_frame(reading, shared.engine.offsets_mm())
        elif scenario is not None:
            time.sleep(1.0 / 30.0)
            reading = scenario.update(now - t0)
            frame = scenario.frame(now - t0)
        else:
            ok, frame = cap.read()
            if not ok:
                shared.log("camera stream ended")
                break
            reading = sensor.update(frame, now)

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
            # desired_slot() goes None and no further slot is requested.
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

        if infant is not None or scenario is not None:
            canvas = frame                     # these draw themselves
        else:
            canvas = sense_draw(frame, reading, sensor)
        ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        state = build_state(shared, reading, now)
        with shared.lock:
            if ok:
                shared.jpeg = encoded.tobytes()
            shared.state_json = json.dumps(state).encode()

    if cap is not None:
        cap.release()
    if mic is not None:
        mic.close()


def build_state(shared: Shared, reading, now: float) -> dict:
    return {
        # the learning panel's feed; absent unless --policy is on
        **({"policy": shared.policy.snapshot()}
           if shared.policy is not None else {}),
        "t": round(now, 3),
        "pose": [round(v, 4) for v in shared.pose],
        "tag": {"present": reading.present,
                "level": round(infant_level(reading), 3),
                "x": round(getattr(reading, "x", 0.0), 3),
                "emotion": getattr(reading, "emotion", ""),
                "name": getattr(reading, "name", ""),
                "alarm": bool(getattr(reading, "alarm", False)),
                "phase": getattr(reading, "phase", "")},
        "jam": shared.jam,
        "cradle": {**shared.engine.snapshot(), **shared.machine.snapshot(now)},
        "events": list(shared.events),
    }


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
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
            if url.path in ("/", "/style.css", "/app.js"):
                # a fixed whitelist, not a static dir -- nothing to traverse
                name, ctype = {
                    "/": ("index.html", "text/html; charset=utf-8"),
                    "/style.css": ("style.css", "text/css; charset=utf-8"),
                    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                }[url.path]
                body = (WEB_DIR / name).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/motions":
                self._json(catalog())
            elif url.path == "/history":
                # a snapshot copy: the sensor thread appends concurrently
                self._json(list(self.shared.history))
            elif url.path == "/motion":
                self._motion(url)
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
        except (BrokenPipeError, ConnectionResetError):
            pass   # a browser tab closed; entirely normal

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
          sense: bool = False, audio_device: str = "plughw:WEBCAM,0",
          no_sound: bool = False, baby: bool = False,
          baby_seed: int | None = None, robot_target: str | None = None,
          personality: bool = False):
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
        from core.phorce_iface import SlotBridge, make_robot
        try:
            robot = make_robot(mock=False, target=robot_target)
            robot.start()
            bridge = SlotBridge(robot, log=shared.log)
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
        args=(shared, camera_index, fake, ros, sense, audio_device, no_sound,
              baby_seed, baby, bridge, personality),
        daemon=True)
    worker.start()
    Handler.shared = shared
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    return server, worker, ros, robot

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--camera-index", type=int, default=0)
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
    parser.add_argument("--policy", choices=("reflex", "claude"), default=None,
                        help="let a soothing policy advise which P1 motion "
                             "each trial uses: 'reflex' = offline taught "
                             "strategy, 'claude' = the LLM (ANTHROPIC_API_KEY)")
    parser.add_argument("--sense", action="store_true",
                        help="real sensing: face+emotion (+mic) drives the "
                             "machine instead of tag state cards")
    parser.add_argument("--audio-device", default="plughw:WEBCAM,0",
                        help="ALSA capture device for --sense")
    parser.add_argument("--no-sound", action="store_true",
                        help="with --sense: face only, no microphone")
    parser.add_argument("--no-ros", action="store_true",
                        help="do not mirror joints to RViz")
    parser.add_argument("--robot", nargs="?", const="robot", default=None,
                        metavar="TARGET",
                        help="also play the machine's motion as PCM slots on "
                             "a phorce target.  Bare --robot = the real robot "
                             "(./robot.sh first, export ROS_DOMAIN_ID=21); "
                             "--robot sim:demo exercises the same path "
                             "against ./sim.sh")
    parser.add_argument("--viz-gain", type=float, default=1.0,
                        help="exaggerate the RViz mirror by this factor "
                             "(display only; the real sway is ~2.5 deg)")
    parser.add_argument("--research", action="store_true",
                        help="unlock the R-grade library modes (sim only; "
                             "--fake already implies it)")
    args = parser.parse_args(argv)
    if args.sense and args.fake:
        parser.error("--sense needs a real camera; it cannot run with --verify")
    if args.baby and (args.sense or args.fake):
        parser.error("--baby is its own world; drop --sense/--verify")
    if args.personality and not args.baby:
        parser.error("--personality is a virtual-infant trait; add --baby")

    # --fake is a desk demo: a synthetic tag, no camera and no rig, so there is
    # nothing an R-grade mode can hurt.  Unlock the whole library there so the
    # dashboard's selector can drive all 50 without a second flag.  This does
    # not put R into automatic behaviour -- TRIAL_LADDER is P1 only, and the
    # machine never commands anything outside it plus M05/M06/M08.  Every other
    # mode (real camera, --sense, --baby) still needs --research explicitly.
    research = args.research or args.fake
    shared = Shared(allow_research=research)
    shared.viz_gain = max(1.0, args.viz_gain)
    if args.policy:
        from core.policy import (ClaudeBrain, ReflexBrain, SoothePolicy,
                                 load_scenarios)
        brain = ClaudeBrain() if args.policy == "claude" else ReflexBrain()
        shared.policy = SoothePolicy(brain, scenarios=load_scenarios())
        shared.machine.advisor = shared.policy.pick
        shared.log(f"soothe policy '{args.policy}' advises trial motions "
                   f"({len(shared.policy.scenarios)} taught scenarios)")
    if not (args.fake or args.baby):
        args.sense = True       # a bare launch means the real recognizer
    server, worker, ros, robot = start(shared, args.port, args.camera_index,
                                       args.fake, use_ros=not args.no_ros,
                                       sense=args.sense,
                                       audio_device=args.audio_device,
                                       no_sound=args.no_sound, baby=args.baby,
                                       baby_seed=args.baby_seed,
                                       robot_target=args.robot,
                                       personality=args.personality)
    source = ("virtual infant" if args.baby
              else "verification scenario" if args.fake
              else f"camera {args.camera_index} (real sensing)")
    print(f"SIGMA dashboard:  http://{lan_ip()}:{args.port}   ({source})")
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

#!/usr/bin/env python3
"""DREAM-Chunk plus the evidence-report cradle machine, live in a web browser.

One stdlib HTTP server:

    /            the dashboard (web/index.html -- canvas cradle sim + panels)
    /events      Server-Sent Events: the whole state as JSON, ~20 Hz
    /frame       MJPEG camera stream with the tag overlay
    /slots       the DREAM motion-slot dictionary, once
    /motions     the M01-M50 library from the evidence report, once
    /jam /play   controls: toggle the jam, play a DREAM slot by hand
    /motion      queue a library command by id (R grade needs --research)
    /auto        the report's state machine on/off (?set=on|off)

Motion behaviour follows docs/infant_robotic_cradle_evidence_report_ko.pdf
(see cradle.py): the default is *not moving*.  The tag is a **state card**
standing in for the infant sensors: WHICH tag you show is the state -- tag_0
calm, tag_1 fussing, tag_2 crying (print them with ``python3 perception/tag.py --make``)
-- losing the tag is a safety-gate failure, and how the tag moves means
nothing.  The CradleMachine runs the report's ladder over that state: gate
first, sleep taper, quiet hold, 30 s sway trials with improvement checks and
caregiver alerts.  The engine's plate offset drives the same joint angles the
canvas and RViz already animate; serve.py itself stays 2D and leaves the 3D
view to the viz.  (--dream-auto restores the old tag-shake -> DREAM-slot
autopilot instead.)

No new dependencies: SSE and MJPEG are both plain HTTP, which is why the
stdlib server is enough.  Open it from any laptop on the same network.

Run::

    python3 serve.py               # webcam + tag, http://<jetson-ip>:8080
    python3 serve.py --fake        # no camera: a synthetic tag drives the loop
    python3 serve.py --research    # unlock the R-grade modes (sim only!)
    python3 tests.py               # all suites, incl. these endpoints
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import cv2

from core.cradle import LIBRARY_BY_ID, CradleMachine, MotionEngine, catalog
from apps.demo import AXIS0, PIVOT, PLAY_DT, Brain, load_chunks, plate_point
from core.slot_table import load_slot_table
from perception.tag import TagTracker, draw_overlay, synthetic_frame

WEB_DIR = Path(__file__).resolve().parent / "web"

# Arm length from CAD: how far the plate travels per radian of arm angle,
# so the engine's millimetre offsets become the joint angle the viz draws.
LEVER_M = float(PIVOT[2] - AXIS0[2])


# --------------------------------------------------------------------------- #
# Shared state between the sensor loop and the HTTP handlers
# --------------------------------------------------------------------------- #
class Shared:
    def __init__(self, brain: Brain, chunks, table,
                 allow_research: bool = False, dream_auto: bool = False) -> None:
        self.brain = brain
        self.chunks = chunks
        self.table = table
        self.engine = MotionEngine(allow_research=allow_research)
        self.machine = CradleMachine(self.engine)
        self.dream_auto = dream_auto
        if dream_auto:
            self.machine.auto = False   # one autopilot at a time
        self.lock = threading.Lock()
        self.state_json = b"{}"
        self.jpeg: Optional[bytes] = None
        self.events: deque[str] = deque(maxlen=14)
        self.decision_seq = 0
        self.decision: Optional[dict] = None
        self.play_request: Optional[int] = None
        self.motion_request: Optional[str] = None
        self.stop = threading.Event()

    def log(self, text: str) -> None:
        self.events.appendleft(f"{time.strftime('%H:%M:%S')}  {text}")


def slot_catalog(shared: Shared) -> list[dict]:
    out = []
    for slot_id in sorted(shared.chunks):
        chunk = shared.chunks[slot_id]
        slot = shared.table.slots.get(slot_id)
        out.append({
            "id": slot_id,
            "name": chunk.name,
            "direction": slot.direction.value if slot else "?",
            "amplitude": slot.amplitude.value if slot else "?",
            "duration": round(chunk.duration_s, 2),
            "target_deg": round(math.degrees(chunk.nominal_end_rad()[0]), 1),
        })
    return out


# --------------------------------------------------------------------------- #
# The sensor loop -- same Brain as demo.py, plus state/JPEG publishing
# --------------------------------------------------------------------------- #
# The tag is a state card, not a puppet: WHICH tag is in view is the infant
# state, and waving it means nothing.  Unknown ids read as calm.
TAG_LEVEL = {0: 0.0, 1: 0.30, 2: 0.60}   # calm / fussing / crying


def tag_level(reading) -> float:
    return TAG_LEVEL.get(reading.tag_id, 0.0) if reading.present else 0.0


# The fake camera acts out a nursery shift with the cards: calm, a fuss the
# trial soothes (sleep taper), calm, a cry that escalates, calm again.
FAKE_SCRIPT = ((20.0, 0), (25.0, 1), (75.0, 0), (35.0, 2), (90.0, 0))
FAKE_CYCLE_S = sum(seconds for seconds, _ in FAKE_SCRIPT)


def fake_frame(t: float):
    """A synthetic state card that drifts gently; the drift carries no meaning."""
    into = t % FAKE_CYCLE_S
    tag_id = FAKE_SCRIPT[-1][1]
    for seconds, candidate in FAKE_SCRIPT:
        if into < seconds:
            tag_id = candidate
            break
        into -= seconds
    x = 0.5 * math.sin(2 * math.pi * t / 60.0)
    y = 0.1 * math.sin(t * 0.7)
    return cv2.cvtColor(synthetic_frame(640, 480, x, y, 120, tag_id=tag_id),
                        cv2.COLOR_GRAY2BGR)


def sensor_loop(shared: Shared, camera_index: int, fake: bool, ros) -> None:
    brain = shared.brain
    tracker = TagTracker()
    cap = None
    if not fake:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            shared.log(f"camera {camera_index} failed -- falling back to --fake")
            fake, cap = True, None

    t0 = time.monotonic()
    was_diverged = False
    while not shared.stop.is_set():
        now = time.monotonic()
        if fake:
            frame = fake_frame(now - t0)
            time.sleep(1.0 / 30.0)
        else:
            ok, frame = cap.read()
            if not ok:
                shared.log("camera stream ended")
                break

        reading = tracker.update(frame, now)
        brain.tick(now)

        # Manual requests from the browser win over any automatic decision.
        with shared.lock:
            wanted, shared.play_request = shared.play_request, None
            motion_req, shared.motion_request = shared.motion_request, None
        if motion_req is not None:
            ok, msg = shared.engine.command(motion_req, now)
            shared.log(msg if ok else "refused: " + msg)
        slot = None
        if wanted is not None and brain.playing is None:
            slot = force_play(brain, shared.chunks, wanted, now)
            if slot:
                shared.log(f"slot {slot} played by hand")
        if slot is None and shared.dream_auto:
            slot = brain.decide(reading, now)
            if slot is not None:
                shared.decision_seq += 1
                shared.decision = decision_payload(shared, brain, now)
                top = brain.last_ranked[0]
                shared.log(f"slot {slot} chosen (cost {top.total:.2f}, "
                           f"dob {brain.dob()[0]:.1f}A)")

        # The evidence-report ladder: the tag's IDENTITY is the infant state
        # (tag_0 calm, tag_1 fuss, tag_2 cry), a lost tag is a gate failure,
        # and tag motion is deliberately ignored.  The engine's plate offset
        # becomes the arm angle the canvas and RViz animate.
        shared.machine.tick(now, reading.present, tag_level(reading), brain.jam)
        for line in shared.machine.events:
            shared.log(line)
        shared.machine.events.clear()
        shared.engine.tick(now)
        if brain.playing is None and not brain.jam and shared.engine.active:
            ap_mm, ml_mm, _ = shared.engine.offsets_mm()
            theta = (ap_mm + ml_mm) / 1000.0 / LEVER_M
            brain.pose = [theta] * 4

        report = brain.monitor.latest()
        if report.diverged and not was_diverged:
            shared.log(f"DIVERGED  slot {report.active_slot} "
                       f"err {report.error_rad:.2f} rad")
        was_diverged = report.diverged

        if ros is not None:
            ros.send_joints(brain.pose)

        canvas = draw_overlay(frame, reading, tracker)
        ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 70])
        state = build_state(shared, brain, reading, now)
        with shared.lock:
            if ok:
                shared.jpeg = encoded.tobytes()
            shared.state_json = json.dumps(state).encode()

    if cap is not None:
        cap.release()


def force_play(brain: Brain, chunks, slot_id: int, now: float) -> Optional[int]:
    """Play one slot on request -- the slot panel's click-to-play."""
    chunk = chunks.get(slot_id)
    if chunk is None or brain.playing is not None:
        return None
    _, traj = chunk.dream(brain.pose, dt=PLAY_DT)
    brain.monitor.begin(chunk, brain.pose, now)
    brain.playing = (chunk, now, traj)
    brain.last_ranked = []
    return slot_id


def decision_payload(shared: Shared, brain: Brain, now: float) -> dict:
    ranked = []
    for i, score in enumerate(brain.last_ranked):
        chunk = shared.chunks.get(score.slot_id)
        arc = []
        if chunk is not None:
            _, y = chunk.dream(brain.pose, dt=0.06)
            arc = [[round(p[0] * 1000, 1), round(p[2] * 1000, 1)]
                   for p in (plate_point(float(th)) for th in y[:, 0])]
        ranked.append({
            "slot": score.slot_id, "cost": round(score.total, 3),
            "consist": round(score.consistency, 3),
            "resist": round(score.resistance, 3),
            "contin": round(score.continuity, 3),
            "task": round(score.task, 3),
            "vetoed": score.vetoed, "arc": arc,
        })
    chosen = next((r["slot"] for r in ranked if not r["vetoed"]), None)
    return {"seq": shared.decision_seq, "t": now, "chosen": chosen, "ranked": ranked}


def build_state(shared: Shared, brain: Brain, reading, now: float) -> dict:
    playing = None
    if brain.playing is not None:
        chunk, t_start, traj = brain.playing
        row = min(len(traj) - 1, int((now - t_start) / PLAY_DT))
        playing = {
            "slot": chunk.slot_id, "name": chunk.name,
            "progress": round(min(1.0, (now - t_start) / max(1e-6, chunk.duration_s)), 3),
            # Where the DREAM says the arms should be right now.  When the
            # cradle is jammed this keeps moving while pose does not -- the
            # browser draws it as the ghost plate.
            "dream_theta": round(float(traj[row][0]), 4),
        }
    report = brain.monitor.latest()
    return {
        "t": round(now, 3),
        "pose": [round(v, 4) for v in brain.pose],
        "tag": {"present": reading.present, "id": reading.tag_id,
                "level": tag_level(reading), "x": round(reading.x, 3),
                "motion": round(reading.motion, 3)},
        "jam": brain.jam,
        "dob": brain.dob()[0],
        "playing": playing,
        "cradle": {**shared.engine.snapshot(), **shared.machine.snapshot(now)},
        "monitor": {
            "slot": report.active_slot, "err": round(report.error_rad, 4),
            "rms": round(report.rms_rad, 4), "diverged": report.diverged,
            "threshold": brain.cfg.diverge_rad,
        },
        "decision": shared.decision,
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
            if url.path == "/":
                body = (WEB_DIR / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/slots":
                self._json(slot_catalog(self.shared))
            elif url.path == "/motions":
                self._json(catalog())
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
                self.shared.brain.jam = not self.shared.brain.jam
                self.shared.log("JAM " + ("ON -- dob 2.5A" if self.shared.brain.jam else "off"))
                self._json({"jam": self.shared.brain.jam})
            elif url.path == "/play":
                slot = int(parse_qs(url.query).get("slot", ["0"])[0])
                if self.shared.engine.active:
                    # one axis, one pattern at a time (report 5.1)
                    self._json({"requested": None,
                                "msg": "motion engine active -- /motion?id=M01 first"})
                else:
                    with self.shared.lock:
                        self.shared.play_request = slot
                    self._json({"requested": slot})
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


def start(shared: Shared, port: int, camera_index: int, fake: bool, use_ros: bool):
    ros = None
    if use_ros:
        try:
            from apps.demo import RosSide
            ros = RosSide()
        except Exception as exc:   # ROS absent or misconfigured: not fatal
            shared.log(f"RViz mirroring off ({type(exc).__name__})")
    worker = threading.Thread(
        target=sensor_loop, args=(shared, camera_index, fake, ros), daemon=True)
    worker.start()
    Handler.shared = shared
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    return server, worker, ros

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--fake", action="store_true",
                        help="no camera: a synthetic tag drives the loop")
    parser.add_argument("--no-ros", action="store_true",
                        help="do not mirror joints to RViz")
    parser.add_argument("--research", action="store_true",
                        help="unlock the R-grade library modes (sim only)")
    parser.add_argument("--dream-auto", action="store_true",
                        help="old autopilot: tag shake picks a DREAM slot "
                             "instead of the report's cradle machine")
    args = parser.parse_args(argv)

    chunks = load_chunks()
    table = load_slot_table("slots.json")
    shared = Shared(Brain(chunks, table), chunks, table,
                    allow_research=args.research, dream_auto=args.dream_auto)
    server, worker, ros = start(shared, args.port, args.camera_index,
                                args.fake, use_ros=not args.no_ros)
    print(f"SIGMA dashboard:  http://{lan_ip()}:{args.port}   "
          f"({'synthetic tag' if args.fake else f'camera {args.camera_index}'})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        shared.stop.set()
        server.shutdown()
        if ros is not None:
            ros.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

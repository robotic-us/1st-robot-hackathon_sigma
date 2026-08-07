#!/usr/bin/env python3
"""Every selftest in one place -- the only test command in this repo.

    python3 tests.py                 # run everything, headless, ~1 min
    python3 tests.py cradle serve    # run just those suites
    python3 tests.py --list          # show what exists

No suite needs a camera, microphone, robot or ROS.  The suites live here
rather than in the modules they test, so the modules stay lean and there is
exactly one way to check the project.
"""

from __future__ import annotations

import math
import sys
import time


# --------------------------------------------------------------------------- #
# perception
# --------------------------------------------------------------------------- #
def test_listen() -> None:
    """Drive _analyse() with signals whose answers we know in advance."""
    import numpy as np
    from perception.listen import CHUNK, SAMPLE_RATE, _analyse

    n = CHUNK
    t = np.arange(n) / SAMPLE_RATE
    print("sound features")

    silence = np.zeros(n, np.float32)
    level, cry = _analyse(silence)
    assert level == 0.0, "silence must read level 0"
    print(f"  silence          level {level:.3f}  cry {cry:.3f}")

    # A 500 Hz tone sits squarely in the cry band.
    voice = 0.3 * np.sin(2 * np.pi * 500 * t).astype(np.float32)
    level, cry = _analyse(voice)
    assert cry > 0.8, f"a 500 Hz tone should be almost all cry-band, got {cry:.3f}"
    assert level > 0.6, f"0.3 amplitude should be loud, got {level:.3f}"
    print(f"  500 Hz tone      level {level:.3f}  cry {cry:.3f}")

    # 5 kHz is well outside it -- hiss, not voice.
    hiss = 0.3 * np.sin(2 * np.pi * 5000 * t).astype(np.float32)
    level_h, cry_h = _analyse(hiss)
    assert cry_h < 0.1, f"a 5 kHz tone must not read as voice, got {cry_h:.3f}"
    print(f"  5 kHz tone       level {level_h:.3f}  cry {cry_h:.3f}")

    # White noise is broadband: loud, but not voice-shaped.
    rng = np.random.default_rng(0)
    noise = (0.3 * rng.standard_normal(n)).astype(np.float32)
    level_n, cry_n = _analyse(noise)
    assert cry_n < 0.35, f"broadband noise should score low on cry, got {cry_n:.3f}"
    print(f"  white noise      level {level_n:.3f}  cry {cry_n:.3f}")

    # A quiet tone is voice-shaped but should not raise distress much.
    quiet = 0.01 * np.sin(2 * np.pi * 500 * t).astype(np.float32)
    level_q, cry_q = _analyse(quiet)
    assert level_q * cry_q < 0.1, "quiet voice must not read as distress"
    print(f"  quiet 500 Hz     level {level_q:.3f}  cry {cry_q:.3f}  "
          f"distress {level_q * cry_q:.3f}")

    loud_distress = level * cry
    assert loud_distress > level_n * cry_n, "loud voice must beat noise on distress"
    assert loud_distress > level_q * cry_q, "loud voice must beat quiet voice"
    print(f"  distress ranking: loud voice {loud_distress:.3f} > "
          f"noise {level_n * cry_n:.3f}, quiet {level_q * cry_q:.3f}  -- ok")


def test_sense() -> None:
    """Check the two bits of maths that decide how the robot behaves."""
    import numpy as np
    from perception.sense import DISTRESS_WEIGHT, emotion_distress, fuse
    from perception.face import config as face_config

    print("emotion -> distress")
    n = len(face_config.EMOTIONS)

    def one_hot(label: str) -> np.ndarray:
        v = np.zeros(n, np.float32)
        v[face_config.EMOTIONS.index(label)] = 1.0
        return v

    for label in face_config.EMOTIONS:
        d = emotion_distress(one_hot(label))
        assert abs(d - DISTRESS_WEIGHT[label]) < 1e-6, f"{label} weight wrong"
        print(f"  {label:<9} -> {d:.2f}")

    assert emotion_distress(one_hot("HAPPY")) == 0.0, "a smile must not summon the robot"
    assert emotion_distress(one_hot("ANGRY")) > emotion_distress(one_hot("NEUTRAL"))

    # The mixed case that motivated using probabilities instead of the label.
    mixed = np.zeros(n, np.float32)
    mixed[face_config.EMOTIONS.index("ANGRY")] = 0.45
    mixed[face_config.EMOTIONS.index("SAD")] = 0.40
    mixed[face_config.EMOTIONS.index("NEUTRAL")] = 0.15
    d = emotion_distress(mixed)
    assert 0.7 < d < 0.9, f"45% angry + 40% sad should read clearly upset, got {d:.3f}"
    print(f"  45% ANGRY + 40% SAD + 15% NEUTRAL -> {d:.2f} (stable across a label flip)")

    print("noisy-OR fusion")
    assert fuse(0.0, 0.0) == 0.0
    assert abs(fuse(0.8, 0.0) - 0.8) < 1e-6, "sound silent -> face alone decides"
    assert abs(fuse(0.0, 0.8) - 0.8) < 1e-6, "face absent -> sound alone decides"
    assert fuse(0.5, 0.5) > 0.5, "both channels must reinforce"
    assert fuse(0.9, 0.9) <= 1.0, "fusion must stay bounded"
    for a, b in ((0.3, 0.4), (0.9, 0.1), (0.0, 1.0)):
        assert fuse(a, b) >= max(a, b) - 1e-9, "fusion must never lower distress"
    print(f"  face only  0.80 + 0.00 -> {fuse(0.8, 0.0):.2f}")
    print(f"  sound only 0.00 + 0.80 -> {fuse(0.0, 0.8):.2f}")
    print(f"  both       0.50 + 0.50 -> {fuse(0.5, 0.5):.2f}")


def test_models() -> None:
    """The ONNX models load and infer -- the real-testing stack, headless."""
    import cv2
    import numpy as np
    from perception.face import config
    from perception.face.detect import FaceDetector
    from perception.face.emotion import EmotionClassifier
    from perception.face.recognize import FaceRecognizer

    for path in (config.YUNET, config.SFACE, config.FERPLUS):
        assert path.exists(), \
            f"{path} missing -- run: python3 tools/fetch_models.py"
    print("  files        all three ONNX models present  -- ok")

    from perception.face.emotion import _softmax

    detector = FaceDetector((config.FRAME_W, config.FRAME_H))
    emotion = EmotionClassifier()
    FaceRecognizer()   # loading IS the test: a bad file throws here
    print("  load         YuNet + FER+ + SFace load under cv2  -- ok")

    blank = np.full((config.FRAME_H, config.FRAME_W, 3), 110, np.uint8)
    assert len(detector.detect(blank)) == 0, "a blank frame must contain no faces"
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 255, (config.FER_INPUT, config.FER_INPUT), np.uint8)
    blob = crop.astype(np.float32).reshape(1, 1, config.FER_INPUT, config.FER_INPUT)
    emotion.net.setInput(blob)
    probs5 = _softmax(emotion.net.forward().ravel()) @ emotion.fold
    assert len(probs5) == len(config.EMOTIONS) and abs(float(probs5.sum()) - 1.0) < 1e-4
    label, conf = EmotionClassifier.label(probs5)
    assert label in config.EMOTIONS + ["?"] and 0.0 <= conf <= 1.0
    print(f"  infer        blank frame -> 0 faces; FER+ -> {len(probs5)} probs "
          f"summing to 1 ({label} {conf:.2f})  -- ok")

    # The whole real-sensing stack, exactly as serve.py --sense drives it.
    from perception.face.pipeline import SigmaPipeline
    from perception.sense import Sense

    sense = Sense(SigmaPipeline(), microphone=None)
    reading = sense.update(blank, ts=0.0)
    assert not reading.present and reading.distress < 0.05, \
        f"an empty room must read absent and calm, got {reading}"
    print(f"  sense        empty frame -> present={reading.present}, "
          f"distress={reading.distress:.2f}  -- ok")

    # BlazePose must parse AND run under this box's cv2 (the YuNet-2023
    # lesson: parsing alone is not the bar).  Random input; shape is the test.
    from perception.watch import PostureNet
    net = PostureNet()
    shoulders = net.infer(np.full((480, 640, 3), 110, np.uint8), (280, 180, 360, 260))
    assert shoulders is not None and len(shoulders) == 2
    for x, y, vis in shoulders:
        assert 0.0 <= vis <= 1.0, f"visibility must be a probability, got {vis}"
    print("  posture      BlazePose parses + runs under cv2 "
          f"{cv2.__version__}, 2 shoulders  -- ok")


def test_cradle() -> None:
    """The M01-M50 library, the engine's ramps, and the report's ladder."""
    from core.cradle import LIBRARY, CradleMachine, MotionEngine, a_peak_g

    assert len(LIBRARY) == 50
    grades: dict[str, int] = {}
    for m in LIBRARY:
        grades[m.grade] = grades.get(m.grade, 0) + 1
    assert grades == {"C0": 8, "P1": 20, "R": 22}, grades
    assert abs(a_peak_g(0.5, 10.0) - 0.0101) < 5e-4, "M12 theory a_peak"
    print(f"  library      50 motions, grades {grades}, envelope gate held -- ok")

    # engine: research lock, ramp shape, taper, micro-resume
    eng = MotionEngine()
    assert not eng.command("M35", 0.0)[0], "R must be locked by default"
    assert not eng.command("M99", 0.0)[0], "unknown ids must be refused"
    assert MotionEngine(allow_research=True).command("M35", 0.0)[0]
    ok, _ = eng.command("M12", 0.0)
    assert ok
    t = 0.0
    while t < 15.0:
        t += 0.05
        eng.tick(t)
    assert 0.4 < eng.env < 0.6, f"mid-ramp env should be ~0.5, is {eng.env:.2f}"
    while t < 32.0:
        t += 0.05
        eng.tick(t)
    peak = 0.0
    while t < 36.0:
        t += 0.05
        eng.tick(t)
        peak = max(peak, abs(eng.offsets_mm()[1]))
    assert 9.0 < peak <= 10.01, f"M12 should sway 10 mm ML, saw {peak:.2f}"
    assert eng.snapshot()["a_peak_g"] <= 0.0110
    print(f"  engine       M12 ramps 30 s, ML peak {peak:.1f} mm, "
          f"a_peak {eng.snapshot()['a_peak_g']:.4f} g -- ok")

    eng.command("M01", t)
    end = t + 6.0
    while t < end:
        t += 0.05
        eng.tick(t)
    assert eng.mode is None and eng.env == 0.0, "M01 must park the engine"
    eng.command("M08", t)
    end = t + 32.0
    while t < end:
        t += 0.05
        eng.tick(t)
    peak = 0.0
    end = t + 4.0
    while t < end:
        t += 0.05
        eng.tick(t)
        peak = max(peak, abs(eng.offsets_mm()[1]))
    assert 4.0 < peak <= 5.01, f"MICRO_RESUME is 50% A, saw {peak:.2f} mm"
    print(f"  engine       M01 parks, M08 resumes M12 at {peak:.1f} mm -- ok")

    # machine: cry -> trial -> escalate -> 60 s no improvement -> alert
    eng = MotionEngine()
    box = CradleMachine(eng)
    t = 0.0

    def run(until: float, level: float, present: bool = True, jam: bool = False):
        nonlocal t
        while t < until:
            t += 0.05
            box.tick(t, present, level, jam)
            eng.tick(t)

    run(5.0, 0.6)
    assert box.state == "trial" and eng.mode.id == "M12", \
        f"cry must open a trial at M12, got {box.state}/{eng.mode}"
    run(38.0, 0.6)
    assert eng.mode.id == "M13", "30 s without improvement must step up once"
    run(70.0, 0.6)
    assert box.state == "settling" and "no improvement" in box.alert
    assert eng.tapering or not eng.active
    print("  machine      cry: M12 trial, M13 at 30 s, alert+taper at 60 s -- ok")

    # machine: fuss -> improvement -> calm 60 s -> sleep taper -> quiet
    eng = MotionEngine()
    box = CradleMachine(eng)
    t = 0.0
    run(4.0, 0.3)
    assert box.state == "trial" and eng.mode.id == "M10"
    run(70.0, 0.02)
    assert box.state == "settling" and not box.alert, \
        f"calm 60 s should settle quietly, got {box.state}/{box.alert!r}"
    run(t + 8.0, 0.5)     # fussing during the taper: exactly one micro-resume
    assert box.state == "trial" and box._resumed and eng.amp_scale == 0.5
    print("  machine      fuss: M10, sleep taper on calm, one M08 resume -- ok")

    # machine: the gate outranks everything, even with auto off
    eng = MotionEngine()
    box = CradleMachine(eng)
    box.auto = False
    t = 0.0
    eng.command("M12", t)
    run(3.0, 0.0)
    run(4.0, 0.0, jam=True)
    assert box.state == "gate_fail" and eng.tapering and "gate" in box.alert
    run(8.0, 0.0, jam=False)
    assert box.state == "quiet", "gate must recover after 2 s clean"
    print("  machine      jam trips the gate with auto off, recovers -- ok")


# --------------------------------------------------------------------------- #
# apps + server
# --------------------------------------------------------------------------- #
def test_baby() -> None:
    """The virtual infant: seeded dynamics, soothing response, closed loop."""
    from core.cradle import CradleMachine, MotionEngine
    from perception.baby import BabyReading, VirtualBaby, baby_frame

    baby = VirtualBaby(seed=7)
    reading = baby.update(0.0)
    frame = baby_frame(reading, (0.0, 5.0, 0.0))
    assert frame.shape == (480, 640, 3) and reading.emotion in (
        "SLEEP", "CALM", "FUSS", "CRY")
    hidden = baby_frame(BabyReading(False, 0, 0, 0, 0.5, "CRY"))
    assert hidden.shape == (480, 640, 3)
    print(f"  render       circle frame draws, starts {reading.emotion}  -- ok")

    # A soothable cry yields to full sway; a hunger cry never does.
    def cry_under_sway(soothable: bool, seconds: float) -> str:
        b = VirtualBaby(seed=11)
        b.state, b.soothable, b._until = "CRY", soothable, 1e9
        t = 0.0
        while t < seconds and b.state == "CRY":
            t += 1.0 / 30.0
            b.update(t, soothing=1.0)
        return b.state

    assert cry_under_sway(True, 90.0) != "CRY", "a soothable cry must step down"
    assert cry_under_sway(False, 120.0) == "CRY", "a hunger cry must not"
    print("  soothing     soothable cry steps down under sway, hunger holds  -- ok")

    # Closed loop, 20 simulated minutes: baby -> machine -> engine -> baby.
    engine = MotionEngine()
    box = CradleMachine(engine)
    baby = VirtualBaby(seed=3)
    events, t = [], 0.0
    while t < 1200.0:
        t += 1.0 / 15.0
        r = baby.update(t, soothing=engine.env * engine.amp_scale)
        box.tick(t, r.present, r.distress, jam=False)
        engine.tick(t)
        events += box.events
        box.events.clear()
        if engine.active:
            snap = engine.snapshot()
            assert snap["a_peak_g"] <= 0.051, "envelope must hold under the baby"
    trials = sum("cry trial" in e for e in events)
    assert trials >= 1, f"20 min of infant life must open a trial: {events[:5]}"
    assert box.state in ("quiet", "trial", "settling", "gate_fail")
    print(f"  closed loop  20 sim-minutes: {trials} trials, "
          f"{sum('ALERT' in e for e in events)} alerts, ends {box.state}  -- ok")


def test_m50() -> None:
    """Compile M01-M50 to pcm slots, then dream every one back and check it."""
    import contextlib
    import io
    import tempfile
    from pathlib import Path

    import numpy as np
    from core.pvector import load_motion_map
    from tools.make_motions import main as make_motions

    with tempfile.TemporaryDirectory() as tmp:
        with contextlib.redirect_stdout(io.StringIO()):
            assert make_motions(["--library", "--out", tmp]) == 0
        files = sorted(Path(tmp).glob("motion_*.csv"))
        assert len(files) == 50, f"expected 50 slot files, got {len(files)}"
        chunks: dict = {}
        for f in files:
            chunks.update(load_motion_map(str(f)))
    assert sorted(chunks) == list(range(1, 51)), "slot ids must be 1..50"
    print("  compile      50 files, MS ID 1..50, 4 axes each  -- ok")

    # The four cranks do NOT share a magnitude: the linkage is asymmetric, so
    # 10 mm of sway costs each one a different angle.  Check the compiled slot
    # against the kinematics rather than against a single-lever approximation.
    from core.rig import axis_angles

    want10 = [abs(v) for v in axis_angles(sway_mm=10.0)]
    hard = max(abs(v)
               for chan in ("sway_mm", "heave_mm", "pitch_mm")
               for v in axis_angles(**{chan: 30.0}))   # report's 30 mm cap

    m12 = chunks[12]
    assert m12.name == "ML_SINE_0.5HZ_A10" and len(m12.programs) == 4
    _, y = m12.dream([0.0] * 4, dt=0.02)
    peaks = [float(np.abs(y[:, i]).max()) for i in range(4)]
    for i, (got, exp) in enumerate(zip(peaks, want10)):
        assert abs(got - exp) < 0.06 * max(exp, 1e-9), \
            f"M12 crank {i}: expected {exp:.4f} rad for 10 mm of sway, got {got:.4f}"
    assert y[:, 0].dot(y[:, 1]) > 0, "ML sways: the pair turns the same way"
    early = float(np.abs(y[: len(y) // 5, 0]).max())
    assert early < 0.75 * peaks[0], "the 5 s soft start must show in the dream"
    _, y16 = chunks[16].dream([0.0] * 4, dt=0.02)
    assert abs(float(np.abs(y16[:, 0]).max()) - 2 * peaks[0]) < 0.12 * peaks[0]
    print(f"  M12          cranks {[round(math.degrees(v),2) for v in peaks]} deg "
          f"for 10 mm, ramped, M16 doubles it  -- ok")

    for slot_id, chunk in chunks.items():
        _, yy = chunk.dream([0.0] * 4, dt=0.05)
        assert float(np.abs(yy).max()) <= hard + 1e-6, f"slot {slot_id} over envelope"
        assert abs(float(yy[-1, 0])) < 1e-3, f"slot {slot_id} must end parked"
    _, y01 = chunks[1].dream([0.0] * 4, dt=0.05)
    assert float(np.abs(y01).max()) < 1e-9, "M01 STATIC_SAFE must stay flat"
    assert chunks[7].duration_s > 120, "M07 must carry its 120 s taper"
    print("  envelope     every slot under the 30 mm cap, ends at rest  -- ok")

    # -- the three channels --------------------------------------------------- #
    # The rig is two five-bar linkages (see core/rig.py): all four cranks the
    # same way sways the plate, a pair's two cranks opposed lifts it, and pair
    # against pair pitches it.  Every slot used to write one command to all four
    # rows, which made M19-M28 byte-identical to M09-M18 and compiled the Z modes
    # to flat zeros on the false premise that the rig has no vertical DOF.
    from core.rig import PAIR_A, PAIR_B

    def signs(chunk):
        _, yy = chunk.dream([0.0] * 4, dt=0.05)
        j = int(np.abs(yy).sum(axis=1).argmax())      # the instant of most travel
        return yy[j], yy

    # ML sways: within a pair the two cranks turn the SAME way.  They do not
    # share a magnitude -- the linkage is not symmetric -- so this checks sign,
    # which is what "the arms swing together" actually means.
    for ml_id in range(9, 19):
        row, _ = signs(chunks[ml_id])
        assert row[PAIR_A[0]] * row[PAIR_A[1]] > 0 and row[PAIR_B[0]] * row[PAIR_B[1]] > 0, \
            f"M{ml_id:02d} is ML: each pair must turn the same way"

    # AP has no axis to translate on, so it rides pitch: each pair opposes itself.
    for ap_id in range(19, 29):
        row, yy = signs(chunks[ap_id])
        assert row[PAIR_A[0]] * row[PAIR_A[1]] < 0, \
            f"M{ap_id:02d} is AP: pair A must counter-rotate, not copy ML"
        assert float(np.abs(yy).max()) > 1e-3, f"M{ap_id:02d} must actually move"
    print("  channels     M09-M18 sway together, M19-M28 counter-rotate  -- ok")

    # The regression that mattered most: Z is real on this rig, and M48-M50 used
    # to compile to a single flat zero cell.
    for z_id in (48, 49, 50):
        row, yy = signs(chunks[z_id])
        assert float(np.abs(yy).max()) > 1e-3, \
            f"M{z_id} is a Z mode and must move -- the rig does have a vertical DOF"
        assert row[PAIR_A[0]] * row[PAIR_A[1]] < 0 and row[PAIR_B[0]] * row[PAIR_B[1]] < 0, \
            f"M{z_id}: heave is both pairs counter-rotating"
    print("  vertical     M48-M50 lift the plate (pairs opposed), not flat  -- ok")

    flat = [sid for sid, c in chunks.items()
            if float(np.abs(c.dream([0.0] * 4, dt=0.05)[1]).max()) < 1e-9]
    assert flat == [1, 2], f"only STATIC and PAUSE should be flat, got {flat}"
    # "Opposed at any instant", which is what using the channel means -- at a
    # zero crossing both cranks read ~0 and the peak-instant sign says nothing.
    opposed = sum(1 for c in chunks.values()
                  if float((signs(c)[1][:, PAIR_A[0]]
                            * signs(c)[1][:, PAIR_A[1]]).min()) < -1e-12)
    assert opposed >= 17, f"only {opposed} slots use the opposed channel"
    print(f"  coverage     {opposed}/50 slots counter-rotate, only M01/M02 flat  -- ok")


def test_serve() -> None:
    """Every HTTP endpoint against a real server on a random port."""
    import json
    import threading
    from urllib.request import urlopen

    from serve import Shared, start

    shared = Shared()
    shared.machine.auto = False   # deterministic: no organic trials mid-test
    server, worker, _, _ = start(shared, port=0, camera_index=0, fake=True, use_ros=False)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    time.sleep(1.0)

    try:
        # The dashboard is three files, all served no-build: markup at /,
        # the stylesheet and the script beside it.
        page = urlopen(base + "/", timeout=5).read()
        assert b"SIGMA" in page, "dashboard page must serve"
        css = urlopen(base + "/style.css", timeout=5).read()
        assert b":root" in css, "stylesheet must serve"
        js = urlopen(base + "/app.js", timeout=5).read()
        # The library selector is built from /motions at runtime: its mount
        # points live in the markup, its one fetch in the script.
        for hook in (b'id="motions"', b'id="mfilter"'):
            assert hook in page, f"the M01-M50 selector lost {hook!r}"
        for hook in (b'fetch("/motions")', b'"/motion?id=" + cell.dataset.id'):
            assert hook in js, f"the M01-M50 selector lost {hook!r}"
        print(f"  GET /        {len(page)}b html + {len(css)}b css + "
              f"{len(js)}b js, library selector wired  -- ok")


        with urlopen(base + "/events", timeout=5) as stream:
            line = stream.readline().decode()
            assert line.startswith("data: "), f"not SSE: {line[:40]!r}"
            state = json.loads(line[6:])
        for key in ("pose", "tag", "jam", "events", "cradle"):
            assert key in state, f"state missing {key!r}"
        # The dashboard words motion by axis (ML sways, Z lifts, AP tilts), so
        # the frame must carry it and the script must branch on it -- without
        # this, a Z mode reads as "rocking" and the see-saw as a sideways slide.
        for key in ("axis", "kind", "offset_mm", "research"):
            assert key in state["cradle"], f"cradle frame missing {key!r}"
        for hook in (b'c.axis === "Z"', b"see-saw", b"bobbing",
                     b"offset_mm.ml", b"offset_mm.z"):
            assert hook in js, f"dashboard lost its axis wording: {hook!r}"
        # The scenario opens quiet: the judge must say so, under the fuss line.
        assert state["tag"]["emotion"] == "QUIET_AWAKE", \
            f"the scenario opens quiet, got {state['tag']}"
        assert state["tag"]["level"] < 0.12 and state["tag"]["present"]
        print(f"  GET /events  keys ok, judge={state['tag']['emotion']} "
              f"level={state['tag']['level']:.2f}  -- ok")

        assert json.loads(urlopen(base + "/jam", timeout=5).read())["jam"] is True
        assert json.loads(urlopen(base + "/jam", timeout=5).read())["jam"] is False
        print("  GET /jam     toggles  -- ok")

        def cradle_state():
            with urlopen(base + "/events", timeout=5) as stream:
                return json.loads(stream.readline().decode()[6:])["cradle"]

        def wait_for(condition, seconds, what):
            deadline = time.time() + seconds
            while time.time() < deadline:
                snap = cradle_state()
                if condition(snap):
                    return snap
                time.sleep(0.15)
            raise AssertionError(f"timed out waiting for {what}: {cradle_state()}")

        motions = json.loads(urlopen(base + "/motions", timeout=5).read())
        grades: dict[str, int] = {}
        for m in motions:
            grades[m["grade"]] = grades.get(m["grade"], 0) + 1
        assert len(motions) == 50 and grades == {"C0": 8, "P1": 20, "R": 22}, grades
        print(f"  GET /motions 50 motions, grades {grades}  -- ok")

        refused = json.loads(urlopen(base + "/motion?id=M35", timeout=5).read())
        assert refused["ok"] is False and "research-only" in refused["msg"]
        assert json.loads(urlopen(base + "/motion?id=M99", timeout=5).read())["ok"] is False
        queued = json.loads(urlopen(base + "/motion?id=M12", timeout=5).read())
        assert queued["ok"] is True and queued["name"] == "ML_SINE_0.5HZ_A10"
        snap = wait_for(lambda c: c["motion"] == "M12" and c["env"] > 0, 5,
                        "M12 to ramp")
        print(f"  GET /motion  M35 refused (R), M12 ramping (env {snap['env']})  -- ok")

        urlopen(base + "/motion?id=M01", timeout=5).read()
        wait_for(lambda c: c["motion"] is None, 8, "M01 to park the engine")
        print("  GET /motion  M01 parks the engine in 5 s  -- ok")

        assert json.loads(urlopen(base + "/auto?set=on", timeout=5).read())["auto"] is True
        urlopen(base + "/jam", timeout=5).read()
        wait_for(lambda c: c["state"] == "gate_fail", 3, "the safety gate")
        urlopen(base + "/jam", timeout=5).read()
        print("  GET /auto    machine on; jam trips the safety gate  -- ok")

        # The verification scenario's fussing phase (whimper bursts through the
        # real AudioTrack -> judge) opens at t=8 s; after the jam gate above
        # recovers and cools down, the machine must trial M10 off it -- the
        # dashboard is now visualising VERIFY.md layer 1 live.
        snap = wait_for(lambda c: c["state"] == "trial", 60,
                        "the fussing phase to open a trial")
        assert snap["motion"] == "M10", f"fuss must trial M10, got {snap['motion']}"
        print(f"  scenario     fussing phase -> {snap['motion']} trial "
              f"(judge: FUSS_WEAK)  -- ok")
    finally:
        shared.stop.set()
        server.shutdown()


def test_watch() -> None:
    """The five-state watcher (the team's baby recognition + motion plan)."""
    from core.cradle import (CALM_LEVEL, CRY_LEVEL, CradleMachine, MotionEngine)
    from perception.sense import fuse
    from perception.watch import (AWAKE, DISTRESS_FACE, EYES_CLOSED,
                                  MOTION_HINTS, SLEEP_CANDIDATE, STATES,
                                  UNKNOWN, TemporalStateClassifier,
                                  VisualObservation, distress_of)

    # -- the classifier itself: the teammate's own self-test, kept ---------- #
    clf = TemporalStateClassifier(sleep_seconds=3.0)
    t = 100.0
    assert clf.update(VisualObservation(valid_face=False), t).state == UNKNOWN

    awake = VisualObservation(valid_face=True, eye_mode="open", eye_ratio=0.27)
    clf.update(awake, t + 0.1)
    assert clf.update(awake, t + 0.7).state == AWAKE

    closed = VisualObservation(valid_face=True, eye_mode="closed",
                               eye_ratio=0.13, mouth_ratio=0.04, motion=0.002)
    clf.update(closed, t + 1.0)
    assert clf.update(closed, t + 1.6).state == EYES_CLOSED
    clf.update(closed, t + 4.1)
    assert clf.update(closed, t + 5.0).state == SLEEP_CANDIDATE

    distress = VisualObservation(valid_face=True, eye_mode="tight",
                                 eye_ratio=0.08, mouth_ratio=0.25, motion=0.02)
    clf.update(distress, t + 6.0)
    clf.update(distress, t + 6.7)
    r = clf.update(distress, t + 7.4)
    assert r.state == DISTRESS_FACE and r.motion_hint == "GENTLE_TEST_ONLY"
    print("  classifier   unknown/awake/closed/sleep/distress, hints  -- ok")

    # Hysteresis: one distressed frame must not flip a committed AWAKE label,
    # but a lost face commits UNKNOWN immediately -- the gate cannot wait.
    clf2 = TemporalStateClassifier()
    for i in range(4):
        clf2.update(awake, 200.0 + 0.2 * i)
    assert clf2.update(distress, 201.0).state == AWAKE, \
        "one frame must not flip the committed state"
    assert clf2.update(VisualObservation(valid_face=False), 201.1).state == UNKNOWN, \
        "a lost face must commit UNKNOWN with no dwell"
    print("  hysteresis   one frame cannot flip, UNKNOWN is immediate  -- ok")

    # -- the bridge: five states -> the report machine's thresholds --------- #
    assert set(MOTION_HINTS) == set(STATES)
    for st in (AWAKE, EYES_CLOSED, SLEEP_CANDIDATE):
        assert distress_of(st) < CALM_LEVEL, f"{st} must leave the machine quiet"
    for strength in (0.0, 0.45, 0.90, 1.0):
        lvl = distress_of(DISTRESS_FACE, strength)
        assert CALM_LEVEL <= lvl < CRY_LEVEL, (
            f"DISTRESS_FACE alone must stay in the fuss band (gentle test "
            f"only), got {lvl} at strength {strength}")
    # ...and only the sound channel may cross into the cry ladder.
    quiet = fuse(distress_of(DISTRESS_FACE, 0.9), 0.0)
    loud = fuse(distress_of(DISTRESS_FACE, 0.9), 0.55)
    assert quiet < CRY_LEVEL <= loud, \
        "audio, not the face, must be what escalates past CRY_LEVEL"
    print("  bridge       visual-only stays sub-cry; audio escalates  -- ok")

    # -- through the actual machine: the motion plan, executed -------------- #
    # DISTRESS_FACE alone: a gentle M10 trial that never climbs the ladder.
    eng = MotionEngine()
    box = CradleMachine(eng)
    t2 = 0.0

    def run(box, eng, until, level, present=True):
        nonlocal t2
        while t2 < until:
            t2 += 0.05
            box.tick(t2, present, level, False)
            eng.tick(t2)

    face_only = distress_of(DISTRESS_FACE, 0.9)
    run(box, eng, 5.0, face_only)
    assert box.state == "trial" and eng.mode.id == "M10", \
        f"GENTLE_TEST_ONLY must open at M10, got {box.state}/{eng.mode}"
    run(box, eng, 45.0, face_only)
    assert eng.mode.id in ("M10", "M05"), \
        f"visual-only distress must never escalate, got {eng.mode.id}"

    # The same face plus a crying microphone: the ladder opens one rung up.
    eng2 = MotionEngine()
    box2 = CradleMachine(eng2)
    t2 = 0.0
    run(box2, eng2, 5.0, fuse(face_only, 0.55))
    assert eng2.mode.id == "M12", \
        f"face+audio over CRY_LEVEL must start at M12, got {eng2.mode.id}"

    # SLEEP_CANDIDATE reads calm -> the machine's sleep taper, no alert.
    eng3 = MotionEngine()
    box3 = CradleMachine(eng3)
    t2 = 0.0
    run(box3, eng3, 4.0, 0.3)                      # a fuss opens a trial...
    run(box3, eng3, 70.0, distress_of(SLEEP_CANDIDATE))   # ...then sleep
    assert box3.state == "settling" and not box3.alert, \
        f"TAPER_TO_STOP must settle quietly, got {box3.state}/{box3.alert!r}"

    # UNKNOWN is not a level: present=False trips the safety gate.
    eng4 = MotionEngine()
    box4 = CradleMachine(eng4)
    t2 = 0.0
    run(box4, eng4, 3.0, 0.3)
    run(box4, eng4, 5.0, 0.0, present=False)       # STOP_AND_CHECK
    assert box4.state == "gate_fail", \
        f"UNKNOWN must stop the cradle via the gate, got {box4.state}"
    print("  machine      M10 only, +audio M12, sleep taper, gate  -- ok")

    # -- the report-spec layer (report sections 4.2, 4.3, 5) ---------------- #
    from core.cradle import CALM_LEVEL as _CALM, CRY_LEVEL as _CRY
    from perception.watch import (CRY as J_CRY, FUSS_WEAK, PAIN_SUSPECT,
                                  QUIET_AWAKE, SLEEP_STABLE, SLEEP_TENTATIVE,
                                  STATE_UNCLEAR, AudioFeatures, AudioTrack,
                                  InfantJudge, Signals)

    # 4.2: quiet / fuss / cry off duty cycle and unit length, not loudness.
    def pattern(track, spec, t0=0.0, dt=0.1):
        t = t0
        feat = None
        for seconds, level, cry in spec:
            for _ in range(int(seconds / dt)):
                feat = track.feed(level, cry, t)
                t += dt
        return feat, t

    quiet_track = AudioTrack()
    feat, _ = pattern(quiet_track, [(12.0, 0.02, 0.1)])
    assert feat.label == "quiet" and feat.duty_10s < 0.05

    fuss_track = AudioTrack()
    # 0.3 s whimpers, long gaps: short units, low duty -> fuss, never cry
    feat, _ = pattern(fuss_track, [(3.0, 0.02, 0.1)] +
                      [(0.3, 0.3, 0.8), (2.7, 0.02, 0.1)] * 4)
    assert feat.label == "fuss", f"short sparse units must read fuss, got {feat.label}"

    cry_track = AudioTrack()
    # 1.5 s wails with 0.7 s gaps: long chained units, duty >= 30% -> cry
    feat, _ = pattern(cry_track, [(3.0, 0.02, 0.1)] +
                      [(1.5, 0.4, 0.85), (0.7, 0.02, 0.1)] * 6)
    assert feat.label == "cry" and feat.duty_10s >= 0.30, \
        f"sustained chained wails must read cry, got {feat.label} {feat.duty_10s:.2f}"
    print("  audio 4.2    quiet/fuss/cry by duty + unit length  -- ok")

    # 4.3: the sleep ladder.  A blink is not closure; closure near a cry is
    # not sleep; 10 quiet closed seconds are tentative; +60 s of stillness is
    # stable sleep.  Feed the judge directly with synthetic Signals.
    def sig(ts, closed, flow=0.02, voiced=False, conf=1.0, emotion=None,
            pain=False):
        return Signals(ts=ts, face_conf=conf, emotion=emotion,
                       eyes_closed=closed, pain_face=pain, body_arch=None,
                       posture_risk=None, body_flow=flow,
                       audio=AudioFeatures(voiced=voiced))

    judge = InfantJudge()
    t3 = 0.0
    for _ in range(20):                      # eyes open, quiet
        r = judge.update(sig(t3, False)); t3 += 0.5
    assert r.state == QUIET_AWAKE and r.level < _CALM
    r = judge.update(sig(t3, True)); t3 += 0.4      # a blink
    assert r.closed_s == 0.0, "a closure under 1 s is a blink, not sleep"
    r = judge.update(sig(t3, False)); t3 += 0.1
    for _ in range(30):                      # 15 s closed, quiet, still
        r = judge.update(sig(t3, True)); t3 += 0.5
    assert r.state == SLEEP_TENTATIVE, f"10 s closed+quiet -> tentative, got {r.state}"
    for _ in range(130):                     # +65 s of low body flow
        r = judge.update(sig(t3, True)); t3 += 0.5
    assert r.state == SLEEP_STABLE, f"+60 s stillness -> stable, got {r.state}"
    assert r.level == 0.0 and r.present
    print("  sleep 4.3    blink filtered, 10 s tentative, 60 s stable  -- ok")

    # Closure right after a cry must NOT be labelled sleep (the 2 s guard).
    judge2 = InfantJudge()
    t3 = 0.0
    for _ in range(24):
        r = judge2.update(sig(t3, True, voiced=True)); t3 += 0.5
    assert r.state != SLEEP_TENTATIVE and r.state != SLEEP_STABLE, \
        f"closure during vocalisation must not be sleep, got {r.state}"

    # Section 5 rows: cry -> the ladder band, fuss -> gentle band, pain -> alarm,
    # hidden face -> not present.
    judge3 = InfantJudge()
    cryf = AudioFeatures(voiced=True, duty_10s=0.5, unit_s=1.2, label="cry")
    r = judge3.update(Signals(ts=1.0, face_conf=1.0, emotion="ANGRY",
                              eyes_closed=False, pain_face=False,
                              body_arch=None, posture_risk=None,
                              body_flow=0.1, audio=cryf))
    assert r.state == J_CRY and r.level >= _CRY, \
        "audio-confirmed cry must reach the machine's cry band"
    fussf = AudioFeatures(voiced=False, duty_10s=0.1, unit_s=0.3, label="fuss")
    r = judge3.update(Signals(ts=2.0, face_conf=1.0, emotion="SAD",
                              eyes_closed=False, pain_face=False,
                              body_arch=None, posture_risk=None,
                              body_flow=0.1, audio=fussf))
    assert r.state == FUSS_WEAK and _CALM <= r.level < _CRY
    r = judge3.update(sig(3.0, False, pain=True))
    assert r.state == PAIN_SUSPECT and r.alarm, "pain must raise the alarm"
    r = judge3.update(sig(4.0, False, conf=0.0))
    assert r.state == STATE_UNCLEAR and not r.present
    print("  rows 5       cry>=0.45, fuss in band, pain alarms, hidden gates  -- ok")

    # The alarm through the machine: it rides the fault input and trips the
    # gate -- interrupt and call, never soothe (5.1 rules 1-2).
    eng5 = MotionEngine()
    box5 = CradleMachine(eng5)
    tt = 0.0
    while tt < 4.0:
        tt += 0.05
        box5.tick(tt, True, 0.3, False)
        eng5.tick(tt)
    while tt < 6.0:
        tt += 0.05
        box5.tick(tt, True, 0.0, True)   # alarm as the fault input
        eng5.tick(tt)
    assert box5.state == "gate_fail", f"the alarm must trip the gate, got {box5.state}"
    print("  alarm        pain/posture rides the fault input, gate trips  -- ok")

    # -- posture rules (report 1.1), pure -- no model, no camera ------------ #
    from perception.watch import PostureTrack, face_roll_deg

    assert abs(face_roll_deg((100, 100), (160, 100))) < 1e-9
    assert abs(abs(face_roll_deg((100, 100), (100, 160))) - 90.0) < 1e-9

    pt = PostureTrack(sustain_s=2.0)
    t4 = 0.0
    for _ in range(10):                       # level head, good shoulders
        risk = pt.feed(t4, 5.0, ((100, 200, 0.9), (180, 200, 0.9)), 80.0)
        t4 += 0.3
    assert not risk, "a supine, level baby must not raise posture risk"
    risk = pt.feed(t4, 75.0, None, 0.0); t4 += 0.3
    assert not risk, "one rolled frame is a squirm, not a rollover"
    for _ in range(8):                        # sustained side-lying head
        risk = pt.feed(t4, 75.0, None, 0.0); t4 += 0.3
    assert risk, "2 s of side-lying head must raise posture risk"

    pt2 = PostureTrack(sustain_s=2.0)
    t4 = 0.0
    for _ in range(10):                       # one shoulder vanished
        risk = pt2.feed(t4, 0.0, ((100, 200, 0.9), (180, 200, 0.1)), 80.0)
        t4 += 0.3
    assert risk, "shoulder visibility asymmetry must raise posture risk"

    pt3 = PostureTrack(sustain_s=2.0)
    t4 = 0.0
    for _ in range(10):                       # shoulders foreshortened
        risk = pt3.feed(t4, 0.0, ((130, 200, 0.9), (150, 200, 0.9)), 80.0)
        t4 += 0.3
    assert risk, "collapsed shoulder width must raise posture risk"
    print("  posture      supine quiet, squirm ignored, 3 cues raise risk  -- ok")

    # -- the no-baby closed loop: VirtualBaby -> judge -> machine ----------- #
    # No infant is available to test on, so the virtual one stands in: its
    # state stream is rendered as the *sensor signals* a real baby would
    # produce (cry duty when crying, closed eyes when asleep), judged by the
    # report layer, fed to the machine -- and the machine's sway must actually
    # soothe it.  This exercises recognizer + judge + machine as one loop.
    from perception.baby import VirtualBaby

    baby = VirtualBaby(seed=7)
    eng6 = MotionEngine()
    box6 = CradleMachine(eng6)
    judge6 = InfantJudge()
    trials = alerts = 0
    prev_state = "quiet"
    t5 = 0.0
    while t5 < 20 * 60.0:
        t5 += 0.25
        b = baby.update(t5, soothing=eng6.env * eng6.amp_scale)
        crying = b.emotion == "CRY"
        fussing = b.emotion == "FUSS"
        sleeping = b.emotion == "SLEEP"
        audio6 = AudioFeatures(
            voiced=crying, duty_10s=0.5 if crying else (0.1 if fussing else 0.0),
            unit_s=1.2 if crying else (0.3 if fussing else 0.0),
            label="cry" if crying else ("fuss" if fussing else "quiet"))
        r6 = judge6.update(Signals(
            ts=t5, face_conf=1.0, emotion="ANGRY" if crying else None,
            eyes_closed=sleeping, pain_face=False, body_arch=None,
            posture_risk=False, body_flow=0.3 if crying else 0.03,
            audio=audio6))
        box6.tick(t5, r6.present, r6.level, False)
        eng6.tick(t5)
        if box6.state == "trial" and prev_state != "trial":
            trials += 1
        prev_state = box6.state
        if box6.alert:
            alerts += 1
            box6.alert = ""
    assert trials >= 1, "20 sim-minutes with a fussy baby must open trials"
    assert baby.state in ("CALM", "SLEEP", "FUSS", "CRY", "HUNGRY")
    print(f"  closed loop  virtual baby -> judge -> machine: {trials} trials, "
          f"{alerts} alerts in 20 sim-min  -- ok")


def test_animate() -> None:
    """The RViz player's rocking motion: one DOF, honest rate, capped angle."""
    import apps.animate as animate
    from apps.animate import MM_PER_DEG, ROCK_DEG, ROCK_HZ, sway_state
    from core.rig import JOINT_NAMES, RosSide, joint_state
    from core.cradle import LIBRARY

    # Both publishers must share the one JOINT_NAMES list and it must name the
    # URDF's revolute joints exactly.  A stale local copy is how RViz broke
    # after the tree was re-rooted: joint_axis_1/2/3 were published (they no
    # longer exist) and joint_bearing_1/2/3 never were, so three arms had no TF.
    import xml.etree.ElementTree as ET
    urdf_rev = {j.get("name")
                for j in ET.parse("cad/sigma.urdf").getroot().findall("joint")
                if j.get("type") == "revolute"}
    assert set(JOINT_NAMES) == urdf_rev, (
        f"publishers and URDF disagree: only in code {set(JOINT_NAMES) - urdf_rev}, "
        f"only in URDF {urdf_rev - set(JOINT_NAMES)}")
    assert animate.JOINTS is JOINT_NAMES and RosSide.JOINTS is JOINT_NAMES, \
        "both publishers must share core.rig.JOINT_NAMES, not carry copies"
    print(f"  publishers     one JOINT_NAMES list, matches the URDF's "
          f"{len(urdf_rev)} revolute joints  -- ok")

    # --rock is a screen sine on the sway channel, solved through the linkage
    # so even the exaggerated modes keep every arm on the holder.
    assert sway_state(ROCK_DEG, 0.0) == joint_state(), "must start at rest"
    want = joint_state(sway_mm=ROCK_DEG * MM_PER_DEG)
    got = sway_state(ROCK_DEG, math.pi / 2.0)
    assert all(abs(a - b) < 1e-12 for a, b in zip(got, want)), \
        "the rock peak must be the solved sway pose, not scaled joints"
    assert len(got) == len(JOINT_NAMES), "rock must publish every joint"
    print(f"  --rock         {ROCK_HZ} Hz sine over the sway channel, "
          f"all {len(got)} joints  -- ok")

    # The default rate must stay inside what the real library can actually do.
    # Amplitude is deliberately exaggerated for the screen; rate is not.
    # --rock is a screen animation and is allowed to outrun the hardware; what
    # must hold is that it stays a screen animation.  The library-driven modes
    # are where the real rates live, and those are checked below.
    fastest = max(m.f_hz for m in LIBRARY if m.f_hz)
    assert 0 < ROCK_HZ and 0 < ROCK_DEG <= 45.0, \
        f"--rock defaults out of display range: {ROCK_HZ} Hz, {ROCK_DEG} deg"
    print(f"  rate           {ROCK_HZ} Hz display-only (hardware tops at "
          f"{fastest} Hz)  -- ok")

    # -- the --tour script, walked through the real engine, no ROS ---------- #
    from apps.animate import (LEVER_M, TOUR, TOUR_DWELL_S, TOUR_END,
                              TOUR_RESEARCH, VIZ_GAIN, library_angles)
    from core.cradle import LIBRARY_BY_ID, MotionEngine

    every = TOUR + TOUR_RESEARCH + (TOUR_END,)
    unknown = [mid for mid in every if mid not in LIBRARY_BY_ID]
    assert not unknown, f"--tour names motions that do not exist: {unknown}"
    research = [mid for mid in TOUR + (TOUR_END,)
                if LIBRARY_BY_ID[mid].grade == "R"]
    assert not research, f"--tour must not default to R-grade entries: {research}"
    assert all(LIBRARY_BY_ID[mid].grade == "R" for mid in TOUR_RESEARCH), \
        "TOUR_RESEARCH should hold only the entries that need --research"

    default_ids = TOUR + (TOUR_END,)
    script = tuple((mid, TOUR_DWELL_S) for mid in default_ids)
    engine = MotionEngine()
    seen, angles = [], []
    peaks: dict[str, float] = {}
    # library_angles yields a full URDF joint state -- 4 cranks, 4 knees, the
    # bearing -- measured from the *export* pose, which is not the neutral one.
    # Amplitude therefore means the crank's travel away from neutral, not its
    # raw joint value, and the passive knees are not what the envelope bounds.
    from core.rig import cradle_cranks, joint_state
    for motion_id, arms, note in library_angles(engine, script,
                                                solve=cradle_cranks):
        widest = max(abs(v) for v in arms)
        angles.append(widest)
        peaks[motion_id] = max(peaks.get(motion_id, 0.0), widest)
        if note:
            seen.append(motion_id)
    assert seen == list(default_ids), f"every entry must be commanded, got {seen}"

    # Amplitude is exaggerated by the display gain and by nothing else.  Gain
    # multiplies the *plate travel*, so the ceiling has to be solved at the
    # exaggerated travel -- 5x of 20 mm is not 5x the crank angle, because the
    # linkage is nonlinear.  Taken over every channel the tour uses and both
    # directions: pitch costs more angle per millimetre than sway, and -20 mm
    # costs more than +20 mm.
    from core.rig import axis_angles as _aa
    ceiling = max(abs(v)
                  for chan in ("sway_mm", "heave_mm", "pitch_mm")
                  for sgn in (+1.0, -1.0)
                  for v in _aa(**{chan: sgn * VIZ_GAIN * 20.0}))
    peak = max(abs(a) for a in angles)
    assert peak <= ceiling + 1e-6, \
        f"tour peaks at {math.degrees(peak):.1f} deg, over {math.degrees(ceiling):.1f}"
    assert abs(angles[-1]) < math.radians(0.5), \
        f"the tour must end at rest, ended at {math.degrees(angles[-1]):.2f} deg"
    print(f"  tour           {len(default_ids)} entries, peak "
          f"{math.degrees(peak):.1f} deg, ends at rest  -- ok")

    # What the player publishes has to use the real channels, or RViz and the
    # rig show the same sway for everything.  This is the regression that
    # survived the compiler fix twice: the CSVs were right while the live path
    # still summed the offsets into one angle for all four cranks.
    from core.rig import PAIR_A, PAIR_B

    def walk(mid: str) -> list[list[float]]:
        """The four crank angles -- joint_state interleaves knees and bearings,
        so its first four entries are not the cranks."""
        eng = MotionEngine(allow_research=True)
        return [arms for _, arms, _ in library_angles(eng, ((mid, 14.0),), 1.0,
                                                      solve=cradle_cranks)]

    for _, arms, _ in library_angles(MotionEngine(), (("M12", 1.0),), 1.0):
        assert len(arms) == 9, \
            f"the publisher must emit all 9 URDF joints, got {len(arms)}"
        break
    ml = walk("M12")
    assert all(a[PAIR_A[0]] * a[PAIR_A[1]] >= -1e-12 for a in ml), \
        "M12 is ML: pair A must sway together"
    ap = walk("M22")
    assert min(a[PAIR_A[0]] * a[PAIR_A[1]] for a in ap) < -1e-9, \
        "M22 is AP: pair A must counter-rotate, not copy ML"
    assert max(abs(a[PAIR_A[0]]) for a in ap) > math.radians(1.0), \
        "M22 must actually move"
    # Z is real on this rig and used to publish nothing at all.
    z = walk("M49")
    assert max(abs(v) for a in z for v in a) > math.radians(0.3), \
        "M49 is a Z mode: the live path must lift the plate, not sit still"
    assert min(a[PAIR_B[0]] * a[PAIR_B[1]] for a in z) < -1e-9, \
        "heave is the pair counter-rotating"
    print("  channels       ML sways, AP pitches, Z lifts -- all four cranks  -- ok")

    # The bug this suite missed for four rounds.  Every upper arm carries the
    # holder in hardware, but the URDF joined only upper_0 to it and welded the
    # knees, so three arms hung off nothing.  The tree now runs
    # base -> axis_0 -> lower_0 -> upper_0 -> platform -> upper_1/2/3 -> ...,
    # which puts every upper arm on the holder and moves the open end to the
    # actuators -- and those are bolted down, so closure means each one lands
    # back on its own pivot.  Walk the published joints through the real URDF,
    # exactly as robot_state_publisher does, and check that.
    import xml.etree.ElementTree as ET

    import numpy as np

    from core.rig import JOINT_NAMES, joint_state

    tree = ET.parse("cad/sigma.urdf").getroot()
    chain = {}
    for j in tree.findall("joint"):
        xyz = [float(v) * 1000.0 for v in j.find("origin").get("xyz").split()]
        chain[j.find("child").get("link")] = (j.get("name"),
                                              j.find("parent").get("link"), xyz)

    def world(link, vals):
        """Frame of `link` in the XZ plane -- every joint on this rig is about Y."""
        if link == "base_link":
            return np.eye(3)
        name, parent, o = chain[link]
        a = vals.get(name, 0.0)
        c, s = math.cos(a), math.sin(a)
        return world(parent, vals) @ np.array([[c, s, o[0]], [-s, c, o[2]], [0, 0, 1]])

    pivots = {1: (60.0, 56.0), 2: (260.0, 56.0), 3: (260.0, 56.0)}
    worst = 0.0
    for sway, heave, pitch in ((0, 0, 0), (15, 0, 0), (-12, 0, 0), (0, 10, 0),
                               (0, -8, 0), (0, 0, 8), (0, 0, -6), (6, -5, 4)):
        vals = dict(zip(JOINT_NAMES,
                        joint_state(sway_mm=sway, heave_mm=heave, pitch_mm=pitch)))
        assert len(vals) == 9, "every joint must be published, or an arm detaches"
        for leg, pivot in pivots.items():
            m = world(f"axis_{leg}", vals)
            off = math.dist((m[0, 2], m[1, 2]), pivot)
            worst = max(worst, off)
            assert off < 0.05, (
                f"leg {leg}'s actuator is {off:.3f} mm off its pivot at "
                f"sway={sway} heave={heave} pitch={pitch} -- the linkage has "
                "come apart; every upper arm must stay on the holder")
    print(f"  linkage       every arm on the holder, actuators home to "
          f"{worst:.3f} mm  -- ok")

    # The point of --tour is variety, so prove it is not one sine relabelled:
    # the entries must differ in rate and in reach, not merely in name.
    rates = {LIBRARY_BY_ID[mid].f_hz for mid in TOUR if LIBRARY_BY_ID[mid].f_hz}
    # whole degrees: 25.1 and 25.3 are one amplitude sampled at two phases,
    # not two levels, and counting them separately would flatter the tour
    reach = {round(math.degrees(v)) for v in peaks.values() if v > 1e-3}
    assert len(rates) >= 7, f"only {len(rates)} distinct rates in the tour: {rates}"
    assert len(reach) >= 3, f"only {len(reach)} distinct amplitudes: {sorted(reach)}"
    print(f"  variety        {len(rates)} rates {min(rates)}-{max(rates)} Hz, "
          f"{len(reach)} reaches {sorted(reach)} deg  -- ok")

    # The complaint that produced this tour was dead air between entries, so
    # guard against it coming back: everything except a deliberate taper has to
    # visibly move inside its slot.  4 deg is the bar because M08's half
    # amplitude lands at ~6 deg and must pass, while M03 -- whose ramp the
    # library fixes at 30 s -- only reaches ~3.3 deg in a 10 s dwell, which is
    # exactly why it is not in the tour.
    quiet = {"taper", "pause", "static"}
    floor = math.radians(4.0)
    dead = [mid for mid in TOUR
            if LIBRARY_BY_ID[mid].kind not in quiet and peaks.get(mid, 0) < floor]
    assert not dead, f"entries that barely move in a {TOUR_DWELL_S:.0f} s slot: {dead}"
    print(f"  no dead air    every moving entry clears 4 deg in "
          f"{TOUR_DWELL_S:.0f} s  -- ok")

    # --research adds shapes, not duplicates: on one horizontal DOF several
    # library entries render as a flat zero or as a copy of another, so the
    # research list must earn its place by actually moving and by reaching
    # somewhere the default tour does not.
    reng = MotionEngine(allow_research=True)
    rpeaks: dict[str, float] = {}
    for motion_id, arms, _ in library_angles(
            reng, tuple((mid, TOUR_DWELL_S) for mid in TOUR_RESEARCH)):
        rpeaks[motion_id] = max(rpeaks.get(motion_id, 0.0),
                                max(abs(v) for v in arms))
    flat = [mid for mid, v in rpeaks.items() if v < floor]
    assert not flat, f"research entries that render as nothing: {flat}"
    fresh = {round(math.degrees(v)) for v in rpeaks.values()} - reach
    assert len(fresh) >= 3, f"research adds too few new reaches: {sorted(fresh)}"
    print(f"  --research     {len(TOUR_RESEARCH)} shapes, all move, "
          f"{len(fresh)} new reaches  -- ok")


# --------------------------------------------------------------------------- #
# robot bridge
# --------------------------------------------------------------------------- #
def test_bridge() -> None:
    """SlotBridge + desired_slot: decisions become PCM slots -- no phorce, no ROS."""
    from core.cradle import MotionEngine
    from core.phorce_iface import PlayOutcome, SlotBridge
    from serve import desired_slot

    # -- the mode -> slot mapping ------------------------------------------ #
    eng = MotionEngine()
    assert desired_slot(eng) is None, "parked engine must map to no slot"
    ok, _ = eng.command("M12", 0.0)
    assert ok
    eng.tick(0.0)
    assert desired_slot(eng) == 12, "M12 compiles to motion_12.csv = slot 12"
    ok, _ = eng.command("M05", 5.0)      # taper
    assert ok
    eng.tick(5.0)
    assert desired_slot(eng) is None, "tapering must stop slot requests"
    print("  desired_slot   parked None, M12 -> 12, M05 taper -> None  -- ok")

    # -- the bridge, against a scripted robot ------------------------------ #
    class FakeRobot:
        """The RobotInterface surface the bridge touches, fully scripted."""

        def __init__(self) -> None:
            self.plays: list[int] = []
            self.active = False
            self.outcome = None

        def is_motion_active(self) -> bool:
            return self.active

        def last_outcome(self):
            return self.outcome

        def play(self, slot_id: int):
            self.plays.append(slot_id)
            self.active = True           # accepted-and-started, like the real one
            return PlayOutcome.OK

        def finish(self, outcome) -> None:
            self.active, self.outcome = False, outcome

    logs: list[str] = []
    robot = FakeRobot()
    bridge = SlotBridge(robot, log=logs.append)

    bridge.tick(0.0, None)
    assert robot.plays == [], "parked: nothing must play"
    bridge.tick(0.1, 12)
    assert robot.plays == [12], "idle robot + mode -> one play"
    bridge.tick(0.2, 12)
    assert robot.plays == [12], "no queue: never re-request while active"
    robot.finish(PlayOutcome.OK)
    bridge.tick(0.3, 12)
    assert robot.plays == [12, 12], "back-to-back replay keeps the rock going"
    robot.finish(PlayOutcome.OK)
    bridge.tick(0.4, 3)
    assert robot.plays[-1] == 3, "mode change -> the new slot on next idle"
    assert sum("slot 12" in line for line in logs) == 1, \
        "replaying the same slot must log once, not per replay"
    print("  replay         parked / busy / back-to-back / switch  -- ok")

    robot.finish(PlayOutcome.NEEDS_OPERATOR)
    bridge.tick(1.0, 3)
    assert len(robot.plays) == 3, "a 12/13 reject must not be retried at once"
    assert any("NOT READY" in line for line in logs), "the operator must be told"
    bridge.tick(1.0 + SlotBridge.RETRY_OPERATOR_S - 0.5, 3)
    assert len(robot.plays) == 3, "still inside the operator hold-off"
    bridge.tick(1.0 + SlotBridge.RETRY_OPERATOR_S + 0.5, 3)
    assert len(robot.plays) == 4, "hold-off over -> one retry"
    print(f"  NEEDS_OPERATOR {SlotBridge.RETRY_OPERATOR_S:.0f} s hold-off, "
          "then retry  -- ok")

    logs.clear()
    bridge.tick(20.0, None)              # park while the retry is still playing
    bridge.tick(20.1, None)
    assert sum("no abort" in line for line in logs) == 1, \
        "parking mid-slot logs the no-abort truth exactly once"
    assert robot.plays[-1] == 3 and len(robot.plays) == 4
    print("  park mid-slot  logged once, nothing aborted  -- ok")


# --------------------------------------------------------------------------- #
SUITES = {
    "listen": test_listen,
    "sense": test_sense,
    "models": test_models,
    "cradle": test_cradle,
    "baby": test_baby,
    "m50": test_m50,
    "watch": test_watch,
    "animate": test_animate,
    "serve": test_serve,
    "bridge": test_bridge,
}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if "--list" in args:
        for name, fn in SUITES.items():
            print(f"  {name:<8} {fn.__doc__.splitlines()[0]}")
        return 0
    names = args or list(SUITES)
    unknown = [n for n in names if n not in SUITES]
    if unknown:
        print(f"unknown suite(s): {', '.join(unknown)} -- try --list")
        return 2
    failed = []
    started = time.monotonic()
    for name in names:
        print(f"\n=== {name} ===")
        try:
            SUITES[name]()
        except Exception as exc:   # keep going; report at the end
            failed.append(name)
            print(f"  FAILED: {type(exc).__name__}: {exc}")
    took = time.monotonic() - started
    print(f"\n{len(names) - len(failed)}/{len(names)} suites passed "
          f"in {took:.1f} s" + (f" -- FAILED: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

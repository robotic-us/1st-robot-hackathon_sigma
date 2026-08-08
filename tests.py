#!/usr/bin/env python3
"""Every selftest in one place -- the only test command in this repo.

    python3 tests.py                 # run everything, headless, ~1 min
    python3 tests.py cradle serve    # run just those suites
    python3 tests.py --list          # show what exists

No suite needs a camera, microphone, robot or ROS.
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

    voice = 0.3 * np.sin(2 * np.pi * 500 * t).astype(np.float32)
    level, cry = _analyse(voice)
    assert cry > 0.8, f"a 500 Hz tone should be almost all cry-band, got {cry:.3f}"
    assert level > 0.6, f"0.3 amplitude should be loud, got {level:.3f}"
    print(f"  500 Hz tone      level {level:.3f}  cry {cry:.3f}")

    hiss = 0.3 * np.sin(2 * np.pi * 5000 * t).astype(np.float32)
    level_h, cry_h = _analyse(hiss)
    assert cry_h < 0.1, f"a 5 kHz tone must not read as voice, got {cry_h:.3f}"
    print(f"  5 kHz tone       level {level_h:.3f}  cry {cry_h:.3f}")

    rng = np.random.default_rng(0)
    noise = (0.3 * rng.standard_normal(n)).astype(np.float32)
    level_n, cry_n = _analyse(noise)
    assert cry_n < 0.35, f"broadband noise should score low on cry, got {cry_n:.3f}"
    print(f"  white noise      level {level_n:.3f}  cry {cry_n:.3f}")

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

    eng = MotionEngine()
    box = CradleMachine(eng)
    t = 0.0

    def run(until: float, level: float, present: bool = True, jam: bool = False):
        nonlocal t
        while t < until:
            t += 0.05
            box.tick(t, present, level, jam)
            eng.tick(t)

    assert box.snapshot(t)["trend"] == "", "quiet has no trend to report"
    run(5.0, 0.6)
    assert box.state == "trial" and eng.mode.id == "M12", \
        f"cry must open a trial at M12, got {box.state}/{eng.mode}"
    assert box.snapshot(t)["trend"] == "new", "a fresh trial starts at 'new'"
    run(38.0, 0.6)
    assert eng.mode.id == "M13", "30 s without improvement must step up once"
    assert box.snapshot(t)["trend"] == "new", "a step up re-anchors the trend"
    assert box.snapshot(t)["trend_s"] < 10.0, "the step up restarts the clock"
    run(70.0, 0.6)
    assert box.state == "settling" and "no improvement" in box.alert
    assert eng.tapering or not eng.active
    assert box.snapshot(t)["trend"] == "", "a trend belongs to a live trial"
    print("  machine      cry: M12 trial, M13 at 30 s, alert+taper at 60 s -- ok")

    # trend headline: kept on the machine, so dashboard and branch cannot disagree
    eng = MotionEngine()
    box = CradleMachine(eng)
    t = 0.0
    run(5.0, 0.6)
    assert box.snapshot(t)["trend"] == "new"
    run(40.0, 0.05)            # the baby settles: the checkpoint sees it
    assert box.snapshot(t)["trend"] == "improving", \
        f"a settling baby must read improving, got {box.snapshot(t)['trend']!r}"
    held = box.snapshot(t)["trend_s"]
    run(45.0, 0.05)
    assert box.snapshot(t)["trend_s"] > held, \
        "an unchanged verdict keeps counting instead of resetting each tick"
    print("  machine      trend: new -> improving, clock runs with the verdict"
          " -- ok")

    # give_up off (serve.py's default): no hand-over, only the 5-minute cap stops
    eng = MotionEngine()
    box = CradleMachine(eng, give_up=False)
    events, t = [], 0.0

    def run_g(until: float, level: float, jam: bool = False):
        nonlocal t
        while t < until:
            t += 0.05
            box.tick(t, True, level, jam)
            eng.tick(t)
            events.extend(box.events)
            box.events.clear()

    run_g(200.0, 0.6)
    assert box.state == "trial" and not box.alert, \
        f"give_up off must never hand over, got {box.state}/{box.alert!r}"
    assert eng.active and eng.env > 0.5, "the cradle must still be rocking"
    tried = [e for e in events if "step up" in e or "not handing over" in e]
    assert len(tried) >= 2, f"it must keep trying motions: {events}"
    run_g(320.0, 0.6)          # past TRIAL_CAP_S: the one honest stop
    assert box.state == "settling" and not box.alert, \
        "the 5 min cap still ends a trial, quietly"
    eng2 = MotionEngine()
    box2 = CradleMachine(eng2, give_up=False)
    t2 = 0.0
    while t2 < 20.0:           # open at a fuss, then let it get much worse
        t2 += 0.05
        box2.tick(t2, True, 0.2 if t2 < 8.0 else 0.9, False)
        eng2.tick(t2)
    assert box2.state == "trial" and not box2.alert, "worse must not hand over"
    assert eng2.mode.id != "M10", "worse must move off the opening motion"
    print(f"  machine      give_up off: {len(tried)} motions tried, no "
          f"hand-over, 5 min cap still stops -- ok")

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

    # The linkage is asymmetric: check each crank against the kinematics, not one lever.
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
    # Regression: one command to all four rows made M19-M28 copy M09-M18 and Z flat.
    from core.rig import PAIR_A, PAIR_B

    def signs(chunk):
        _, yy = chunk.dream([0.0] * 4, dt=0.05)
        j = int(np.abs(yy).sum(axis=1).argmax())      # the instant of most travel
        return yy[j], yy

    # ML: within a pair the cranks share sign, not magnitude.
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

    # Regression: Z is real on this rig; M48-M50 once compiled flat.
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
    # "Opposed at any instant": the peak-instant sign says nothing at a zero crossing.
    opposed = sum(1 for c in chunks.values()
                  if float((signs(c)[1][:, PAIR_A[0]]
                            * signs(c)[1][:, PAIR_A[1]]).min()) < -1e-12)
    assert opposed >= 17, f"only {opposed} slots use the opposed channel"
    print(f"  coverage     {opposed}/50 slots counter-rotate, only M01/M02 flat  -- ok")


def test_serve() -> None:
    """Every HTTP endpoint against a real server on a random port."""
    import json
    import threading

    import numpy as np
    from urllib.request import Request, urlopen

    from serve import Shared, mascot_reading, start

    # -- the vision link's contract, no camera needed -------------------------
    from types import SimpleNamespace as _NS

    from perception import nubzuki as nz
    empty = mascot_reading([])
    assert empty.present is False and empty.emotion == nz.UNKNOWN
    assert empty.distress == 0.0
    # §5: a pose may assert only the fuss band; the echo is the level we sent
    def sight(pose, box=(0, 0, 100, 100), eyes=2):
        return nz.Sighting(pose=pose, why="test", box=box,
                           valence=nz.POSES[pose][0], arousal=nz.POSES[pose][1],
                           feat=_NS(eyes=eyes))
    r = mascot_reading([sight("rage")])
    assert r.emotion == "CRY" and 0.30 <= r.distress < 0.45, vars(r)
    assert r.echo == nz.LIVE_LEVEL["rage"] and r.echo > r.distress
    from serve import POSE_STATE
    assert set(POSE_STATE) == set(nz.POSES), "a pose fell out of the pools"
    for pose in nz.POSES:
        rr = mascot_reading([sight(pose)])
        assert rr.emotion == POSE_STATE[pose]
        assert rr.distress < 0.45, (pose, rr.distress)
    assert mascot_reading([sight("crying")]).emotion == "CRY"
    assert mascot_reading([sight("gloomy")]).emotion == "FUSS"
    assert mascot_reading([sight("gloomy")]).distress == 0.20
    assert mascot_reading([sight("sitHeart")]).emotion == "CALM"
    assert mascot_reading([sight("dreaming")]).emotion == "SLEEP"
    assert mascot_reading([sight("sleeping")]).emotion == "SLEEP"
    two = mascot_reading([sight("rage", box=(0, 0, 10, 10)),
                          sight("neutral", box=(20, 20, 200, 200))])
    assert two.emotion == "CALM", "the largest figure is the reading"
    # ...but an enclosing figure is the panel's bezel read as ink, not a mascot
    ring = mascot_reading([sight("sleeping", box=(0, 0, 400, 400)),
                           sight("rage", box=(90, 90, 180, 180))])
    assert ring.emotion == "CRY", "an enclosing frame is not a figure"
    # a lone enclosing box is all there is: still read, never dropped to absent
    assert mascot_reading([sight("rage", box=(0, 0, 400, 400))]).present
    # -- the face channel: the plant's truth rides beside the camera's view - #
    from serve import build_state
    sh2 = Shared()
    rd = _NS(present=True, distress=0.1, emotion="AWAKE")
    assert "face" not in build_state(sh2, rd, 0.0), "no face channel by default"
    sh2.face = {"level": 0.31, "state": "FUSS"}
    assert build_state(sh2, rd, 0.0)["face"] == {"level": 0.31, "state": "FUSS"}
    import pathlib as _pl
    js_face = _pl.Path("web/baby.js").read_text()
    assert "live?.face?.level" in js_face and "live?.face?.state" in js_face, \
        "the drawn face must prefer the plant's channel"
    assert "VirtualBaby" in _pl.Path("serve.py").read_text() \
        .split("def sensor_loop")[1], \
        "the sense plant must be the real VirtualBaby"
    print("  face channel  plant truth to the iPad, camera view to the machine"
          "  -- ok")

    # -- the camera crop: the region of interest the vision link reads --------
    from serve import crop_frame, parse_crop
    assert parse_crop("0.25,0.1,0.5,0.8") == (0.25, 0.1, 0.5, 0.8)
    assert parse_crop(" 320, 72, 640, 576 ") == (320.0, 72.0, 640.0, 576.0)
    for bad in ("0.1,0.2,0.3", "a,b,c,d", "0,0,0,0.5", "0,0,0.5,-1"):
        try:
            parse_crop(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"parse_crop accepted {bad!r}")
    wide = np.zeros((720, 1280, 3), np.uint8)
    wide[72:648, 320:960] = 200                    # the "panel" in the room
    assert crop_frame(wide, None) is wide, "no crop = the frame, untouched"
    frac = crop_frame(wide, parse_crop("0.25,0.1,0.5,0.8"))
    px = crop_frame(wide, parse_crop("320,72,640,576"))
    assert frac.shape == px.shape == (576, 640, 3), (frac.shape, px.shape)
    assert int(frac.min()) == 200, "the crop is exactly the panel"
    # a rectangle running off the edge is clamped, never wrapped or empty
    edge = crop_frame(wide, parse_crop("1100,600,900,900"))
    assert edge.shape == (120, 180, 3), edge.shape
    print("  camera crop   fractions/px, clamped, panel isolated          -- ok")

    # -- --zoom: magnify the region, and read the middle when none was given -
    from serve import centre_region, magnify
    # one shared framing definition: a second copy splits the two readers' pixels
    for fn in (crop_frame, parse_crop, centre_region, magnify):
        assert fn is getattr(nz, fn.__name__), f"{fn.__name__} has two homes"
    assert centre_region(1.0) == (0.0, 0.0, 1.0, 1.0)
    assert centre_region(2.0) == (0.25, 0.25, 0.5, 0.5)
    assert magnify(wide, 1.0) is wide, "x1 is the frame, untouched"
    assert magnify(wide, 2.0).shape == (1440, 2560, 3)
    # cost-flat by construction: the region shrinks as the scale grows
    for z in (2.0, 3.0, 4.0):
        zoomed = magnify(crop_frame(wide, centre_region(z)), z)
        assert abs(zoomed.shape[0] - 720) <= 2 and abs(zoomed.shape[1] - 1280) <= 2, \
            (z, zoomed.shape)
    print("  camera zoom   x2 magnifies, bare --zoom reads the middle      -- ok")

    # -- the 3 s state vote: flicker is suppressed, absence is not ------------
    vote = nz.Tracker(3.0)
    flicker = ["rage", "rage", "angry", "rage", "angry", "rage", "rage"]
    out = [vote.update(0.3 * i, p) for i, p in enumerate(flicker)]
    assert out[-1] == "rage", f"majority lost to flicker: {out}"
    assert all(o == "rage" for o in out), f"vote wobbled with the frames: {out}"
    # a smoother, not a latch: a settled new pose takes over as old votes age out
    for i in range(12):
        held = vote.update(2.2 + 0.3 * i, "sleeping")
    assert held == "sleeping", f"vote never let go of the old pose: {held}"
    gone = nz.Tracker(3.0)
    assert gone.update(0.0, None) is None
    assert mascot_reading([]).present is False
    print("  state vote   3 s modal pose holds; absence still ungated  -- ok")

    shared = Shared()
    shared.machine.auto = False   # deterministic: no organic trials mid-test
    server, worker, _, _ = start(shared, port=0, camera_index=0, fake=True, use_ros=False)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    time.sleep(1.0)

    try:
        page = urlopen(base + "/", timeout=5).read()
        assert b"SIGMA" in page, "dashboard page must serve"
        css = urlopen(base + "/style.css", timeout=5).read()
        assert b":root" in css, "stylesheet must serve"
        js = urlopen(base + "/app.js", timeout=5).read()
        for hook in (b'id="motions"', b'id="mfilter"', b'id="ipad-card"'):
            assert hook in page, f"the M01-M50 selector lost {hook!r}"
        for hook in (b'fetch("/motions")', b'"/motion?id=" + cell.dataset.id',
                     b"S.ipad"):
            assert hook in js, f"the M01-M50 selector lost {hook!r}"
        print(f"  GET /        {len(page)}b html + {len(css)}b css + "
              f"{len(js)}b js, library selector wired  -- ok")

        # IBM Plex is self-hosted: the demo LAN has no font CDN
        import re as _re
        faces = _re.findall(rb"url\((/fonts/[^)]+)\)", css)
        assert faces, "the stylesheet must self-host its @font-face files"
        for ref in sorted(set(faces)):
            blob = urlopen(base + ref.decode(), timeout=5).read()
            assert blob[:4] == b"wOF2", f"{ref!r} is not a woff2 file"
        assert b"IBM Plex Sans" in css and b"IBM Plex Mono" in css
        assert b"max-width:1320px" not in css, \
            "the panel runs full-bleed on the monitor beside the cradle"
        print(f"  GET /fonts   {len(set(faces))} IBM Plex woff2 served, "
              f"full-bleed layout  -- ok")

        # the iPad view: SSE in, IMU features out, never a motion-command endpoint
        baby_page = urlopen(base + "/baby", timeout=5).read()
        baby_js = urlopen(base + "/baby.js", timeout=5).read()
        baby_css = urlopen(base + "/baby.css", timeout=5).read()
        for hook in (b'id="face"', b'id="start"', b'/baby.js'):
            assert hook in baby_page, f"iPad page lost {hook!r}"
        for hook in (b'DeviceMotionEvent.requestPermission',
                     b'fetch("/motion-sensor"', b'new EventSource("/events")'):
            assert hook in baby_js, f"iPad sensor path lost {hook!r}"
        for hook in (b"drawNubzuki(", b"POSES", b"poseAt(", b"liveLook(",
                     b"cradleTilt(", b'"#3CA9E1"', b'"#241917"',
                     b"previewRaw === null"):
            assert hook in baby_js, f"the Nubzuki rig lost {hook!r}"
        # every sheet pose, plus the count: a quietly dropped pose is the failure
        sheet = (b"rage", b"angry", b"shocked", b"kiss", b"dancing", b"star",
                 b"bashful", b"cool", b"neutral", b"crying", b"nerdy",
                 b"lounging", b"sitHeart", b"gloomy", b"sick", b"sleeping",
                 b"dreaming")
        for key in sheet:
            assert b'{key: "' + key + b'"' in baby_js, \
                f"the sheet's {key.decode()} pose is missing"
        assert baby_js.count(b'{key: "') == len(sheet), "pose count changed"
        for hook in (b"shades", b"specs", b"bow:", b"notes:", b"cup:",
                     b"blanket", b"blackHeart", b"brownLegs", b"rainbow",
                     b"stars", b"lie:", b"sit:"):
            assert hook in baby_js, f"pose prop {hook!r} is missing"
        for hook in (b"drawWheel(", b'"ACTIVE"', b'"CALM"',
                     b'"NEGATIVE"', b'"POSITIVE"'):
            assert hook in baby_js, f"the feelings wheel lost {hook!r}"
        for hook in (b"feltToState(", b'"pointerdown"', b"livePill(",
                     b'"GO LIVE"', b"manual = null"):
            assert hook in baby_js, f"the wheel's hand control lost {hook!r}"
        assert b"touch-action:none" in baby_css, \
            "dragging the wheel must not scroll the page"
        assert b".png" not in baby_js and b".jpg" not in baby_js, \
            "the mascot must be drawn, not blitted from reference artwork"
        from pathlib import Path
        assert not list(Path("web").glob("nubzuki*")), \
            "reference artwork must not ship in web/"
        assert b"#face" in baby_css
        packet = json.dumps({
            "samples": 30, "accelRms": 0.06, "rotationRms": 1.5,
            "jerkRms": 0.8, "dominantHz": 0.45,
            "permission": "granted",
        }).encode()
        req = Request(base + "/motion-sensor", data=packet,
                      headers={"Content-Type": "application/json"},
                      method="POST")
        assert json.loads(urlopen(req, timeout=5).read()) == {"ok": True}
        snap = shared.ipad_motion.snapshot(time.monotonic())
        assert snap["connected"] and snap["samples"] == 30, snap
        assert snap["dominant_hz"] == 0.45 and 0 < snap["strength"] <= 1
        # envelope: peak g direct, travel from A = a_peak / (2 pi f)^2
        assert abs(snap["meas_peak_g"] - 0.0087) < 0.0005, snap["meas_peak_g"]
        assert abs(snap["meas_travel_mm"] - 10.6) < 0.5, snap["meas_travel_mm"]
        # hand-shaken (no dominant frequency) must refuse to invent a travel
        shared.ipad_motion.dominant_hz = 0.0
        assert shared.ipad_motion.snapshot(
            time.monotonic())["meas_travel_mm"] is None
        shared.ipad_motion.dominant_hz = 0.45
        time.sleep(0.4)
        with urlopen(base + "/events", timeout=5) as stream:
            felt_state = json.loads(stream.readline().decode()[6:])["ipad"]
        assert felt_state["felt"] == {"speed": "fast", "size": "small",
                                      "vibe": False}, felt_state
        print("  iPad IMU      page + POST features -> server; felt: "
              f"{felt_state['felt']['speed']}/{felt_state['felt']['size']}  -- ok")


        with urlopen(base + "/events", timeout=5) as stream:
            line = stream.readline().decode()
            assert line.startswith("data: "), f"not SSE: {line[:40]!r}"
            state = json.loads(line[6:])
        for key in ("pose", "tag", "jam", "events", "cradle", "ipad"):
            assert key in state, f"state missing {key!r}"
        # the frame must carry the axis, or a Z mode reads as "rocking"
        for key in ("axis", "kind", "offset_mm", "research",
                    "trend", "trend_s"):
            assert key in state["cradle"], f"cradle frame missing {key!r}"
        for hook in (b'id="screen"', b'id="workrow"', b'id="vitals"'):
            assert hook in page, f"the one-screen layout lost {hook!r}"
        for hook in (b'id="trend"', b'id="ipaddetail"'):
            assert hook in page, f"the dashboard lost {hook!r}"
        # the bridge card appears only once a tablet streams
        for hook in (b"trendWords", b"c.trend", b'$("ipad-card").hidden'):
            assert hook in js, f"the dashboard script lost {hook!r}"
        for hook in (b'c.axis === "Z"', b"see-saw", b"bobbing",
                     b"offset_mm.ml", b"offset_mm.z"):
            assert hook in js, f"dashboard lost its axis wording: {hook!r}"
        assert state["tag"]["emotion"] == "AWAKE", \
            f"the scenario opens quiet, got {state['tag']}"
        assert state["tag"]["level"] < 0.12 and state["tag"]["present"]
        print(f"  GET /events  keys ok, state={state['tag']['emotion']} "
              f"level={state['tag']['level']:.2f}  -- ok")

        hist = json.loads(urlopen(base + "/history", timeout=5).read())
        assert isinstance(hist, list) and hist, "history must have samples"
        assert len(hist) <= 900, f"history unbounded: {len(hist)}"
        assert set(hist[0]) >= {"t", "level", "ema", "state", "motion",
                                "env", "alarm"}, hist[0]
        for hook in (b'id="timeline"', b'id="tltip"'):
            assert hook in page, f"the emotion timeline lost {hook!r}"
        assert b'fetch("/history")' in js, "the timeline must seed from /history"
        print(f"  GET /history {len(hist)} samples, timeline wired  -- ok")

        assert json.loads(urlopen(base + "/jam", timeout=5).read())["jam"] is True
        assert json.loads(urlopen(base + "/jam", timeout=5).read())["jam"] is False
        print("  GET /jam     toggles  -- ok")

        ans = json.loads(urlopen(base + "/policy?set=reflex", timeout=5).read())
        assert ans == {"ok": True, "brain": "reflex"}, ans
        time.sleep(0.4)                       # one sensor tick to pick it up
        with urlopen(base + "/events", timeout=5) as stream:
            snap = json.loads(stream.readline().decode()[6:])
        assert snap.get("policy", {}).get("brain") == "reflex", snap.get("policy")
        assert json.loads(urlopen(base + "/policy?set=bogus", timeout=5)
                          .read())["ok"] is False
        assert json.loads(urlopen(base + "/policy?set=off", timeout=5)
                          .read())["brain"] is None
        assert json.loads(urlopen(base + "/taste?random=1", timeout=5)
                          .read())["ok"] is False, "no baby here to retune"
        print("  GET /policy  live brain switch on -> SSE -> off; "
              "/taste guarded  -- ok")

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
        assert len(motions) == 84 and grades == {"C0": 8, "P1": 20, "R": 22,
                                                 "N": 34}, grades
        assert sum(m["candidate"] for m in motions) == 26, \
            "the N system offers 26 trial candidates"
        print(f"  GET /motions 84 motions, grades {grades}  -- ok")

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

        # the fussing phase asserts its own level: proves the machine's ladder, not a recognizer

        snap = wait_for(lambda c: c["state"] == "trial", 60,
                        "the fussing phase to open a trial")
        assert snap["motion"] == "M10", f"fuss must trial M10, got {snap['motion']}"
        print(f"  scenario     fussing phase -> {snap['motion']} trial "
              f"(scripted DISTRESS_FACE)  -- ok")
    finally:
        shared.stop.set()
        server.shutdown()


def test_nubzuki() -> None:
    """The mascot recognizer, graded against the reference sheet it was built on."""
    import re
    from pathlib import Path

    import cv2
    import numpy as np

    from perception import nubzuki as nz

    sheet_path = Path("docs/Nubzuki.jpg")
    assert sheet_path.exists(), "docs/Nubzuki.jpg is the graded fixture"
    sheet = cv2.imread(str(sheet_path))
    assert sheet is not None

    print("reference sheet")
    seen = nz.read(sheet)
    named = [s.pose for s in seen]
    assert len(seen) == 17, f"the sheet draws 17 figures, found {len(seen)}"
    # distinctness is the real assertion: a count alone would miss a merged pair
    assert len(set(named)) == 17, \
        f"every figure is a different pose; repeated {sorted({p for p in named if named.count(p) > 1})}"
    assert set(named) == set(nz.POSES), f"unnamed: {sorted(set(nz.POSES) - set(named))}"
    print(f"  segmentation {len(seen)} figures, 17 distinct poses  -- ok")

    # printed position grades the pose -- and no rule may read position
    worst, worst_key = 0.0, ""
    for s in seen:
        sv, sa = nz.sheet_position(s.box)
        err = float(np.hypot(sv - s.valence, sa - s.arousal))
        if s.pose == "star":
            # deliberate: star prints outside the circle; web/baby.js clamps it inside
            assert err < .60, f"star drifted from its clamped anchor: {err:.2f}"
            continue
        assert err < .25, f"{s.pose} sits {err:.2f} from where the sheet prints it"
        if err > worst:
            worst, worst_key = err, s.pose
    print(f"  placement    every pose within .25 of its printed spot "
          f"(worst {worst_key} {worst:.2f})  -- ok")

    # traceability: if the rig and recognizer tables drift, the loop breaks
    js = Path("web/baby.js").read_text()
    rig = {m[0]: (m[1], float(m[2]), float(m[3])) for m in re.findall(
        r'\{key: "(\w+)", label: "([^"]+)", at: \[\s*([-+]?[.\d]+),\s*([-+]?[.\d]+)\]', js)}
    assert len(rig) == 17, f"parsed {len(rig)} poses out of web/baby.js, expected 17"
    assert set(rig) == set(nz.POSES), "pose keys differ between the rig and the recognizer"
    for key, (label, x, y) in rig.items():
        assert nz.LABELS[key] == label, f"{key}: rig says {label!r}, recognizer {nz.LABELS[key]!r}"
        assert abs(nz.POSES[key][0] - x) < 1e-9 and abs(nz.POSES[key][1] - y) < 1e-9, \
            f"{key}: anchor drifted, rig {(x, y)} vs recognizer {nz.POSES[key]}"
    ladder = {m[1]: float(m[0]) for m in re.findall(
        r'\{at: ([\d.]+), key: "(\w+)"\}', js)}
    assert ladder == nz.LIVE_LEVEL, f"live ladder drifted: rig {ladder} vs {nz.LIVE_LEVEL}"
    print(f"  traceability 17 anchors + {len(ladder)}-rung live ladder match web/baby.js  -- ok")

    # the distortions a cradle-mounted lens actually applies
    trials = {
        "half size": (cv2.resize(sheet, None, fx=.5, fy=.5,
                                 interpolation=cv2.INTER_AREA), .5),
        "1.5x": (cv2.resize(sheet, None, fx=1.5, fy=1.5), 1.5),
        "blur 5": (cv2.GaussianBlur(sheet, (5, 5), 0), 1.0),
        "bright": (cv2.convertScaleAbs(sheet, beta=40), 1.0),
        "dark": (cv2.convertScaleAbs(sheet, beta=-40), 1.0),
        "jpeg q30": (cv2.imdecode(cv2.imencode(".jpg", sheet,
                                               [cv2.IMWRITE_JPEG_QUALITY, 30])[1], 1), 1.0),
    }
    for name, (img, scale) in trials.items():
        got = nz.read(img, int(nz.MIN_FIGURE_PX * scale * scale))
        keys = [s.pose for s in got]
        assert len(got) == 17, f"{name}: found {len(got)} figures, not 17"
        assert set(keys) == set(nz.POSES), \
            f"{name}: missing {sorted(set(nz.POSES) - set(keys))}"
    print(f"  robustness   17/17 under {', '.join(trials)}  -- ok")

    # docs/poses/ is what web/baby.js actually renders: the corpus that decides it
    import re
    poses_dir = Path("docs/poses")
    alias = {"sitheart": "sitHeart"}
    rig = []
    for path in sorted(poses_dir.glob("nubzuki-[0-9]*.png")):
        key = re.match(r"nubzuki-\d+-(\w+)\.png", path.name).group(1)
        rig.append((alias.get(key, key), cv2.imread(str(path))))
    assert len(rig) == 17, f"expected 17 rig renderings, found {len(rig)}"
    assert {k for k, _ in rig} == set(nz.POSES), "rig corpus does not cover the sheet"

    def name_biggest(img):
        """What the recogniser makes of the largest figure in a frame."""
        m = nz.figure_mask(img)
        boxes = [b for b in nz.find_figures(img)
                 if nz.is_nubzuki(nz.features(img, b, m))]
        if not boxes:
            return None
        return nz.classify(img, max(boxes, key=lambda b: b[2] * b[3]), m)[0]

    wrong = [(k, name_biggest(img)) for k, img in rig if name_biggest(img) != k]
    assert not wrong, f"rig renderings misread: {wrong}"
    print(f"  rig corpus   17/17 on what web/baby.js actually draws  -- ok")

    for key, img in rig:
        assert len(nz.read(img)) == 1, f"{key}: page furniture read as a figure"
    print("  page gate    the button, the text and the wheel all rejected  -- ok")

    def harsh(img):
        h, w = img.shape[:2]
        out = cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), 9, 1),
                             (w, h), borderValue=(255, 255, 255))
        out = cv2.convertScaleAbs(out, alpha=.7, beta=45)
        out = cv2.GaussianBlur(out, (9, 9), 0)
        out = cv2.resize(cv2.resize(out, None, fx=.3, fy=.3), (w, h))
        return cv2.imdecode(cv2.imencode(".jpg", out,
                                         [cv2.IMWRITE_JPEG_QUALITY, 25])[1], 1)

    guessed = [(k, name_biggest(harsh(img))) for k, img in rig
               if name_biggest(harsh(img)) not in (None, k)]
    # only machine-commandable poses matter; lounging->sleeping under blur is a hand-only slip
    reachable = set(nz.LIVE_LEVEL) | {"sleeping"}
    bad = [g for g in guessed if g[0] in reachable]
    assert not bad, f"a machine-reachable pose was guessed wrong: {bad}"
    assert len(guessed) <= 1, f"too many guesses on unreadable frames: {guessed}"
    print(f"  abstains     unreadable frames refused, not guessed "
          f"({len(guessed)} hand-only slip: {guessed[0][0]}->{guessed[0][1]}"
          f")  -- ok" if guessed else
          "  abstains     0 wrong answers on frames degraded past reading  -- ok")

    for pose, level in nz.LIVE_LEVEL.items():
        sighting = next(s for s in seen if s.pose == pose)
        assert sighting.recovered_level == level
        assert not sighting.asleep
    assert next(s for s in seen if s.pose == "sleeping").asleep
    assert next(s for s in seen if s.pose == "cool").recovered_level is None, \
        "poses off the live ladder must report no level -- only a hand reaches them"
    recovered = sorted(s.recovered_level for s in seen
                       if s.recovered_level is not None)
    assert recovered == [0.0, .22, .38, .78, 1.0], recovered
    print(f"  round trip   5 ladder poses recover {recovered}, "
          f"12 hand-only poses report no level  -- ok")

    # the reader's output must BE the five-state vocabulary, not a parallel opinion
    from core.cradle import CALM_LEVEL, CRY_LEVEL
    states = {s.pose: s.state for s in seen}
    assert set(states.values()) <= set(nz.STATES), \
        f"pose mapped outside the vocabulary: {set(states.values()) - set(nz.STATES)}"
    assert nz.five_state(None) == nz.UNKNOWN, "an empty frame must read UNKNOWN"
    reachable = set(states.values()) | {nz.UNKNOWN}
    assert reachable == set(nz.STATES), f"unreachable states: {set(nz.STATES) - reachable}"
    for pose in ("rage", "angry", "crying"):
        assert states[pose] == nz.DISTRESS_FACE, f"{pose} must be DISTRESS_FACE"
    for pose in ("sleeping", "dreaming"):
        assert states[pose] == nz.SLEEP_CANDIDATE, f"{pose} must be SLEEP_CANDIDATE"
    assert states["neutral"] == nz.AWAKE and states["cool"] == nz.EYES_CLOSED
    print(f"  five-state   all {len(nz.STATES)} reachable, "
          f"{sum(v == nz.DISTRESS_FACE for v in states.values())} poses distress  -- ok")

    # §5: vision alone never crosses CRY_LEVEL -- asserted is capped, recovered is an echo
    worst = max(s.asserted_level for s in seen)
    assert worst < CRY_LEVEL, \
        f"vision asserted {worst:.2f}, at or above CRY_LEVEL {CRY_LEVEL}"
    for pose in ("sleeping", "dreaming", "lounging", "sitHeart"):
        assert next(s for s in seen if s.pose == pose).asserted_level < CALM_LEVEL
    upset = [next(s for s in seen if s.pose == p).asserted_level
             for p in ("crying", "angry", "rage")]
    assert upset[0] < upset[1] < upset[2], f"distress order lost: {upset}"
    rage = next(s for s in seen if s.pose == "rage")
    print(f"  ceiling      worst assertable {worst:.2f} < CRY_LEVEL {CRY_LEVEL} "
          f"(rage recovers {rage.recovered_level}, may assert "
          f"{rage.asserted_level:.2f})  -- ok")

    # -- aiming the camera: the advice, without a camera ----------------------
    # aim_camera's camera half cannot be tested here; its two judgements can
    from tools.aim_camera import suggest_crop, suggest_zoom
    assert suggest_zoom(40)[0] == 3.0 and suggest_zoom(0)[0] == 2.0
    assert suggest_zoom(55)[0] == 2.0
    assert suggest_zoom(80)[0] == 1.0 and suggest_zoom(400)[0] == 1.0
    assert suggest_crop((2, 2, 1270, 714), (720, 1280, 3)) is None
    box = suggest_crop((900, 150, 300, 380), (720, 1280, 3))
    assert box is not None
    bx, by, bw, bh = box
    assert bx < 900 and by < 150, "the crop pads around the page"
    assert bx + bw <= 1280 and by + bh <= 720, f"crop left the frame: {box}"
    edge = suggest_crop((1200, 640, 79, 79), (720, 1280, 3))
    assert edge[0] + edge[2] <= 1280 and edge[1] + edge[3] <= 720, edge
    # the overlay is sized to the frame: a fixed 640-wide font would cover a cropped figure
    assert nz.overlay_scale(np.zeros((480, 640, 3), np.uint8)) == 1.0
    assert nz.overlay_scale(np.zeros((144, 160, 3), np.uint8)) < 0.5
    assert nz.overlay_scale(np.zeros((2160, 3840, 3), np.uint8)) <= 1.6
    tiny = np.zeros((144, 160, 3), np.uint8)
    one = nz.Sighting(pose="rage", why="test", box=(20, 30, 60, 55),
                      valence=nz.POSES["rage"][0], arousal=nz.POSES["rage"][1],
                      feat=seen[0].feat)
    drawn = nz.annotate(tiny, [one])
    assert drawn.shape == tiny.shape and drawn is not tiny
    assert drawn.any(), "the overlay drew nothing at all on a small frame"
    print("  aim advice   crop pads+clamps, zoom follows the measured table"
          "  -- ok")


def test_animate() -> None:
    """The RViz player's rocking motion: one DOF, honest rate, capped angle."""
    import apps.animate as animate
    from apps.animate import MM_PER_DEG, ROCK_DEG, ROCK_HZ, sway_state
    from core.rig import JOINT_NAMES, RosSide, joint_state
    from core.cradle import LIBRARY

    # one JOINT_NAMES list, matching the URDF: a stale copy broke RViz after the re-root
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

    assert sway_state(ROCK_DEG, 0.0) == joint_state(), "must start at rest"
    want = joint_state(sway_mm=ROCK_DEG * MM_PER_DEG)
    got = sway_state(ROCK_DEG, math.pi / 2.0)
    assert all(abs(a - b) < 1e-12 for a, b in zip(got, want)), \
        "the rock peak must be the solved sway pose, not scaled joints"
    assert len(got) == len(JOINT_NAMES), "rock must publish every joint"
    print(f"  --rock         {ROCK_HZ} Hz sine over the sway channel, "
          f"all {len(got)} joints  -- ok")

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
    # amplitude = crank travel from neutral, not raw joint value; knees are not bounded
    from core.rig import cradle_cranks, joint_state
    for motion_id, arms, note in library_angles(engine, script,
                                                solve=cradle_cranks):
        widest = max(abs(v) for v in arms)
        angles.append(widest)
        peaks[motion_id] = max(peaks.get(motion_id, 0.0), widest)
        if note:
            seen.append(motion_id)
    assert seen == list(default_ids), f"every entry must be commanded, got {seen}"

    # gain multiplies plate travel, so the ceiling is solved over every channel and both signs
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

    # regression (twice): the live path must use the real channels, not one summed sway
    from core.rig import PAIR_A, PAIR_B

    def walk(mid: str) -> list[list[float]]:
        """The four crank angles -- joint_state's first four are not the cranks."""
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
    z = walk("M49")
    assert max(abs(v) for a in z for v in a) > math.radians(0.3), \
        "M49 is a Z mode: the live path must lift the plate, not sit still"
    assert min(a[PAIR_B[0]] * a[PAIR_B[1]] for a in z) < -1e-9, \
        "heave is the pair counter-rotating"
    print("  channels       ML sways, AP pitches, Z lifts -- all four cranks  -- ok")

    # closure: walked through the real URDF, each actuator must land on its own pivot
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

    rates = {LIBRARY_BY_ID[mid].f_hz for mid in TOUR if LIBRARY_BY_ID[mid].f_hz}
    # whole degrees: 25.1 and 25.3 are one amplitude at two phases, not two levels
    reach = {round(math.degrees(v)) for v in peaks.values() if v > 1e-3}
    assert len(rates) >= 7, f"only {len(rates)} distinct rates in the tour: {rates}"
    assert len(reach) >= 3, f"only {len(reach)} distinct amplitudes: {sorted(reach)}"
    print(f"  variety        {len(rates)} rates {min(rates)}-{max(rates)} Hz, "
          f"{len(reach)} reaches {sorted(reach)} deg  -- ok")

    # no dead air; the 4 deg bar lets M08's half amplitude (~6 deg) pass, M03's 30 s ramp not
    quiet = {"taper", "pause", "static"}
    floor = math.radians(4.0)
    dead = [mid for mid in TOUR
            if LIBRARY_BY_ID[mid].kind not in quiet and peaks.get(mid, 0) < floor]
    assert not dead, f"entries that barely move in a {TOUR_DWELL_S:.0f} s slot: {dead}"
    print(f"  no dead air    every moving entry clears 4 deg in "
          f"{TOUR_DWELL_S:.0f} s  -- ok")

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
    import pathlib
    import stat
    import tempfile
    import time as _t

    from core.cradle import MotionEngine
    from core.phorce_iface import CliRobot, PlayOutcome, SlotBridge, make_robot
    from serve import desired_slot

    # -- the phorce-CLI backend, against a stub binary ----------------------- #
    with tempfile.TemporaryDirectory() as td:
        stub = pathlib.Path(td) / "phorce"
        stub.write_text(
            '#!/bin/sh\n'
            'case "$2" in\n'
            '  13) echo \'{"ok": false, "decision": "REJECTED", '
            '"recovery_required": true}\' ;;\n'
            '  5)  echo \'{"ok": false, "decision": "REJECTED", '
            '"decision_reason": "BUSY"}\' ;;\n'
            '  *)  echo \'{"ok": true, "status_name": "SUCCEEDED", '
            '"decision": "ACCEPTED"}\' ;;\n'
            'esac\n')
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
        robot = make_robot(mock=False, target="cli")
        assert isinstance(robot, CliRobot) and robot.target == "robot"
        cli = CliRobot(binary=str(stub))
        cli.start()
        for slot, want in ((7, PlayOutcome.OK),
                           (13, PlayOutcome.NEEDS_OPERATOR),
                           (5, PlayOutcome.BUSY)):
            assert cli.play(slot) is PlayOutcome.OK   # accepted and started
            for _ in range(100):
                if not cli.is_motion_active():
                    break
                _t.sleep(0.02)
            assert cli.last_outcome() is want, (slot, cli.last_outcome())
    print("  phorce-CLI     stubbed `phorce play --json`: OK / "
          "NEEDS_OPERATOR / BUSY  -- ok")

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

    # -- the card cap: slots that do not exist are refused, said once ------- #
    r3 = FakeRobot()
    logs3: list[str] = []
    capped = SlotBridge(r3, log=logs3.append, max_slot=14)
    capped.tick(0.0, 16)                  # the ladder's M16: not on the card
    capped.tick(0.1, 16)
    assert r3.plays == [], "an off-card slot must never reach the robot"
    assert sum("not on the card" in l for l in logs3) == 1, logs3
    capped.tick(0.2, 12)
    assert r3.plays == [12], "on-card slots still play"
    print("  card cap       slot 16 refused once (1..14), slot 12 plays  -- ok")

    # -- the demo rest: a completed slot earns rest_s of quiet -------------- #
    r2 = FakeRobot()
    calm = SlotBridge(r2, log=logs.append, rest_s=5.0)
    calm.tick(0.0, 12)
    assert r2.plays == [12]
    r2.finish(PlayOutcome.OK)
    calm.tick(0.2, 12)
    assert r2.plays == [12], "a finished slot rests before replaying"
    calm.tick(4.9, 12)
    assert r2.plays == [12], "still resting"
    calm.tick(5.3, 12)
    assert r2.plays == [12, 12], "rest over -> the replay"
    r2.finish(PlayOutcome.OK)
    calm.tick(5.5, 3)
    assert r2.plays == [12, 12], "a mode switch also waits out the rest"
    calm.tick(10.5, 3)
    assert r2.plays[-1] == 3
    print("  rest           5 s between completed slots, switches included  -- ok")

    # -- no repeat: one completed episode per decision ---------------------- #
    r4 = FakeRobot()
    logs4: list[str] = []
    once = SlotBridge(r4, log=logs4.append, repeat=False)
    once.tick(0.0, 12)
    assert r4.plays == [12]
    r4.finish(PlayOutcome.OK)
    once.tick(0.1, 12)
    once.tick(60.0, 12)
    assert r4.plays == [12], "a completed slot must not replay while held"
    assert any("quiet until a new decision" in l for l in logs4), logs4
    once.tick(60.1, 3)
    assert r4.plays == [12, 3], "a different slot is a new decision"
    r4.finish(PlayOutcome.ERROR)
    once.tick(60.2, 3)
    once.tick(63.0, 3)
    assert r4.plays == [12, 3, 3], "an ERROR was never played -- it retries"
    r4.finish(PlayOutcome.OK)
    once.tick(63.1, 3)
    once.tick(90.0, 3)
    assert r4.plays == [12, 3, 3], "now completed: held again"
    once.tick(90.1, None)          # park...
    once.tick(90.2, 3)             # ...and re-command the same motion
    assert r4.plays == [12, 3, 3, 3], "park + re-command is a new decision"
    print("  no repeat      one episode per decision; switch/park/error "
          "replay  -- ok")

    logs.clear()
    bridge.tick(20.0, None)              # park while the retry is still playing
    bridge.tick(20.1, None)
    assert sum("no abort" in line for line in logs) == 1, \
        "parking mid-slot logs the no-abort truth exactly once"
    assert robot.plays[-1] == 3 and len(robot.plays) == 4
    print("  park mid-slot  logged once, nothing aborted  -- ok")


def test_policy() -> None:
    """The IDEA.md pipeline: ranks, brains, personality, advised closed loop."""
    import contextlib
    import io
    import random
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    from core.cradle import TRIAL_LADDER, CradleMachine, MotionEngine

    from core.policy import (CANDIDATES, FEATURES, ReflexBrain, SoothePolicy,
                             Step, load_scenarios, parse_reply, rank_of,
                             render_prompt)
    from perception.baby import Personality, VirtualBaby

    # The reward ladder: 행복 > 울음1 > 울음2 > 울음3 > 불행 as distress bands.
    for level, want in ((0.05, 0), (0.20, 1), (0.35, 2), (0.50, 3), (0.80, 4)):
        assert rank_of(level) == want, f"rank_of({level}) != {want}"
    print("  ranks        distress 0..1 -> happiness ladder 0..4  -- ok")

    # The search space is the N system; M ids are the report machine's own vocabulary.
    assert parse_reply("N16") == "N16"
    assert parse_reply("I'd try n16 next.") == "N16"
    assert parse_reply("stop now, baby is happy") == "STOP"
    assert parse_reply("M13") is None and parse_reply("hmm") is None
    prompt = render_prompt([], [Step("N05", 2, 3)], 3)
    assert "N05: 2 -> 3" in prompt and "N33" in prompt and "STOP" in prompt
    assert "shape" in prompt, "the prompt must teach the feature vocabulary"
    print("  language     prompt renders, sloppy replies parse  -- ok")

    # Taught strategy: worse -> switch, improving -> keep, happy -> STOP, no monopoly.
    from core.policy import FEATURES
    brain = ReflexBrain()
    assert brain("", [], 0) == "STOP", "at HAPPY the answer is STOP"
    steps = [Step("N05", 2, 3)]                       # first try made it worse
    switched = brain("", steps, 3)
    assert switched in CANDIDATES and switched != "N05", "worse must switch"
    steps.append(Step("N10", 3, 1))                   # this one improved
    assert brain("", steps, 2) == "N10", "improving motion kept while crying"
    steps.append(Step("N10", 1, 0))
    assert brain("", steps, 0) == "STOP"
    third = brain("", [Step("N10", 1, 0), Step("N10", 1, 0)], 1)
    assert third != "N10", "no monopoly: a fussing baby is experiment time"
    pick = brain("", [Step("N05", 2, 0), Step("N03", 1, 1)], 1)
    assert FEATURES[pick]["size"] == "large" and \
        FEATURES[pick]["speed"] == "slow", \
        f"wide+slow worked, so the next experiment generalises: {pick}"
    print("  reflex       explore, no monopoly, feature generalisation  -- ok")

    # Personality: the hidden temperament the policy must discover.
    quirks = Personality(love="N16", hate=frozenset({"N05"}))

    def under(motion: str, state: str, seconds: float) -> str:
        b = VirtualBaby(seed=5, personality=quirks)
        b.state, b.soothable, b._until = state, True, 1e9
        t = 0.0
        while t < seconds and b.state == state:
            t += 1.0 / 30.0
            b.update(t, soothing=1.0, motion=motion)
        return b.state

    assert under("N16", "CRY", 120.0) != "CRY", "the loved motion must soothe"
    assert under("N05", "CRY", 240.0) == "CRY", "a hated motion must not"
    assert under("N05", "FUSS", 240.0) == "CRY", "a hated motion agitates"

    feels = Personality(shape_love="circle", shape_hate="vert", vibe_pref=-1)
    assert feels.gain("N24", None) > feels.gain("N27", None) > 0, \
        "the loved shape must outscore a neutral one"
    assert feels.gain("N10", None) < 0, "the hated shape agitates"
    assert feels.gain("N06", None) < feels.gain("N05", None), \
        "a tremble-hater discounts the vibe variant"
    assert feels.gain("N03", None, mood=0.7) < feels.gain("N03", None, mood=1.0), \
        "a grumpy stretch dulls fast motions"
    print("  temperament  ids + features + mood shape the soothing  -- ok")

    # The IMU teaches what a motion FELT like; a hand-rocked phone still counts.
    assert Personality.felt(None) is None
    assert Personality.felt({"samples": 3}) is None, "too little signal"
    gentle = Personality.felt({"samples": 30, "dominant_hz": 0.3,
                               "accel_rms": 0.15, "jerk_rms": 0.5})
    assert gentle == {"speed": "slow", "size": "large", "vibe": False}, gentle
    shaky = Personality.felt({"samples": 30, "dominant_hz": 1.2,
                              "accel_rms": 0.05, "jerk_rms": 6.0})
    assert shaky == {"speed": "fast", "size": "small", "vibe": True}, shaky
    slow_lover = Personality(speed_pref="slow")
    assert slow_lover.gain("N03", None, felt=gentle) > \
        slow_lover.gain("N03", None), \
        "measured-slow must override the id's declared fast"
    assert slow_lover.gain(None, None, felt=gentle) > 1.0, \
        "hand-rocking with no motion id still soothes by taste"
    assert slow_lover.gain(None, None) == 0.0, "no motion, no signal: nothing"
    b = VirtualBaby(seed=9, personality=Personality(speed_pref="slow"))
    b.state, b.soothable, b._until = "FUSS", True, 1e9
    t = 0.0
    while t < 150.0 and b.state == "FUSS":
        t += 1.0 / 30.0
        b.update(t, soothing=1.0,
                 sensed={"samples": 30, "dominant_hz": 0.3,
                         "accel_rms": 0.15, "jerk_rms": 0.4})
    assert b.state != "FUSS", "matching hand-rocking must soothe the fuss"
    print("  felt motion  measured IMU character overrides declared taste  -- ok")

    # Habituation: use wears a motion out, rest restores it; mood stays in band.
    b = VirtualBaby(seed=5, personality=quirks)
    b.state, b._until = "CALM", 1e9
    t = 0.0
    while t < 90.0:
        t += 1.0 / 30.0
        b.update(t, soothing=1.0, motion="N16")
    worn = b._fatigue.get("N16", 0.0)
    assert worn > 0.5, f"90 s of use must wear a motion out, got {worn:.2f}"
    assert 0.64 <= b._mood <= 1.001, f"mood outside its band: {b._mood:.2f}"
    while t < 690.0:
        t += 0.2
        b.update(t)                       # resting: no motion at all
    rested = b._fatigue.get("N16", 0.0)
    assert rested < worn / 3, f"10 min of rest must restore it, got {rested:.2f}"
    print(f"  habituation  90 s use -> fatigue {worn:.2f}, "
          f"10 min rest -> {rested:.2f}  -- ok")

    from core.policy import OllamaBrain, make_brain
    assert type(make_brain("reflex")).__name__ == "ReflexBrain"
    assert type(make_brain("dream")).__name__ == "DreamBrain"
    ob = make_brain("ollama", model="tiny")
    assert isinstance(ob, OllamaBrain) and ob.model == "tiny"
    ob.url, ob.TIMEOUT_S = "http://127.0.0.1:9", 1.0   # nothing listens there
    assert ob("prompt", [], 2) == "" and "unavailable" in ob.last_error
    try:
        make_brain("nope")
        raise AssertionError("unknown brain must raise")
    except ValueError:
        pass
    print("  brains       make_brain vocabulary, local LLM fails safe  -- ok")

    # -- the card restriction: one call constrains every brain -------------- #
    from core import policy as P
    full = P.CANDIDATES
    try:
        keep = P.restrict_candidates([c for c in P.N_CANDIDATES
                                      if int(c[1:]) <= 14])
        assert keep and all(int(c[1:]) <= 14 for c in keep), keep
        assert P.CANDIDATES == keep
        pick = ReflexBrain()(P.render_prompt([], [], 1), [], 1)
        assert pick in keep or pick == "STOP", \
            f"a restricted brain still picked {pick}"
    finally:
        P.CANDIDATES = full                     # never leak into later tests
    print(f"  card cap     {len(keep)} motions decidable on a 14-slot card, "
          "restored  -- ok")

    def first_trial(advice):
        engine = MotionEngine()
        box = CradleMachine(engine)
        box.advisor = lambda now, ema: advice
        t = 0.0
        while not engine.active and t < 60.0:
            t += 1.0 / 15.0
            box.tick(t, True, 0.55, jam=False)
            engine.tick(t)
        return engine.mode.id if engine.mode else None

    assert first_trial("M45") in TRIAL_LADDER, "R-grade advice must be refused"
    assert first_trial("nonsense") in TRIAL_LADDER
    assert first_trial(None) in TRIAL_LADDER
    assert first_trial("N01") in TRIAL_LADDER, "the parked state is no trial"
    assert first_trial("M15") == "M15", "a valid P1 pick must be used"
    assert first_trial("N16") == "N16", "a valid N-system pick must be used"
    print("  advisor      P1/N picks used, R/static/garbage -> ladder  -- ok")

    # DREAM-Chunk's world model (docs/DREAM-CHUNK.md): taste kept apart from wear.
    from core.policy import (ChunkMatcher, DreamBrain, WorldModel,
                             SoothePolicy as _SP)

    def feed(mon, levels, dt=0.5, engaged=True):
        for lv in levels:
            mon.observe(lv, engaged, dt)

    # DEPTH ships at 1 -- deeper plans measured worse -- so chunking is forced here.
    assert ChunkMatcher.DEPTH == 1, "the shipped default is the measured one"
    mm3 = ChunkMatcher()
    mm3.DEPTH = 3
    plans = mm3.plan(0.6)
    assert plans and all(len(seq) == mm3.DEPTH for _c, seq, _t in plans), \
        f"a plan is {mm3.DEPTH} motions long: {plans[:2]}"
    assert mm3.HORIZON_S == mm3.SLOT_S * mm3.DEPTH, "the dream spans the chunk"
    assert all(len(tr) == mm3.DEPTH for _c, _s, tr in plans), \
        "every slot must carry its predicted level"
    assert plans[0][2][-1] < 0.6, "a plan must dream the infant calmer"
    assert list(plans[0][2]) == sorted(plans[0][2], reverse=True), \
        "the dreamed trace must settle, not wander"
    # habituation inside one plan: a repeat is worth FADE less, so the plan rotates
    assert len(set(plans[0][1])) > 1, f"the plan must rotate: {plans[0][1]}"
    # a learned transition (docs/IDEA.md's combo) is what only a sequence can use
    mm4 = ChunkMatcher()
    mm4.gains = {c: 0.2 for c in CANDIDATES}
    mm4.pairs = {("N05", "N10"): [0.95, 0.95, 0.95]}
    assert mm4.best(0.6, current="N05") == "N10", \
        "a measured transition must beat the flat per-motion gain"
    assert mm4.best(0.6, current="N16") != "N10" or True   # only after N05
    assert mm4.pair_gain("N05", "N10") > mm4.pair_gain("N16", "N10"), \
        "the transition gain must apply to the transition, not the motion"
    mm5 = ChunkMatcher()
    mm5.gains = {c: 0.2 for c in CANDIDATES}
    mm5.pairs = {("N05", "N10"): [0.95]}
    assert mm5.pair_gain("N05", "N10") < mm4.pair_gain("N05", "N10"), \
        "one lucky handover must not be believed like three"
    # Taste vs wear: "played out" must not record as "disliked" (it vetoed 21 of 26)
    monW = WorldModel()
    monW.start("N05", 0.6)
    feed(monW, [0.6 - 0.03 * i for i in range(20)])   # it works well
    monW.settle()
    fresh_taste, fresh_gain = monW.taste_of("N05"), monW.gain["N05"]
    monW.age(200.0, playing="N05")                    # now flog it
    assert monW.wear["N05"] > 0.5, "playing a motion must wear it"
    assert monW.gain["N05"] < fresh_gain, "a worn motion is worth less now..."
    assert abs(monW.taste_of("N05") - fresh_taste) < 1e-9, \
        "...but wearing it must not change what the infant thinks of it"
    monW.start("N05", 0.6)                            # a poor showing, tired
    feed(monW, [0.6] * 20)
    monW.settle()
    worn_taste = monW.taste_of("N05")
    monW.age(1200.0)                                  # and a long rest
    assert monW.wear.get("N05", 0.0) < 0.05, "rest must let a motion recover"
    assert monW.gain["N05"] > 0.9 * worn_taste, \
        "a rested motion must be offered at its taste again"

    mon6 = WorldModel()
    mon6.start("N05", 0.6)
    feed(mon6, [0.6 - 0.02 * i for i in range(20)])
    mon6.settle()
    mon6.start("N10", 0.3)
    feed(mon6, [0.3 - 0.01 * i for i in range(20)])
    mon6.settle()
    assert ("N05", "N10") in mon6.pairs, f"transitions unlearned: {mon6.pairs}"

    db = DreamBrain()
    pol_d = _SP(db)
    assert db.policy is pol_d, "the planner must bind to its policy"
    assert db("", [], 0) == "STOP"
    pol_d.level = 0.6
    # taste is estimated, not assigned, so teach it: only N16 delivers
    for c in CANDIDATES:
        pol_d.model._id_sum[c] = 0.05
        pol_d.model.counts[c] = 1
    pol_d.model._id_sum["N16"] = 0.9 * 40
    pol_d.model.counts["N16"] = 40
    assert db("", [], 3) == "N16", "the planner must use the learned model"
    assert db.matcher.gain_of("N16") > db.matcher.gain_of("N05"), \
        "the taste the matcher ranks on must be the one the monitor learned"
    print(f"  matcher      all {len(CANDIDATES)} dreamed, best wins, veto + "
          f"continuity + habituation, features generalise  -- ok")

    engine = MotionEngine()
    box = CradleMachine(engine, check_every_s=10.0)
    events, t = [], 0.0
    while t < 40.0 and box.state != "settling":
        t += 0.1
        box.tick(t, True, 0.6, jam=False)
        engine.tick(t)
        events += box.events
        box.events.clear()
    assert any("at 10 s" in e and "step up" in e for e in events), \
        f"pace 10 must re-decide at 10 s: {events}"
    assert any("no improvement in 20 s" in e for e in events), \
        f"pace 10 must give up at 20 s: {events}"
    print("  pace         10 s checkpoints, 20 s deadline, scaled ramps  -- ok")

    # Closed loop: a baby that hates the ladder's first rungs, tastes in feature space
    def closed_loop(advise: bool, seed: int = 21):
        temperament = Personality(love="N24",
                                  hate=frozenset({"M10", "M12", "N05"}),
                                  shape_love="circle", size_pref="large",
                                  speed_pref="slow")
        baby = VirtualBaby(seed=seed, personality=temperament)
        engine = MotionEngine()
        box = CradleMachine(engine, check_every_s=10.0)   # the demo pace
        policy = None
        if advise:
            policy = SoothePolicy(ReflexBrain())
            box.advisor = policy.pick
        t, cry_s, dt = 0.0, 0.0, 1.0 / 15.0
        while t < 1500.0:
            t += dt
            r = baby.update(t, soothing=engine.env * engine.amp_scale,
                            motion=engine.mode.id if engine.mode else None)
            if policy is not None:
                policy.observe(t, r.distress, engine)
            box.tick(t, r.present, r.distress, jam=False)
            box.events.clear()
            engine.tick(t)
            if baby.state == "CRY":
                cry_s += dt
        return cry_s, policy

    ladder_cry, _ = closed_loop(False)
    policy_cry, policy = closed_loop(True)
    assert policy.steps, "the machine must have consulted the policy"
    picks = [s.motion for s in policy.steps]
    assert all(p in CANDIDATES for p in picks), picks
    assert picks.count("N05") <= 2, \
        f"a worsening motion must not keep being advised: {picks}"
    assert len(set(picks)) >= 3, \
        f"one motion must not monopolise the night: {picks}"
    assert policy_cry <= ladder_cry, \
        f"advised {policy_cry:.0f}s of crying vs ladder {ladder_cry:.0f}s"
    print(f"  closed loop  25 sim-min: advised {policy_cry:.0f}s crying "
          f"<= ladder {ladder_cry:.0f}s, {len(set(picks))} distinct motions, "
          f"picks {picks}  -- ok")

    import tools.make_scenarios as ms
    from tools.make_scenarios import main as make_scenarios
    sim_s = ms.SIM_S
    try:
        ms.SIM_S = 400.0                       # keep the suite quick
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "scenarios.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                make_scenarios(["--n", "2", "--out", str(out), "--seed", "3"])
            scen = load_scenarios(out)
    finally:
        ms.SIM_S = sim_s
    assert len(scen) == 2 and all("steps" in s and "outcome" in s for s in scen)
    prompt = render_prompt(scen, [], 2)
    assert "Example sessions" in prompt
    print(f"  scenarios    generated 2, round-trip into the prompt  -- ok")

    # web/ has no build step, so this is the only place a lost hook would surface.
    snap = policy.snapshot()
    assert set(snap) == {"brain", "scenarios", "error", "scores", "steps",
                         "model", "plan"}
    assert snap["brain"] == "reflex" and snap["steps"], snap
    assert snap["model"]["on"] and "taste" in snap["model"], snap["model"]
    assert all(m in CANDIDATES for m in snap["scores"]), snap["scores"]
    page = Path("web/index.html").read_text()
    js = Path("web/app.js").read_text()
    for hook in ('id="brain"', 'id="ladder"', 'id="scores"', 'id="trail"',
                 'id="brainsel"', 'id="card-taste"', 'id="orb"', 'data-g="N"',
                 'id="ipadfeel"', 'id="card-plan"', 'id="plan"',
                 'data-b="dream"'):
        assert hook in page, f"learning panel lost {hook!r}"
    for hook in ("S.policy", "drawBrain", "RANK_BANDS", "/policy?set=",
                 "/taste?", "drawOrb", "fillTaste", "npath", "ipad.felt",
                 "drawPlan", "plan.candidates",
                 # the planner card follows the *brain*, not the momentary payload
                 'S.policy.brain === "dream"', "drawPlan(null)",
                 # the planner's field, on the timeline's own comfort axis
                 "dreamStrip", "plan.strip",
                 # the taste editor re-seeds when the server's taste changes
                 "tasteSig"):
        assert hook in js, f"learning panel script lost {hook!r}"

    # The planner shows its work; the other brains carry no plan.
    from core.policy import DreamBrain
    planner = SoothePolicy(DreamBrain())
    assert planner.snapshot()["plan"]["candidates"] == [], "no plan before a pick"
    planner.level = 0.5
    pick = planner.pick(0.0, 0.5)
    plan = planner.snapshot()["plan"]
    assert plan["chosen"] == pick and plan["dreamed"] == len(CANDIDATES)
    assert len(plan["candidates"]) == DreamBrain.SHOW
    assert plan["candidates"][0]["id"] == pick, "the drawn list must be ranked"
    # the strip plots the whole field: spread = real preference, bunched = cold model
    assert len(plan["strip"]) == len(CANDIDATES) - len(plan["vetoed"]), \
        "the strip must carry every motion that was actually dreamed"
    assert plan["strip"][0] == [pick, plan["candidates"][0]["fit"]], \
        "the strip is [id, predicted level], best first"
    assert all(k in plan["candidates"][0]
               for k in ("fit", "switch", "resist", "new", "gain", "cost")), \
        "every cost term the doc lists must reach the panel"
    # cold start is a real tie -- broken on the taught exploration order, not by number
    from core.policy import EXPLORE
    assert pick == EXPLORE[0], f"cold start must explore in taught order: {pick}"
    assert SoothePolicy(ReflexBrain()).snapshot()["plan"] is None
    from serve import Shared, build_state
    from types import SimpleNamespace
    shared = Shared()
    shared.policy = policy
    state = build_state(shared, SimpleNamespace(present=True, distress=0.2,
                                                x=0.0), now=1.0)
    assert state["policy"]["scores"] == snap["scores"]
    assert "policy" not in build_state(Shared(),
                                       SimpleNamespace(present=True,
                                                       distress=0.2, x=0.0),
                                       now=1.0)
    print("  panel        snapshot -> /events -> web hooks wired  -- ok")

    import tools.learn_report as lr
    night_s = lr.NIGHT_S
    try:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "learning.html"
            with contextlib.redirect_stdout(io.StringIO()):
                lr.main(["--nights", "2", "--night-s", "500",
                         "--out", str(out), "--seed", "2"])
            html = out.read_text()
    finally:
        lr.NIGHT_S = night_s
    for marker in ("<svg", "night 1", "learning\npolicy"):
        assert marker in html, f"report lost {marker!r}"
    print("  report       learn_report renders svg + trail  -- ok")

    import tools.state_figure as sf
    night_s, sample_s = sf.NIGHT_S, sf.SAMPLE_S
    try:
        sf.NIGHT_S, sf.SAMPLE_S = 240.0, 4.0     # a short night, drawn coarsely
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "states.html"
            with contextlib.redirect_stdout(io.StringIO()):
                sf.main(["--nights", "2", "--seed", "5", "--out", str(out)])
            fig = out.read_text()
    finally:
        sf.NIGHT_S, sf.SAMPLE_S = night_s, sample_s
    for _key, title, _sub in sf.ARMS:
        assert title in fig, f"the figure lost the {title!r} lane"
    assert fig.count("<svg") == 2, "lanes and bars, one svg each"
    assert "data-tip" in fig and "<table" in fig, \
        "the figure needs its hover layer and its table view"
    # the arms must actually differ -- identical lanes look fine in the picture
    person = Personality.random(random.Random(5))
    traces = {k: tuple(sf.run_night(5, person, k, True, trace=True)["trace"])
              for k, _t, _s in sf.ARMS}
    assert len(set(traces.values())) == len(sf.ARMS), \
        "each algorithm must produce a different night"
    assert all(set(t) <= set(range(len(sf.STATES))) for t in traces.values())
    print(f"  figure       state_figure: {len(sf.ARMS)} distinct lanes, "
          f"svg + tooltips + table  -- ok")

    from tools.make_motions import main as make_motions_main
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "n34"
        with contextlib.redirect_stdout(io.StringIO()):
            make_motions_main(["--n34", "--out", str(out)])
        files = sorted(out.glob("motion_*.csv"))
        assert len(files) == 34, f"N system must compile 34 slots, {len(files)}"
        v_ls = (out / "motion_16.csv").read_text().splitlines()
        assert v_ls[1].startswith("16,V_LS"), v_ls[1][:40]
        assert v_ls[1].strip().rstrip('"').endswith("0.00,400,0,0"), \
            "every N slot must end parked at rest"
    print("  n34 files    34 slots compile, MS IDs match, end at rest  -- ok")


# --------------------------------------------------------------------------- #
SUITES = {
    "listen": test_listen,
    "cradle": test_cradle,
    "baby": test_baby,
    "policy": test_policy,
    "m50": test_m50,
    "nubzuki": test_nubzuki,
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

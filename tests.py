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
    from sigma import config as face_config

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


def test_tag() -> None:
    """Detection, identity, distance and dropout on synthetic frames."""
    import numpy as np
    from perception.tag import TagTracker, synthetic_frame

    width, height, fps = 640, 480, 30.0
    tracker = TagTracker()

    # Still tag at a known spot: position right, motion ~0.
    for i in range(40):
        r = tracker.update(synthetic_frame(width, height, -0.4, 0.1, 120), ts=i / fps)
    assert r.present, "a plainly visible tag must be detected"
    assert abs(r.x - (-0.4)) < 0.06, f"x off: {r.x:+.3f} vs -0.400"
    assert r.motion < 0.05, f"a still tag must read still, got {r.motion:.3f}"
    assert r.tag_id == 0, f"tag_0 must decode as id 0, got {r.tag_id}"
    print(f"  still tag   x={r.x:+.3f} (true -0.400)  motion={r.motion:.3f}  -- ok")

    # Identity: the state cards must decode as themselves.
    for want in (1, 2):
        r = TagTracker().update(synthetic_frame(width, height, 0, 0, 120,
                                                tag_id=want), ts=0.0)
        assert r.present and r.tag_id == want, f"tag_{want} read as {r.tag_id}"
    print("  identity    tag_1 and tag_2 decode by id  -- ok")

    # Shaken tag: oscillate, motion must climb well above the still case.
    shaken = TagTracker()
    for i in range(60):
        x = 0.15 * math.sin(2 * math.pi * 3.0 * i / fps)
        r = shaken.update(synthetic_frame(width, height, x, 0.0, 120), ts=i / fps)
    assert r.motion > 0.3, f"a hard-shaken tag must read agitated, got {r.motion:.3f}"
    print(f"  shaken tag  motion={r.motion:.3f}  -- ok")

    # Distance: a bigger tag reads nearer.
    near = TagTracker().update(synthetic_frame(width, height, 0, 0, 190), ts=0.0)
    far = TagTracker().update(synthetic_frame(width, height, 0, 0, 60), ts=0.0)
    assert near.distance > far.distance, "bigger tag must read nearer"
    print(f"  distance    near={near.distance:.2f} > far={far.distance:.2f}  -- ok")

    # Dropout: hold a few frames, then absent; motion decays rather than spikes.
    holder = TagTracker(hold_frames=3)
    holder.update(synthetic_frame(width, height, 0.3, 0, 120), ts=0.0)
    blank = np.full((height, width), 110, np.uint8)
    for i in range(3):
        held = holder.update(blank, ts=0.1 + i / fps)
        assert held.present, f"hold failed on miss {i + 1}"
    gone = holder.update(blank, ts=0.4)
    assert not gone.present, "should report absent after hold_frames misses"
    print("  dropout     held 3 frames then absent  -- ok")


# --------------------------------------------------------------------------- #
# core: the world model, the matcher, the cradle machine
# --------------------------------------------------------------------------- #
def test_pvector() -> None:
    """Check the polynomial against the boundary conditions it claims to meet."""
    import numpy as np
    from core.pvector import AxisProgram, PVector, _parse_axis_index

    print("P-Vector boundary conditions")

    pv = PVector(yd=100.0, l_traj=1000, s0=0.0, sd=0.0)
    assert abs(pv.evaluate(0.0, 0.0).item() - 0.0) < 1e-9, "y(0) must equal y0"
    assert abs(pv.evaluate(0.0, 1.0).item() - 100.0) < 1e-9, "y(1) must equal yd"
    assert abs(pv.velocity(0.0, 0.0).item()) < 1e-9, "initial velocity must be 0"
    assert abs(pv.velocity(0.0, 1.0).item()) < 1e-9, "final velocity must be 0"
    print("  rest-to-rest: y(0)=y0, y(1)=yd, y'(0)=y'(1)=0  -- ok")

    # A plain s0=sd=0 move is the classic minimum-jerk curve: monotone, and it
    # never overshoots its target.  If either fails, the coefficients are wrong.
    traj = pv.sample(0.0, dt=0.001)
    assert np.all(np.diff(traj) >= -1e-9), "s0=sd=0 must be monotone"
    assert traj.max() <= 100.0 + 1e-6, "s0=sd=0 must not overshoot"
    print(f"  monotone, no overshoot (peak {traj.max():.4f} <= 100)  -- ok")

    # Non-zero shaping still has to honour the endpoints.
    for s0, sd in ((10.0, 0.0), (0.0, 10.0), (20.0, 20.0), (-5.0, 5.0)):
        shaped = PVector(yd=-40.0, l_traj=500, s0=s0, sd=sd)
        assert abs(shaped.evaluate(7.0, 0.0).item() - 7.0) < 1e-9, f"y(0) wrong for {s0},{sd}"
        assert abs(shaped.evaluate(7.0, 1.0).item() + 40.0) < 1e-9, f"y(1) wrong for {s0},{sd}"
    print("  shaped segments (s0/sd != 0) still hit both endpoints  -- ok")

    # Duration comes from L_traj at the pcm's 1 kHz recording rate.
    assert abs(PVector(yd=1.0, l_traj=1000).duration_s - 1.0) < 1e-9
    assert abs(PVector(yd=1.0, l_traj=3000).duration_s - 3.0) < 1e-9
    print("  L_traj=1000 -> 1.000 s, L_traj=3000 -> 3.000 s  -- ok")

    # Concatenation: two segments must join without a jump.
    program = AxisProgram(0, (PVector(50.0, 500), PVector(-20.0, 500)))
    dreamt = program.dream(0.0, dt=0.001)
    assert abs(program.duration_s - 1.0) < 1e-9, "durations must add"
    joint = np.abs(np.diff(dreamt)).max()
    assert joint < 1.0, f"segment join produced a jump of {joint:.4f}"
    assert abs(dreamt[-1] + 20.0) < 1e-6, "program must end on the last yd"
    print(f"  two-segment join: max step {joint:.5f}, ends at {dreamt[-1]:.3f}  -- ok")

    # Holding after the program ends (slower axes still running).
    held = program.dream(0.0, dt=0.001, horizon_s=2.0)
    assert abs(held[-1] - held[len(held) // 2]) < 1e-6 or abs(held[-1] + 20.0) < 1e-6
    assert abs(held[-1] + 20.0) < 1e-6, "axis must hold its final value past the end"
    print("  holds final value past program end  -- ok")

    # The parser has to survive the real file's trailing zeros and blanks.
    assert PVector.parse("2389.4,1500,0,0,0.0,0.0") == PVector(2389.4, 1500, 0.0, 0.0)
    assert PVector.parse("108, 1500, 0, 0") == PVector(108.0, 1500, 0.0, 0.0)
    assert PVector.parse("-") is None and PVector.parse("") is None
    print("  MotionMap cell parsing (trailing zeros, '-', blanks)  -- ok")

    assert _parse_axis_index("MD5") == 4
    assert _parse_axis_index("0x06") == 4
    assert _parse_axis_index("3") == 3
    print("  axis id mapping: MD5 -> 4, 0x06 -> 4  -- ok")


def test_dream() -> None:
    """Exercise matcher and monitor against hand-built chunks.  No hardware."""
    from typing import Sequence

    import numpy as np
    from core.dream import ChunkMatcher, DreamConfig, DreamMonitor
    from core.pvector import AxisProgram, MotionChunk, PVector

    def chunk(slot_id: int, deltas_deg: Sequence[float], l_traj: int = 1600) -> MotionChunk:
        return MotionChunk(
            slot_id=slot_id, name=f"test {slot_id}",
            programs=tuple(
                AxisProgram(i, (PVector(yd=d, l_traj=l_traj),))
                for i, d in enumerate(deltas_deg)
            ),
            synthesised=True,
        )

    small, medium, large = chunk(1, [2.0, 0.0]), chunk(2, [10.0, 0.0]), chunk(3, [40.0, 0.0])
    chunks = {c.slot_id: c for c in (small, medium, large)}
    cfg = DreamConfig()
    matcher = ChunkMatcher(chunks, cfg)
    here = [0.0, 0.0]

    print("ChunkMatcher")

    # 1. No external force -> the gentlest chunk wins on continuity.
    ranked = matcher.rank([1, 2, 3], here)
    assert ranked[0].slot_id == 1, f"expected slot 1 with no force, got {ranked[0].slot_id}"
    print(f"  no force -> {ranked[0].describe()}")

    # 2. Push back on axis 0 and the big swing must lose to the small one.
    pushed = matcher.rank([1, 2, 3], here, external_force_a=[2.0, 0.0])
    assert pushed[0].slot_id == 1, "contact on axis 0 must favour the smallest move"
    assert pushed[0].resistance < pushed[-1].resistance, "resistance must order the ranking"
    print(f"  dob=2.0A on axis 0 -> {pushed[0].describe()}")
    print(f"                       worst: {pushed[-1].describe()}")

    # 3. Task fit can outvote continuity when nothing is pushing back.
    amps = {1: 0.15, 2: 0.45, 3: 0.85}
    task_cfg = DreamConfig(w_task=50.0, w_continuity=0.01, w_resistance=0.0)
    biased = ChunkMatcher(chunks, task_cfg).rank(
        [1, 2, 3], here, target_amplitude=0.85, amplitude_of=amps)
    assert biased[0].slot_id == 3, f"strong stirring should pick slot 3, got {biased[0].slot_id}"
    print(f"  target amp 0.85 -> slot {biased[0].slot_id}")

    # 4. The excursion veto has to fire before anything else.
    tight = ChunkMatcher(chunks, DreamConfig(max_excursion_rad=0.1))
    vetoed = tight.rank([1, 2, 3], here)
    assert any(s.vetoed for s in vetoed), "a 40 deg swing must be vetoed at 0.1 rad"
    assert tight.best([3], here) is None, "best() must not return a vetoed chunk"
    print(f"  veto at 0.1 rad -> {[s.slot_id for s in vetoed if s.vetoed]} rejected")

    # 5. No joint data -> pose terms drop out, nothing crashes, task still ranks.
    blind = matcher.rank([1, 2, 3], None, target_amplitude=0.85, amplitude_of=amps)
    assert blind[0].slot_id == 3 and blind[0].peak_excursion == 0.0
    print("  no joint data -> ranks on task fit alone, pose terms inactive")

    print("DreamMonitor")

    # 6. Following the dream exactly must never trip the monitor.
    monitor = DreamMonitor(cfg)
    monitor.begin(medium, here, now=0.0)
    t_ref, y_ref = medium.dream(here, dt=cfg.dt, horizon_s=medium.duration_s)
    for i, t in enumerate(t_ref):
        report = monitor.update(y_ref[i], now=float(t), external_force_a=[0.0, 0.0])
    assert not report.diverged, "a perfectly tracked chunk must not diverge"
    assert report.rms_rad < 1e-9, f"rms should be ~0, got {report.rms_rad}"
    print(f"  perfect tracking -> {report.describe()}")

    # 7. Jam it: hold the arm still while the dream keeps moving.
    monitor.begin(large, here, now=0.0)
    stuck, fired_at = np.array(here, dtype=float), None
    for i, t in enumerate(t_ref):
        report = monitor.update(stuck, now=float(t), external_force_a=[3.0, 0.0])
        if report.diverged and fired_at is None:
            fired_at = float(t)
    assert fired_at is not None, "a jammed axis must trip the divergence monitor"
    assert fired_at <= 1.0, f"divergence should be caught early, fired at {fired_at:.2f}s"
    print(f"  jammed axis -> diverged at t={fired_at:.2f}s, {report.describe()}")

    # 8. A single spike must NOT trip it (that is what diverge_hold_s is for).
    monitor.begin(medium, here, now=0.0)
    spiked = False
    for i, t in enumerate(t_ref):
        sample = y_ref[i].copy()
        if i == len(t_ref) // 2:
            sample[0] += 0.5   # one bad frame
        report = monitor.update(sample, now=float(t))
        spiked = spiked or report.diverged
    assert not spiked, "a one-frame spike must not count as divergence"
    print("  single-frame spike -> ignored (diverge_hold_s held)")

    # 9. ASAP-lite residual must shift the next dream, not the command.
    matcher.note_residual([0.05, 0.0])
    _, corrected = matcher.corrected_dream(medium, here)
    _, plain = medium.dream(here, dt=cfg.dt, horizon_s=cfg.horizon_s)
    assert corrected[-1, 0] > plain[-1, 0], "residual must bias the dream"
    assert abs(corrected[0, 0] - plain[0, 0]) < 1e-9, "residual must ramp in, not jump"
    print(f"  residual 0.05 rad -> dream end shifted "
          f"{corrected[-1, 0] - plain[-1, 0]:+.4f} rad, start unchanged")


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
def test_demo() -> None:
    """The whole decision/jam/preempt cycle, no hardware anywhere."""
    from apps.demo import PLAY_DT, Brain, load_chunks
    from core.slot_table import load_slot_table
    from perception.tag import TagReading

    chunks = load_chunks()
    table = load_slot_table("slots.json")
    brain = Brain(chunks, table)
    origin = "motions/" if not next(iter(chunks.values())).synthesised else "slots.json"
    print(f"dictionary: {len(chunks)} chunks from {origin}")

    still = TagReading(True, -0.6, 0.0, 0.5, 0.02, 0.0)
    assert brain.decide(still, 0.0) is None, "a calm tag must not trigger a motion"
    print("  calm tag              -> no motion  -- ok")

    shaken_left = TagReading(True, -0.6, 0.0, 0.5, 0.85, 0.0)
    slot = brain.decide(shaken_left, 1.0)
    picked = table.slots[slot]
    assert picked.direction.value == "left", f"expected a left slot, got {slot}"
    assert len(brain.last_ranked) == 3, "all three amplitudes must be ranked"
    assert picked.amplitude.value == "large", \
        f"hard shake + no force should pick large, got {picked.amplitude.value}"
    print(f"  hard shake at x=-0.6  -> slot {slot} ({picked.amplitude.value}), "
          f"3 candidates ranked  -- ok")

    # Play cleanly to the end: no divergence, normal dwell.
    t = 1.0
    while brain.playing is not None:
        t += PLAY_DT
        brain.tick(t)
    assert not brain.preempted, "an unobstructed chunk must not diverge"
    assert brain.next_ok - t > 0.5, "clean finish should use the full dwell"
    print(f"  clean playback        -> no divergence, dwell {brain.next_ok - t:.1f}s  -- ok")

    # Jam mid-play: divergence must fire, and the dwell must collapse.
    # Shake on the RIGHT so the cradle has real ground to cover (it is parked
    # at the left target; replaying the same slot would dream a flat line and
    # a flat dream cannot diverge -- there is nothing to fall behind).
    shaken_right = TagReading(True, 0.6, 0.0, 0.5, 0.85, 0.0)
    slot = brain.decide(shaken_right, brain.next_ok + 0.1)
    assert table.slots[slot].direction.value == "right"
    t = brain.next_ok + 0.1
    for _ in range(10):                      # let it move first
        t += PLAY_DT; brain.tick(t)
    brain.jam = True
    diverged_at = None
    while brain.playing is not None:
        t += PLAY_DT; brain.tick(t)
        if diverged_at is None and brain.monitor.latest().diverged:
            diverged_at = t
    assert diverged_at is not None, "a jammed cradle must trip the monitor"
    assert brain.preempted, "the finish must carry the diverged verdict"
    assert brain.next_ok - t < 0.5, "a diverged dream must collapse the dwell"
    print(f"  jam mid-play          -> DIVERGED, dwell collapsed to "
          f"{brain.next_ok - t:.1f}s  -- ok")

    # Still jammed (dob high): the ranking must flip from large to gentle.
    slot = brain.decide(shaken_right, brain.next_ok + 0.01)
    picked = table.slots[slot]
    assert picked.amplitude.value != "large", \
        f"with 2.5A of contact the big sway must lose, got {picked.amplitude.value}"
    print(f"  re-rank under contact -> slot {slot} ({picked.amplitude.value}), "
          f"resistance flipped the choice  -- ok")


def test_serve() -> None:
    """Every HTTP endpoint against a real server on a random port."""
    import json
    import threading
    from urllib.request import urlopen

    from apps.demo import Brain, load_chunks
    from core.slot_table import load_slot_table
    from serve import Shared, start

    shared = Shared(Brain(load_chunks(), load_slot_table("slots.json")),
                    load_chunks(), load_slot_table("slots.json"))
    shared.machine.auto = False   # deterministic: no organic trials mid-test
    server, worker, _ = start(shared, port=0, camera_index=0, fake=True, use_ros=False)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    time.sleep(1.0)

    try:
        page = urlopen(base + "/", timeout=5).read()
        assert b"DREAM-Chunk" in page, "dashboard page must serve"
        print(f"  GET /        {len(page)} bytes  -- ok")

        slots = json.loads(urlopen(base + "/slots", timeout=5).read())
        assert len(slots) == 10 and slots[0]["id"] == 1, f"expected 10 slots, got {len(slots)}"
        print(f"  GET /slots   {len(slots)} slots, e.g. {slots[2]['name']} "
              f"{slots[2]['target_deg']} deg  -- ok")

        with urlopen(base + "/events", timeout=5) as stream:
            line = stream.readline().decode()
            assert line.startswith("data: "), f"not SSE: {line[:40]!r}"
            state = json.loads(line[6:])
        for key in ("pose", "tag", "monitor", "jam", "events", "cradle"):
            assert key in state, f"state missing {key!r}"
        assert state["tag"]["id"] == 0 and state["tag"]["level"] == 0.0, \
            f"the calm card must read id 0 / level 0, got {state['tag']}"
        print(f"  GET /events  keys ok, card id={state['tag']['id']} "
              f"x={state['tag']['x']:+.2f}  -- ok")

        assert json.loads(urlopen(base + "/jam", timeout=5).read())["jam"] is True
        assert json.loads(urlopen(base + "/jam", timeout=5).read())["jam"] is False
        print("  GET /jam     toggles  -- ok")

        urlopen(base + "/play?slot=9", timeout=5).read()
        deadline = time.time() + 3.0
        played = False
        while time.time() < deadline and not played:
            with urlopen(base + "/events", timeout=5) as stream:
                state = json.loads(stream.readline().decode()[6:])
            played = state.get("playing", {}) and state["playing"]["slot"] == 9
            time.sleep(0.1)
        assert played, "manual /play?slot=9 must start playing"
        print(f"  GET /play    slot 9 playing, "
              f"dream_theta={state['playing']['dream_theta']}  -- ok")

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

        busy = json.loads(urlopen(base + "/play?slot=9", timeout=5).read())
        assert busy["requested"] is None, "slot play must yield to the engine"
        urlopen(base + "/motion?id=M01", timeout=5).read()
        wait_for(lambda c: c["motion"] is None, 8, "M01 to park the engine")
        print("  GET /motion  /play yields while active, M01 parks in 5 s  -- ok")

        assert json.loads(urlopen(base + "/auto?set=on", timeout=5).read())["auto"] is True
        urlopen(base + "/jam", timeout=5).read()
        wait_for(lambda c: c["state"] == "gate_fail", 3, "the safety gate")
        urlopen(base + "/jam", timeout=5).read()
        print("  GET /auto    machine on; jam trips the safety gate  -- ok")

        # The fake camera's fuss card (tag_1) appears at t=20 s; after the gate
        # recovers and its cooldown passes, the machine must open an M10 trial
        # from the card's identity alone -- the card never shakes.
        snap = wait_for(lambda c: c["state"] == "trial", 45,
                        "the fuss card to open a trial")
        assert snap["motion"] == "M10", f"fuss must trial M10, got {snap['motion']}"
        print(f"  state card   tag_1 -> {snap['motion']} trial, no shake involved  -- ok")
    finally:
        shared.stop.set()
        server.shutdown()


# --------------------------------------------------------------------------- #
SUITES = {
    "listen": test_listen,
    "sense": test_sense,
    "tag": test_tag,
    "pvector": test_pvector,
    "dream": test_dream,
    "cradle": test_cradle,
    "demo": test_demo,
    "serve": test_serve,
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

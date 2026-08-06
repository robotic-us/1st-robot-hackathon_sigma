#!/usr/bin/env python3
"""The infant-cradle motion library (M01-M50) and its safety state machine.

A transcription of docs/infant_robotic_cradle_evidence_report_ko.pdf into
runnable form:

* ``LIBRARY``       the 50 motions of section 7 -- C0 stop/transition
                    commands, P1 horizontal-sway trial candidates, R
                    research-only modes (refused unless explicitly allowed)
* ``MotionEngine``  plays them: phase-continuous sine synthesis with S-curve
                    amplitude ramps (min 5 s, default 30 s).  The kinematic
                    envelope of section 6 (0.2-0.8 Hz, A <= 30 mm hard,
                    base a_peak <= 0.05 g) is asserted over the whole library
                    at import time -- the report's V0 gate: bad units or axes
                    must not load.
* ``CradleMachine`` the priority ladder of section 5: safety gate first, then
                    stable-sleep taper, quiet-awake hold ("the default is not
                    moving"), and 30 s cry trials with improvement checks,
                    single-step escalation, one micro-resume, and caregiver
                    alerts.

Nothing here touches HTTP, cameras or ROS, and time is injected everywhere,
so the selftest drives a whole trial with a fake clock in milliseconds.
serve.py wires the machine to the tag reading (a stand-in for the infant
sensors) and the engine's offset to the cradle pose.

Tested by ``python3 tests.py cradle``.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional

GRAVITY = 9.80665

# --- the kinematic envelope, report section 6.1 ----------------------------- #
F_MIN_HZ, F_MAX_HZ = 0.2, 0.8   # verified-candidate frequency band
A_HARD_MM = 30.0                # hard one-way amplitude ceiling
A_PEAK_MAX_G = 0.05             # engineering pre-trial ceiling, not clinical
RAMP_MIN_S = 5.0                # fastest allowed start/taper (emergencies)
RAMP_DEFAULT_S = 30.0           # SOFT_START_30 / TAPER_30 default


def a_peak_g(f_hz: float, a_mm: float) -> float:
    """Theoretical base peak acceleration of a sine, in g (report 6.1)."""
    return (2.0 * math.pi * f_hz) ** 2 * (a_mm / 1000.0) / GRAVITY


def smoothstep(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


# --------------------------------------------------------------------------- #
# The library, report section 7
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Motion:
    id: str
    name: str
    kind: str          # static|pause|soft_start|taper|micro_resume|sine|
                       # diagonal|ellipse|circle|lissajous|pseudo_walk|
                       # adaptive_a|adaptive_f
    grade: str         # C0 = stop/transition, P1 = trial candidate, R = research
    desc: str
    axis: str = ""     # ML | AP | Z | APML
    f_hz: float = 0.0
    a_mm: float = 0.0
    f2_hz: float = 0.0  # secondary component (lissajous)
    a2_mm: float = 0.0  # secondary amplitude (ellipse minor, lissajous)
    sign: float = 1.0   # diagonal +-45 deg, circle CW/CCW
    ramp_s: float = 0.0  # soft_start / taper lengths

    def worst_components(self) -> list[tuple[float, float]]:
        """(f, A) pairs at their envelope-worst, for the import-time gate."""
        if self.kind == "sine":
            return [(self.f_hz, self.a_mm)]
        if self.kind in ("diagonal", "circle"):
            return [(self.f_hz, self.a_mm)]
        if self.kind == "ellipse":
            return [(self.f_hz, self.a_mm), (self.f_hz, self.a2_mm)]
        if self.kind == "lissajous":
            return [(self.f_hz, self.a_mm), (self.f2_hz, self.a2_mm)]
        if self.kind == "pseudo_walk":
            return [(0.7, self.a_mm)]          # band-limited 0.4-0.7, |A|<=10
        if self.kind == "adaptive_a":
            return [(self.f_hz, 15.0)]         # A steps 5 -> 10 -> 15
        if self.kind == "adaptive_f":
            return [(0.7, self.a_mm)]          # f steps 0.3 -> 0.5 -> 0.7
        return []                              # C0 commands: no oscillation


def _build_library() -> list[Motion]:
    lib: list[Motion] = []
    add = lib.append

    # 7.1 safety / transition commands
    add(Motion("M01", "STATIC_SAFE", "static", "C0",
               "hold at rest -- the default and the sleep state"))
    add(Motion("M02", "PAUSE_OBSERVE", "pause", "C0",
               "current amplitude to 0, then observe 3-10 s after a startle "
               "or outside noise", ramp_s=RAMP_MIN_S))
    add(Motion("M03", "SOFT_START_30", "soft_start", "C0",
               "selected mode 0 to 100% over 30 s -- how a cry trial begins",
               ramp_s=30.0))
    add(Motion("M04", "SOFT_START_60", "soft_start", "C0",
               "selected mode 0 to 100% over 60 s -- sensitive starts",
               ramp_s=60.0))
    add(Motion("M05", "TAPER_30", "taper", "C0",
               "current amplitude to 0 over 30 s (emergencies override to 5 s)",
               ramp_s=30.0))
    add(Motion("M06", "TAPER_60", "taper", "C0",
               "presumed-sleep deceleration", ramp_s=60.0))
    add(Motion("M07", "TAPER_120", "taper", "C0",
               "gentler sleep deceleration", ramp_s=120.0))
    add(Motion("M08", "MICRO_RESUME", "micro_resume", "C0",
               "previous mode at 50% amplitude, 30 s ramp -- one retry when "
               "fussing returns during a taper", ramp_s=30.0))

    # 7.2 / 7.3 the P1 single-axis horizontal sine candidates
    for base, axis in ((9, "ML"), (19, "AP")):
        for i, f in enumerate((0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)):
            add(Motion(f"M{base + i:02d}", f"{axis}_SINE_{f:.1f}HZ_A10",
                       "sine", "P1", "weak-fuss/cry single-axis trial",
                       axis=axis, f_hz=f, a_mm=10.0))
        for i, f in enumerate((0.5, 0.6, 0.7)):
            add(Motion(f"M{base + 7 + i:02d}", f"{axis}_SINE_{f:.1f}HZ_A20",
                       "sine", "P1", "only when A10 improves but not enough",
                       axis=axis, f_hz=f, a_mm=20.0))

    # 7.4 diagonal horizontal sines (research)
    for mid, sign, f, a in (("M29", +1, 0.3, 10.0), ("M30", -1, 0.3, 10.0),
                            ("M31", +1, 0.4, 15.0), ("M32", -1, 0.4, 15.0),
                            ("M33", +1, 0.5, 20.0), ("M34", -1, 0.5, 20.0)):
        deg = "+45" if sign > 0 else "-45"
        add(Motion(mid, f"DIAG{deg}_{f:.1f}HZ_A{a:.0f}", "diagonal", "R",
                   "direction-responsiveness comparison", axis="APML",
                   f_hz=f, a_mm=a, sign=sign))

    # 7.5 ellipse / circle centre trajectories (research)
    for mid, aap, aml, f in (("M35", 15.0, 5.0, 0.3), ("M36", 15.0, 5.0, 0.5),
                             ("M37", 20.0, 10.0, 0.4), ("M38", 20.0, 10.0, 0.5)):
        add(Motion(mid, f"ELLIPSE_AP{aap:.0f}_ML{aml:.0f}_{f:.1f}HZ",
                   "ellipse", "R", "two-axis curve comparison", axis="APML",
                   f_hz=f, a_mm=aap, a2_mm=aml))
    for mid, sign, f in (("M39", +1, 0.3), ("M40", -1, 0.3),
                         ("M41", +1, 0.5), ("M42", -1, 0.5)):
        cw = "CW" if sign > 0 else "CCW"
        add(Motion(mid, f"CIRCLE_{cw}_R10_{f:.1f}HZ", "circle", "R",
                   "centre orbit without rotation", axis="APML",
                   f_hz=f, a_mm=10.0, sign=sign))

    # 7.6 composite / adaptive / vertical research modes
    add(Motion("M43", "LISSAJOUS_ML", "lissajous", "R",
               "aperiodicity comparison", axis="ML",
               f_hz=0.5, a_mm=10.0, f2_hz=0.25, a2_mm=5.0))
    add(Motion("M44", "LISSAJOUS_AP", "lissajous", "R",
               "axis-symmetry comparison", axis="AP",
               f_hz=0.5, a_mm=10.0, f2_hz=0.25, a2_mm=5.0))
    add(Motion("M45", "PSEUDO_WALK", "pseudo_walk", "R",
               "0.4-0.7 Hz band-limited transport-response probe",
               axis="ML", a_mm=10.0))
    add(Motion("M46", "ADAPTIVE_A", "adaptive_a", "R",
               "0.5 Hz, A 5-10-15 mm stepped every 30 s (improving only)",
               axis="ML", f_hz=0.5, a_mm=15.0))
    add(Motion("M47", "ADAPTIVE_F", "adaptive_f", "R",
               "A10, f 0.3-0.5-0.7 Hz stepped every 30 s (improving only)",
               axis="ML", a_mm=10.0))
    for mid, f in (("M48", 0.3), ("M49", 0.5), ("M50", 0.7)):
        add(Motion(mid, f"Z_SINE_{f:.1f}HZ_A3", "sine", "R",
                   "vertical minimum-amplitude study", axis="Z",
                   f_hz=f, a_mm=3.0))

    # V0 gate: bad units/axes/params must not even load (report 9.1).
    assert len(lib) == 50, f"library must hold 50 motions, has {len(lib)}"
    for m in lib:
        for f, a in m.worst_components():
            if not (F_MIN_HZ <= f <= F_MAX_HZ):
                raise ValueError(f"{m.id}: {f} Hz outside {F_MIN_HZ}-{F_MAX_HZ}")
            if a > A_HARD_MM:
                raise ValueError(f"{m.id}: A={a} mm over the {A_HARD_MM} mm hard cap")
            if a_peak_g(f, a) > A_PEAK_MAX_G:
                raise ValueError(f"{m.id}: a_peak {a_peak_g(f, a):.4f} g over "
                                 f"{A_PEAK_MAX_G} g")
    return lib


LIBRARY = _build_library()
LIBRARY_BY_ID = {m.id: m for m in LIBRARY}


def catalog() -> list[dict]:
    """The /motions payload: the whole library with its theory numbers."""
    out = []
    for m in LIBRARY:
        peak = max((a_peak_g(f, a) for f, a in m.worst_components()), default=0.0)
        out.append({
            "id": m.id, "name": m.name, "kind": m.kind, "grade": m.grade,
            "axis": m.axis, "f_hz": m.f_hz, "a_mm": m.a_mm,
            "a_peak_g": round(peak, 4), "desc": m.desc,
        })
    return out


# --------------------------------------------------------------------------- #
# The engine: library entry -> live (ap, ml, z) offset in millimetres
# --------------------------------------------------------------------------- #
class MotionEngine:
    """Phase-continuous playback with S-curve amplitude ramps.

    ``command()`` accepts any library id: C0 entries act on the current mode
    (soft start, taper, resume), oscillation entries become the current mode.
    R-grade entries are refused unless ``allow_research`` -- the report bars
    them from any automatic infant mode.
    """

    def __init__(self, allow_research: bool = False) -> None:
        self.allow_research = allow_research
        self.mode: Optional[Motion] = None
        self.resumable: Optional[Motion] = None   # for M08 after a taper
        self.amp_scale = 1.0                      # 0.5 after MICRO_RESUME
        self.env = 0.0                            # amplitude envelope, 0..1
        self._ramp: Optional[tuple] = None        # (t0, e0, t1, e1)
        self._phase = [0.0, 0.0, 0.0]             # rad, per component
        self.mode_t = 0.0
        self._t: Optional[float] = None

    @property
    def active(self) -> bool:
        return self.mode is not None

    @property
    def tapering(self) -> bool:
        return self._ramp is not None and self._ramp[3] <= 0.0

    # -- commands ------------------------------------------------------------ #
    def command(self, motion_id: str, now: float,
                ramp_s: Optional[float] = None) -> tuple[bool, str]:
        m = LIBRARY_BY_ID.get(str(motion_id).upper())
        if m is None:
            return False, f"unknown motion {motion_id!r}"
        if m.grade == "R" and not self.allow_research:
            return False, (f"{m.id} {m.name} is research-only (R): refused "
                           "without --research (report 7.6)")

        if m.kind in ("static", "pause", "taper"):
            if self.mode is None:
                return True, f"{m.id} {m.name}: already static"
            seconds = max(RAMP_MIN_S, ramp_s if ramp_s is not None else m.ramp_s)
            self._ramp = (now, self.env, now + seconds, 0.0)
            return True, f"{m.id} {m.name}: amplitude to 0 over {seconds:.0f} s"

        if m.kind == "soft_start":
            target = self.mode or self.resumable
            if target is None:
                return False, f"{m.id}: no mode selected -- send an oscillation id"
            return self._start(target, now, m.ramp_s, scale=1.0)

        if m.kind == "micro_resume":
            target = self.mode or self.resumable
            if target is None:
                return False, f"{m.id}: nothing to resume"
            ok, _ = self._start(target, now, m.ramp_s, scale=0.5)
            return ok, (f"{m.id} {m.name}: {target.id} at 50% amplitude, "
                        f"{m.ramp_s:.0f} s ramp")

        # an oscillation entry becomes the current mode
        return self._start(m, now,
                           max(RAMP_MIN_S, ramp_s if ramp_s is not None
                               else RAMP_DEFAULT_S), scale=1.0)

    def _start(self, m: Motion, now: float, ramp_s: float,
               scale: float) -> tuple[bool, str]:
        if self.mode is None:
            self._phase = [0.0, 0.0, 0.0]
        self.mode = m
        self.mode_t = 0.0
        self.amp_scale = scale
        self._ramp = (now, self.env, now + max(RAMP_MIN_S, ramp_s), 1.0)
        return True, (f"{m.id} {m.name}: soft start over "
                      f"{max(RAMP_MIN_S, ramp_s):.0f} s")

    # -- time ---------------------------------------------------------------- #
    def tick(self, now: float) -> None:
        dt = 0.0 if self._t is None else max(0.0, min(0.1, now - self._t))
        self._t = now
        if self._ramp is not None:
            t0, e0, t1, e1 = self._ramp
            self.env = e0 + (e1 - e0) * smoothstep((now - t0) / max(1e-6, t1 - t0))
            if now >= t1:
                self.env = e1
                self._ramp = None
                if e1 <= 0.0:                     # taper finished: park
                    self.resumable, self.mode = self.mode, None
        if self.mode is not None:
            self.mode_t += dt
            for i, f in enumerate(self._freqs()):
                self._phase[i] += 2.0 * math.pi * f * dt

    def _freqs(self) -> tuple[float, ...]:
        m = self.mode
        if m is None:
            return ()
        if m.kind == "lissajous":
            return (m.f_hz, m.f2_hz)
        if m.kind == "pseudo_walk":
            return (0.45, 0.55, 0.65)
        if m.kind == "adaptive_f":
            return ((0.3, 0.5, 0.7)[min(2, int(self.mode_t // 30.0))],)
        return (m.f_hz,)

    def _amplitude_mm(self) -> float:
        """The current primary one-way amplitude, adaptive schedules applied."""
        m = self.mode
        if m is None:
            return 0.0
        if m.kind == "adaptive_a":
            steps = (5.0, 10.0, 15.0)
            k = min(2, int(self.mode_t // 30.0))
            a = steps[k]
            into = self.mode_t - 30.0 * k
            if k < 2 and into > 25.0:             # blend the last 5 s of a step
                a += (steps[k + 1] - steps[k]) * smoothstep((into - 25.0) / 5.0)
            return a
        return m.a_mm

    def offsets_mm(self) -> tuple[float, float, float]:
        """(ap, ml, z) centre offset right now, millimetres."""
        m = self.mode
        s = self.env * self.amp_scale
        if m is None or s <= 0.0:
            return 0.0, 0.0, 0.0
        p = self._phase
        a = self._amplitude_mm() * s
        if m.kind in ("sine", "adaptive_a", "adaptive_f"):
            v = a * math.sin(p[0])
            return ((v, 0.0, 0.0) if m.axis == "AP" else
                    (0.0, 0.0, v) if m.axis == "Z" else (0.0, v, 0.0))
        if m.kind == "diagonal":
            v = a * math.sin(p[0]) / math.sqrt(2.0)
            return v, m.sign * v, 0.0
        if m.kind == "ellipse":
            return a * math.sin(p[0]), m.a2_mm * s * math.cos(p[0]), 0.0
        if m.kind == "circle":
            return a * math.cos(p[0]), m.sign * a * math.sin(p[0]), 0.0
        if m.kind == "lissajous":
            v1, v2 = a * math.sin(p[0]), m.a2_mm * s * math.sin(p[1])
            return (v1, v2, 0.0) if m.axis == "AP" else (v2, v1, 0.0)
        if m.kind == "pseudo_walk":
            v = (4.5 * math.sin(p[0]) + 3.5 * math.sin(p[1] + 2.1)
                 + 2.0 * math.sin(p[2] + 4.2)) * s
            return 0.0, max(-a, min(a, v)), 0.0
        return 0.0, 0.0, 0.0

    def snapshot(self) -> dict:
        ap, ml, z = self.offsets_mm()
        m = self.mode
        f = self._freqs()[0] if m is not None else 0.0
        a_now = self._amplitude_mm() * self.env * self.amp_scale
        return {
            "motion": m.id if m else None,
            "name": m.name if m else "STATIC",
            "grade": m.grade if m else "C0",
            # what sort of thing this is (sine/taper/pause/...), so a reader
            # can describe it in words without parsing the name
            "kind": m.kind if m else "static",
            "f_hz": round(f, 2),
            "a_mm": round(a_now, 2),
            "env": round(self.env * self.amp_scale, 3),
            "tapering": self.tapering,
            "offset_mm": {"ap": round(ap, 2), "ml": round(ml, 2), "z": round(z, 2)},
            "a_peak_g": round(a_peak_g(f, a_now), 4),
            # so the library selector can show which R entries it would refuse
            # rather than letting the click fail with no explanation
            "research": self.allow_research,
        }


# --------------------------------------------------------------------------- #
# The state machine, report section 5
# --------------------------------------------------------------------------- #
CALM_LEVEL = 0.12          # matches demo.CALM_FLOOR: below this, nobody fusses
CRY_LEVEL = 0.45           # above this the trial starts one rung up
TRIAL_LADDER = ("M10", "M12", "M13", "M16")   # fuss -> cry -> escalation (ML)

FUSS_SUSTAIN_S = 1.5       # cry must be sustained before a trial (report 4.2)
GATE_ABSENT_S = 0.7        # face gone this long trips the gate
GATE_RECOVER_S = 2.0
CHECK_EVERY_S = 30.0       # trial checkpoints
NO_IMPROVE_S = 60.0        # no improvement by here: taper + caregiver
WORSE_SUSTAIN_S = 5.0
TRIAL_CAP_S = 300.0        # improving trials still end at 5 min
SLEEP_CALM_S = 60.0        # calm this long during a trial = presumed sleep
COOLDOWN_END_S = 30.0
COOLDOWN_ABORT_S = 90.0


class CradleMachine:
    """The report's priority ladder over a MotionEngine.

    Inputs per tick: is the face (tag) visible, a 0..1 distress level, and
    the jam flag standing in for an arm-desync/E-stop.  The safety gate runs
    even with ``auto`` off; ``auto`` only enables the trial/sleep behaviour.
    """

    def __init__(self, engine: MotionEngine) -> None:
        self.engine = engine
        self.auto = True
        self.state = "quiet"       # quiet | trial | settling | gate_fail
        self.ema = 0.0
        self.alert = ""
        self.events: list[str] = []   # drained by the caller into its log
        self._t: Optional[float] = None
        self._absent_t = 0.0
        self._fuss_t = 0.0
        self._calm_t = 0.0
        self._worse_t = 0.0
        self._gate_ok_t = 0.0
        self._trial_t0 = 0.0
        self._check_t = 0.0
        self._deadline = 0.0
        self._baseline = 0.0
        self._rung = 0
        self._resumed = False
        self._cooldown_until = 0.0

    def _log(self, text: str) -> None:
        self.events.append(text)

    def _alert(self, text: str) -> None:
        self.alert = text
        self._log("ALERT caregiver: " + text)

    def snapshot(self, now: float) -> dict:
        return {
            "state": self.state,
            "auto": self.auto,
            "ema": round(self.ema, 3),
            "trial_s": round(now - self._trial_t0, 1) if self.state == "trial" else 0,
            "alert": self.alert,
        }

    # -- one tick of the ladder ---------------------------------------------- #
    def tick(self, now: float, present: bool, level: float, jam: bool) -> None:
        dt = 0.0 if self._t is None else max(0.0, min(0.2, now - self._t))
        self._t = now
        self.ema += (max(0.0, level) - self.ema) * min(1.0, dt / 0.4)
        self._absent_t = 0.0 if present else self._absent_t + dt
        self._fuss_t = self._fuss_t + dt if self.ema >= CALM_LEVEL else 0.0
        self._calm_t = self._calm_t + dt if self.ema < CALM_LEVEL else 0.0

        # 1. the safety gate outranks everything, auto or not (report 5.1)
        gate_bad = jam or self._absent_t > GATE_ABSENT_S
        if gate_bad:
            self._gate_ok_t = 0.0
            if self.engine.active and not self.engine.tapering:
                self.engine.command("M05", now, ramp_s=RAMP_MIN_S)
                self._alert("safety gate failed (jam/face lost) -- "
                            "amplitude to 0 in 5 s")
            if self.state != "gate_fail":
                self.state = "gate_fail"
                self._log("gate FAIL: " + ("jam" if jam else "face lost"))
            return
        if self.state == "gate_fail":
            self._gate_ok_t += dt
            if self._gate_ok_t >= GATE_RECOVER_S:
                self.state = "quiet"
                self._cooldown_until = now + 10.0
                self._log("gate recovered -- observing before any restart")
            return

        if not self.auto:
            return

        # 2. quiet awake / sleep: the default value of automation is M01
        if self.state == "quiet":
            if self._fuss_t >= FUSS_SUSTAIN_S and now >= self._cooldown_until:
                self._start_trial(now)
            return

        if self.state == "trial":
            if self._calm_t >= SLEEP_CALM_S:
                self.engine.command("M06", now)
                self.state = "settling"
                self._log("calm 60 s -- presumed sleep, M06 taper")
                return
            if self.ema > 1.3 * self._baseline + 0.05:
                self._worse_t += dt
                if self._worse_t >= WORSE_SUSTAIN_S:
                    self._abort(now, "worse during trial")
                    return
            else:
                self._worse_t = 0.0
            if now - self._check_t >= CHECK_EVERY_S:
                self._check_t = now
                if self.ema <= 0.7 * self._baseline:
                    self._deadline = now + NO_IMPROVE_S
                    self._log(f"trial improving (ema {self.ema:.2f}) -- holding "
                              f"{TRIAL_LADDER[self._rung]}")
                elif (self.ema >= CRY_LEVEL
                      and self._rung + 1 < len(TRIAL_LADDER)):
                    self._rung += 1
                    step = TRIAL_LADDER[self._rung]
                    self.engine.command(step, now, ramp_s=10.0)
                    self._log(f"no improvement at 30 s -- one step up to {step}")
            if now >= self._deadline:
                self._abort(now, "no improvement in 60 s")
            elif now - self._trial_t0 >= TRIAL_CAP_S:
                self.engine.command("M05", now)
                self.state = "settling"
                self._cooldown_until = now + COOLDOWN_END_S
                self._log("5 min trial cap -- taper")
            return

        if self.state == "settling":
            if (self._fuss_t >= FUSS_SUSTAIN_S and not self._resumed
                    and now >= self._cooldown_until
                    and (self.engine.mode or self.engine.resumable)):
                ok, _ = self.engine.command("M08", now)
                if ok:
                    self._resumed = True
                    self.state = "trial"
                    self._trial_t0 = self._check_t = now
                    self._deadline = now + NO_IMPROVE_S
                    self._baseline = max(self.ema, 0.05)
                    self._log("fussing during taper -- one micro-resume (M08)")
                    return
            if not self.engine.active:
                self.state = "quiet"
                self._cooldown_until = max(self._cooldown_until,
                                           now + COOLDOWN_END_S)

    def _start_trial(self, now: float) -> None:
        self._rung = 1 if self.ema >= CRY_LEVEL else 0
        step = TRIAL_LADDER[self._rung]
        self.engine.command(step, now, ramp_s=RAMP_DEFAULT_S)
        self.state = "trial"
        self._trial_t0 = self._check_t = now
        self._deadline = now + NO_IMPROVE_S
        self._baseline = max(self.ema, 0.05)
        self._worse_t = 0.0
        self._resumed = False
        self._log(f"cry trial: {step} soft start (level {self.ema:.2f})")

    def _abort(self, now: float, why: str) -> None:
        self.engine.command("M05", now)
        self.state = "settling"
        self._cooldown_until = now + COOLDOWN_ABORT_S
        self._alert(f"{why} -- taper and hand over")

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", action="store_true",
                        help="print the 50-motion library and exit")
    args = parser.parse_args(argv)
    for row in catalog():
        print(f"{row['id']}  {row['grade']:<2} {row['name']:<24} "
              f"{row['f_hz'] or '':<5} {row['a_mm'] or '':<5} "
              f"{row['a_peak_g']:.4f} g  {row['desc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

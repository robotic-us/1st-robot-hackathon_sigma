#!/usr/bin/env python3
"""The IDEA.md soothing policy: a learner that picks which motion to try.

docs/IDEA.md in runnable form.  Every infant has a fixed, hidden motion
personality (loves one entry, hates a couple, never changes), but the policy
is only shown the happiness ladder -- 행복 > 울음1 > 울음2 > 울음3 > 불행,
here ranks 0..4 over the machine's 0..1 distress input.  From that reward
alone it must explore the P1 candidates, notice what worsens, exploit what
works and say STOP at HAPPY.

The policy is an *advisor*, never a commander: ``CradleMachine.advisor``
asks it which P1 motion a trial should use, validates the answer, and keeps
every safety decision (when to trial, escalate, abort, taper, alert) for
itself.  A useless or unsafe answer simply falls back to the report ladder.

Two interchangeable brains produce the answer from the same prompt/history:

* ``ReflexBrain``   the taught strategy as plain code -- deterministic,
                    offline, doubles as the test double and the scenario
                    generator's teacher (tools/make_scenarios.py).
* ``ClaudeBrain``   the same decision asked of Claude over the Anthropic
                    SDK, few-shot taught with the generated scenarios.

Run::

    python3 serve.py --baby --personality --policy reflex
    python3 serve.py --baby --personality --policy claude   # ANTHROPIC_API_KEY
    python3 tests.py policy
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from core.cradle import CALM_LEVEL, CRY_LEVEL

# The reward ladder of docs/IDEA.md, discretised from the 0..1 distress the
# machine already runs on.  Band edges reuse the report thresholds plus the
# virtual baby's own FUSS/CRY base levels (perception/baby.py STATES).
RANKS = ("HAPPY", "FUSS", "CRY1", "CRY2", "MISERABLE")
BANDS = (CALM_LEVEL, 0.30, CRY_LEVEL, 0.62)

# The ten "모션 1~10" of the idea: the P1 ML sine candidates.  AP renders as
# pitch on this rig and R grades are barred from automatic behaviour, so the
# ML band is the whole safe search space.
CANDIDATES = tuple(f"M{n:02d}" for n in range(9, 19))
# Exploration order for untried motions: the report's own trial ladder first
# (0.3/0.5/0.6 Hz), then outward through the rest of the band.
EXPLORE = ("M10", "M12", "M13", "M09", "M11", "M14", "M15", "M16", "M17", "M18")

SCENARIOS_PATH = Path("data/scenarios.jsonl")


def rank_of(level: float) -> int:
    """0..1 distress -> 0..4 happiness rank (0 = HAPPY, 4 = MISERABLE)."""
    for i, edge in enumerate(BANDS):
        if level < edge:
            return i
    return len(BANDS)


@dataclass
class Step:
    """One advised motion and what it did to the rank (None = still playing)."""

    motion: str
    rank_before: int
    rank_after: Optional[int] = None

    @property
    def delta(self) -> Optional[int]:
        """Positive = improved (rank went down)."""
        if self.rank_after is None:
            return None
        return self.rank_before - self.rank_after


# --------------------------------------------------------------------------- #
# Prompt <-> reply: the language both brains share
# --------------------------------------------------------------------------- #
def render_prompt(scenarios: list[dict], steps: list[Step], rank: int) -> str:
    """The full decision prompt: rules, taught scenarios, this session, ask."""
    lines = [
        "You pick soothing motions for a robotic infant cradle.",
        "Each turn: choose ONE motion, it plays ~30 s, then you see the",
        "infant's comfort rank.  Ranks: 0 HAPPY > 1 FUSS > 2 CRY1 > 3 CRY2",
        "> 4 MISERABLE -- lower is better.  Goal: reach rank 0, then STOP.",
        f"Motions: {', '.join(CANDIDATES)} (gentle side-sways, different",
        "speeds/strengths).  Each infant has FIXED hidden preferences: some",
        "motions soothe fast, one or two make things worse, none of it",
        "changes within a session.",
        "",
        "Strategy: keep a motion that improved the rank; if it worsened or",
        "did nothing, switch -- best known motion first, else one untried;",
        "at rank 0 answer STOP.",
    ]
    if scenarios:
        lines += ["", "Example sessions (other infants):"]
        for sc in scenarios:
            trail = ", ".join(f"{s['motion']} {s['before']}->{s['after']}"
                              for s in sc.get("steps", []))
            lines.append(f"  {trail} => {sc.get('outcome', '?')}")
    lines += ["", "This infant so far:"]
    if steps:
        for s in steps:
            after = "?" if s.rank_after is None else s.rank_after
            lines.append(f"  {s.motion}: {s.rank_before} -> {after}")
    else:
        lines.append("  (nothing tried yet)")
    lines += [f"Current rank: {rank}",
              "",
              "Answer with exactly one token: a motion id or STOP."]
    return "\n".join(lines)


def parse_reply(text: str) -> Optional[str]:
    """'I'd try M13.' -> 'M13'; 'stop' -> 'STOP'; garbage -> None."""
    hit = re.search(r"\bM(0\d|1\d)\b", str(text).upper())
    if hit and hit.group(0) in CANDIDATES:
        return hit.group(0)
    if "STOP" in str(text).upper():
        return "STOP"
    return None


def load_scenarios(path: Path = SCENARIOS_PATH, limit: int = 6) -> list[dict]:
    """The few-shot corpus from tools/make_scenarios.py; [] when absent."""
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out[:limit]


# --------------------------------------------------------------------------- #
# Brains: (prompt, steps, rank) -> reply text
# --------------------------------------------------------------------------- #
class ReflexBrain:
    """The taught strategy as code: explore, back off, exploit, stop."""

    def __call__(self, prompt: str, steps: list[Step], rank: int) -> str:
        if rank == 0:
            return "STOP"
        scores: dict[str, list[int]] = {}
        for s in steps:
            if s.delta is not None:
                scores.setdefault(s.motion, []).append(s.delta)
        # Keep a motion that just worked.
        if steps and steps[-1].delta is not None and steps[-1].delta > 0:
            return steps[-1].motion
        # Exploit the best motion that has ever improved things.
        best, best_avg = None, 0.0
        for motion, deltas in scores.items():
            avg = sum(deltas) / len(deltas)
            if avg > best_avg:
                best, best_avg = motion, avg
        if best is not None:
            return best
        # Nothing has worked yet: explore an untried candidate.
        for motion in EXPLORE:
            if motion not in scores:
                return motion
        # Everything tried, nothing improved: least bad one.
        return max(scores, key=lambda m: sum(scores[m]) / len(scores[m]))


CLAUDE_SYSTEM = (
    "You are the motion-selection policy of a research infant-soothing "
    "cradle.  You answer with exactly one token: a motion id (M09..M18) or "
    "STOP.  A separate safety state machine validates everything you say; "
    "you only ever influence which pre-approved gentle motion is tried."
)


class ClaudeBrain:
    """The same decision asked of Claude (Anthropic SDK, lazy import).

    Failures return "" -- the policy then answers None and the machine
    falls back to the report ladder, so a network hiccup can never stall
    the nursery loop.
    """

    def __init__(self, model: str = "claude-opus-5") -> None:
        import anthropic   # only --policy claude pays this dependency
        self._anthropic = anthropic
        self._client = anthropic.Anthropic()
        self.model = model
        self.last_error = ""

    def __call__(self, prompt: str, steps: list[Step], rank: int) -> str:
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=4000,
                output_config={"effort": "low"},
                system=CLAUDE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            if response.stop_reason == "refusal":
                self.last_error = "refusal"
                return ""
            return next((b.text for b in response.content
                         if b.type == "text"), "")
        except Exception as exc:   # any API failure: fall back to the ladder
            self.last_error = f"{type(exc).__name__}: {exc}"
            return ""


# --------------------------------------------------------------------------- #
# The policy: history keeper + the machine's advisor
# --------------------------------------------------------------------------- #
class SoothePolicy:
    """Wire ``pick`` into ``CradleMachine.advisor`` and feed ``observe``.

    ``observe(now, level, engine)`` runs every sensor tick.  A step's outcome
    is the rank last seen while its motion was *engaged* -- playing at
    strength, not ramping in, not tapering out -- and the step settles when
    the engine parks.  Anything the baby does outside engagement (recovering
    on its own between trials, calming during a taper) is deliberately not
    credited to the motion; getting this wrong once taught the policy to
    love a motion the baby hated.
    ``pick(now, ema)`` is called by the machine at trial start and at
    escalation checkpoints; it closes the open step, asks the brain, and
    returns a candidate id or None (machine then uses its own ladder).
    """

    ENGAGED_ENV = 0.2      # below this the motion is not really acting yet

    def __init__(self, brain: Callable[[str, list, int], str],
                 scenarios: Optional[list[dict]] = None) -> None:
        self.brain = brain
        self.scenarios = scenarios or []
        self.steps: list[Step] = []
        self._seen: Optional[int] = None   # last rank while engaged

    def _close(self, fallback: int) -> None:
        if self.steps and self.steps[-1].rank_after is None:
            self.steps[-1].rank_after = (
                self._seen if self._seen is not None else fallback)
        self._seen = None

    def observe(self, now: float, level: float, engine=None) -> None:
        if not self.steps or self.steps[-1].rank_after is not None:
            return
        rank = rank_of(level)
        if engine is None or not engine.active:
            self._close(rank)             # motion stopped: settle the step
            return
        if (not engine.tapering
                and engine.env * engine.amp_scale > self.ENGAGED_ENV):
            self._seen = rank

    def scores(self) -> dict[str, float]:
        """Per-motion mean rank improvement over the closed steps (>0 helps)."""
        deltas: dict[str, list[int]] = {}
        for s in self.steps:
            if s.delta is not None:
                deltas.setdefault(s.motion, []).append(s.delta)
        return {m: round(sum(d) / len(d), 2) for m, d in deltas.items()}

    def snapshot(self) -> dict:
        """The /events payload: what the dashboard's learning panel draws."""
        return {
            "brain": type(self.brain).__name__.replace("Brain", "").lower(),
            "scenarios": len(self.scenarios),
            "scores": self.scores(),
            "steps": [{"motion": s.motion, "before": s.rank_before,
                       "after": s.rank_after} for s in self.steps[-24:]],
        }

    def pick(self, now: float, ema: float) -> Optional[str]:
        rank = rank_of(ema)
        self._close(rank)
        reply = self.brain(render_prompt(self.scenarios, self.steps, rank),
                           self.steps, rank)
        motion = parse_reply(reply)
        if motion in CANDIDATES:
            self.steps.append(Step(motion, rank))
            self._seen = None
            return motion
        return None                   # STOP/garbage: the machine decides

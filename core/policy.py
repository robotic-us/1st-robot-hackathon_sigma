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

Three interchangeable brains produce the answer from the same prompt/history
(switchable live from the dashboard via serve.py's /policy endpoint):

* ``ReflexBrain``   the taught strategy as plain code -- deterministic,
                    offline, doubles as the test double and the scenario
                    generator's teacher (tools/make_scenarios.py).
* ``OllamaBrain``   the same decision asked of a local LLM over Ollama's
                    HTTP API (stdlib urllib; OLLAMA_URL / OLLAMA_MODEL).
* ``ClaudeBrain``   the same decision asked of Claude over the Anthropic
                    SDK (paid API; needs ANTHROPIC_API_KEY).

Run::

    python3 serve.py --baby --personality --policy reflex
    python3 serve.py --baby --personality --policy ollama   # local LLM
    python3 serve.py --baby --personality --policy claude   # ANTHROPIC_API_KEY
    python3 tests.py policy
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from core.cradle import CALM_LEVEL, CRY_LEVEL, N_CANDIDATES, N_LIBRARY

# The reward ladder of docs/IDEA.md, discretised from the 0..1 distress the
# machine already runs on.  Band edges reuse the report thresholds plus the
# virtual baby's own FUSS/CRY base levels (perception/baby.py STATES).
RANKS = ("HAPPY", "FUSS", "CRY1", "CRY2", "MISERABLE")
BANDS = (CALM_LEVEL, 0.30, CRY_LEVEL, 0.62)

# The search space: the team's 34-motion system (docs/motion-system.png),
# minus the parked state and the self-stopping decay entries.  Each
# candidate carries its features -- shape, size, speed, tremble -- which is
# what lets a brain generalise ("slow+large works on this baby") instead of
# memorising 26 ids one by one.
CANDIDATES = N_CANDIDATES
FEATURES = {m.id: {"shape": m.shape, "size": m.size, "speed": m.speed,
                   "vibe": "vibe" if m.vibe else "plain"}
            for m in N_LIBRARY}
# Cold-start exploration order: one calm representative of each family
# first (wide slow strokes), then the fast ones, then the tremble variants.
EXPLORE = ("N05", "N10", "N16", "N20", "N24", "N27", "N30", "N33", "N13",
           "N18", "N21", "N03", "N08", "N14", "N19", "N23", "N26", "N29",
           "N32", "N06", "N11", "N17", "N02", "N04", "N09", "N15")

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
        "Each turn: choose ONE motion, it plays one short interval (about",
        "10-30 s), then you see the infant's comfort rank.  Ranks: 0 HAPPY",
        "> 1 FUSS > 2 CRY1 > 3 CRY2",
        "> 4 MISERABLE -- lower is better.  Goal: reach rank 0, then STOP.",
        "",
        "Motions (id: shape-size-speed, +T = with tremble):",
    ]
    tags = [f"{c}:{FEATURES[c]['shape']}"
            + (f"-{FEATURES[c]['size']}-{FEATURES[c]['speed']}"
               if FEATURES[c]["size"] else "")
            + ("+T" if FEATURES[c]["vibe"] == "vibe" else "")
            for c in CANDIDATES]
    for i in range(0, len(tags), 5):
        lines.append("  " + "  ".join(tags[i:i + 5]))
    lines += [
        "",
        "Each infant has FIXED hidden preferences over these FEATURES --",
        "a favourite shape, a hated one, small-vs-large, fast-vs-slow, and",
        "loving or hating the tremble.  Generalise: if wide+slow shapes",
        "work, other wide+slow shapes likely will.",
        "",
        "Strategy: while the infant merely fusses (rank 1), experiment --",
        "keep a winner at most twice in a row, then try the most promising",
        "untried motion.  While it cries (rank 2+), play the best known",
        "remedy.  Motions wear out with heavy use (habituation): when the",
        "favourite fades, rotate and come back later.  At rank 0: STOP.",
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
    """'I'd try N16.' -> 'N16'; 'stop' -> 'STOP'; garbage -> None."""
    for hit in re.findall(r"\b[MN]\d{2}\b", str(text).upper()):
        if hit in CANDIDATES:
            return hit
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
    """The taught strategy as code: experiment, generalise, exploit, stop.

    Three rules shape it:

    * **Explore while fussing, exploit while crying.**  A mild fuss (rank 1)
      is cheap experiment time -- a winner is kept at most ``MAX_RUN`` times
      in a row before something untried gets a turn, which is what keeps one
      lucky motion from monopolising the whole night.  A crying baby
      (rank >= 2) always gets the best known remedy.
    * **Generalise over features.**  Untried motions are ranked by the mean
      outcome of their shape/size/speed/tremble values across everything
      tried so far -- learn "wide and slow works" from N05 and N16 already
      points at N20 and N27.
    * **Recent outcomes only.**  Scores are each motion's last three
      attempts, because habituation is real: a worn-out favourite must fall
      out of favour, and a rested one must be allowed back.
    """

    RECENT = 3
    MAX_RUN = 2

    def __call__(self, prompt: str, steps: list[Step], rank: int) -> str:
        if rank == 0:
            return "STOP"
        deltas: dict[str, list[int]] = {}
        for s in steps:
            if s.delta is not None:
                deltas.setdefault(s.motion, []).append(s.delta)
        recent = {m: sum(d[-self.RECENT:]) / len(d[-self.RECENT:])
                  for m, d in deltas.items()}
        untried = [c for c in CANDIDATES if c not in deltas]
        last = steps[-1] if steps else None
        run = 0
        for s in reversed(steps):
            if last is not None and s.motion == last.motion:
                run += 1
            else:
                break
        last_won = last is not None and last.delta is not None and last.delta > 0
        best = max(recent, key=recent.get) if recent else None

        if rank >= 2:                      # crying: best known remedy, now
            if last_won:
                return last.motion
            if best is not None and recent[best] > 0:
                return best
            return self._promising(untried, deltas) if untried \
                else (best or EXPLORE[0])
        # merely fussing: keep a winner briefly, then experiment
        if last_won and run < self.MAX_RUN:
            return last.motion
        if untried:
            return self._promising(untried, deltas)
        if best is not None:
            return best
        return EXPLORE[0]

    @staticmethod
    def _promising(untried: list, deltas: dict) -> str:
        """The untried candidate whose features have scored best so far."""
        fscore: dict[tuple, list[int]] = {}
        for m, ds in deltas.items():
            for f, v in FEATURES[m].items():
                fscore.setdefault((f, v), []).extend(ds)

        def predicted(c: str) -> float:
            known = [sum(fscore[k]) / len(fscore[k])
                     for k in ((f, v) for f, v in FEATURES[c].items())
                     if k in fscore]
            return sum(known) / len(known) if known else 0.0

        order = {m: i for i, m in enumerate(EXPLORE)}
        return max(untried,
                   key=lambda c: (predicted(c), -order.get(c, len(order))))


BRAIN_SYSTEM = (
    "You are the motion-selection policy of a research infant-soothing "
    "cradle.  You answer with exactly one token: a motion id (M09..M18) or "
    "STOP.  A separate safety state machine validates everything you say; "
    "you only ever influence which pre-approved gentle motion is tried."
)


class OllamaBrain:
    """The same decision asked of a *local* LLM over Ollama's HTTP API.

    stdlib-only (urllib), so it costs no dependency and no money; point it
    at another host with OLLAMA_URL.  Failures return "" -- the policy
    then answers None and the machine falls back to the report ladder, so
    a missing/slow local model can never stall the nursery loop (the one
    cost is this call's own timeout, bounded well under the machine's
    30 s checkpoint cadence).
    """

    TIMEOUT_S = 10.0

    def __init__(self, url: Optional[str] = None,
                 model: Optional[str] = None) -> None:
        import os
        self.url = (url or os.environ.get("OLLAMA_URL",
                                          "http://127.0.0.1:11434")).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
        self.last_error = ""

    def __call__(self, prompt: str, steps: list[Step], rank: int) -> str:
        import urllib.request
        body = json.dumps({
            "model": self.model, "stream": False,
            "messages": [{"role": "system", "content": BRAIN_SYSTEM},
                         {"role": "user", "content": prompt}],
            "options": {"num_predict": 24},
        }).encode()
        try:
            req = urllib.request.Request(
                self.url + "/api/chat", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.TIMEOUT_S) as resp:
                out = json.load(resp)
            self.last_error = ""
            return out.get("message", {}).get("content", "")
        except Exception as exc:   # unreachable/missing model: use the ladder
            self.last_error = (f"local LLM unavailable at {self.url} "
                               f"({type(exc).__name__})")
            return ""


class ClaudeBrain:
    """The same decision asked of Claude (Anthropic SDK, lazy import).

    Failures return "" -- the policy then answers None and the machine
    falls back to the report ladder, so a network hiccup can never stall
    the nursery loop.
    """

    def __init__(self, model: str = "claude-opus-5") -> None:
        import anthropic   # only the claude brain pays this dependency
        self._anthropic = anthropic
        # bounded: this runs on the sensor thread between frames
        self._client = anthropic.Anthropic(timeout=15.0, max_retries=0)
        self.model = model
        self.last_error = ""

    def __call__(self, prompt: str, steps: list[Step], rank: int) -> str:
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=4000,
                output_config={"effort": "low"},
                system=BRAIN_SYSTEM,
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


def make_brain(kind: str, model: Optional[str] = None):
    """'reflex' | 'ollama' | 'claude' -> a brain, or raise with a clear why.

    The one place serve.py's /policy switcher and --policy flag resolve a
    name; keep the vocabulary here so the dashboard and CLI never drift.
    """
    kind = (kind or "").lower()
    if kind == "reflex":
        return ReflexBrain()
    if kind == "ollama":
        return OllamaBrain(model=model)
    if kind == "claude":
        return ClaudeBrain(model=model) if model else ClaudeBrain()
    raise ValueError(f"unknown brain {kind!r} (reflex | ollama | claude)")


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
            "error": getattr(self.brain, "last_error", ""),
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

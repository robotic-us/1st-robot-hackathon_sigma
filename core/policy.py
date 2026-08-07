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
import math
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
    """'reflex' | 'dream' | 'ollama' | 'claude' -> a brain, or raise with a why.

    The one place serve.py's /policy switcher and --policy flag resolve a
    name; keep the vocabulary here so the dashboard and CLI never drift.
    """
    kind = (kind or "").lower()
    if kind == "reflex":
        return ReflexBrain()
    if kind == "dream":
        return DreamBrain()
    if kind == "ollama":
        return OllamaBrain(model=model)
    if kind == "claude":
        return ClaudeBrain(model=model) if model else ClaudeBrain()
    raise ValueError(f"unknown brain {kind!r} "
                     "(reflex | dream | ollama | claude)")


# --------------------------------------------------------------------------- #
# DREAM-Chunk, with the infant as the plant (docs/dream-chunk.md)
# --------------------------------------------------------------------------- #
@dataclass
class DreamConfig:
    """Knobs, in the shape docs/dream-chunk.md's own tuning table uses."""

    tau_s: float = 8.0       # expected soothing time constant
    prior_gain: float = 0.35 # untried motion: expect this much of the way down
    # The unit of a meaningful comfort change: the fuss-to-cry span.  Scoring
    # a play as (y0 - y_end) / y0 instead looks reasonable and is not -- the
    # machine only asks for a motion while the infant is fussing, so y0 is
    # small, improvement is capped at +1 and worsening is unbounded.  Measured
    # over an hour that bias alone drove the mean residual to -0.09 against a
    # median of +0.04 and vetoed 21 of 26 motions, 19 of them ones the infant
    # genuinely liked.  A fixed scale is symmetric; a ratio is not.
    scale: float = 0.33
    # What the policy assumes about wearing a motion out.  These are its own
    # assumption, deliberately NOT read from perception/baby.py -- the model
    # has to work on an infant whose constants nobody knows.
    wear_s: float = 90.0     # engaged seconds to wear a motion out
    rest_s: float = 240.0    # recovery constant while it is not playing
    wear_floor: float = 0.25 # even worn out, a motion keeps this much effect
    veto_n: int = 5          # samples before a motion may be vetoed at all
    # Taste is estimated hierarchically -- per *feature* first, per id second.
    # A 90-minute session yields ~175 plays over 26 motions, so a single id
    # gets ~7 noisy samples and its mean is worthless (measured rank
    # correlation with the infant's real taste: +0.03).  The same plays give
    # every feature *value* ten times that, and the features are what the
    # taste is actually made of: pooling there scores +0.44, and shrinking the
    # id estimate toward its features by n/(n+K) reaches +0.56.
    shrink_k: float = 40.0
    # Exploration, as a confidence bound rather than a flat bonus for being
    # untried.  The estimator is honest but *low-contrast*: measured taste
    # spans about [-0.23, +0.09] where the infant's real gains span [0, 2.5],
    # so a ranking that is only ~0.6 correlated gets exploited as if it were
    # certain, the cradle settles on one motion and the infant habituates to
    # it.  Rotation is what actually soothes -- ReflexBrain's MAX_RUN gets
    # -21% against the ladder on rotation alone -- so the planner has to keep
    # a reason to move on.  c * sqrt(ln(N)/n) is that reason, and it fades as
    # the evidence for a motion accumulates.
    explore_c: float = 0.06


class WorldModel:
    """What this infant makes of each motion -- DREAM-Chunk's world model,
    re-anchored on the baby instead of the arm.

    ``docs/dream-chunk.md`` maps DREAM-Chunk onto phorce motion slots; that
    version dreams the *arm*, and the arm is not what we are trying to
    improve.  Here the state is distress, the sensor is the judge's live 0..1
    level -- the same one the machine runs on -- and what is learned is:

    ``taste``   what the infant thinks of a motion.  Fixed, per docs/IDEA.md,
                and estimated hierarchically: pooled over the motion's
                *features* first, its own id second, because a session gives
                one id ~7 noisy plays and each feature value ten times that.
    ``wear``    how worn out a motion is right now.  Transient, and computed
                from this model's own play history rather than learned -- it
                knows what it played and for how long.
    ``pairs``   what a motion is worth *after* another one: docs/IDEA.md's
                combo, the one preference a per-motion score cannot hold.

    Keeping taste and wear apart is the whole point.  Recording "played until
    it stopped working" as "disliked" is permanent, and a motion never offered
    again never gets to show it recovered -- an hour of that vetoed 21 of 26
    motions, 19 of them ones the infant genuinely liked.

    The paper's other half -- a divergence tube that cut a motion the moment
    it left its dreamed curve -- was built, measured at +2.9 % upset, and
    removed.  ``docs/dream-chunk.md`` keeps the numbers.
    """

    PAIR_KEEP = 4        # samples kept per transition, newest wins

    def __init__(self, cfg: Optional[DreamConfig] = None) -> None:
        self.cfg = cfg or DreamConfig()
        self.motion: Optional[str] = None
        # The world model, in two halves that must not be confused:
        #   taste -- what this infant thinks of a motion.  Fixed, per IDEA.md.
        #   wear  -- how worn out it is right now.  Transient; it recovers.
        # Learning one as the other is what poisons a carried model: a motion
        # played until it stopped working gets recorded as disliked, is never
        # offered again, and so never gets the chance to show it recovered.
        # Taste is not stored; it is *estimated* from running sums, per motion
        # and per feature value, and the two are blended by how much evidence
        # the motion itself has.  Plain means, not an EWMA: the taste is fixed
        # by assumption, so every sample of it still counts.
        self._id_sum: dict[str, float] = {}
        self._f_sum: dict[tuple, float] = {}
        self._f_n: dict[tuple, int] = {}
        self.wear: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        # (previous, next) -> recent samples.  The sequential half of the
        # model: what this infant makes of one motion *following* another.
        self.pairs: dict[tuple, list] = {}
        self._prev: Optional[str] = None
        self._y0 = 0.0
        self._last = 0.0
        self._wear0 = 0.0
        self._engaged_s = 0.0

    def feature_prior(self, motion: str) -> float:
        """What this motion's *features* have been worth -- the pooled half."""
        seen = [self._f_sum[k] / self._f_n[k]
                for k in ((f, v) for f, v in FEATURES.get(motion, {}).items())
                if k in self._f_n]
        return sum(seen) / len(seen) if seen else self.cfg.prior_gain

    def taste_of(self, motion: str) -> float:
        """The fixed half: the id's own mean, shrunk toward its features."""
        n = self.counts.get(motion, 0)
        prior = self.feature_prior(motion)
        if not n:
            return prior
        w = n / (n + self.cfg.shrink_k)
        return w * (self._id_sum[motion] / n) + (1.0 - w) * prior

    @property
    def taste(self) -> dict:
        """Every candidate's fixed worth -- untried ones by their features."""
        return {m: self.taste_of(m) for m in CANDIDATES}

    @property
    def gain(self) -> dict:
        """What a motion is worth *right now*: taste, less how worn it is."""
        return {m: t * (1.0 - self.wear.get(m, 0.0))
                for m, t in self.taste.items()}

    def age(self, dt: float, playing: Optional[str] = None) -> None:
        """Wear builds on what is playing and recovers on everything else.

        The policy can compute this from its own actions -- it knows what it
        played and for how long -- so it never has to *learn* the transient
        and mistake it for the permanent.
        """
        if dt <= 0.0:
            return
        decay = math.exp(-dt / self.cfg.rest_s)
        for m in list(self.wear):
            if m != playing:
                self.wear[m] *= decay
                if self.wear[m] < 1e-3:
                    del self.wear[m]
        if playing:
            self.wear[playing] = min(1.0, self.wear.get(playing, 0.0)
                                     + dt / self.cfg.wear_s)

    # -- the world model ----------------------------------------------------- #
    def note_residual(self, motion: str, y0: float, y_end: float,
                      engaged_s: float, prev: Optional[str] = None) -> None:
        """ASAP in reduced form: measured-minus-dreamed corrects the *model*.

        The doc's point exactly -- ASAP adds its residual to the actuator
        command and we have no such channel, so the dream is corrected and
        the command is not.  The same measurement updates both halves: what
        this motion is worth, and what it is worth *after that one*.
        """
        if engaged_s < 1.0 or y0 <= 1e-3:
            return
        settled = 1.0 - math.exp(-engaged_s / self.cfg.tau_s)
        # symmetric in comfort units, not as a fraction of a small y0
        moved = (y0 - max(0.0, y_end)) / self.cfg.scale
        observed = min(1.0, max(-1.0, moved / max(0.2, settled)))
        # ...and divided out by how worn the motion already was, so a tired
        # motion's poor showing lands on `wear`, where it recovers, instead of
        # on `taste`, where it would be permanent
        rested = max(self.cfg.wear_floor, 1.0 - self._wear0)
        achieved = min(1.0, max(-1.0, observed / rested))
        self._id_sum[motion] = self._id_sum.get(motion, 0.0) + achieved
        self.counts[motion] = self.counts.get(motion, 0) + 1
        # ...and the same sample teaches every feature the motion has, which
        # is where the evidence is dense enough to mean anything
        for f, v in FEATURES.get(motion, {}).items():
            self._f_sum[(f, v)] = self._f_sum.get((f, v), 0.0) + achieved
            self._f_n[(f, v)] = self._f_n.get((f, v), 0) + 1
        if prev:
            seen = self.pairs.setdefault((prev, motion), [])
            seen.append(achieved)
            del seen[:-self.PAIR_KEEP]

    # -- the loop ------------------------------------------------------------ #
    def start(self, motion: str, level: float) -> None:
        self.motion = motion
        self._y0 = self._last = max(0.0, level)
        self._wear0 = self.wear.get(motion, 0.0)   # how tired it already was
        self._engaged_s = 0.0

    def observe(self, level: float, engaged: bool, dt: float) -> None:
        # wear ages on every tick, playing or not -- resting is when a motion
        # recovers, and the model only knows that if it keeps the clock
        self.age(dt, self.motion if (engaged and self.motion) else None)
        if self.motion is None or not engaged or dt <= 0.0:
            return
        self._engaged_s += dt
        self._last = max(0.0, level)

    def settle(self) -> None:
        """Close the open slot: fold its residual into the world model."""
        if self.motion is not None:
            self.note_residual(self.motion, self._y0, self._last,
                               self._engaged_s, prev=self._prev)
            self._prev = self.motion      # what the next slot follows
        self.motion = None

    def snapshot(self) -> dict:
        return {"on": True, "motion": self.motion,
                "engaged_s": round(self._engaged_s, 1),
                # both halves, because confusing them is the failure mode
                "taste": {m: round(t, 2) for m, t in self.taste.items()},
                "wear": {m: round(w, 2) for m, w in self.wear.items()
                         if w >= 0.05},
                "gain": {m: round(g, 2) for m, g in self.gain.items()},
                "pairs": {f"{a}>{b}": round(sum(v) / len(v), 2)
                          for (a, b), v in self.pairs.items() if v}}


class ChunkMatcher:
    """DREAM-Chunk's *other* half: dream every candidate, rank, take the best.

    ``WorldModel`` answers "is the motion playing still working?".  This
    answers the question the paper is actually named for -- *reactive action
    matching*: roll all 26 candidates forward through the world model from
    where the infant is right now, and score them.  The doc's cost breakdown,
    re-anchored:

    ``task fit``      predicted distress at the horizon under that motion's
                      learned gain -- the whole point, and the only term that
                      needs the world model
    ``continuity``    the arm version avoids a jump in joint space; here a
                      change of motion is itself a small disturbance to a
                      settling baby, so switching pays a penalty
    ``resistance``    ``dob_a`` is the world pushing back on the current plan.
                      The infant's version of pushing back is habituation, so
                      recent use of a candidate counts against it
    ``veto``          the excursion guard's analogue: a motion this baby has
                      *measurably* been made worse by is never returned

    Unknown candidates are scored by feature generalisation (shape/size/
    speed/tremble), the same trick ``ReflexBrain`` uses, plus an optimism
    bonus -- untried motions are worth learning about.
    """

    SLOT_S = 20.0        # one motion holds this long before the next
    # Plans of one.  The machinery below dreams schedules of any depth -- the
    # paper's actual mechanism -- but measured over 100 paired nights, depth 3
    # costs +12.0% upset / +37.3% crying against depth 1 (and +11.8% with the
    # slot matched to the machine's own 10 s cadence, so it is not a mismatched
    # constant).  The transitions that would justify sequencing are 1 pair in
    # 676 and turn up ~5 times in 100 nights.  Raise it to see the tree branch;
    # do not raise it expecting a calmer infant.
    DEPTH = 1            # motions per dreamed plan -- the chunk
    BEAM = 2             # plans kept per opening (26^3 exhaustive is wasteful)
    W_SWITCH = 0.020     # continuity: the cost of changing motion at all
    # A single play is a coin flip -- the infant's own dwell process moves the
    # level as much as any motion does -- so the bar for exiling a motion has
    # to clear that noise.  At -0.15 over 3 samples an hour's run vetoed 21 of
    # 26 motions, 19 of them ones the infant liked.
    VETO_GAIN = -0.40    # measured to make this baby worse: never returned
    FADE = 0.55          # each repeat inside one plan is worth this much less
    PAIR_TRUST = 3.0     # samples before a learned transition is fully believed

    @property
    def HORIZON_S(self) -> float:
        """How far the dream reaches: the whole chunk, not one slot."""
        return self.SLOT_S * self.DEPTH

    def __init__(self, gains: Optional[dict] = None,
                 cfg: Optional[DreamConfig] = None,
                 pairs: Optional[dict] = None,
                 wear: Optional[dict] = None,
                 counts: Optional[dict] = None) -> None:
        # `gains` is *taste* -- the fixed half.  `wear` is the transient half,
        # kept apart so a tired motion is passed over today and offered again
        # tomorrow, while a disliked one stays out.
        self.gains = gains if gains is not None else {}
        self.wear = wear if wear is not None else {}
        self.counts = counts if counts is not None else {}
        # (previous, next) -> [gain samples].  docs/IDEA.md's own example is a
        # *sequential* taste -- "2모션 하다가 4모션으로 가는걸 좋아함" -- and no
        # amount of per-motion scoring can represent it.
        self.pairs = pairs if pairs is not None else {}
        self.cfg = cfg or DreamConfig()
        self.last: list[tuple] = []       # the last ranking, for the log/UI
        self.plans: list[tuple] = []      # the last dreamed plans
        self.vetoed: list[str] = []       # excluded, and why the dashboard says so
        self.at_level = 0.0               # the level it dreamed from

    def bonus(self, motion: str, used: int = 0) -> float:
        """The confidence bound: how much this motion's estimate might be
        understating it.  Wide while the evidence is thin, and it shrinks as
        the motion is played -- including within the plan being dreamed."""
        n = self.counts.get(motion, 0) + used
        total = sum(self.counts.values()) + 1
        return self.cfg.explore_c * math.sqrt(math.log(total + 1) / (1 + n))

    def gain_of(self, motion: str) -> float:
        """The world model's taste for a candidate.

        ``gains`` already carries every candidate -- ``WorldModel`` fills
        untried ones in from their features -- so there is no inference to do
        here.  Keeping it in one place is deliberate: two half-implementations
        of feature generalisation is how they drift apart.
        """
        return self.gains.get(motion, self.cfg.prior_gain)

    def effective(self, prev: Optional[str], motion: str,
                  extra_wear: float = 0.0) -> float:
        """What the motion is worth right now: taste, less how worn it is."""
        worn = min(1.0, self.wear.get(motion, 0.0) + extra_wear)
        return self.pair_gain(prev, motion) * (1.0 - worn)

    def pair_gain(self, prev: Optional[str], motion: str) -> float:
        """The gain of playing ``motion`` *after* ``prev``.

        A transition is only believed once it has been seen a few times --
        until then this is the plain per-motion gain, so one lucky handover
        cannot invent a combo that is not there.
        """
        base = self.gain_of(motion)
        seen = self.pairs.get((prev, motion)) if prev else None
        if not seen:
            return base
        w = min(1.0, len(seen) / self.PAIR_TRUST)
        return (1.0 - w) * base + w * (sum(seen) / len(seen))

    def dream(self, motion: str, level: float,
              horizon_s: Optional[float] = None,
              prev: Optional[str] = None, fade: float = 1.0) -> float:
        """Predicted distress after ``horizon_s`` of this motion."""
        h = self.SLOT_S if horizon_s is None else horizon_s
        g = max(-1.0, min(1.0, self.effective(prev, motion) * fade))
        return max(0.0, level * (1.0 - g * (1.0 - math.exp(-h / self.cfg.tau_s))))

    def _live(self, level: float) -> list[str]:
        """The candidates worth dreaming; the rest are vetoed and recorded.

        The veto reads *taste*, never the worn-down value, and only once a
        motion has been seen enough times to mean it -- a permanent exile on
        one bad stretch is how a model talks itself out of every motion the
        infant actually likes.
        """
        self.vetoed, self.at_level = [], level
        live = []
        for c in CANDIDATES:
            if (self.gain_of(c) <= self.VETO_GAIN
                    and self.counts.get(c, 0) >= self.cfg.veto_n):
                self.vetoed.append(c)           # vetoed, like an excursion
            else:
                live.append(c)
        return live

    def plan(self, level: float, current: Optional[str] = None,
             steps: Optional[list] = None,
             depth: Optional[int] = None) -> list[tuple]:
        """Dream whole **plans**, not single motions: [(cost, seq, trace)].

        This is the part that makes it action *chunking*.  A chunk here is a
        schedule -- three motions, ``SLOT_S`` each -- and the cost is the mean
        distress the infant is predicted to sit at *across* it, so a plan that
        settles early beats one that only ends well.  Two structures exist
        that a one-motion-at-a-time score cannot see:

        * **habituation** -- repeating a motion inside a plan is worth
          ``FADE`` less each time, so rotation beats hammering a favourite;
        * **transitions** -- a learned (prev, next) gain, which is exactly
          docs/IDEA.md's combo.

        Beam search: 26^3 is 17k plans for nothing, and the beam keeps the
        search bounded on the sensor thread.  Only the first motion is
        committed -- the machine asks again at its next checkpoint, so this
        is a receding horizon, not a schedule anyone is stuck with.
        """
        depth = self.DEPTH if depth is None else depth
        live = self._live(level)
        order = {m: i for i, m in enumerate(EXPLORE)}
        settle = 1.0 - math.exp(-self.SLOT_S / self.cfg.tau_s)

        # (cost, level, prev, seq, used, trace)
        beam = [(0.0, level, current, (), {}, ())]
        for _step in range(depth):
            grown = []
            for cost, y, prev, seq, used, trace in beam:
                for c in live:
                    fade = self.FADE ** used.get(c, 0)
                    g = max(-1.0, min(1.0, self.effective(prev, c) * fade))
                    y2 = max(0.0, y * (1.0 - g * settle))
                    # the objective is *time spent upset*, not the end state:
                    # the mean level across this slot, trapezoid
                    step_cost = (y + y2) / 2.0
                    step_cost += 0.0 if c == prev else self.W_SWITCH
                    # the confidence bound, narrowing as the plan itself
                    # would gather evidence about this motion
                    step_cost -= self.bonus(c, used.get(c, 0))
                    grown.append((cost + step_cost, y2, c, seq + (c,),
                                  {**used, c: used.get(c, 0) + 1},
                                  trace + (round(y2, 3),)))
            grown.sort(key=lambda r: (r[0], order.get(r[3][0], len(order)),
                                      r[3]))
            # Beam *per first motion*, not globally: the first motion is the
            # only thing committed, so the search must not let one promising
            # opening crowd every alternative out of the beam (and the
            # dashboard would otherwise draw six plans with the same prefix).
            kept, seen_root = [], {}
            for row in grown:
                root = row[3][0]
                if seen_root.get(root, 0) >= self.BEAM:
                    continue
                seen_root[root] = seen_root.get(root, 0) + 1
                kept.append(row)
            beam = kept
        plans = [(cost / max(1, depth), seq, trace)
                 for cost, _y, _p, seq, _u, trace in beam]
        self.plans = plans
        # the one-motion view the cost breakdown is drawn from
        self.last = self.rank(level, current, steps)
        return plans

    def rank(self, level: float, current: Optional[str] = None,
             steps: Optional[list] = None) -> list[tuple]:
        """[(cost, motion, terms)] ascending -- one slot, the doc's terms."""
        live = self._live(level)
        out = []
        for c in live:
            fit = self.dream(c, level, prev=current)
            switch = 0.0 if c == current else self.W_SWITCH
            # `resist` was a stand-in for habituation before the model kept
            # its own wear; it is now what the dream already accounts for
            resist = round(self.wear.get(c, 0.0), 3)
            new = -self.bonus(c)
            out.append((fit + switch + new, c,
                        {"fit": round(fit, 3), "switch": switch,
                         "resist": resist, "new": round(new, 3),
                         "gain": round(self.pair_gain(current, c), 2)}))
        # Cold start is one big tie: with nothing measured, every untried
        # candidate inherits the same feature-inferred gain and the same
        # terms.  Break it on EXPLORE -- one calm representative of each
        # family first -- rather than on the id, which would just walk the
        # library in numerical order.
        order = {m: i for i, m in enumerate(EXPLORE)}
        out.sort(key=lambda r: (r[0], order.get(r[1], len(order)), r[1]))
        self.last = out
        return out

    def best(self, level: float, current: Optional[str] = None,
             steps: Optional[list] = None) -> Optional[str]:
        """The first motion of the best plan -- the rest is re-decided later."""
        plans = self.plan(level, current, steps)
        return plans[0][1][0] if plans and plans[0][1] else None


class DreamBrain:
    """A brain that *plans* rather than reacts: ``ChunkMatcher.best()``.

    The same interface as the other three, so ``make_brain('dream')`` and the
    dashboard's live switcher treat it like any other -- and the machine still
    validates whatever it says.  It ignores the rendered prompt: this brain
    reads the world model directly rather than a description of it.
    """

    def __init__(self) -> None:
        self.policy = None
        self.matcher = ChunkMatcher()
        self.last_error = ""
        self.chosen = ""
        self.playing = None

    def bind(self, policy) -> None:
        """``SoothePolicy`` hands itself over: the gains live in its monitor."""
        self.policy = policy

    def _gains(self) -> tuple[dict, dict, dict, dict]:
        """The world model: taste, transitions, wear, and sample counts."""
        monitor = getattr(self.policy, "model", None)
        if monitor is not None:
            return (monitor.taste, monitor.pairs, monitor.wear,
                    monitor.counts)
        # no monitor: fall back to the coarse rank deltas the history carries
        deltas: dict[str, list[float]] = {}
        for s in getattr(self.policy, "steps", []):
            if s.delta is not None:
                deltas.setdefault(s.motion, []).append(s.delta / 4.0)
        return ({m: sum(d) / len(d) for m, d in deltas.items()}, {}, {},
                {m: len(d) for m, d in deltas.items()})

    def __call__(self, prompt: str, steps: list[Step], rank: int) -> str:
        if rank == 0:
            self.chosen = ""
            return "STOP"
        (self.matcher.gains, self.matcher.pairs, self.matcher.wear,
         self.matcher.counts) = self._gains()
        level = getattr(self.policy, "level", None)
        if level is None:                      # no live level: use the band
            level = ([0.06] + list(BANDS))[min(rank, len(BANDS))]
        self.playing = steps[-1].motion if steps else None
        self.chosen = self.matcher.best(level, self.playing, steps) or ""
        return self.chosen

    SHOW = 6        # rows the dashboard draws; the rest are a count

    KEEP_PER_OPENING = 2    # so the drawn tree actually branches

    def _by_opening(self, plans: list) -> list:
        """The plans worth drawing: the best few openings, each with its best
        continuations.  The first motion is all that gets committed, so the
        openings are the real choice; keeping a couple of continuations each
        is what makes the dashboard's picture a tree rather than a fan."""
        out: list = []
        per: dict = {}
        for cost, seq, trace in plans:
            if not seq:
                continue
            root = seq[0]
            if root not in per and len(per) >= self.SHOW:
                continue
            if per.get(root, 0) >= self.KEEP_PER_OPENING:
                continue
            per[root] = per.get(root, 0) + 1
            out.append({"seq": list(seq), "cost": round(cost, 3),
                        "trace": list(trace)})
        return out

    def plan_snapshot(self) -> dict:
        """What the planner imagined, for the dashboard to draw.

        The whole point of a *planning* brain is that its reasoning is
        inspectable -- unlike the LLM brains, every number here is one it
        actually used.  ``fit`` is the dreamed distress at the horizon;
        the rest are the doc's other cost terms, kept separate so the
        picture shows *why* the winner won.
        """
        ranked = self.matcher.last
        return {
            "level": round(self.matcher.at_level, 3),
            "horizon_s": self.matcher.HORIZON_S,
            "slot_s": self.matcher.SLOT_S,
            "chosen": getattr(self, "chosen", ""),
            "playing": getattr(self, "playing", None),
            "dreamed": len(ranked) + len(self.matcher.vetoed),
            "vetoed": list(self.matcher.vetoed),
            # The chunk: whole schedules, best first.  Only the first motion
            # is committed, so the list is the best plan per distinct
            # *opening* -- six variations on one opening would tell the
            # reader nothing about the choice actually being made.
            "plans": self._by_opening(self.matcher.plans),
            "candidates": [
                {"id": mid, "cost": round(cost, 3), **terms}
                for cost, mid, terms in ranked[:self.SHOW]
            ],
        }


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
                 scenarios: Optional[list[dict]] = None,
                 model: bool = True) -> None:
        self.brain = brain
        self.scenarios = scenarios or []
        self.steps: list[Step] = []
        self._seen: Optional[int] = None   # last rank while engaged
        # What the infant makes of each motion.  A planning brain reads it;
        # the machine never does -- every safety decision stays over there.
        self.model = WorldModel() if model else None
        self._t: Optional[float] = None
        self.level = 0.0       # the live distress a planning brain reads
        if hasattr(brain, "bind"):     # DreamBrain plans off the world model
            brain.bind(self)

    def _close(self, fallback: int) -> None:
        if self.steps and self.steps[-1].rank_after is None:
            self.steps[-1].rank_after = (
                self._seen if self._seen is not None else fallback)
        self._seen = None
        if self.model is not None:
            self.model.settle()

    def observe(self, now: float, level: float, engine=None) -> None:
        dt = 0.0 if self._t is None else max(0.0, min(0.5, now - self._t))
        self._t = now
        if not self.steps or self.steps[-1].rank_after is not None:
            return
        rank = rank_of(level)
        if engine is None or not engine.active:
            self._close(rank)             # motion stopped: settle the step
            return
        engaged = (not engine.tapering
                   and engine.env * engine.amp_scale > self.ENGAGED_ENV)
        if engaged:
            self._seen = rank
        if self.model is not None:
            self.model.observe(level, engaged, dt)

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
            "model": self.model.snapshot() if self.model else {"on": False},
            # a planning brain can show its work; the others have none to show
            "plan": (self.brain.plan_snapshot()
                     if hasattr(self.brain, "plan_snapshot") else None),
        }

    def pick(self, now: float, ema: float) -> Optional[str]:
        rank = rank_of(ema)
        self.level = ema
        self._close(rank)
        reply = self.brain(render_prompt(self.scenarios, self.steps, rank),
                           self.steps, rank)
        motion = parse_reply(reply)
        if motion in CANDIDATES:
            self.steps.append(Step(motion, rank))
            self._seen = None
            if self.model is not None:
                self.model.start(motion, ema)
            return motion
        return None                   # STOP/garbage: the machine decides

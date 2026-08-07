#!/usr/bin/env python3
"""Generate the IDEA.md scenario corpus: taught soothing sessions as JSONL.

docs/IDEA.md says "여러가지의 시나리오를 우리가 만들고 LLM 에게 학습 시킴" --
this is the making.  For each scenario a virtual infant gets a random hidden
Personality (perception/baby.py), and the ReflexBrain (the taught strategy,
core/policy.py) soothes it through the *real* CradleMachine closed loop.
Each finished session is one JSONL line:

    {"personality": {...}, "steps": [{"motion", "before", "after"}, ...],
     "outcome": "calmed" | "unsettled"}

The steps are what the LLM is taught from (few-shot in ClaudeBrain's
prompt); the personality is kept only so a human can audit that the
recorded behaviour makes sense -- the LLM never sees it.

Usage::

    python3 tools/make_scenarios.py                 # 12 -> data/scenarios.jsonl
    python3 tools/make_scenarios.py --n 4 --out /tmp/s.jsonl --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

if __package__ in (None, ""):   # direct run: put the repo root on sys.path
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cradle import CradleMachine, MotionEngine
from core.policy import ReflexBrain, SoothePolicy
from perception.baby import Personality, VirtualBaby

SIM_S = 1500.0        # per scenario: enough for several trials
PACE_S = 10.0         # the demo rig plays each motion for ~10 s
DT = 1.0 / 15.0


def run_scenario(seed: int) -> dict:
    """One closed-loop session -> the JSONL record."""
    rng = random.Random(seed)
    personality = Personality.random(rng)
    baby = VirtualBaby(seed=seed, personality=personality)
    engine = MotionEngine()
    machine = CradleMachine(engine, check_every_s=PACE_S)
    policy = SoothePolicy(ReflexBrain())
    machine.advisor = policy.pick

    t = 0.0
    while t < SIM_S:
        t += DT
        reading = baby.update(
            t, soothing=engine.env * engine.amp_scale,
            motion=engine.mode.id if engine.mode else None)
        policy.observe(t, reading.distress, engine)
        machine.tick(t, reading.present, reading.distress, jam=False)
        machine.events.clear()
        engine.tick(t)
    policy.observe(t + DT, baby.level, engine=None)    # settle the last step

    steps = [{"motion": s.motion, "before": s.rank_before,
              "after": s.rank_after}
             for s in policy.steps if s.rank_after is not None]
    calmed = any(s["after"] == 0 for s in steps)
    return {"personality": {"love": personality.love,
                            "hate": sorted(personality.hate),
                            "combo": list(personality.combo)},
            "steps": steps,
            "outcome": "calmed" if calmed else "unsettled"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=12,
                        help="how many scenarios to generate")
    parser.add_argument("--out", type=Path, default=Path("data/scenarios.jsonl"))
    parser.add_argument("--seed", type=int, default=0,
                        help="base seed; scenario k uses seed+k")
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for k in range(args.n):
        record = run_scenario(args.seed + k)
        lines.append(json.dumps(record))
        print(f"  scenario {k}: {len(record['steps'])} steps, "
              f"{record['outcome']}  (loves {record['personality']['love']})")
    args.out.write_text("\n".join(lines) + "\n")
    print(f"{args.n} scenarios -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

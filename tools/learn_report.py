#!/usr/bin/env python3
"""Prove the IDEA.md learning works: N simulated nights -> one HTML report.

The claim on the slide is "5일밤 뒤면 바로 잘 진정시킴" -- after a few nights
the cradle knows *this* baby.  This tool runs that experiment end to end:
one infant with a fixed hidden temperament (perception/baby.py), the same
policy carried across every night (its step history IS the personalisation),
and, for the control arm, the report's fixed trial ladder on identical
nights.  Everything goes through the real CradleMachine -- same gates, same
30 s checkpoints, same abort rules.

Output: ``data/learning.html`` -- self-contained (inline SVG, no libraries,
light/dark via prefers-color-scheme, styled like the live dashboard) -- plus
the same numbers on stdout.  Open the file in any browser or beamer.

Usage::

    python3 tools/learn_report.py                   # 5 nights -> data/learning.html
    python3 tools/learn_report.py --nights 7 --seed 4 --out /tmp/r.html
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

if __package__ in (None, ""):   # direct run: put the repo root on sys.path
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cradle import CradleMachine, MotionEngine
from core.policy import ReflexBrain, SoothePolicy
from perception.baby import Personality, VirtualBaby

NIGHT_S = 1800.0     # simulated seconds per night
DT = 0.1


def run_night(seed: int, personality: Personality,
              policy: SoothePolicy | None) -> dict:
    """One night through the real closed loop -> distress totals."""
    baby = VirtualBaby(seed=seed, personality=personality)
    engine = MotionEngine()
    machine = CradleMachine(engine)
    if policy is not None:
        machine.advisor = policy.pick
    t, upset_s = 0.0, 0.0
    while t < NIGHT_S:
        t += DT
        r = baby.update(t, soothing=engine.env * engine.amp_scale,
                        motion=engine.mode.id if engine.mode else None)
        if policy is not None:
            policy.observe(t, r.distress, engine)
        machine.tick(t, r.present, r.distress, jam=False)
        machine.events.clear()
        engine.tick(t)
        if baby.state in ("FUSS", "CRY"):
            upset_s += DT
    if policy is not None:
        policy.observe(t + DT, baby.level, engine=None)    # settle last step
    return {"upset_s": upset_s}


# --------------------------------------------------------------------------- #
# The page: dashboard-styled, inline SVG, zero dependencies
# --------------------------------------------------------------------------- #
CSS = """
:root{--page:#fff;--ink:#111;--ink2:#3d3d3d;--muted:#6f6f6f;
      --baseline:#b5b5b5;--hairline:#e2e2e2;--raised:#f2f2f2}
@media(prefers-color-scheme:dark){
 :root{--page:#000;--ink:#fff;--ink2:#c9c9c9;--muted:#8a8a8a;
       --baseline:#454545;--hairline:#1f1f1f;--raised:#161616}}
*{box-sizing:border-box;margin:0}
body{background:var(--page);color:var(--ink);max-width:860px;margin:0 auto;
     padding:36px 24px 60px;
     font:14px/1.55 "Helvetica Neue","Liberation Sans","Segoe UI",Arial,sans-serif}
h1{font-size:22px;letter-spacing:-.01em}
h1 small{font-weight:400;color:var(--muted);font-size:14px}
h2{font-size:11px;color:var(--muted);text-transform:uppercase;
   letter-spacing:.1em;border-top:1px solid var(--baseline);
   padding-top:8px;margin:30px 0 10px}
.tiles{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:18px}
.tile{border-top:1px solid var(--baseline);padding-top:8px}
.tile .k{font-size:10px;color:var(--muted);text-transform:uppercase;
         letter-spacing:.1em}
.tile .v{font-size:26px;font-weight:600;font-variant-numeric:tabular-nums}
.tile .v small{font-size:11px;font-weight:400;color:var(--muted)}
.legend{display:flex;gap:18px;font-size:11px;color:var(--muted);margin:6px 0 2px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;
          margin-right:6px;vertical-align:-1px}
svg{width:100%;height:auto;display:block}
svg text{font:10px ui-monospace,Menlo,Consolas,monospace;fill:var(--muted)}
svg text.val{fill:var(--ink2)}
svg .axis{stroke:var(--baseline)}
svg .ladder{fill:var(--muted)}
svg .policy{fill:var(--ink)}
table{border-collapse:collapse;font-size:12px;width:100%;
      font-variant-numeric:tabular-nums}
td,th{text-align:left;padding:4px 12px 4px 0;
      border-bottom:1px dotted var(--hairline)}
th{font-size:10px;color:var(--muted);text-transform:uppercase;
   letter-spacing:.08em;font-weight:500}
td b{font-family:ui-monospace,Menlo,Consolas,monospace;font-weight:500}
.note{font-size:11px;color:var(--muted);margin-top:8px;max-width:64ch}
.trail{font:11px/1.8 ui-monospace,Menlo,Consolas,monospace;color:var(--ink2)}
"""


def bars_svg(nights: list[dict]) -> str:
    """Grouped bars: crying minutes per night, ladder vs learning policy.

    One y scale, thin marks with 2 px rounded tops anchored to the baseline,
    a direct value label on every bar (few marks), identity by legend + fixed
    order (ladder always left), never colour alone.
    """
    n = len(nights)
    width, height, pad_l, pad_b, pad_t = 840, 240, 8, 26, 18
    plot_h = height - pad_b - pad_t
    peak = max(max(x["ladder"], x["policy"]) for x in nights) or 1.0
    group_w = (width - 2 * pad_l) / n
    bar_w = min(46.0, group_w * 0.28)
    gap = 2.0                                     # the 2 px surface spacer
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="minutes upset per night, fixed ladder versus '
             f'learning policy">']
    for i, night in enumerate(nights):
        cx = pad_l + group_w * (i + 0.5)
        for k, key in enumerate(("ladder", "policy")):
            v = night[key]
            h = 0.0 if peak <= 0 else (v / peak) * plot_h
            x = cx - bar_w - gap / 2 + k * (bar_w + gap)
            y = height - pad_b - h
            parts.append(
                f'<path class="{key}" d="M{x:.1f} {height - pad_b:.1f} '
                f'v{-max(0.0, h - 2):.1f} q0 -2 2 -2 h{bar_w - 4:.1f} '
                f'q2 0 2 2 v{max(0.0, h - 2):.1f} z"/>' if h >= 2 else
                f'<rect class="{key}" x="{x:.1f}" y="{y:.1f}" '
                f'width="{bar_w:.1f}" height="{h:.1f}"/>')
            parts.append(f'<text class="val" x="{x + bar_w / 2:.1f}" '
                         f'y="{y - 4:.1f}" text-anchor="middle">'
                         f'{v:.1f}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{height - 8}" '
                     f'text-anchor="middle">night {i + 1}</text>')
    parts.append(f'<line class="axis" x1="{pad_l}" y1="{height - pad_b}" '
                 f'x2="{width - pad_l}" y2="{height - pad_b}"/>')
    parts.append("</svg>")
    return "".join(parts)


def build_html(personality: Personality, nights: list[dict],
               policy: SoothePolicy, args) -> str:
    total_l = sum(x["ladder"] for x in nights)
    total_p = sum(x["policy"] for x in nights)
    saved = 0.0 if total_l <= 0 else (1.0 - total_p / total_l) * 100.0
    trail_rows = "".join(
        f'<tr><td>night {i + 1}</td><td class="trail">{x["picks"] or "—"}</td>'
        f'<td>{x["ladder"]:.1f}</td><td>{x["policy"]:.1f}</td></tr>'
        for i, x in enumerate(nights))
    score_rows = "".join(
        f'<tr><td><b>{m}</b></td><td>{"+" if v > 0 else ""}{v:.2f}</td>'
        f'<td>{"helps" if v > 0 else "makes it worse" if v < 0 else "no effect"}'
        f'</td></tr>'
        for m, v in sorted(policy.scores().items(), key=lambda kv: -kv[1]))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SIGMA · does the learning work?</title>
<style>{CSS}</style></head><body>
<h1>SIGMA — the cradle learns this baby
  <small>· {args.nights} simulated nights · docs/IDEA.md pipeline</small></h1>
<p class="note">One virtual infant with a fixed hidden temperament
({personality.describe()}). The learning policy carries its memory across
nights; the control arm re-runs the identical nights with the report's fixed
trial ladder. Same safety machine, same checkpoints, same abort rules.
The policy is never shown the temperament — only the happiness ranks.</p>

<div class="tiles">
 <div class="tile"><div class="k">upset, night 1 → night {args.nights}
  (policy)</div><div class="v">{nights[0]["policy"]:.0f} →
  {nights[-1]["policy"]:.0f} <small>min</small></div></div>
 <div class="tile"><div class="k">total upset time, all nights</div>
  <div class="v">{total_p:.0f} <small>vs {total_l:.0f} min on the
  ladder</small></div></div>
 <div class="tile"><div class="k">distress avoided</div>
  <div class="v">{saved:.0f}<small> %</small></div></div>
</div>

<h2>Minutes upset (fussing + crying) per night</h2>
<div class="legend"><span><i style="background:var(--muted)"></i>fixed
ladder</span><span><i style="background:var(--ink)"></i>learning
policy</span></div>
{bars_svg(nights)}

<h2>What the policy tried, night by night</h2>
<table><tr><th>night</th><th>advised motions (rank before→after)</th>
<th>ladder min upset</th><th>policy min upset</th></tr>{trail_rows}</table>

<h2>What it believes about this baby now</h2>
<table><tr><th>motion</th><th>mean rank change</th><th>reading</th></tr>
{score_rows}</table>
<p class="note">Ground truth it was never told: {personality.describe()}.
Report generated by tools/learn_report.py --seed {args.seed}
--nights {args.nights} --night-s {args.night_s:.0f}.</p>
</body></html>
"""


def main(argv=None) -> int:
    global NIGHT_S
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nights", type=int, default=5)
    parser.add_argument("--night-s", type=float, default=NIGHT_S,
                        help="simulated seconds per night")
    parser.add_argument("--seed", type=int, default=2,
                        help="fixes the nights (and, with --temperament "
                             "random, the temperament)")
    parser.add_argument("--temperament", choices=("tricky", "random"),
                        default="tricky",
                        help="'tricky' = the IDEA.md demo case: the baby "
                             "hates exactly the ladder's first two rungs and "
                             "loves its third; 'random' draws one from --seed "
                             "(an easy baby may show no gap -- honestly so)")
    parser.add_argument("--out", type=Path, default=Path("data/learning.html"))
    args = parser.parse_args(argv)

    NIGHT_S = args.night_s
    # The demo temperament collides head-on with the fixed ladder (M10, M12,
    # M13, M16): rungs one and two agitate this baby, rung three is its
    # favourite.  The policy is never told any of this.
    personality = (Personality(love="M13", hate=frozenset({"M10", "M12"}),
                               combo=("M09", "M13"))
                   if args.temperament == "tricky"
                   else Personality.random(random.Random(args.seed)))
    policy = SoothePolicy(ReflexBrain())   # one memory across every night

    nights = []
    print(f"temperament (hidden from the policy): {personality.describe()}")
    for k in range(args.nights):
        seed = args.seed * 1000 + k
        ladder = run_night(seed, personality, policy=None)
        before = len(policy.steps)
        learned = run_night(seed, personality, policy=policy)
        picks = "  ".join(
            f"{s.motion} {s.rank_before}→{'?' if s.rank_after is None else s.rank_after}"
            for s in policy.steps[before:])
        nights.append({"ladder": ladder["upset_s"] / 60.0,
                       "policy": learned["upset_s"] / 60.0,
                       "picks": picks})
        print(f"  night {k + 1}: ladder {nights[-1]['ladder']:5.1f} min upset"
              f" | policy {nights[-1]['policy']:5.1f} min | {picks or '—'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_html(personality, nights, policy, args))
    total_l = sum(x["ladder"] for x in nights)
    total_p = sum(x["policy"] for x in nights)
    print(f"totals: ladder {total_l:.1f} min, policy {total_p:.1f} min "
          f"({(1 - total_p / max(total_l, 1e-9)) * 100:.0f}% less upset time)")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

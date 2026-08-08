#!/usr/bin/env python3
"""The infant's state through the night, under all four decision algorithms.

Three panels -- a state trace per algorithm, mean upset over N nights, the
numbers -- as self-contained HTML with inline SVG (the demo LAN has no CDN).
Every arm runs the real CradleMachine, so only the choice of motion differs.

    python3 tools/state_figure.py                    # data/states.html
    python3 tools/state_figure.py --nights 20 --seed 4
    python3 tools/state_figure.py --memoryless       # the pre-cumulative baby
"""

from __future__ import annotations

import argparse
import os
import random
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cradle import CradleMachine, MotionEngine
from core.policy import DreamBrain, ReflexBrain, SoothePolicy
from perception.baby import Personality, VirtualBaby

NIGHT_S = 1800.0     # simulated seconds per night
PACE_S = 10.0        # the demo rig plays each motion for ~10 s
DT = 0.1
SAMPLE_S = 2.0       # one state sample every 2 s for the trace

STATES = ("SLEEP", "CALM", "FUSS", "CRY")     # the ordinal axis, bottom to top
WORDS = {"SLEEP": "asleep", "CALM": "calm", "FUSS": "fussing", "CRY": "crying"}

ARMS = (
    ("ladder", "Report ladder",
     "the evidence report's fixed escalation — no brain"),
    ("reflex", "Local algorithm",
     "the taught heuristic: explore while fussing, exploit while crying"),
    ("planner", "DREAM-Chunk planner",
     "dreams all 26 motions forward through a learned comfort model"),
)


# --------------------------------------------------------------------------- #
# The simulation
# --------------------------------------------------------------------------- #
def run_night(seed: int, personality: Personality, arm: str,
              cumulative: bool = True, trace: bool = False,
              brain=None) -> dict:
    """One night through the real closed loop.  Same seed = the same infant.

    ``brain`` overrides how the arm's brain is built (tools/ablation.py hands
    in de-tuned ones); the loop stays here so no measurement runs a copy."""
    baby = VirtualBaby(seed=seed, personality=personality,
                       cumulative=cumulative)
    engine = MotionEngine()
    machine = CradleMachine(engine, check_every_s=PACE_S, give_up=False)
    policy = None
    if arm != "ladder":
        make = brain or (DreamBrain if arm == "planner" else ReflexBrain)
        policy = SoothePolicy(make())
        machine.advisor = policy.pick

    t, samples, held = 0.0, [], {s: 0.0 for s in STATES}
    next_sample = 0.0
    while t < NIGHT_S:
        t += DT
        r = baby.update(t, soothing=engine.env * engine.amp_scale,
                        motion=engine.mode.id if engine.mode else None)
        if policy is not None:
            policy.observe(t, r.distress, engine)
        machine.tick(t, r.present, r.distress, jam=False)
        machine.events.clear()
        engine.tick(t)
        held[baby.state] = held.get(baby.state, 0.0) + DT
        if trace and t >= next_sample:
            samples.append(STATES.index(baby.state))
            next_sample += SAMPLE_S
    return {"held_min": {s: v / 60.0 for s, v in held.items()},
            "upset_min": (held["FUSS"] + held["CRY"]) / 60.0,
            "cry_min": held["CRY"] / 60.0,
            "trace": samples}


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #
CSS = """
:root{color-scheme:light dark;
      --page:#fff;--ink:#100F0F;--ink2:#3d3d3d;--muted:#6f6f6f;
      --hair:#e2e2e2;--band:#f4f2ee;--trace:#100F0F;
      --upset:#E8705F;--cry:#AF3029;--tip:#fff;--tipline:#d7d5cf}
@media(prefers-color-scheme:dark){
 :root{--page:#000;--ink:#FFFCF0;--ink2:#CECDC3;--muted:#878580;
       --hair:#1f1f1f;--band:#141312;--trace:#FFFCF0;
       --upset:#E8705F;--cry:#AF3029;--tip:#1C1B1A;--tipline:#343331}}
*{box-sizing:border-box;margin:0}
body{background:var(--page);color:var(--ink);max-width:980px;margin:0 auto;
     padding:36px 24px 60px;
     font:14px/1.55 "IBM Plex Sans","Helvetica Neue","Liberation Sans",Arial,sans-serif}
h1{font-size:22px;letter-spacing:-.01em}
h1 small{font-weight:400;color:var(--muted);font-size:14px;display:block;
         margin-top:5px;letter-spacing:0}
h2{font-size:11px;color:var(--muted);text-transform:uppercase;
   letter-spacing:.1em;border-top:1px solid var(--hair);
   padding-top:8px;margin:34px 0 6px}
p.note{font-size:12px;color:var(--muted);max-width:70ch;margin:6px 0 14px}
svg{width:100%;height:auto;display:block;overflow:visible}
svg text{font:10px "IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;
         fill:var(--muted)}
svg text.lane{font-family:"IBM Plex Sans",sans-serif;font-size:12px;
              font-weight:600;fill:var(--ink)}
svg text.sub{font-family:"IBM Plex Sans",sans-serif;font-size:11px;
             fill:var(--muted)}
svg text.val{font-size:11px;fill:var(--ink2)}
.legend{display:flex;gap:16px;font-size:11px;color:var(--muted);margin:10px 0 2px;
        flex-wrap:wrap}
.legend i{display:inline-block;width:11px;height:11px;border-radius:2px;
          margin-right:6px;vertical-align:-1px}
table{border-collapse:collapse;font-size:12px;width:100%;margin-top:10px;
      font-variant-numeric:tabular-nums}
td,th{text-align:right;padding:5px 10px 5px 0;border-bottom:1px dotted var(--hair)}
th:first-child,td:first-child{text-align:left}
th{font-size:10px;color:var(--muted);text-transform:uppercase;
   letter-spacing:.08em;font-weight:500}
td b{font-family:"IBM Plex Mono",ui-monospace,monospace;font-weight:500}
#tip{position:fixed;z-index:9;pointer-events:none;display:none;
     background:var(--tip);border:1px solid var(--tipline);border-radius:7px;
     padding:6px 9px;font-size:11px;line-height:1.5;color:var(--ink);
     box-shadow:0 2px 10px rgba(0,0,0,.18);white-space:nowrap}
footer{margin-top:34px;font-size:11px;color:var(--muted)}
"""

TIP_JS = """
(function(){
 var tip=document.getElementById('tip');
 function show(e){var t=e.target.getAttribute('data-tip');if(!t)return;
   tip.innerHTML=t;tip.style.display='block';move(e);}
 function move(e){var p=12;var w=tip.offsetWidth,h=tip.offsetHeight;
   var x=e.clientX+p,y=e.clientY+p;
   if(x+w>innerWidth-8)x=e.clientX-w-p;
   if(y+h>innerHeight-8)y=e.clientY-h-p;
   tip.style.left=x+'px';tip.style.top=y+'px';}
 function hide(){tip.style.display='none';}
 document.addEventListener('mouseover',show);
 document.addEventListener('mousemove',function(e){
   if(tip.style.display==='block')move(e);});
 document.addEventListener('mouseout',hide);
})();
"""


def esc(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def lanes_svg(runs: dict, night_s: float) -> str:
    """Four small multiples: state (ordinal y) against time (shared x)."""
    W, LANE_H, GAP = 940, 96, 52
    LEFT, RIGHT, TOP = 74, 16, 26
    SUBX = LEFT + 190
    H = TOP + (len(ARMS) - 1) * (LANE_H + GAP) + LANE_H + 26
    plot_w = W - LEFT - RIGHT
    row_h = LANE_H / len(STATES)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" '
           f'aria-label="infant state over one simulated night under four '
           f'algorithms">']

    for lane, (key, title, sub) in enumerate(ARMS):
        y0 = TOP + lane * (LANE_H + GAP)
        run = runs[key]
        upset_top = y0
        upset_h = row_h * 2
        out.append(f'<rect x="{LEFT}" y="{upset_top:.1f}" width="{plot_w}" '
                   f'height="{upset_h:.1f}" fill="var(--band)"/>')
        for i, state in enumerate(STATES):
            by = y0 + (len(STATES) - 1 - i) * row_h
            out.append(f'<text x="{LEFT - 8}" y="{by + row_h / 2 + 3:.1f}" '
                       f'text-anchor="end">{WORDS[state]}</text>')
        out.append(f'<line x1="{LEFT}" y1="{y0 + upset_h:.1f}" '
                   f'x2="{W - RIGHT}" y2="{y0 + upset_h:.1f}" '
                   f'stroke="var(--hair)"/>')
        out.append(f'<text class="lane" x="{LEFT}" y="{y0 - 12:.1f}">'
                   f'{esc(title)}</text>')
        out.append(f'<text class="sub" x="{SUBX}" y="{y0 - 12:.1f}">'
                   f'{esc(sub)}</text>')

        trace = run["trace"]
        step = plot_w / max(1, len(trace) - 1)
        d = []
        for i, s in enumerate(trace):
            x = LEFT + i * step
            y = y0 + (len(STATES) - 1 - s) * row_h + row_h / 2
            if i == 0:
                d.append(f"M{x:.1f} {y:.1f}")
            else:
                d.append(f"H{x:.1f}")
                d.append(f"V{y:.1f}")
        out.append(f'<path d="{" ".join(d)}" fill="none" stroke="var(--trace)" '
                   f'stroke-width="2" stroke-linejoin="round"/>')

        run_start = None
        for i, s in enumerate(trace + [0]):
            upset = s >= 2 if i < len(trace) else False
            if upset and run_start is None:
                run_start = i
            elif not upset and run_start is not None:
                x1 = LEFT + run_start * step
                x2 = LEFT + (i - 1) * step + step
                mins = (i - run_start) * SAMPLE_S / 60.0
                tip = (f"{esc(title)}<br>upset from "
                       f"{run_start * SAMPLE_S / 60:.0f} to "
                       f"{i * SAMPLE_S / 60:.0f} min &middot; "
                       f"<b>{mins:.1f} min</b>")
                out.append(f'<rect x="{x1:.1f}" y="{y0:.1f}" '
                           f'width="{max(1.0, x2 - x1):.1f}" height="{LANE_H}" '
                           f'fill="transparent" data-tip="{tip}"/>')
                run_start = None

        # per-lane headline: *this night*, since the bars below carry the average
        out.append(f'<text class="val" x="{W - RIGHT}" y="{y0 - 12:.1f}" '
                   f'text-anchor="end">this night: {run["upset_min"]:.1f} min '
                   f'upset &#183; {run["cry_min"]:.1f} crying</text>')
        out.append(f'<line x1="{LEFT}" y1="{y0 + LANE_H:.1f}" x2="{W - RIGHT}" '
                   f'y2="{y0 + LANE_H:.1f}" stroke="var(--hair)"/>')

    ylast = TOP + (len(ARMS) - 1) * (LANE_H + GAP) + LANE_H
    total_min = night_s / 60.0
    tick = 10 if total_min > 20 else 5
    minute = 0
    while minute <= total_min:
        x = LEFT + plot_w * (minute / total_min)
        out.append(f'<text x="{x:.1f}" y="{ylast + 17:.1f}" '
                   f'text-anchor="middle">{minute} min</text>')
        minute += tick
    out.append("</svg>")
    return "".join(out)


def bars_svg(means: dict, nights: int) -> str:
    """Upset minutes per night, stacked: the segments sum to the bar."""
    W, ROW, GAP, LEFT, RIGHT = 940, 26, 16, 168, 130
    H = len(ARMS) * (ROW + GAP)
    plot_w = W - LEFT - RIGHT
    top = max(m["upset_min"] for m in means.values()) * 1.06
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="mean minutes '
           f'per night spent fussing and crying, by algorithm">']
    base = means["ladder"]["upset_min"]
    for i, (key, title, _sub) in enumerate(ARMS):
        y = i * (ROW + GAP)
        m = means[key]
        fuss_min = m["upset_min"] - m["cry_min"]
        w_fuss = plot_w * fuss_min / top
        w_cry = plot_w * m["cry_min"] / top
        out.append(f'<text class="sub" x="{LEFT - 10}" y="{y + ROW / 2 + 4:.0f}" '
                   f'text-anchor="end">{esc(title)}</text>')
        tip_f = (f"{esc(title)}<br><b>{fuss_min:.1f} min</b> fussing"
                 f"<br>mean of {nights} nights")
        out.append(f'<rect x="{LEFT}" y="{y}" width="{w_fuss:.1f}" '
                   f'height="{ROW}" rx="4" fill="var(--upset)" '
                   f'data-tip="{tip_f}"/>')
        # +2: the gap between stacked segments
        tip_c = (f"{esc(title)}<br><b>{m['cry_min']:.1f} min</b> crying"
                 f"<br>mean of {nights} nights")
        out.append(f'<rect x="{LEFT + w_fuss + 2:.1f}" y="{y}" '
                   f'width="{max(0.0, w_cry - 2):.1f}" height="{ROW}" rx="4" '
                   f'fill="var(--cry)" data-tip="{tip_c}"/>')
        delta = 100.0 * (m["upset_min"] - base) / base if base else 0.0
        label = (f'{m["upset_min"]:.1f} min'
                 + ("" if key == "ladder" else f"  ({delta:+.0f}%)"))
        out.append(f'<text class="val" x="{LEFT + w_fuss + w_cry + 10:.1f}" '
                   f'y="{y + ROW / 2 + 4:.0f}">{label}</text>')
    out.append("</svg>")
    return "".join(out)


def table_html(means: dict, nights: int) -> str:
    rows = []
    for key, title, _sub in ARMS:
        m = means[key]
        held = m["held_min"]
        rows.append(
            f"<tr><td>{esc(title)}</td>"
            + "".join(f"<td><b>{held.get(s, 0.0):.1f}</b></td>" for s in STATES)
            + f"<td><b>{m['upset_min']:.1f}</b></td></tr>")
    head = "".join(f"<th>{WORDS[s]}</th>" for s in STATES)
    return (f"<table><thead><tr><th>algorithm</th>{head}"
            f"<th>upset total</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
            f"<p class='note'>Minutes per 30-minute night, mean of {nights} "
            f"nights. &ldquo;Upset&rdquo; is fussing plus crying.</p>")


def build_html(runs: dict, means: dict, nights: int, seed: int,
               personality: Personality, cumulative: bool) -> str:
    model = ("cumulative settling" if cumulative
             else "memoryless (pre-2026-08 model)")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SIGMA · the infant's night, by algorithm</title>
<style>{CSS}</style></head><body>
<h1>The infant's night, by algorithm
<small>One simulated infant &mdash; same seed, same hidden temperament &mdash;
lived through once per decision algorithm. Everything runs through
the real CradleMachine, so the only difference is who chooses the motion.</small>
</h1>

<h2>One night, state by state</h2>
<p class="note">State is read by <b>height</b>: asleep at the bottom, crying at
the top. The bands behind the trace are reference, not the encoding &mdash; the
same ladder the dashboard draws. Hover an upset stretch for its length.</p>
{lanes_svg(runs, NIGHT_S)}

<h2>How much of the night was spent upset</h2>
<p class="note">One night is an anecdote, so these bars are the mean of
{nights} nights &mdash; each one a different infant, every algorithm facing the
same set. The two segments sum to the upset time; the percentage is against
the report ladder.</p>
<div class="legend">
  <span><i style="background:var(--upset)"></i>fussing</span>
  <span><i style="background:var(--cry)"></i>crying</span>
</div>
{bars_svg(means, nights)}

<h2>The numbers</h2>
{table_html(means, nights)}

<footer>
Generated by <b>tools/state_figure.py</b> &middot; seed {seed} &middot;
{nights} nights &times; {NIGHT_S / 60:.0f} simulated minutes &middot;
pace {PACE_S:.0f}s &middot; infant model: {model}<br>
Temperament of the traced night: {esc(personality.describe())}<br>
The algorithms and the measurement behind them: <b>docs/DREAM-CHUNK.md</b>.
</footer>
<div id="tip" role="status"></div>
<script>{TIP_JS}</script>
</body></html>
"""


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nights", type=int, default=12,
                        help="nights averaged for the bars (default 12)")
    parser.add_argument("--seed", type=int, default=3,
                        help="seed for the traced night and the night set")
    parser.add_argument("--memoryless", action="store_true",
                        help="use the pre-cumulative infant model, the one the "
                             "planner ties the heuristic on")
    parser.add_argument("--out", default="data/states.html")
    args = parser.parse_args(argv)
    cumulative = not args.memoryless

    keys = [a[0] for a in ARMS]
    # the traced night: one infant, lived once per algorithm
    person = Personality.random(random.Random(args.seed))
    runs = {k: run_night(args.seed, person, k, cumulative, trace=True)
            for k in keys}

    # the bars: the mean over a fresh set of infants
    totals = {k: {"upset_min": 0.0, "cry_min": 0.0,
                  "held_min": {s: 0.0 for s in STATES}} for k in keys}
    for i in range(args.nights):
        seed = args.seed * 1000 + i
        p = Personality.random(random.Random(seed))
        for k in keys:
            r = run_night(seed, p, k, cumulative)
            totals[k]["upset_min"] += r["upset_min"] / args.nights
            totals[k]["cry_min"] += r["cry_min"] / args.nights
            for s in STATES:
                totals[k]["held_min"][s] += r["held_min"].get(s, 0.0) / args.nights
        print(f"  night {i + 1}/{args.nights}", end="\r", flush=True)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write(build_html(runs, totals, args.nights, args.seed, person,
                            cumulative))
    print(f"wrote {out}")
    for k, title, _ in ARMS:
        m = totals[k]
        print(f"  {title:<22} {m['upset_min']:5.1f} min upset  "
              f"{m['cry_min']:5.1f} crying")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

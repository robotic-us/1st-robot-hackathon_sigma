#!/usr/bin/env python3
"""The mascot recognizer's answers, drawn on the wheel it answers in.

Per figure on docs/Nubzuki.jpg: a hollow ring where the sheet *prints* it (its
own circumplex is the ground truth), a filled dot at the pose the classifier
*named*, the line between them the error; colour is the five-state vocabulary
the decision layer sees.  Self-contained HTML, inline SVG (demo LAN has no CDN).

    python3 tools/nubzuki_wheel.py                  # data/nubzuki_wheel.html
    python3 tools/nubzuki_wheel.py --image shot.png # grade any frame instead
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from core.cradle import CALM_LEVEL, CRY_LEVEL
from perception import nubzuki as nz

W = 660
R = 250
CX = CY = W // 2

# SVG twins of nubzuki.STATE_COLOUR, which is BGR because cv2 draws it.
STATE_CSS = {
    nz.AWAKE: "#4f9d4a",
    nz.EYES_CLOSED: "#b08528",
    nz.SLEEP_CANDIDATE: "#5f7fc4",
    nz.DISTRESS_FACE: "#c4443a",
    nz.UNKNOWN: "#8a8a8a",
}


def at(v: float, a: float) -> tuple[float, float]:
    """Circumplex coordinates -> SVG. y is already screen-down on the sheet."""
    return CX + v * R, CY + a * R


def wheel_svg(seen: list) -> str:
    out = [f'<svg viewBox="0 0 {W} {W}" role="img" '
           f'aria-label="the 17 mascot poses on the ACTIVE/CALM by '
           f'NEGATIVE/POSITIVE circumplex, each linked to where the reference '
           f'sheet prints it">']
    out.append(f'<circle cx="{CX}" cy="{CY}" r="{R}" class="ring"/>')
    out.append(f'<line x1="{CX}" y1="{CY-R}" x2="{CX}" y2="{CY+R}" class="axis"/>')
    out.append(f'<line x1="{CX-R}" y1="{CY}" x2="{CX+R}" y2="{CY}" class="axis"/>')
    for text, x, y, anchor in (("ACTIVE", CX, CY - R - 12, "middle"),
                               ("CALM", CX, CY + R + 24, "middle"),
                               ("NEGATIVE", CX - R - 12, CY, "end"),
                               ("POSITIVE", CX + R + 12, CY, "start")):
        out.append(f'<text x="{x:.0f}" y="{y:.0f}" text-anchor="{anchor}" '
                   f'class="ax">{text}</text>')

    # The live ladder: the machine's own path across the wheel, in level order.
    ladder = sorted(nz.LIVE_LEVEL, key=lambda k: nz.LIVE_LEVEL[k])
    pts = " ".join(f"{at(*nz.POSES[k])[0]:.1f},{at(*nz.POSES[k])[1]:.1f}"
                   for k in ladder)
    out.append(f'<polyline points="{pts}" class="ladder"/>')

    for s in seen:
        sx, sy = at(*nz.sheet_position(s.box))
        ax, ay = at(s.valence, s.arousal)
        colour = STATE_CSS[s.state]
        out.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ax:.1f}" '
                   f'y2="{ay:.1f}" class="err"/>')
        out.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="4" '
                   f'class="printed"/>')
        out.append(f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="7" '
                   f'fill="{colour}" class="named"/>')
        dx = 12 if s.valence >= 0 else -12
        anchor = "start" if s.valence >= 0 else "end"
        out.append(f'<text x="{ax + dx:.1f}" y="{ay + 4:.1f}" '
                   f'text-anchor="{anchor}" class="lbl">{s.label}</text>')
    out.append("</svg>")
    return "\n".join(out)


def table_html(seen: list) -> str:
    rows = []
    for s in sorted(seen, key=lambda s: (s.state, -(s.recovered_level or -1))):
        sv, sa = nz.sheet_position(s.box)
        err = ((sv - s.valence) ** 2 + (sa - s.arousal) ** 2) ** .5
        rec = "&mdash;" if s.recovered_level is None else f"{s.recovered_level:.2f}"
        rows.append(
            f'<tr><td><span class="dot" style="background:{STATE_CSS[s.state]}">'
            f'</span>{s.label}</td><td class="st">{s.state}</td>'
            f'<td class="n">{rec}</td><td class="n">{s.asserted_level:.2f}</td>'
            f'<td class="n">{err:.2f}</td><td class="why">{s.why}</td></tr>')
    return ("<table><thead><tr><th>pose</th><th>five-state</th>"
            "<th>recovered</th><th>may assert</th><th>error</th>"
            "<th>decided by</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def build_html(seen: list, source: str) -> str:
    named = len({s.pose for s in seen})
    worst = max((((nz.sheet_position(s.box)[0] - s.valence) ** 2
                  + (nz.sheet_position(s.box)[1] - s.arousal) ** 2) ** .5)
                for s in seen) if seen else 0.0
    ceiling = max((s.asserted_level for s in seen), default=0.0)
    legend = "".join(
        f'<span class="key"><span class="dot" style="background:{c}"></span>'
        f'{state}</span>' for state, c in STATE_CSS.items() if state != nz.UNKNOWN)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SIGMA &middot; what the mascot recognizer sees</title>
<style>
:root {{ --bg:#fbfaf8; --fg:#23201d; --dim:#6d675f; --line:#d9d3ca;
         --grid:#e6e1d8; --card:#fff; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#17161a; --fg:#e9e6e1; --dim:#9a938a; --line:#3a3630;
           --grid:#2a2724; --card:#1e1c20; }}
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:28px 20px 60px; background:var(--bg); color:var(--fg);
        font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif; }}
main {{ max-width:900px; margin:0 auto; }}
h1 {{ font-size:22px; margin:0 0 4px; letter-spacing:-.01em; }}
p.sub {{ margin:0 0 22px; color:var(--dim); }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
         padding:18px; margin-bottom:20px; }}
svg {{ width:100%; height:auto; display:block; }}
.ring {{ fill:none; stroke:var(--grid); stroke-width:1.5; }}
.axis {{ stroke:var(--grid); stroke-width:1; stroke-dasharray:5 5; }}
.ax {{ fill:var(--dim); font-size:12px; letter-spacing:.12em; }}
.lbl {{ fill:var(--fg); font-size:11.5px; }}
.err {{ stroke:var(--dim); stroke-width:1.4; opacity:.55; }}
.printed {{ fill:none; stroke:var(--dim); stroke-width:1.4; }}
.named {{ stroke:var(--card); stroke-width:1.5; }}
.ladder {{ fill:none; stroke:var(--dim); stroke-width:1.2; stroke-dasharray:2 6;
           opacity:.5; }}
.keys {{ display:flex; flex-wrap:wrap; gap:14px; margin-top:10px;
         color:var(--dim); font-size:13px; }}
.key {{ display:inline-flex; align-items:center; gap:6px; }}
.dot {{ width:10px; height:10px; border-radius:50%; display:inline-block;
        margin-right:7px; vertical-align:-1px; }}
.wrap {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; min-width:620px; }}
th, td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); }}
th {{ color:var(--dim); font-weight:600; font-size:12px; letter-spacing:.04em;
      text-transform:uppercase; }}
td.n {{ text-align:right; font-variant-numeric:tabular-nums; }}
td.st, td.why {{ color:var(--dim); }}
td.st {{ font-size:12.5px; }}
.stat {{ display:flex; flex-wrap:wrap; gap:26px; margin:0 0 18px; }}
.stat div {{ font-size:13px; color:var(--dim); }}
.stat b {{ display:block; font-size:21px; color:var(--fg);
           font-variant-numeric:tabular-nums; }}
</style></head><body><main>
<h1>What the mascot recognizer sees</h1>
<p class="sub">Every figure in <code>{source}</code>, read by
<code>perception/nubzuki.py</code> and plotted on the sheet&rsquo;s own
circumplex. The hollow ring is where the sheet <em>prints</em> the figure &mdash;
independent ground truth, because the sheet is an emotion chart. The filled dot
is the pose the classifier <em>named</em>. The line between them is the error.</p>

<div class="stat">
  <div><b>{named}/17</b>poses named</div>
  <div><b>{worst:.2f}</b>worst placement error</div>
  <div><b>{ceiling:.2f}</b>highest assertable level</div>
  <div><b>{CRY_LEVEL}</b>CRY_LEVEL &mdash; vision may not cross it</div>
</div>

<div class="card">{wheel_svg(seen)}
<div class="keys">{legend}
<span class="key"><span class="dot" style="border:1.4px solid var(--dim);
background:transparent"></span>where the sheet prints it</span></div></div>

<div class="card"><div class="wrap">{table_html(seen)}</div>
<p class="sub" style="margin:14px 0 0">
<b>recovered</b> is the distress level that would have drawn the pose &mdash; an
echo of what the machine sent, and the right number for checking the loop.
<b>may assert</b> is what this reading is allowed to claim as a
<em>visual</em> observation: capped at 0.42, below CRY_LEVEL {CRY_LEVEL}, because
&sect;5 says vision alone never escalates. Calm poses sit under CALM_LEVEL
{CALM_LEVEL}. The two columns differing is the point, not a bug.</p></div>
</main></body></html>"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", default="docs/Nubzuki.jpg")
    parser.add_argument("--out", default="data/nubzuki_wheel.html")
    args = parser.parse_args(argv)

    bgr = cv2.imread(args.image)
    if bgr is None:
        print(f"cannot read {args.image}", file=sys.stderr)
        return 2
    seen = nz.read(bgr)
    if not seen:
        print(f"no Nubzuki found in {args.image}", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(seen, args.image), encoding="utf-8")
    print(f"{out}  ({len(seen)} figures, "
          f"{len({s.pose for s in seen})} distinct poses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

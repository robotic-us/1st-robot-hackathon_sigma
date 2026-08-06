# webapp — the dashboard as a Next.js app

Same dashboard as `web/index.html`, same three zones, built as a React app.
It is a **client of `serve.py`**, not a replacement for it: every reading and
every command still comes from the Python server, which owns the camera, the
state machine and the engine.

```
serve.py :8080  ──SSE /events, MJPEG /frame, /jam /auto /motion /play──▶  this app :3000
```

## Running it

Needs Node 18.18+ (this repo has no Node toolchain by default — see below).

```bash
npm install                       # first time only
python3 serve.py --baby           # terminal 1: the cradle, on :8080
npm run dev                       # terminal 2: this app, on :3000
```

Then open <http://localhost:3000>.

If the cradle runs on another machine:

```bash
SIGMA_API=http://jetson.local:8080 npm run dev
```

## Getting Node without root

Ubuntu 22.04's `apt` only offers Node 12, which is far too old. `nvm` installs
into your home directory and needs no `sudo`:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
exec $SHELL -l
nvm install 20
```

## How it talks to serve.py

`next.config.mjs` rewrites `/events`, `/frame`, `/slots`, `/motions`, `/jam`,
`/auto`, `/motion` and `/play` through to `SIGMA_API`. The browser therefore
only ever talks to its own origin, and `serve.py` never has to hand out CORS
headers — which matters, because `/jam`, `/motion` and `/auto` command a
machine that moves.

## Layout

| Zone | Answers |
|---|---|
| hero | how is the baby, what is the cradle doing, and why |
| strip | by how much — each number against the limit that bounds it |
| foot | camera, manual override, recent events |

Engineering-grade numbers (θ, `dob_a`, ASAP residual, slot ranking) live in the
drawer at the bottom and nowhere else. The rule for the rest of the page: a
reader should never need to know what "amplitude envelope" or "0.018 g" means
to understand what the cradle is doing.

The visual treatment is the plain one, shared with `web/index.html`:
monochrome and square — no hue, no border radius. Severity is an ink ramp
(fainter = calmer, stronger = worse) and every marked element sits beside a
text label, so nothing is lost with the colour. Both grounds (paper white and
near-black) come via `prefers-color-scheme`.

The three zones survive but are arranged as a stage: the monitoring circle
holds the centre of the page, and the readouts, camera, keypad and log stack
in a 340 px right-hand panel. There are no headers — the connection dot is
the only trace of the old masthead; panel blocks are introduced by their
heavy top rules, not titles. The layout is pure CSS grid placement, so the
component markup is identical to before. Tokens and the rationale live at
the top of `app/globals.css`; components never restate a hex.

## Files

| Path | What it is |
|---|---|
| `lib/types.ts` | the shape of one SSE frame — a transcription of `build_state()` |
| `lib/words.ts` | plain English for every state, plus the report thresholds |
| `lib/useCradle.ts` | the SSE subscription; hands back a value *and* a ref |
| `components/Orb.tsx` | the living blob; reads the ref so 60 fps never re-renders |
| `components/Cards.tsx` | hero overlay, safety chip, the four-number strip |
| `components/Controls.tsx` | camera thumbnail and the manual buttons |
| `components/Internals.tsx` | the drawer |

`lib/words.ts` duplicates the thresholds in `core/cradle.py` rather than
deriving them. If the report's numbers move, they move in both places.

## What this does not replace

`web/` (`index.html` + `style.css` + `app.js`) is still the dashboard
`serve.py` serves on :8080, still has
no build step, and is still what `tests.py` checks. This app is additive.

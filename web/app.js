/* The dashboard's one script.  Served by serve.py at /app.js; no build
   step.  Everything below reads the SSE stream and paints the page. */
"use strict";
let S = null, slots = [], lastSeq = -1;
const $ = id => document.getElementById(id);
const clamp = (v, lo, hi) => v < lo ? lo : v > hi ? hi : v;

/* State → status role. Every colored dot sits beside its text label. */
const ROLE = {SLEEP:"--accent", CALM:"--good", HAPPY:"--good",
              NEUTRAL:"--muted", FUSS:"--warn", SAD:"--warn", SURPRISE:"--warn",
              CRY:"--serious", ANGRY:"--serious",
              /* the five-state watcher (perception/watch.py).  DISTRESS_FACE
                 is a visual pattern, not confirmed crying, so it wears warn,
                 not serious -- the words say "looks upset", not "crying". */
              AWAKE:"--good", EYES_CLOSED:"--accent",
              SLEEP_CANDIDATE:"--accent", DISTRESS_FACE:"--warn",
              UNKNOWN:"--muted",
              QUIET_AWAKE:"--good", STARTLE:"--warn", FUSS_WEAK:"--warn",
              CRY:"--serious", STRONG_DISTRESS:"--serious",
              PAIN_SUSPECT:"--critical", DROWSY:"--accent",
              SLEEP_TENTATIVE:"--accent", SLEEP_STABLE:"--accent",
              STATE_UNCLEAR:"--muted"};

/* The palette only changes when the colour scheme does, so read each token
   once and drop the cache on a theme flip -- the hero asks for colours sixty
   times a second and getComputedStyle is not free. */
let TOKENS = {};
const cssv = name => name in TOKENS ? TOKENS[name]
  : (TOKENS[name] = getComputedStyle(document.documentElement)
                      .getPropertyValue(name).trim());
matchMedia("(prefers-color-scheme: dark)")
  .addEventListener("change", () => { TOKENS = {}; });
const stateColor = st => cssv(ROLE[st] || "--muted");

/* One meter update: width on the fill, hue on the track it sits in.  Setting
   --fill on the track is what keeps the two halves from drifting apart -- the
   unfilled remainder is a dim step of the same colour, never a neutral gray,
   so a bar at 12% still reads as "serious" rather than "mostly empty". */
function setMeter(id, frac, color) {
  const fill = $(id);
  fill.style.width = (clamp(frac, 0, 1) * 100) + "%";
  fill.parentNode.style.setProperty("--fill", color);
}

/* #rrggbb -> rgba(), for the glow gradients. */
function rgba(hex, a) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.replace(/./g, "$&$&") : h, 16);
  return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`;
}

/* Report thresholds, mirrored from core/cradle.py so the words on this page
   and the machine's own decisions agree. */
const CALM_LEVEL = 0.12, CRY_LEVEL = 0.45;
const NO_IMPROVE_S = 60, CHECK_EVERY_S = 30, GATE_RECOVER_S = 2;
const SWAY_CAP_MM = 30, ACC_CAP_G = 0.05, LEVER_MM = 227;

/* ---- plain words -------------------------------------------------------- */
const BABY_WORD = {SLEEP:"Sleeping", CALM:"Calm", HAPPY:"Happy",
                   NEUTRAL:"Settled", FUSS:"Fussing", SAD:"Fussing",
                   SURPRISE:"Startled", CRY:"Crying", ANGRY:"Crying",
                   AWAKE:"Awake", EYES_CLOSED:"Eyes closed",
                   SLEEP_CANDIDATE:"Drifting off", DISTRESS_FACE:"Looks upset",
                   UNKNOWN:"Can't see the baby",
                   /* the report-spec judge (perception/watch.py, report §5) */
                   QUIET_AWAKE:"Quiet and awake", STARTLE:"Startled",
                   FUSS_WEAK:"Fussing", CRY:"Crying",
                   STRONG_DISTRESS:"Very upset", PAIN_SUSPECT:"Needs you now",
                   DROWSY:"Getting sleepy", SLEEP_TENTATIVE:"Falling asleep",
                   SLEEP_STABLE:"Sleeping", STATE_UNCLEAR:"Can't see the baby"};

function babyWord(tag) {
  if (!tag.present) return "Can't see the baby";
  if (tag.emotion) return BABY_WORD[tag.emotion] || tag.emotion;
  const lvl = tag.level ?? 0;          // state-card mode: no face, just a level
  return lvl >= CRY_LEVEL ? "Crying" : lvl >= CALM_LEVEL ? "Fussing" : "Calm";
}

/* A rate in hertz means nothing to most readers; a sway every N seconds does.
   The rig has three motions, and the words must not blur them: ML sways side
   to side, Z lifts and lowers, AP tilts like a see-saw (the rig has no second
   horizontal axis, so that is how those entries actually move). */
function cradleWords(c) {
  if (c.tapering) return ["slowing to a stop", "easing the motion down to nothing"];
  switch (c.kind) {
    case "static": return ["holding still", "not moving"];
    case "pause": return ["pausing to watch", "stopped for a moment to see what happens"];
    case "taper": return ["slowing to a stop", "easing the motion down to nothing"];
    case "soft_start": return ["starting gently", "easing the rocking up"];
    case "micro_resume": return ["easing back in", "the same motion at half strength"];
  }
  const pace = c.f_hz < 0.35 ? "slow" : c.f_hz < 0.55 ? "steady"
             : c.f_hz < 0.70 ? "brisk" : "quick";
  const every = c.f_hz > 0 ? (1 / c.f_hz).toFixed(1) : "–";
  const mm = c.a_mm.toFixed(0);
  if (c.axis === "Z")
    return [`${pace} bobbing`, `rises and settles every ${every} s · ${mm} mm`];
  if (c.axis === "AP")
    return [`${pace} see-saw tilt`, `tips head-to-toe every ${every} s · ${mm} mm`];
  if (c.axis === "APML")
    return [`${pace} sway and tilt`, `swaying and tipping together, every ${every} s`];
  const wide = c.a_mm >= 15 ? ", wide" : "";
  return [`${pace}${wide} rocking`,
          `one sway every ${every} s · ${mm} mm each way`];
}

/* What the state machine will do next, in the reader's terms. */
function nextWords(c) {
  if (c.state === "gate_fail")
    return "Stopped for safety. Rocking only restarts once the baby has been "
         + `visible again for ${GATE_RECOVER_S} seconds.`;
  if (!c.auto)
    return "Automatic care is off. The cradle does only what you press, though "
         + "the safety gate still overrides it.";
  if (c.state === "trial") {
    const left = Math.max(0, NO_IMPROVE_S - c.trial_s);
    return `Trying this for ${Math.round(c.trial_s)} s. It gets checked every `
         + `${CHECK_EVERY_S} s, and if it still hasn't helped in `
         + `${Math.round(left)} s the cradle stops and calls for you.`;
  }
  if (c.state === "settling")
    return "The baby has been calm for a while, so the cradle is winding down "
         + "towards sleep.";
  return "Watching. If the baby starts fussing, the cradle begins rocking on "
       + "its own.";
}

/* ---- wiring ------------------------------------------------------------- */
fetch("/slots").then(r => r.json()).then(list => {
  slots = list;
  $("slots").innerHTML = list.map(s =>
    `<div class="slot" id="slot${s.id}" onclick="fetch('/play?slot=${s.id}')">
       <b>${s.id} · ${s.name}</b>${s.target_deg}° · ${s.duration}s</div>`).join("");
});

/* The whole library, straight from core/cradle.py's catalog().  Clicking one
   goes through /motion, which validates grade and refuses R without --research
   -- so the browser never decides what is allowed, it only reports the reply. */
let motions = [], mGrade = "", mQuery = "", mResearch = null;

fetch("/motions").then(r => r.json()).then(list => {
  motions = list;
  drawMotions();
});

/* The catalog is ours, not user input, but it lands in an attribute -- one
   quote in a future desc would silently break the grid. */
const esc = s => String(s).replace(/[&<>"]/g,
  c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function drawMotions() {
  const q = mQuery.trim().toLowerCase();
  const show = motions.filter(m =>
    (!mGrade || m.grade === mGrade) &&
    (!q || m.id.toLowerCase().includes(q) || m.name.toLowerCase().includes(q)));
  const research = mResearch === null ? false : mResearch;
  $("motions").innerHTML = show.map(m => {
    /* The engine refuses R unless serve.py started with --research; show that
       up front rather than letting the click fail silently. */
    const locked = m.grade === "R" && !research;
    const hz = m.f_hz ? `${m.f_hz.toFixed(1)} Hz` : "—";
    const mm = m.a_mm ? `${m.a_mm.toFixed(0)} mm` : "";
    const tip = `${m.id} ${m.name} · ${m.kind}${m.axis ? " · " + m.axis : ""}
${m.desc}${locked ? "\n\nresearch-only (R) — needs serve.py --research" : ""}`;
    return `<div class="m g-${m.grade}${locked ? " locked" : ""}"
                 data-id="${esc(m.id)}" title="${esc(tip)}">
              <b>${esc(m.id)} · ${esc(m.name)}</b>${hz}${mm ? " · " + mm : ""}
              ${m.axis ? " · " + esc(m.axis) : ""} · ${m.grade}</div>`;
  }).join("");
  $("mcount").textContent = `${show.length} of ${motions.length}`;
  markActiveMotion();
}

/* Delegated, so it survives every re-render of the grid. */
$("motions").onclick = e => {
  const cell = e.target.closest(".m");
  if (!cell) return;
  fetch("/motion?id=" + cell.dataset.id).then(r => r.json()).then(r => {
    $("mmsg").textContent = r.ok
      ? `queued ${r.queued} · ${r.name}` : r.msg;
    $("mmsg").classList.toggle("bad", !r.ok);
  });
};

$("mq").oninput = e => { mQuery = e.target.value; drawMotions(); };
document.querySelectorAll("#mfilter [data-g]").forEach(b => b.onclick = () => {
  mGrade = b.dataset.g;
  document.querySelectorAll("#mfilter [data-g]")
          .forEach(x => x.classList.toggle("on", x === b));
  drawMotions();
});

function markActiveMotion() {
  const now = S && S.cradle ? S.cradle.motion : null;
  document.querySelectorAll("#motions .m").forEach(el =>
    el.classList.toggle("on", el.dataset.id === now));
}

const es = new EventSource("/events");
es.onopen  = () => { $("conn").textContent = "live"; $("conn").classList.add("live"); };
es.onerror = () => { $("conn").textContent = "reconnecting"; $("conn").classList.remove("live"); };
es.onmessage = e => {
  S = JSON.parse(e.data);
  if (S.decision && S.decision.seq !== lastSeq) {
    lastSeq = S.decision.seq;
    showRanking(S.decision);
  }
  updatePanels();
};

$("jam").onclick = () => fetch("/jam");
$("auto").onclick = () =>
  fetch("/auto?set=" + (S && S.cradle.auto ? "off" : "on"));
document.querySelectorAll("[data-m]").forEach(b =>
  b.onclick = () => fetch("/motion?id=" + b.dataset.m));
document.addEventListener("keydown", e => { if (e.key === "j") fetch("/jam"); });

function updatePanels() {
  const c = S.cradle, lvl = S.tag.level ?? 0, ema = c.ema ?? lvl;
  const who = babyWord(S.tag);
  const [doing, detail] = cradleWords(c);
  const col = S.tag.present ? stateColor(S.tag.emotion) : cssv("--critical");

  // zone 1 -- the hero says it in one sentence
  $("hdot").style.background = c.state === "gate_fail" ? cssv("--critical") : col;
  $("hbaby").textContent = who;
  $("hcradle").textContent = c.state === "gate_fail" ? "stopping for safety" : doing;
  $("hdetail").textContent = detail;
  $("hwhy").textContent = nextWords(c);

  // zone 2 -- four numbers, each against the limit that makes it mean something
  $("lvlval").textContent = lvl.toFixed(2);
  // With no face there is nothing to read, so the bar must not keep asserting
  // a level -- a stale confident-looking reading is worse than none.
  setMeter("lvlbar", lvl, !S.tag.present ? cssv("--baseline")
    : lvl >= CRY_LEVEL ? cssv("--serious")
    : lvl >= CALM_LEVEL ? cssv("--warn") : cssv("--good"));

  $("envval").innerHTML = c.env > 0.01
      ? Math.round(c.env * 100) + " <small>%</small>" : "– <small>idle</small>";
  setMeter("envbar", c.env, c.tapering ? cssv("--warn") : cssv("--accent"));

  // Three distinct motions -- summing them (the old code) added see-saw
  // millimetres to sway millimetres.  The stat shows whichever is largest,
  // named, against the report's 30 mm per-channel cap.
  const chans = [[Math.abs(c.offset_mm.ml), "swing"],
                 [Math.abs(c.offset_mm.z), "lift"],
                 [Math.abs(c.offset_mm.ap), "tilt"]];
  const [mm, chanWord] = chans.reduce((a, b) => (b[0] > a[0] ? b : a));
  const swayFrac = clamp(mm / SWAY_CAP_MM, 0, 1);
  const accFrac = clamp(c.a_peak_g / ACC_CAP_G, 0, 1);
  const worst = Math.max(swayFrac, accFrac);
  const safeCol = worst > 0.8 ? cssv("--warn") : cssv("--good");
  $("swayval").innerHTML =
      `${mm.toFixed(1)} <small>of ${SWAY_CAP_MM} mm ${mm > 0.05 ? chanWord : ""}</small>`;
  setMeter("swaybar", swayFrac, safeCol);
  $("accval").innerHTML = `${c.a_peak_g.toFixed(3)} <small>of ${ACC_CAP_G} g</small>`;
  setMeter("accbar", accFrac, safeCol);

  $("safedot").style.background = c.state === "gate_fail" ? cssv("--critical") : safeCol;
  $("safesay").textContent = c.state === "gate_fail"
      ? "safety gate tripped — winding down"
      : worst > 0.8 ? "close to the limit, still inside it"
                    : "well within safe limits";

  // zone 3 -- alert, buttons, log
  $("alert").textContent = c.alert ? "⚠ " + c.alert : "";
  $("alert").classList.toggle("on", !!c.alert);
  $("jam").classList.toggle("on", S.jam);
  $("jam").innerHTML = S.jam ? "<b>Jammed — release</b><i>j</i>"
                             : "<b>Simulate a jam</b><i>j</i>";
  $("auto").classList.toggle("on", c.auto);
  $("auto").innerHTML = `<b>Automatic care</b><i>${c.auto ? "on" : "off"}</i>`;
  $("log").innerHTML = S.events.map(t => `<div>${t}</div>`).join("");

  // The library grid: follow the active entry, and redraw once the first frame
  // tells us whether --research is on (the fetch usually lands before it does).
  if (mResearch !== c.research) { mResearch = c.research; drawMotions(); }
  else markActiveMotion();

  // the drawer -- raw numbers, for anyone who wants them
  $("rawstate").textContent = c.state + (c.auto ? "" : " · auto off")
                            + (c.trial_s ? ` · ${c.trial_s}s` : "");
  $("rawmotion").textContent = `${c.motion || "M01"} ${c.name} · ${c.grade}`;
  $("rawmode").textContent = `${c.f_hz} Hz · ${c.a_mm.toFixed(1)} mm · `
                           + `env ${c.env.toFixed(3)}`;
  $("rawlevel").textContent = `${lvl.toFixed(2)} · ${ema.toFixed(2)}`;
  $("thetaval").textContent = S.pose[0].toFixed(4) + " rad";
  $("apeak").textContent = c.a_peak_g.toFixed(4) + " g";
  $("err").textContent = S.monitor.err.toFixed(3) + " rad"
                       + (S.monitor.diverged ? " · DIVERGED" : "");
  const frac = clamp(S.monitor.err / (S.monitor.threshold * 2), 0, 1);
  setMeter("errbar", frac,
           S.monitor.diverged ? cssv("--critical") : cssv("--accent"));
  $("dob").textContent = S.dob.toFixed(1) + " A";
  $("rms").textContent = S.monitor.rms.toFixed(3) + " rad";
  $("playname").textContent = S.playing
      ? `playing slot ${S.playing.slot} · ${S.playing.name}` : "idle";
  $("playpct").textContent = S.playing
      ? Math.round(S.playing.progress * 100) + "%" : "";
  for (const s of slots) {
    const el = $("slot" + s.id);
    if (el) el.classList.toggle("play", !!(S.playing && S.playing.slot === s.id));
  }
}

function showRanking(d) {
  $("rank").querySelector("tbody").innerHTML = d.ranked.map(r => `
    <tr class="${r.vetoed ? "veto" : (r.slot === d.chosen ? "win" : "")}">
      <td>${r.slot}</td><td>${r.vetoed ? "—" : r.cost.toFixed(2)}</td>
      <td>${r.consist.toFixed(2)}</td><td>${r.resist.toFixed(2)}</td>
      <td>${r.contin.toFixed(2)}</td><td>${r.task.toFixed(2)}</td>
      <td>${r.vetoed ? "veto" : (r.slot === d.chosen ? "play" : "")}</td>
    </tr>`).join("");
}

/* ---- The hero: the infant as something alive.

   The outline is a blob, not a circle -- a handful of sine lobes turning at
   different speeds, which is enough to read as breathing tissue instead of a
   spinning polygon.  Everything that animates is driven by state the machine
   actually publishes: distress sets how far and how fast the outline wobbles,
   the envelope sets how brightly it burns, a taper settles it, and the body
   rides the real plate offset.  A lost face goes dashed and red.

   The one thing not measured is the breathing cadence: it is a visual pulse
   to make the shape feel inhabited, NOT a respiration reading.  Do not let it
   grow into one without a sensor behind it.

   When a DREAM slot is playing, a dashed ghost marks where the dream says the
   plate should be.  Jam the cradle and the two part -- divergence made
   visible.  The orb sits high in the frame so the sentence below it never
   fights the shape for space. ---- */
const cv = $("babycv"), g = cv.getContext("2d");
const view = { off: 0, lift: 0, tilt: 0, ghost: 0, lvl: 0, env: 0 };
const STILL = matchMedia("(prefers-reduced-motion: reduce)").matches;
const LOBES = [{k: 2, a: .024, w: .31}, {k: 3, a: .030, w: -.55},
               {k: 5, a: .017, w: .80}, {k: 7, a: .010, w: -1.15}];

/* One closed outline: radius R, deformed by the lobes at time t. */
function blob(cx, cy, R, t, amp, speed) {
  const N = 96;
  g.beginPath();
  for (let i = 0; i <= N; i++) {
    const th = i / N * Math.PI * 2;
    let d = 0;
    for (const l of LOBES) d += l.a * Math.sin(l.k * th + l.w * speed * t);
    const rr = R * (1 + d * amp);
    const x = cx + rr * Math.cos(th), y = cy + rr * Math.sin(th);
    i ? g.lineTo(x, y) : g.moveTo(x, y);
  }
  g.closePath();
}

function drawBaby() {
  requestAnimationFrame(drawBaby);
  if (!S || document.hidden) return;

  // back the canvas with real device pixels, so the edge stays crisp
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(cv.clientWidth * dpr), h = Math.round(cv.clientHeight * dpr);
  if (!w || !h) return;
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }

  const c = S.cradle, W = cv.width, H = cv.height, span = Math.min(W, H * 1.35);
  const t = performance.now() / 1000;
  // Three motions, three readings: ML sways the body sideways, Z lifts it,
  // AP tips the plate it rides on.  The old code summed ap+ml into one x
  // offset, which drew a see-saw as a sideways slide.
  const sway = c.offset_mm.ml, lift = c.offset_mm.z, tilt = c.offset_mm.ap;
  view.lvl += ((S.tag.level ?? 0) - view.lvl) * 0.15;
  view.env += ((c.env || 0) - view.env) * 0.10;
  const lvl = view.lvl;

  // agitation: distress drives it, a taper calms it, lost face unsettles it
  const settle = c.tapering ? 0.45 : 1;
  const amp = STILL ? 0.20 : (0.45 + 1.5 * lvl) * settle * (S.tag.present ? 1 : 1.6);
  const speed = (0.9 + 2.4 * lvl) * settle;
  const breath = STILL ? 0 : 0.045 * Math.sin(2 * Math.PI * (0.8 + 1.0 * lvl) * t);
  const R = span * (0.15 + 0.16 * lvl) * (1 + breath);

  // Exaggerated like the RViz mirror -- true sway is ~10 mm and would be a
  // few pixels -- but never wide enough to push the body out of the panel.
  const pxmm = Math.min(span * 0.011, Math.max(0, W / 2 - R - 16 * dpr) / SWAY_CAP_MM);
  view.off += (sway * pxmm - view.off) * 0.35;
  view.lift += (lift * pxmm - view.lift) * 0.35;
  view.tilt += (tilt - view.tilt) * 0.35;
  view.ghost += ((S.playing ? S.playing.dream_theta * LEVER_MM : sway) * pxmm
                 - view.ghost) * 0.35;

  const room = Math.max(0, W / 2 - R * 1.05 - 6 * dpr);
  // sits above centre: the sentence owns the bottom third of the hero.
  // Lift raises the body (canvas y grows downward), capped so it never
  // collides with the safety chip or the sentence.
  const rise = clamp(view.lift, -H * 0.06, H * 0.06);
  const cy = H * 0.40 - rise, cx = W / 2 + clamp(view.off, -room, room);
  const col = S.tag.present ? stateColor(S.tag.emotion) : cssv("--critical");
  g.clearRect(0, 0, W, H);

  // The plate the baby rides: the one place a see-saw tilt is visible, since
  // rotating an amorphous blob reads as nothing.  Angle from the real
  // geometry -- the two mount pairs sit 200 mm apart -- exaggerated the same
  // way the sway is, then clamped so it stays a gesture, not a ramp.
  const plateHalf = R * 1.15;
  const ang = clamp(Math.atan2(2 * view.tilt, 200) * 3, -0.22, 0.22);
  g.save();
  g.translate(cx, cy + R * 1.22);
  g.rotate(-ang);
  g.beginPath();
  g.moveTo(-plateHalf, 0); g.lineTo(plateHalf, 0);
  g.lineWidth = 3 * dpr;
  g.lineCap = "round";
  g.strokeStyle = cssv("--baseline");
  g.stroke();
  g.restore();

  // the dream's plate position, drawn only once it visibly parts from the real
  if (S.playing && Math.abs(view.ghost - view.off) > 2 * dpr) {
    blob(W / 2 + clamp(view.ghost, -room, room), cy, R, t, amp, speed);
    g.lineWidth = 1.5 * dpr;
    g.strokeStyle = S.monitor.diverged ? cssv("--critical") : cssv("--baseline");
    g.setLineDash([6 * dpr, 6 * dpr]);
    g.stroke();
    g.setLineDash([]);
  }

  // glow: how hard the cradle is working, wrapped around the body
  const aura = g.createRadialGradient(cx, cy, R * 0.5, cx, cy, R * 1.8);
  aura.addColorStop(0, rgba(col, 0.13 + 0.20 * view.env));
  aura.addColorStop(1, rgba(col, 0));
  g.fillStyle = aura;
  g.beginPath(); g.arc(cx, cy, R * 1.8, 0, Math.PI * 2); g.fill();

  // body, an inner outline turning the other way for depth, then the edge
  blob(cx, cy, R, t, amp, speed);
  g.fillStyle = rgba(col, 0.07);
  g.fill();

  blob(cx, cy, R * 0.62, -t, amp * 1.35, speed);
  g.lineWidth = 1.5 * dpr;
  g.strokeStyle = rgba(col, 0.30);
  g.stroke();

  blob(cx, cy, R, t, amp, speed);
  g.lineWidth = 5 * dpr;
  g.strokeStyle = col;
  if (!S.tag.present) g.setLineDash([12 * dpr, 10 * dpr]);
  g.stroke();
  g.setLineDash([]);
}
requestAnimationFrame(drawBaby);

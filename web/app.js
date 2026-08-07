/* The dashboard's one script.  Served by serve.py at /app.js; no build step.
   Reads the SSE stream (~20 Hz) and paints the page; the emotion timeline
   seeds itself from /history so a freshly opened tab shows the last 15 min.

   Colour rules: one calm blue accent for the cradle's own actions, a five
   step comfort ramp (--r0..--r4) for the baby, slate for sleep.  Colour is
   never the only carrier -- every coloured element sits beside its word. */
"use strict";
let S = null;
const $ = id => document.getElementById(id);
const clamp = (v, lo, hi) => v < lo ? lo : v > hi ? hi : v;

/* Change-guarded innerHTML writer: the stream ticks ~20x a second and most
   frames change nothing; rewriting identical HTML kills text selection. */
function put(id, html) {
  const el = $(id);
  if (el.dataset.v !== html) { el.dataset.v = html; el.innerHTML = html; }
}

/* CSS tokens, cached until the colour scheme flips. */
let TOKENS = {};
const cssv = name => name in TOKENS ? TOKENS[name]
  : (TOKENS[name] = getComputedStyle(document.documentElement)
                      .getPropertyValue(name).trim());
matchMedia("(prefers-color-scheme: dark)")
  .addEventListener("change", () => { TOKENS = {}; tlDirty = true; });

/* Report thresholds, mirrored from core/cradle.py and core/policy.py so the
   words on this page and the machine's decisions agree. */
const CALM_LEVEL = 0.12, CRY_LEVEL = 0.45;
const GATE_RECOVER_S = 2;   // trial cadence now rides in c.check_s / c.no_improve_s
const SWAY_CAP_MM = 30, ACC_CAP_G = 0.05;
const RANK_BANDS = [0.12, 0.30, 0.45, 0.62];          // rank edges (policy.py)
const RANK_WORD = ["happy", "fussing", "crying", "crying hard", "very upset"];
const RANK_SHORT = ["happy", "fuss", "cry 1", "cry 2", "awful"];
const rankOf = v => { for (let i = 0; i < RANK_BANDS.length; i++)
                        if (v < RANK_BANDS[i]) return i;
                      return RANK_BANDS.length; };
const rankColor = i => cssv(["--r0", "--r1", "--r2", "--r3", "--r4"][i] || "--faint");

/* Baby state -> colour token.  The ramp is comfort, slate is sleep, and the
   critical red is reserved for pain/lost-face/gate. */
const ROLE = {SLEEP:"--sleep", EYES_CLOSED:"--sleep", DROWSY:"--sleep",
              SLEEP_CANDIDATE:"--sleep", SLEEP_TENTATIVE:"--sleep",
              SLEEP_STABLE:"--sleep",
              CALM:"--r0", HAPPY:"--r0", AWAKE:"--r0", QUIET_AWAKE:"--r0",
              NEUTRAL:"--r0",
              FUSS:"--r1", SAD:"--r1", SURPRISE:"--r1", STARTLE:"--r1",
              FUSS_WEAK:"--r1", DISTRESS_FACE:"--r1",
              CRY:"--r3", ANGRY:"--r3", STRONG_DISTRESS:"--r3",
              PAIN_SUSPECT:"--critical",
              UNKNOWN:"--faint", STATE_UNCLEAR:"--faint"};
const stateColor = st => cssv(ROLE[st] || "--faint");

const BABY_WORD = {SLEEP:"Sleeping", CALM:"Calm", HAPPY:"Happy",
                   NEUTRAL:"Settled", FUSS:"Fussing", SAD:"Fussing",
                   SURPRISE:"Startled", CRY:"Crying", ANGRY:"Crying",
                   AWAKE:"Awake", EYES_CLOSED:"Eyes closed",
                   SLEEP_CANDIDATE:"Drifting off", DISTRESS_FACE:"Looks upset",
                   UNKNOWN:"Can't see the baby",
                   QUIET_AWAKE:"Quiet and awake", STARTLE:"Startled",
                   FUSS_WEAK:"Fussing", STRONG_DISTRESS:"Very upset",
                   PAIN_SUSPECT:"Needs you now", DROWSY:"Getting sleepy",
                   SLEEP_TENTATIVE:"Falling asleep", SLEEP_STABLE:"Sleeping",
                   STATE_UNCLEAR:"Can't see the baby"};

function babyWord(tag) {
  if (!tag.present) return "Can't see the baby";
  if (tag.emotion) return BABY_WORD[tag.emotion] || tag.emotion;
  const lvl = tag.level ?? 0;
  return lvl >= CRY_LEVEL ? "Crying" : lvl >= CALM_LEVEL ? "Fussing" : "Calm";
}

/* A rate in hertz means nothing to most readers; a sway every N seconds
   does.  Three real motions on this rig: ML sways side to side, Z lifts
   (bobbing), AP tilts like a see-saw. */
const SHAPE_WORD = {still: "gentle tremble", horiz: "side-to-side sway",
                    vert: "up-and-down bob", vert_fall: "bob with a quick drop",
                    v: "V-shaped swing", v_fall: "V-swing with a quick drop",
                    parab: "scooping swing", dwell: "scoop resting at each end",
                    circle: "circling", ellipse: "oval glide",
                    inf: "figure-eight", arc: "pendulum arc"};

function cradleWords(c) {
  if (c.tapering) return ["slowing to a stop", "easing the motion down to nothing"];
  if (c.kind === "npath") {   // the N01-N34 system: word it by its shape
    const pace = c.f_hz <= 0 ? "" : c.f_hz < 0.33 ? "slow " : c.f_hz < 0.5 ? "steady " : "quick ";
    const every = c.f_hz > 0 ? ` · one pass every ${(1 / c.f_hz).toFixed(1)} s` : "";
    return [pace + (SHAPE_WORD[c.shape] || "gentle motion"),
            `${c.a_mm.toFixed(0)} mm${every}`];
  }
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
  return [`${pace}${wide} rocking`, `one sway every ${every} s · ${mm} mm each way`];
}

function nextWords(c) {
  if (c.state === "gate_fail")
    return "Stopped for safety. Rocking only restarts once the baby has been "
         + `visible again for ${GATE_RECOVER_S} seconds.`;
  if (!c.auto)
    return "Automatic care is off. The cradle does only what you press, "
         + "though the safety gate still overrides it.";
  if (c.state === "trial") {
    const check = c.check_s ?? 30, noImp = c.no_improve_s ?? 2 * check;
    const left = Math.max(0, noImp - c.trial_s);
    return `Trying this for ${Math.round(c.trial_s)} s. Re-decided every `
         + `${Math.round(check)} s; if nothing helps for ${Math.round(left)} s `
         + "the cradle stops and calls for you.";
  }
  if (c.state === "settling")
    return "The baby has been calm for a while, so the cradle is winding "
         + "down towards sleep.";
  return "Watching. If the baby starts fussing, the cradle begins rocking "
       + "on its own.";
}

function setMeter(id, frac, color) {
  const fill = $(id);
  fill.style.width = (clamp(frac, 0, 1) * 100) + "%";
  fill.parentNode.style.setProperty("--fill", color);
}

/* ---- the motion library drawer ------------------------------------------ */
let motions = [], mGrade = "", mQuery = "", mResearch = null;

fetch("/motions").then(r => r.json()).then(list => {
  motions = list;
  drawMotions();
  fillTaste(list);
});

const esc = s => String(s).replace(/[&<>"]/g,
  c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function drawMotions() {
  const q = mQuery.trim().toLowerCase();
  const show = motions.filter(m =>
    (!mGrade || m.grade === mGrade) &&
    (!q || m.id.toLowerCase().includes(q) || m.name.toLowerCase().includes(q)));
  const research = mResearch === null ? false : mResearch;
  $("motions").innerHTML = show.map(m => {
    const locked = m.grade === "R" && !research;
    const hz = m.f_hz ? ` · ${m.f_hz.toFixed(1)} Hz` : "";
    const mm = m.a_mm ? ` · ${m.a_mm.toFixed(0)} mm` : "";
    const tip = `${m.id} ${m.name}${hz}${mm} · ${m.kind}${m.axis ? " · " + m.axis : ""}
${m.desc}${locked ? "\n\nresearch-only (R) — needs serve.py --research" : ""}`;
    return `<div class="m g-${m.grade}${locked ? " locked" : ""}"
                 data-id="${esc(m.id)}" title="${esc(tip)}"><b>${esc(m.id)}</b></div>`;
  }).join("");
  $("mcount").textContent = `${show.length} of ${motions.length}`;
  markActiveMotion();
}

$("motions").onclick = e => {
  const cell = e.target.closest(".m");
  if (cell) fetch("/motion?id=" + cell.dataset.id);
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

/* ---- the stream ---------------------------------------------------------- */
const es = new EventSource("/events");
es.onopen  = () => { $("conn").textContent = "live";
                     $("conn").classList.add("live");
                     $("conn").classList.remove("down"); };
es.onerror = () => { $("conn").textContent = "reconnecting — numbers may be stale";
                     $("conn").classList.remove("live");
                     $("conn").classList.add("down"); };
es.onmessage = e => { S = JSON.parse(e.data); pushSample(); updatePanels(); };

$("jam").onclick = () => fetch("/jam");
$("auto").onclick = () => fetch("/auto?set=" + (S && S.cradle.auto ? "off" : "on"));
document.querySelectorAll("[data-m]").forEach(b =>
  b.onclick = () => fetch("/motion?id=" + b.dataset.m));
document.addEventListener("keydown", e => {
  if (e.target.matches("input, textarea, select")) return;
  if (e.key === "j") fetch("/jam");
});

/* ---- the panels ---------------------------------------------------------- */
function updatePanels() {
  const c = S.cradle, lvl = S.tag.level ?? 0, ema = c.ema ?? lvl;
  const [doing, detail] = cradleWords(c);
  const col = S.tag.present ? stateColor(S.tag.emotion) : cssv("--critical");

  // baby card
  $("hdot").style.background = c.state === "gate_fail" ? cssv("--critical") : col;
  put("hbaby", esc(babyWord(S.tag)));
  put("hbabysub", esc(`recognizer: ${S.tag.emotion || "level only"}`
                      + (S.tag.name ? ` · ${S.tag.name}` : "")));
  put("lvlval", lvl.toFixed(2));
  setMeter("lvlbar", lvl, !S.tag.present ? cssv("--faint")
    : lvl >= CRY_LEVEL ? cssv("--r3")
    : lvl >= CALM_LEVEL ? cssv("--r1") : cssv("--r0"));
  const rank = rankOf(ema);
  put("ladder", RANK_SHORT.map((w, i) =>
      `<span class="r${i}${i === rank ? " on" : ""}">${w}</span>`).join(""));

  // cradle card
  put("hcradle", esc(c.state === "gate_fail" ? "stopping for safety" : doing));
  put("hdetail", esc(detail));
  put("envval", c.env > 0.01
      ? Math.round(c.env * 100) + " <small>%</small>" : "– <small>idle</small>");
  setMeter("envbar", c.env, c.tapering ? cssv("--warn") : cssv("--accent"));
  put("rawmotion", esc(`${c.motion || "M01"} · ${c.name}`));

  // safety card -- the largest of the three channels, named, vs the caps
  const chans = [[Math.abs(c.offset_mm.ml), "swing"],
                 [Math.abs(c.offset_mm.z), "lift"],
                 [Math.abs(c.offset_mm.ap), "tilt"]];
  const [mm, chanWord] = chans.reduce((a, b) => (b[0] > a[0] ? b : a));
  const swayFrac = clamp(mm / SWAY_CAP_MM, 0, 1);
  const accFrac = clamp(c.a_peak_g / ACC_CAP_G, 0, 1);
  const worst = Math.max(swayFrac, accFrac);
  const safeCol = worst > 0.8 ? cssv("--warn") : cssv("--good");
  put("swayval", `${mm.toFixed(1)} <small>of ${SWAY_CAP_MM} mm`
                 + ` ${mm > 0.05 ? chanWord : ""}</small>`);
  setMeter("swaybar", swayFrac, safeCol);
  put("accval", `${c.a_peak_g.toFixed(3)} <small>of ${ACC_CAP_G} g</small>`);
  setMeter("accbar", accFrac, safeCol);
  put("hwhy", esc(nextWords(c)));

  const alarm = !!S.tag.alarm;
  $("safedot").style.background =
      alarm || c.state === "gate_fail" ? cssv("--critical") : safeCol;
  put("safesay", alarm ? "pain/posture alarm — stopping"
      : c.state === "gate_fail" ? "safety gate tripped — winding down"
      : worst > 0.8 ? "close to the limit, still inside it"
                    : "within safe limits");
  $("scen").textContent = S.tag.phase ? `scenario · ${S.tag.phase}` : "";

  // cradle-mounted iPad IMU; this is physical exposure for the virtual baby,
  // not a substitute for the safety envelope above.
  const ipad = S.ipad || {connected:false, strength:0};
  $("ipaddot").style.background = ipad.connected ? cssv("--good") : cssv("--faint");
  put("ipadstate", ipad.connected ? "sensor live" : "not connected");
  put("ipadstrength", ipad.connected
      ? `${Math.round(ipad.strength * 100)} <small>%</small>` : "–");
  setMeter("ipadbar", ipad.connected ? ipad.strength : 0,
           ipad.connected ? cssv("--accent") : cssv("--faint"));
  put("ipadaccel", ipad.connected
      ? `${ipad.accel_rms.toFixed(3)} <small>m/s²</small>` : "–");
  put("ipadhz", ipad.connected
      ? `${ipad.dominant_hz.toFixed(2)} <small>Hz</small>` : "–");
  // the server's felt-classification: what this motion IS to the baby's
  // taste (speed/size/tremble), measured rather than commanded
  const felt = ipad.felt || null;
  put("ipadfeel", felt
      ? [felt.speed, felt.size, felt.vibe ? "trembling" : ""]
          .filter(Boolean).join(" · ") || "barely moving"
      : "–");

  // header controls + alert
  $("alert").textContent = c.alert ? "⚠ " + c.alert : "";
  $("alert").classList.toggle("on", !!c.alert);
  $("jam").classList.toggle("on", S.jam);
  $("jam").setAttribute("aria-pressed", S.jam);
  put("jam", S.jam ? "<b>Jammed — release</b><i>j</i>"
                   : "<b>Simulate a jam</b><i>j</i>");
  $("auto").classList.toggle("on", c.auto);
  $("auto").setAttribute("aria-pressed", c.auto);
  put("auto", `<b>Automatic care</b><i>${c.auto ? "on" : "off"}</i>`);

  // event log (newest first; identical on most frames)
  const logKey = S.events.length + "|" + (S.events[0] || "");
  if (logKey !== put.logKey) {
    put.logKey = logKey;
    put("log", S.events.map(t => `<div>${esc(t)}</div>`).join(""));
  }

  drawBrain(rank);
  drawTaste();

  if (mResearch !== c.research) { mResearch = c.research; drawMotions(); }
  else markActiveMotion();

  // internals drawer
  put("rawstate", esc(c.state + (c.auto ? "" : " · auto off")
                      + (c.trial_s ? ` · ${c.trial_s}s` : "")));
  put("rawmode", esc(`${c.f_hz} Hz · ${c.a_mm.toFixed(1)} mm · env ${c.env.toFixed(3)}`));
  put("rawlevel", esc(`${lvl.toFixed(2)} · ${ema.toFixed(2)}`));
  put("thetaval", S.pose[0].toFixed(4) + " rad");
  put("apeak", c.a_peak_g.toFixed(4) + " g");
  put("rawjudge", esc(S.tag.emotion || "–"));
  put("rawphase", esc(S.tag.phase || "live camera"));
}

/* ---- the decision brain (docs/IDEA.md, core/policy.py) ------------------- */
const BRAIN_NAME = {reflex: "Local algorithm",
                    ollama: "Local LLM",
                    claude: "Claude (Anthropic API)"};
const BRAIN_SUB = {reflex: "taught strategy in code — free, offline, deterministic",
                   ollama: "LLM on this machine via Ollama — free, private; "
                         + "falls back to the ladder if unreachable",
                   claude: "paid cloud LLM — falls back to the ladder on any "
                         + "API failure"};

/* Switch brains live: /policy swaps the advisor (fresh memory each time). */
document.querySelectorAll("#brainsel [data-b]").forEach(b => b.onclick = () =>
  fetch("/policy?set=" + b.dataset.b));

function drawBrain(rank) {
  const p = S.policy;
  const current = p ? p.brain : "off";
  document.querySelectorAll("#brainsel [data-b]").forEach(b =>
    b.classList.toggle("on", b.dataset.b === current));
  if (!p) {
    put("brainname", "Fixed ladder");
    put("brainkind", "the report's own escalation (M10 → M12 → M13 → M16) — "
                   + "safe, but it never learns this baby");
    put("brainerr", "");
    put("brainrank", esc(RANK_WORD[rank]));
    put("brainlast", "—");
    put("brainscen", "—");
    put("feed", `<div class="empty">Fixed-ladder mode: the report's state
      machine picks motions (M10 → M12 → M13 → M16) with no memory of this
      baby. Start serve.py with <b>--policy reflex</b> (local algorithm) or
      <b>--policy claude</b> (LLM) to watch a learning brain here.</div>`
      + ladderEvents());
    put("scores", `<div class="empty">nothing to learn in fixed-ladder mode</div>`);
    put("trail", "");
    return;
  }

  // the brain card
  put("brainname", esc(BRAIN_NAME[p.brain] || p.brain));
  put("brainkind", esc(BRAIN_SUB[p.brain] || ""));
  put("brainerr", p.error ? "⚠ " + esc(p.error) + " — using the ladder" : "");
  put("brainrank", esc(RANK_WORD[rank]));
  put("brainscen", String(p.scenarios));
  const last = p.steps[p.steps.length - 1];
  put("brainlast", last
      ? esc(`${last.motion} — ${last.after === null ? "playing now"
             : verdictWord(last.before - last.after)}`)
      : "none yet");

  // the decision feed, newest first
  put("feed", p.steps.length ? p.steps.slice().reverse().map(s => {
    const pending = s.after === null || s.after === undefined;
    const d = pending ? null : s.before - s.after;
    const badge = pending ? `<span class="badge live">playing…</span>`
      : d > 0 ? `<span class="badge ok">improved</span>`
      : d < 0 ? `<span class="badge bad">made it worse</span>`
      : `<span class="badge flat">no change</span>`;
    const after = pending ? "…" : RANK_WORD[s.after];
    return `<div class="dcard"><div class="mid">tried <b>${s.motion}</b>
        <div class="why">${RANK_WORD[s.before]} → ${after}</div></div>${badge}</div>`;
  }).join("") : `<div class="empty">no decisions yet — the brain is asked
      when the baby first fusses</div>`);

  // learned preferences: diverging bars, helps right / worsens left
  const rows = Object.entries(p.scores).sort((a, b) => b[1] - a[1]);
  put("scores", rows.length ? rows.map(([m, v]) => {
    const w = clamp(Math.abs(v) / 2, 0, 1) * 50;
    const bar = v >= 0 ? `<i class="up" style="width:${w}%"></i>`
                       : `<i class="dn" style="width:${w}%"></i>`;
    return `<div class="srow"><b>${m}</b><span class="dbar">${bar}</span>
        <span class="verdict">${v > 0 ? "+" : ""}${v.toFixed(1)} · ${verdictWord(v)}</span></div>`;
  }).join("") : `<div class="empty">nothing tried yet — learning starts at
      the first fuss</div>`);
  put("trail", p.steps.slice(-6).map(s =>
      `${s.motion} ${s.before}→${s.after ?? "…"}`).join("  ·  "));
}
const verdictWord = d => d > 0 ? "helps" : d < 0 ? "worse" : "no effect";

/* Fixed-ladder mode still shows its decisions: the machine's own log lines. */
function ladderEvents() {
  const hits = (S.events || []).filter(t =>
      /cry trial|step up|micro-resume|taper|hand over/.test(t)).slice(0, 5);
  return hits.map(t => `<div class="dcard"><div class="mid mono why">${esc(t)}</div></div>`).join("");
}

/* ---- the temperament editor (virtual infant only) ------------------------ */
/* The selects fill from /motions once it arrives: the N-system trial
   candidates, each labeled with its features so picking a taste means
   something ("N16 · v large slow"). */
function fillTaste(list) {
  const cands = list.filter(m => m.candidate);
  const opt = m => `<option value="${m.id}">${m.id} · ${m.shape}`
    + (m.size ? ` ${m.size} ${m.speed}` : "") + (m.vibe ? " +tremble" : "")
    + `</option>`;
  for (const [id, blank] of [["t-love", false], ["t-hate1", true],
                             ["t-hate2", true], ["t-c1", true], ["t-c2", true]])
    $(id).innerHTML = (blank ? '<option value="">—</option>' : "")
      + cands.map(opt).join("");
}
$("t-apply").onclick = () => {
  const hate = [$("t-hate1").value, $("t-hate2").value].filter(Boolean).join(",");
  const c1 = $("t-c1").value, c2 = $("t-c2").value;
  fetch(`/taste?love=${$("t-love").value}`
        + (hate ? `&hate=${hate}` : "")
        + (c1 && c2 ? `&combo=${c1},${c2}` : ""));
};
$("t-rand").onclick = () => fetch("/taste?random=1");

let tasteSeeded = false;
function drawTaste() {
  const t = S.taste;
  $("card-taste").hidden = t === undefined;
  if (t === undefined) return;
  put("t-now", t.love
    ? "now: loves " + t.love
      + (t.hate && t.hate.length ? " · hates " + t.hate.join("/") : "")
      + (t.combo && t.combo.length === 2
         ? ` · likes ${t.combo[0]}→${t.combo[1]}` : "")
      + " · motions also wear out with use (habituation)"
    : "no temperament yet — every motion feels the same; hit Randomize");
  if (!tasteSeeded && t.love) {    // seed the selects once from the live truth
    tasteSeeded = true;
    $("t-love").value = t.love;
    $("t-hate1").value = (t.hate && t.hate[0]) || "";
    $("t-hate2").value = (t.hate && t.hate[1]) || "";
    $("t-c1").value = (t.combo && t.combo[0]) || "";
    $("t-c2").value = (t.combo && t.combo[1]) || "";
  }
}

/* ---- the living circle ----------------------------------------------------
   A blob, not a circle: sine lobes turning at different speeds read as
   breathing tissue.  Everything it does is published state -- distress sets
   the wobble, the envelope sets the glow, the body rides the real sway --
   except the breathing cadence, which is a visual pulse, not a respiration
   reading.  Do not let it become one without a sensor behind it. */
const orb = $("orb"), og = orb.getContext("2d");
const STILL = matchMedia("(prefers-reduced-motion: reduce)").matches;
const LOBES = [{k: 2, a: .024, w: .31}, {k: 3, a: .030, w: -.55},
               {k: 5, a: .017, w: .80}, {k: 7, a: .010, w: -1.15}];
const ov = { off: 0, lvl: 0, env: 0 };

function rgba(hex, a) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.replace(/./g, "$&$&") : h, 16);
  return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`;
}

function blobPath(cx, cy, R, t, amp, speed) {
  const N = 72;
  og.beginPath();
  for (let i = 0; i <= N; i++) {
    const th = i / N * Math.PI * 2;
    let d = 0;
    for (const l of LOBES) d += l.a * Math.sin(l.k * th + l.w * speed * t);
    const rr = R * (1 + d * amp);
    const px = cx + rr * Math.cos(th), py = cy + rr * Math.sin(th);
    i ? og.lineTo(px, py) : og.moveTo(px, py);
  }
  og.closePath();
}

function drawOrb() {
  requestAnimationFrame(drawOrb);
  if (!S || document.hidden) return;
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(orb.clientWidth * dpr), h = Math.round(orb.clientHeight * dpr);
  if (!w || !h) return;
  if (orb.width !== w || orb.height !== h) { orb.width = w; orb.height = h; }

  const c = S.cradle, t = performance.now() / 1000;
  ov.lvl += ((S.tag.level ?? 0) - ov.lvl) * 0.15;
  ov.env += ((c.env || 0) - ov.env) * 0.10;
  const settle = c.tapering ? 0.45 : 1;
  const amp = STILL ? 0.2 : (0.45 + 1.5 * ov.lvl) * settle * (S.tag.present ? 1 : 1.6);
  const speed = (0.9 + 2.4 * ov.lvl) * settle;
  const breath = STILL ? 0 : 0.05 * Math.sin(2 * Math.PI * (0.8 + ov.lvl) * t);
  const R = h * (0.30 + 0.10 * ov.lvl) * (1 + breath);
  // ride the real sway, exaggerated like the RViz mirror, never off-card
  const room = Math.max(0, w / 2 - R * 1.3 - 4 * dpr);
  ov.off += (c.offset_mm.ml * (w * 0.012) - ov.off) * 0.35;
  const cx = w / 2 + clamp(ov.off, -room, room), cy = h * 0.52;
  const col = S.tag.present ? stateColor(S.tag.emotion) : cssv("--critical");

  og.clearRect(0, 0, w, h);
  const aura = og.createRadialGradient(cx, cy, R * 0.5, cx, cy, R * 1.9);
  aura.addColorStop(0, rgba(col, 0.16 + 0.22 * ov.env));
  aura.addColorStop(1, rgba(col, 0));
  og.fillStyle = aura;
  og.beginPath(); og.arc(cx, cy, R * 1.9, 0, Math.PI * 2); og.fill();

  blobPath(cx, cy, R, t, amp, speed);
  og.fillStyle = rgba(col, 0.10);
  og.fill();
  blobPath(cx, cy, R * 0.62, -t, amp * 1.35, speed);
  og.lineWidth = 1.2 * dpr;
  og.strokeStyle = rgba(col, 0.35);
  og.stroke();
  blobPath(cx, cy, R, t, amp, speed);
  og.lineWidth = 3 * dpr;
  og.strokeStyle = col;
  if (!S.tag.present) og.setLineDash([10 * dpr, 8 * dpr]);
  og.stroke();
  og.setLineDash([]);
}
requestAnimationFrame(drawOrb);

/* ---- the emotion timeline ------------------------------------------------ */
/* Samples {t, level, ema, motion, alarm}: seeded once from /history (the
   server keeps 15 min at 1 Hz), then appended live from the stream.  The
   chart is the page's centrepiece: comfort bands as the backdrop, the raw
   level dotted, the decision-driving trend solid, motion starts flagged. */
const WINDOW_S = 900;
const HIST = [];
let tlDirty = true, hoverX = null;

fetch("/history").then(r => r.json()).then(rows => {
  HIST.unshift(...rows.map(r => ({t: r.t, level: r.level, ema: r.ema,
                                  motion: r.motion, alarm: r.alarm})));
  HIST.sort((a, b) => a.t - b.t);
  tlDirty = true;
});

function pushSample() {
  const lastT = HIST.length ? HIST[HIST.length - 1].t : -1e9;
  if (S.t - lastT < 0.5) return;
  HIST.push({t: S.t, level: S.tag.level ?? 0, ema: S.cradle.ema ?? 0,
             motion: S.cradle.motion,
             alarm: !!S.tag.alarm || S.cradle.state === "gate_fail"});
  while (HIST.length && HIST[0].t < S.t - WINDOW_S - 30) HIST.shift();
  tlDirty = true;
}

const tl = $("timeline"), tg = tl.getContext("2d");
const tip = $("tltip");
tl.addEventListener("mousemove", e => {
  hoverX = e.offsetX; tlDirty = true;
});
tl.addEventListener("mouseleave", () => {
  hoverX = null; tip.hidden = true; tlDirty = true;
});

function drawTimeline() {
  requestAnimationFrame(drawTimeline);
  if (!tlDirty || document.hidden) return;
  tlDirty = false;

  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(tl.clientWidth * dpr), h = Math.round(tl.clientHeight * dpr);
  if (!w || !h) return;
  if (tl.width !== w || tl.height !== h) { tl.width = w; tl.height = h; }

  const padL = 6 * dpr, padR = 52 * dpr, padT = 22 * dpr, padB = 20 * dpr;
  const pw = w - padL - padR, ph = h - padT - padB;
  const tNow = HIST.length ? HIST[HIST.length - 1].t : WINDOW_S;
  const x = t => padL + clamp((t - (tNow - WINDOW_S)) / WINDOW_S, 0, 1) * pw;
  const y = v => padT + (1 - clamp(v, 0, 1)) * ph;
  tg.clearRect(0, 0, w, h);
  tg.font = `${10 * dpr}px ${getComputedStyle(document.body).fontFamily}`;

  // comfort bands: the ladder as the backdrop, named on the right edge
  const edges = [0, ...RANK_BANDS, 1];
  for (let i = 0; i < 5; i++) {
    const yTop = y(edges[i + 1]), yBot = y(edges[i]);
    tg.globalAlpha = 0.14;                    // tuned for the black page
    tg.fillStyle = rankColor(i);
    tg.fillRect(padL, yTop, pw, yBot - yTop);
    tg.globalAlpha = 1;
    tg.fillStyle = rankColor(i);
    tg.textAlign = "left";
    tg.fillText(RANK_SHORT[i], padL + pw + 8 * dpr,
                Math.min(yBot - 3 * dpr, (yTop + yBot) / 2 + 3 * dpr));
    tg.strokeStyle = cssv("--border");
    tg.lineWidth = 1;
    tg.beginPath(); tg.moveTo(padL, yTop); tg.lineTo(padL + pw, yTop); tg.stroke();
  }

  // time axis: a tick every 5 minutes (edge labels hug their edge)
  tg.fillStyle = cssv("--faint");
  for (let back = WINDOW_S; back >= 0; back -= 300) {
    const xx = x(tNow - back);
    tg.textAlign = back === WINDOW_S ? "left" : back === 0 ? "right" : "center";
    tg.fillText(back ? `${back / 60}m ago` : "now", xx, h - 6 * dpr);
  }

  // motion starts: dashed accent flags with the motion id
  tg.textAlign = "center";
  let lastLabelX = -1e9;
  for (let i = 1; i < HIST.length; i++) {
    const cur = HIST[i], prev = HIST[i - 1];
    if (cur.motion && cur.motion !== prev.motion && cur.t >= tNow - WINDOW_S) {
      const xx = x(cur.t);
      tg.strokeStyle = cssv("--accent");
      tg.lineWidth = 1 * dpr;
      tg.setLineDash([3 * dpr, 4 * dpr]);
      tg.beginPath(); tg.moveTo(xx, padT); tg.lineTo(xx, padT + ph); tg.stroke();
      tg.setLineDash([]);
      if (xx - lastLabelX > 26 * dpr) {          // skip labels that would pile up
        tg.fillStyle = cssv("--accent");
        tg.fillText(cur.motion, xx, padT - 8 * dpr);
        lastLabelX = xx;
      }
    }
    if (cur.alarm && !prev.alarm) {              // gate/alarm marker
      const xx = x(cur.t);
      tg.fillStyle = cssv("--critical");
      tg.beginPath();
      tg.moveTo(xx, padT - 4 * dpr);
      tg.lineTo(xx - 4 * dpr, padT - 12 * dpr);
      tg.lineTo(xx + 4 * dpr, padT - 12 * dpr);
      tg.closePath(); tg.fill();
    }
  }

  // the two series: raw level dotted, decision trend solid
  const line = (key, color, width, dash) => {
    tg.strokeStyle = color; tg.lineWidth = width * dpr;
    tg.setLineDash(dash.map(d => d * dpr));
    tg.beginPath();
    let started = false;
    for (const s of HIST) {
      if (s.t < tNow - WINDOW_S) continue;
      const xx = x(s.t), yy = y(s[key]);
      started ? tg.lineTo(xx, yy) : tg.moveTo(xx, yy);
      started = true;
    }
    tg.stroke(); tg.setLineDash([]);
  };
  line("level", cssv("--faint"), 1.4, [2, 3]);
  line("ema", cssv("--ink"), 2, []);

  // hover: crosshair + tooltip on the nearest sample
  if (hoverX !== null) {
    const tx = tNow - WINDOW_S + ((hoverX * dpr - padL) / pw) * WINDOW_S;
    let best = null, bestD = 1e9;
    for (const s of HIST) {
      const d = Math.abs(s.t - tx);
      if (d < bestD) { bestD = d; best = s; }
    }
    if (best && bestD < 30) {
      const xx = x(best.t);
      tg.strokeStyle = cssv("--muted");
      tg.lineWidth = 1;
      tg.beginPath(); tg.moveTo(xx, padT); tg.lineTo(xx, padT + ph); tg.stroke();
      tg.fillStyle = cssv("--ink");
      tg.beginPath(); tg.arc(xx, y(best.ema), 3.5 * dpr, 0, Math.PI * 2); tg.fill();
      const ago = Math.max(0, Math.round(tNow - best.t));
      tip.innerHTML = `<b>${ago < 60 ? ago + " s" : Math.round(ago / 60) + " min"} ago</b><br>`
        + `level ${best.level.toFixed(2)} · trend ${best.ema.toFixed(2)} `
        + `(${RANK_SHORT[rankOf(best.ema)]})<br>`
        + (best.motion ? `playing ${best.motion}` : "cradle still")
        + (best.alarm ? " · ⚠ safety" : "");
      tip.hidden = false;
      const wrap = $("tlwrap").getBoundingClientRect();
      const px = clamp(xx / dpr + 12, 4, wrap.width - tip.offsetWidth - 4);
      tip.style.left = px + "px";
      tip.style.top = Math.max(4, y(best.ema) / dpr - tip.offsetHeight - 12) + "px";
    } else { tip.hidden = true; }
  }
}
requestAnimationFrame(drawTimeline);
new ResizeObserver(() => { tlDirty = true; }).observe($("tlwrap"));


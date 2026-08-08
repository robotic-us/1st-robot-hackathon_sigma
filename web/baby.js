"use strict";

const canvas = document.getElementById("face");
const g = canvas.getContext("2d");
const start = document.getElementById("start");
const setup = document.getElementById("setup");
const sensorText = document.getElementById("sensor");
const linkText = document.getElementById("link");
const linkDot = document.getElementById("linkdot");
const emotionText = document.getElementById("emotion");
const levelText = document.getElementById("level");

let live = null;
let permission = "waiting";
let motionStarted = false;
let posting = false;
const samples = [];
const KEEP_MS = 2200;

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function lerp(a, b, t) { return a + (b - a) * t; }
function smooth01(v) {
  const x = clamp(v, 0, 1);
  return x * x * (3 - 2 * x);
}

// ?level=0.65 is a visual-test hook; it never changes server state.
const previewRaw = new URLSearchParams(location.search).get("level");
const previewLevel = previewRaw === null ? NaN : Number(previewRaw);
const previewState = new URLSearchParams(location.search).get("state");
const hasPreview = Number.isFinite(previewLevel);
// ?pose=crying pins one *pure* pose, no blend: perception/nubzuki.py's bench hook.
const previewPose = new URLSearchParams(location.search).get("pose");

function stateWord(state, level) {
  if (state === "SLEEP") return "Sleeping";
  if (state === "CALM" || level < .12) return "Calm";
  if (state === "FUSS" || level < .45) return "Fussing";
  return level >= .72 ? "Crying hard" : "Crying";
}

let captionWords = "", captionLevel = "";
function setCaption(words, levelText_) {
  if (words !== captionWords) emotionText.textContent = captionWords = words;
  if (levelText_ !== captionLevel) levelText.textContent = captionLevel = levelText_;
}

// Sized from the visible rectangle, or the CSS box and backing store disagree on a phone.
function fitViewport() {
  const vv = window.visualViewport;
  if (!vv) return;
  const root = document.querySelector("main");
  if (root) { root.style.height = vv.height + "px"; root.style.width = vv.width + "px"; }
}
if (window.visualViewport) {
  visualViewport.addEventListener("resize", fitViewport);
  visualViewport.addEventListener("scroll", fitViewport);
}
fitViewport();

const events = new EventSource("/events");
events.onopen = () => {
  linkText.textContent = "Connected to Jetson";
  linkDot.classList.add("on");
};
events.onerror = () => {
  linkText.textContent = "Reconnecting to Jetson";
  linkDot.classList.remove("on");
};
events.onmessage = event => {
  live = JSON.parse(event.data);
  // the render loop owns the caption; writing it here too would flicker
  const ipad = live.ipad;
  if (motionStarted && ipad?.connected) {
    sensorText.textContent = `Sensor live · ${(ipad.accel_rms || 0).toFixed(3)} m/s² · `
      + `${(ipad.dominant_hz || 0).toFixed(2)} Hz`;
  }
};

function motionHandler(event) {
  const a = event.acceleration;
  const r = event.rotationRate;
  if (!a && !r) return;
  const now = performance.now();
  samples.push({
    t: now,
    x: Number(a?.x) || 0, y: Number(a?.y) || 0, z: Number(a?.z) || 0,
    ra: Number(r?.alpha) || 0, rb: Number(r?.beta) || 0,
    rg: Number(r?.gamma) || 0
  });
  while (samples.length && samples[0].t < now - KEEP_MS) samples.shift();
}

/* ---- rough handling -> outrage ----------------------------------------- */
// The envelope tops out at 0.05 g, so accel above it is a hand shaking it.
let lastFeat = null;
let shake = 0;
const SHAKE_FROM = 2.0, SHAKE_FULL = 7.0;   // m/s2 RMS: begins / full rage
function shakeUpdate() {
  const rmsNow = (motionStarted && lastFeat) ? lastFeat.accelRms : 0;
  const target = clamp((rmsNow - SHAKE_FROM) / (SHAKE_FULL - SHAKE_FROM), 0, 1);
  shake += (target - shake) * (target > shake ? 0.30 : 0.015);
  if (shake < 0.005) shake = 0;
  return shake;
}

function rms(values) {
  return values.length
    ? Math.sqrt(values.reduce((sum, v) => sum + v * v, 0) / values.length) : 0;
}

function dominantFrequency(rows) {
  if (rows.length < 8) return 0;
  const axes = ["x", "y", "z"];
  const stats = axes.map(axis => {
    const mean = rows.reduce((s, p) => s + p[axis], 0) / rows.length;
    const variance = rows.reduce((s, p) => s + (p[axis] - mean) ** 2, 0) / rows.length;
    return {axis, mean, variance};
  });
  const chosen = stats.sort((a, b) => b.variance - a.variance)[0];
  if (chosen.variance < 1e-5) return 0;
  const crossings = [];
  for (let i = 1; i < rows.length; i++) {
    const before = rows[i - 1][chosen.axis] - chosen.mean;
    const after = rows[i][chosen.axis] - chosen.mean;
    if (before <= 0 && after > 0) crossings.push(rows[i].t);
  }
  if (crossings.length < 2) return 0;
  const seconds = (crossings.at(-1) - crossings[0]) / 1000;
  return seconds > 0 ? clamp((crossings.length - 1) / seconds, 0, 20) : 0;
}

function features() {
  const rows = samples.slice();
  const accel = rows.map(p => Math.hypot(p.x, p.y, p.z));
  const rotation = rows.map(p => Math.hypot(p.ra, p.rb, p.rg));
  const jerk = [];
  for (let i = 1; i < rows.length; i++) {
    const dt = (rows[i].t - rows[i - 1].t) / 1000;
    if (dt > 0) jerk.push(Math.hypot(
      rows[i].x - rows[i - 1].x,
      rows[i].y - rows[i - 1].y,
      rows[i].z - rows[i - 1].z) / dt);
  }
  return {
    samples: rows.length,
    accelRms: rms(accel),
    rotationRms: rms(rotation),
    jerkRms: rms(jerk),
    dominantHz: dominantFrequency(rows),
    permission
  };
}

async function postFeatures() {
  if (!motionStarted || posting) return;
  posting = true;
  try {
    const feat = features();
    lastFeat = feat;                    // the shake meter reads this too
    await fetch("/motion-sensor", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(feat),
      cache: "no-store"
    });
  } catch (_) {
    sensorText.textContent = "Could not send motion data to Jetson.";
  } finally {
    posting = false;
  }
}
setInterval(postFeatures, 200);

start.addEventListener("click", async () => {
  start.disabled = true;
  try {
    if (!window.isSecureContext) {
      throw new Error("Motion sensing needs HTTPS on iPad.");
    }
    if (typeof DeviceMotionEvent === "undefined") {
      throw new Error("This browser does not expose a motion sensor.");
    }
    if (typeof DeviceMotionEvent.requestPermission === "function") {
      permission = await DeviceMotionEvent.requestPermission();
      if (permission !== "granted") throw new Error("Motion permission was denied.");
    } else {
      permission = "granted";
    }
    window.addEventListener("devicemotion", motionHandler, {passive: true});
    motionStarted = true;
    start.textContent = "Motion sensor running";
    sensorText.textContent = "Waiting for the first motion sample…";
  } catch (error) {
    permission = "denied";
    sensorText.textContent = error.message || String(error);
    start.disabled = false;
  }
});

/* ---- Nubzuki, a vector rig ----------------------------------------------
   Units: head half-width = 100, every constant a percentage of it.  The sheet
   is a circumplex: poses are anchors, the face a blend of nearby ones. */

const INK = "#241917";
const BLUE = "#3CA9E1";
const PINK = "#E2007E";
const RAGE_RED = "#C4161C";
const BROWN = "#7A4B24";
const TAU = Math.PI * 2;
const DEG = Math.PI / 180;

const ART = {
  headRx: 100, headRy: 49,
  eyeDx: 38.4, eyeDy: 2, eyeR: 17.2, pupilR: 9.9,
  shoulderX: 20, shoulderY: 66, armLen: 51, armW: 26,
  torsoHw: 34, torsoTop: 50, torsoBot: 126, torsoR: 26,
  legDx: 17, legTop: 112, legBot: 140, legW: 30,
  groundY: 148, shadowRx: 56, shadowRy: 12,
  ink: 4.2
};
// Design box the figure is fitted into, in head units.
const BOX = {x: -168, y: -92, w: 336, h: 272};
// The figure's middle inside the clear band, as a fraction of that band.
const FACE_CENTRE_Y = .46;

// Every parameter a pose can set; poses list only what they change.
const BASE = {
  eyeOpen: 1, eyeBig: 0, eyesOff: 0, brow: 0, blush: 0, blueBlush: 0,
  tear: 0, shine: 0, hunch: 0, flush: 0, dusk: 0, rage: 0, rainbow: 0,
  stars: 0, energy: .1, armUp: 0, armsIn: 0, armsCross: 0, heart: 0,
  heartTrail: 0, sit: 0, lie: 0, shades: 0, specs: 0, bow: 0, notes: 0,
  cup: 0, blanket: 0, groundHeart: 0, blackHeart: 0, brownLegs: 0
};
const POSE_KEYS = Object.keys(BASE);

// Sheet coordinates: x is valence (POSITIVE right), y screen-down (ACTIVE < 0).
const POSES = [
  {key: "rage", label: "Raging", at: [-.73, -.68],
   p: {eyeOpen: .5, brow: 1, rage: 1, energy: 1, armUp: .18, hunch: .12}},
  {key: "angry", label: "Angry", at: [-.50, -.58],
   p: {eyeOpen: .55, brow: .92, flush: .9, blush: .5, energy: .82, armUp: .22}},
  {key: "shocked", label: "Startled", at: [-.05, -.78],
   p: {eyeBig: 1, energy: .72, armsIn: 1, hunch: .1}},
  {key: "kiss", label: "Blowing a kiss", at: [.29, -.58],
   p: {heart: 1, armUp: .9, blush: .28, energy: .45}},
  {key: "dancing", label: "Dancing", at: [.65, -.78],
   p: {bow: 1, notes: 1, energy: .95, armUp: .45, hunch: .2}},
  {key: "star", label: "Celebrating", at: [.88, -.46],
   p: {rainbow: 1, stars: 1, eyesOff: 1, energy: 1, armUp: .55}},
  {key: "bashful", label: "Bashful", at: [.55, -.33],
   p: {blueBlush: 1, heart: .6, armsIn: .55, energy: .3, hunch: .22}},
  {key: "cool", label: "Playing it cool", at: [-.43, -.08],
   p: {shades: 1, armsCross: 1, energy: .28}},
  {key: "neutral", label: "Neutral", at: [-.03, -.14],
   p: {energy: .15}},
  {key: "crying", label: "Crying", at: [-.90, .06],
   p: {shine: 1, tear: 1, hunch: .7, sit: 1, energy: .55, armsIn: .6,
       groundHeart: 1}},
  {key: "nerdy", label: "Thinking", at: [.08, .25],
   p: {specs: 1, heart: .75, energy: .14, hunch: .16}},
  {key: "lounging", label: "Lounging", at: [.73, .13],
   p: {lie: .8, heartTrail: .5, energy: .16}},
  {key: "sitHeart", label: "Content", at: [.45, .45],
   p: {sit: 1, heart: 1, energy: .1, hunch: .28}},
  {key: "gloomy", label: "Gloomy", at: [-.71, .45],
   p: {dusk: 1, sit: 1, hunch: .62, energy: .07, blackHeart: 1}},
  {key: "sick", label: "Poorly", at: [-.33, .70],
   p: {eyeOpen: .32, blush: 1, brownLegs: 1, cup: 1, energy: .09, hunch: .34}},
  {key: "sleeping", label: "Sleeping", at: [.20, .83],
   p: {eyeOpen: 0, lie: 1, energy: .04, hunch: .5}},
  {key: "dreaming", label: "Dreaming", at: [.70, .71],
   p: {eyeOpen: 0, lie: 1, blanket: 1, heartTrail: 1, energy: .05, hunch: .5}}
];

const POSE_BY_KEY = {};
for (const pose of POSES) POSE_BY_KEY[pose.key] = pose;

// the machine's wardrobe per plant state -- mirrors serve.py POSE_STATE
const STATE_POOL = {
  CALM: ["neutral", "sitHeart", "lounging", "nerdy", "bashful", "kiss",
         "dancing", "star"],
  FUSS: ["gloomy", "sick", "cool", "shocked"],
  CRY: ["crying", "angry", "rage"],
  SLEEP: ["sleeping", "dreaming"],
};
let posePick = null;

// Resolved here rather than at parse time: POSES does not exist yet up there.
const pinnedPose = previewPose ? POSE_BY_KEY[previewPose] || null : null;

function fullParams(pose) {
  const out = Object.assign({}, BASE);
  for (const key in pose.p) out[key] = pose.p[key];
  return out;
}

// The Jetson's one distress number becomes a blend of the two *named* poses
// either side of it; sampling the wheel field instead lets stray poses leak in.
const LIVE_LADDER = [
  {at: 0, key: "sitHeart"}, {at: .22, key: "neutral"},
  // crying at .38, not .50: at .50 the FUSS band drew as neutral to the camera.
  {at: .38, key: "crying"}, {at: .78, key: "angry"}, {at: 1, key: "rage"}
];

function liveLook(level, asleep) {
  let lo = LIVE_LADDER[0], hi = LIVE_LADDER[LIVE_LADDER.length - 1];
  for (let i = 1; i < LIVE_LADDER.length; i++) {
    if (level <= LIVE_LADDER[i].at) {
      lo = LIVE_LADDER[i - 1];
      hi = LIVE_LADDER[i];
      break;
    }
  }
  const span = hi.at - lo.at;
  const t = span > 0 ? smooth01((level - lo.at) / span) : 0;
  const a = POSE_BY_KEY[lo.key], b = POSE_BY_KEY[hi.key];
  const s = POSE_BY_KEY.sleeping;
  const pa = fullParams(a), pb = fullParams(b), ps = fullParams(s);
  const params = {};
  for (const key of POSE_KEYS) {
    params[key] = lerp(lerp(pa[key], pb[key], t), ps[key], asleep);
  }
  return {
    params,
    at: {x: lerp(lerp(a.at[0], b.at[0], t), s.at[0], asleep),
         y: lerp(lerp(a.at[1], b.at[1], t), s.at[1], asleep)}
  };
}

// Blend reach; the falloff is steep so standing next to an anchor reads as it.
const POSE_REACH = .5;

function poseAt(x, y) {
  const out = Object.assign({}, BASE);
  let total = 0;
  const weights = POSES.map(pose => {
    const d = Math.hypot(x - pose.at[0], y - pose.at[1]);
    const w = d >= POSE_REACH ? 0 : Math.pow(1 - d / POSE_REACH, 4);
    total += w;
    return w;
  });
  if (total <= 0) {                       // nowhere near anything: nearest wins
    let best = 0, bestD = Infinity;
    POSES.forEach((pose, i) => {
      const d = Math.hypot(x - pose.at[0], y - pose.at[1]);
      if (d < bestD) { bestD = d; best = i; }
    });
    weights[best] = total = 1;
  }
  POSES.forEach((pose, i) => {
    const w = weights[i] / total;
    if (w <= 0) return;
    for (const key of POSE_KEYS) {
      const v = pose.p[key];
      out[key] += ((v === undefined ? BASE[key] : v) - BASE[key]) * w;
    }
  });
  return out;
}

function dominantPose(x, y) {
  let best = POSES[0], bestD = Infinity;
  for (const pose of POSES) {
    const d = Math.hypot(x - pose.at[0], y - pose.at[1]);
    if (d < bestD) { bestD = d; best = pose; }
  }
  return best;
}

// Frame-rate independent easing: `rate` is the fraction closed per second.
function ease(current, target, rate, dt) {
  return current + (target - current) * (1 - Math.exp(-rate * dt));
}

const R = Object.assign({}, BASE, {
  lookX: 0, lookY: 0, lookTx: 0, lookTy: 0, nextLook: 0,
  blinkUntil: -1, nextBlink: 1.4, blink: 0,
  lean: 0, bob: 0, breath: 0, feltX: 0, feltY: 0
});
const bits = [];   // tears, hearts and "z"s, one small pool
let seeded = 0;

/* --- path helpers ------------------------------------------------------- */

// Returns hex, not rgb(), so a mix can be fed straight back into another.
function mix(a, b, t) {
  const k = clamp(t, 0, 1);
  const pa = [1, 3, 5].map(i => parseInt(a.substr(i, 2), 16));
  const pb = [1, 3, 5].map(i => parseInt(b.substr(i, 2), 16));
  return "#" + pa.map((v, i) => Math.round(v + (pb[i] - v) * k)
                                   .toString(16).padStart(2, "0")).join("");
}

function roundRectPath(ctx, x, y, w, h, r) {
  const k = Math.min(r, w / 2, h / 2);
  ctx.moveTo(x + k, y);
  ctx.lineTo(x + w - k, y);
  ctx.arcTo(x + w, y, x + w, y + k, k);
  ctx.lineTo(x + w, y + h - k);
  ctx.arcTo(x + w, y + h, x + w - k, y + h, k);
  ctx.lineTo(x + k, y + h);
  ctx.arcTo(x, y + h, x, y + h - k, k);
  ctx.lineTo(x, y + k);
  ctx.arcTo(x, y, x + k, y, k);
  ctx.closePath();
}

function capsulePath(ctx, x0, y0, x1, y1, r) {
  const a = Math.atan2(y1 - y0, x1 - x0);
  ctx.moveTo(x0 + r * Math.cos(a + Math.PI / 2), y0 + r * Math.sin(a + Math.PI / 2));
  ctx.arc(x0, y0, r, a + Math.PI / 2, a + Math.PI * 1.5);
  ctx.arc(x1, y1, r, a - Math.PI / 2, a + Math.PI / 2);
  ctx.closePath();
}

function heartPath(ctx, cx, cy, s) {
  ctx.beginPath();
  ctx.moveTo(cx, cy + s * .95);
  ctx.bezierCurveTo(cx - s * 1.38, cy - s * .16, cx - s * .74, cy - s * 1.18, cx, cy - s * .30);
  ctx.bezierCurveTo(cx + s * .74, cy - s * 1.18, cx + s * 1.38, cy - s * .16, cx, cy + s * .95);
  ctx.closePath();
}

function tearPath(ctx, x, y, r) {
  ctx.beginPath();
  ctx.moveTo(x, y - r * 1.9);
  ctx.bezierCurveTo(x + r * .96, y - r * .34, x + r, y + r * .56, x, y + r);
  ctx.bezierCurveTo(x - r, y + r * .56, x - r * .96, y - r * .34, x, y - r * 1.9);
  ctx.closePath();
}

function starPath(ctx, cx, cy, outer, inner, points, rot) {
  ctx.beginPath();
  for (let i = 0; i < points * 2; i++) {
    const a = rot + i * Math.PI / points;
    const rr = i % 2 ? inner : outer;
    const x = cx + Math.cos(a) * rr, y = cy + Math.sin(a) * rr;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.closePath();
}

function shinePath(ctx, r, spikes, phase) {
  const n = spikes * 2;
  const at = i => {
    const a = (i % n) / n * TAU + phase;
    const rr = i % 2 ? r * 1.13 : r * .94;   // odd indices are bezier controls
    return [Math.cos(a) * rr, Math.sin(a) * rr];
  };
  const first = at(0);
  ctx.beginPath();
  ctx.moveTo(first[0], first[1]);
  for (let k = 0; k < spikes; k++) {
    const c = at(2 * k + 1), p = at(2 * k + 2);
    ctx.quadraticCurveTo(c[0], c[1], p[0], p[1]);
  }
  ctx.closePath();
}

// The flat-colour look the recognizer matches on: one ink outline around the
// *union* of the subpaths -- stroke at 2x ink, then refill, erases the seams.
function inkFill(ctx, trace, fill) {
  ctx.beginPath();
  trace(ctx);
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.lineWidth = ART.ink * 2;
  ctx.strokeStyle = INK;
  ctx.stroke();
  ctx.fillStyle = INK;
  ctx.fill();
  ctx.beginPath();
  trace(ctx);
  ctx.fillStyle = fill;
  ctx.fill();
}

/* --- the figure --------------------------------------------------------- */

function handOf(side, lift, wobble) {
  const angle = (-5 + 52 * lift + wobble) * DEG;
  const len = (ART.armLen - 9 * lift) * (1 - .34 * R.lie);
  const out = {
    x: side * (ART.shoulderX + len * Math.cos(angle)),
    y: ART.shoulderY - len * Math.sin(angle)
  };
  out.x = lerp(out.x, side * 11, R.armsIn);
  out.y = lerp(out.y, 52, R.armsIn);
  out.x = lerp(out.x, -side * 18, R.armsCross);
  out.y = lerp(out.y, 84, R.armsCross);
  return out;
}

function armTrace(side, P) {
  const hand = side < 0 ? P.handL : P.handR;
  return ctx => capsulePath(ctx, side * ART.shoulderX, ART.shoulderY,
                            hand.x, hand.y, ART.armW / 2);
}

function armsTrace(P) {
  return ctx => { armTrace(-1, P)(ctx); armTrace(1, P)(ctx); };
}

function legsTrace(P) {
  return ctx => {
    capsulePath(ctx, -ART.legDx, P.legTop, -ART.legDx + P.kick, P.legBot,
                ART.legW / 2);
    capsulePath(ctx, ART.legDx, P.legTop, ART.legDx - P.kick, P.legBot,
                ART.legW / 2);
  };
}

function bodyTrace(P) {
  return ctx => {
    roundRectPath(ctx, -ART.torsoHw, ART.torsoTop, ART.torsoHw * 2,
                  P.torsoBot - ART.torsoTop, ART.torsoR);
    legsTrace(P)(ctx);
    armsTrace(P)(ctx);
  };
}

function applyHead(ctx, P) {
  ctx.translate(P.headX, P.headY);
  ctx.rotate(P.headTilt * DEG);
  ctx.scale(P.headSx, P.headSy);
}

function headTrace(ctx) {
  ctx.ellipse(0, 0, ART.headRx, ART.headRy, 0, 0, TAU);
}

// A full stroke: clipping the front arm *on* the head outline stripes the face.
const CLIP_PAD = ART.ink * 2;

function headDepth(x, y, P) {
  return Math.hypot((x - P.headX) / (ART.headRx + CLIP_PAD),
                    (y - P.headY) / (ART.headRy + CLIP_PAD));
}

// The stretch of an arm that belongs in *front* of the head: hand back to
// where its centreline leaves the head, plus overrun to park the end cap
// outside the clip.  A whole arm redrawn in front leaves a stray rim line.
function frontArm(side, P) {
  const shoulderX = side * ART.shoulderX;
  const hand = side < 0 ? P.handL : P.handR;
  if (headDepth(hand.x, hand.y, P) >= 1) return null;
  const reach = Math.max(1, Math.hypot(shoulderX - hand.x,
                                       ART.shoulderY - hand.y));
  let t = 0;
  while (t < 1 && headDepth(lerp(hand.x, shoulderX, t),
                            lerp(hand.y, ART.shoulderY, t), P) < 1) t += 1 / 32;
  const back = Math.min(1, t + (ART.armW / 2 + 4 * ART.ink) / reach);
  return ctx => capsulePath(ctx, hand.x, hand.y,
                            lerp(hand.x, shoulderX, back),
                            lerp(hand.y, ART.shoulderY, back), ART.armW / 2);
}

/* --- the face ----------------------------------------------------------- */

// The mood wash, painted over the white head into whatever region is clipped
// (so a lid can put it back over itself): rainbow, flush pink, dusk purple.
function paintWashes(ctx) {
  if (R.rainbow > .01) {
    const bow = ctx.createLinearGradient(-ART.headRx, 0, ART.headRx, 0);
    const stops = ["#e8402a", "#f6a12a", "#f2e33a", "#4ec04e", "#3aa7e0",
                   "#8f52c8", "#e2007e"];
    stops.forEach((c, i) => bow.addColorStop(i / (stops.length - 1), c));
    ctx.globalAlpha = R.rainbow;
    ctx.fillStyle = bow;
    ctx.fillRect(-ART.headRx, -ART.headRy, ART.headRx * 2, ART.headRy * 2);
    ctx.globalAlpha = 1;
  }
  // The flush gives way as rage reddens the skin; stacked, the two grey out.
  const flush = R.flush * (1 - .75 * R.rage);
  if (flush > .01) {
    const wash = ctx.createLinearGradient(0, ART.headRy * .1, 0, ART.headRy);
    wash.addColorStop(0, "rgba(226,0,126,0)");
    wash.addColorStop(1, `rgba(226,0,126,${(.80 * flush).toFixed(3)})`);
    ctx.fillStyle = wash;
    ctx.fillRect(-ART.headRx, -ART.headRy, ART.headRx * 2, ART.headRy * 2);
  }
  if (R.dusk > .01) {
    const dusk = ctx.createLinearGradient(0, -ART.headRy, 0, ART.headRy * .2);
    dusk.addColorStop(0, `rgba(92,62,164,${(.80 * R.dusk).toFixed(3)})`);
    dusk.addColorStop(1, "rgba(92,62,164,0)");
    ctx.fillStyle = dusk;
    ctx.fillRect(-ART.headRx, -ART.headRy, ART.headRx * 2, ART.headRy * 2);
  }
  const cheeks = [[R.blush, "255,116,158"], [R.blueBlush, "72,132,226"]];
  for (const [amount, rgb] of cheeks) {
    if (amount <= .02) continue;
    for (const side of [-1, 1]) {
      // Squash the context, not the arc, so the gradient squashes with it.
      ctx.save();
      ctx.translate(side * ART.eyeDx, ART.eyeDy + ART.eyeR * 1.15);
      ctx.scale(1, .58);
      const cheek = ctx.createRadialGradient(0, 0, 0, 0, 0, ART.eyeR * 1.5);
      cheek.addColorStop(0, `rgba(${rgb},${(.95 * amount).toFixed(3)})`);
      cheek.addColorStop(1, `rgba(${rgb},0)`);
      ctx.fillStyle = cheek;
      ctx.beginPath();
      ctx.arc(0, 0, ART.eyeR * 1.5, 0, TAU);
      ctx.fill();
      ctx.restore();
    }
  }
}

function drawEye(ctx, side, P) {
  const ex = side * ART.eyeDx, ey = ART.eyeDy;
  const open = clamp(R.eyeOpen * (1 - R.blink), 0, 1);
  const r = ART.eyeR * (1 + .10 * R.shine + .90 * R.eyeBig);
  const pupil = ART.pupilR * (1 + .22 * R.shine + 1.05 * R.eyeBig);

  if (R.shine > .02) {
    ctx.save();
    ctx.translate(ex, ey);
    ctx.globalAlpha = .90 * R.shine;
    ctx.fillStyle = "#fff";
    shinePath(ctx, r, 9, P.t * .55 + side);
    ctx.fill();
    ctx.restore();
  }

  ctx.fillStyle = "#fff";
  ctx.beginPath();
  ctx.arc(ex, ey, r, 0, TAU);
  ctx.fill();

  const px = ex + R.lookX * r * .30;
  const py = ey + R.lookY * r * .30 + R.tear * 2.2;
  ctx.fillStyle = INK;
  ctx.beginPath();
  ctx.arc(px, py, pupil, 0, TAU);
  ctx.fill();

  ctx.fillStyle = "rgba(255,255,255,.94)";
  ctx.beginPath();
  ctx.arc(px - r * .17, py - r * .21, pupil * (.20 + .22 * R.shine), 0, TAU);
  ctx.fill();

  // The lid is head-coloured, so a closing eye loses its white; it repaints the
  // wash, or flat blue would scrub the flush off the surrounding head.
  const lidY = -r - 2 + (1 - open) * (2 * r + 4);
  ctx.save();
  ctx.beginPath();
  ctx.arc(ex, ey, r * 1.36 + 2, 0, TAU);
  ctx.clip();
  ctx.save();
  ctx.translate(ex, ey);
  ctx.rotate(side * R.brow * 19 * DEG);
  ctx.beginPath();
  ctx.rect(-r * 2.6, lidY - r * 4, r * 5.2, r * 4);
  ctx.restore();
  ctx.clip();
  ctx.fillStyle = P.skin;
  ctx.fillRect(ex - r * 3, ey - r * 5, r * 6, r * 10);
  paintWashes(ctx);
  ctx.restore();

  // ...and the lash line fades in behind it, so a blink still reads at speed.
  if (open < .5) {
    ctx.save();
    ctx.translate(ex, ey);
    ctx.globalAlpha = 1 - open * 2;
    ctx.lineCap = "butt";
    ctx.lineWidth = 5.0;
    for (const [style, from, to] of [["#fff", -1.00, -.72], [INK, -.76, .76],
                                     ["#fff", .72, 1.00]]) {
      ctx.strokeStyle = style;
      ctx.beginPath();
      ctx.moveTo(r * .95 * from, -2.2 * from);
      ctx.lineTo(r * .95 * to, -2.2 * to);
      ctx.stroke();
    }
    ctx.restore();
  }
}

function drawEyewear(ctx) {
  if (R.shades > .02) {
    ctx.save();
    ctx.globalAlpha = clamp(R.shades * 1.5, 0, 1);
    ctx.fillStyle = INK;
    ctx.strokeStyle = INK;
    ctx.lineWidth = 3.4;
    ctx.lineCap = "round";
    for (const side of [-1, 1]) {
      ctx.beginPath();
      ctx.ellipse(side * ART.eyeDx, ART.eyeDy + 1, ART.eyeR * 1.5,
                  ART.eyeR * .92, side * .06, 0, TAU);
      ctx.fill();
      // temple arm, hooking back toward the head's edge
      ctx.beginPath();
      ctx.moveTo(side * (ART.eyeDx + ART.eyeR * 1.4), ART.eyeDy - 4);
      ctx.quadraticCurveTo(side * (ART.eyeDx + ART.eyeR * 2.3), ART.eyeDy - 14,
                           side * (ART.eyeDx + ART.eyeR * 2.6), ART.eyeDy - 8);
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.moveTo(-ART.eyeDx + ART.eyeR * 1.3, ART.eyeDy);
    ctx.lineTo(ART.eyeDx - ART.eyeR * 1.3, ART.eyeDy);
    ctx.lineWidth = 5;
    ctx.stroke();
    ctx.restore();
  }
  if (R.specs > .02) {
    ctx.save();
    ctx.globalAlpha = clamp(R.specs * 1.5, 0, 1);
    ctx.strokeStyle = INK;
    ctx.lineWidth = 2.6;
    for (const side of [-1, 1]) {
      ctx.beginPath();
      ctx.arc(side * ART.eyeDx, ART.eyeDy, ART.eyeR * 1.18, 0, TAU);
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.moveTo(-ART.eyeDx + ART.eyeR * 1.18, ART.eyeDy);
    ctx.lineTo(ART.eyeDx - ART.eyeR * 1.18, ART.eyeDy);
    ctx.stroke();
    ctx.restore();
  }
}

function drawFace(ctx, P) {
  // Clipped just inside the head, so lids and washes never reach the outline.
  ctx.save();
  ctx.beginPath();
  ctx.ellipse(0, 0, ART.headRx - 1, ART.headRy - 1, 0, 0, TAU);
  ctx.clip();
  paintWashes(ctx);
  if (R.eyesOff < .98) {
    ctx.globalAlpha = 1 - R.eyesOff;
    drawEye(ctx, -1, P);
    drawEye(ctx, 1, P);
    drawEyewear(ctx);
    ctx.globalAlpha = 1;
  }
  ctx.restore();
}

/* --- props -------------------------------------------------------------- */

function drawHeadProps(ctx, P) {
  if (R.bow > .02) {
    ctx.save();
    ctx.globalAlpha = clamp(R.bow * 1.6, 0, 1);
    ctx.translate(-ART.headRx * .60, -ART.headRy * .62);
    ctx.rotate(-.35 + Math.sin(P.t * 2.2) * .05);
    ctx.lineJoin = "round";
    ctx.lineWidth = ART.ink * 1.6;
    ctx.strokeStyle = INK;
    ctx.beginPath();
    for (const side of [-1, 1]) {
      ctx.ellipse(side * 12, 0, 12, 9, side * .5, 0, TAU);
    }
    ctx.stroke();
    ctx.fillStyle = PINK;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(0, 0, 5, 0, TAU);
    ctx.fillStyle = PINK;
    ctx.fill();
    ctx.restore();
  }
  if (R.stars > .02) {
    ctx.save();
    ctx.globalAlpha = clamp(R.stars * 1.5, 0, 1);
    ctx.lineJoin = "round";
    ctx.lineWidth = ART.ink * 1.4;
    ctx.strokeStyle = INK;
    const spin = Math.sin(P.t * 1.3) * .12;
    starPath(ctx, ART.headRx * .95, -ART.headRy * .35, 26, 11, 5, -1.2 + spin);
    ctx.fillStyle = "#F5C518";
    ctx.stroke();
    ctx.fill();
    for (const [sx, sy, s, col] of [[-.28, -.30, 13, "#fff"],
                                    [.16, -.10, 10, "#fff"],
                                    [-.62, .05, 9, "#e2007e"]]) {
      starPath(ctx, ART.headRx * sx, ART.headRy * sy, s, s * .42, 5, .3 + spin);
      ctx.fillStyle = col;
      ctx.fill();
    }
    ctx.restore();
  }
}

function drawHeldHeart(ctx, P) {
  // Put away by shrinking, not fading: ink at half alpha over blue reads as dirt.
  if (R.heart <= .12) return;
  ctx.save();
  ctx.globalAlpha = clamp((R.heart - .12) * 4, 0, 1);
  ctx.translate(P.handL.x - 12, P.handL.y - 12);
  ctx.rotate(-18 * DEG + Math.sin(P.t * 1.7) * .10);
  const pulse = (.45 + .55 * R.heart) * (1 + .07 * Math.sin(P.t * 3.1));
  ctx.lineJoin = "round";
  ctx.lineWidth = ART.ink * 2;
  ctx.strokeStyle = INK;
  heartPath(ctx, 0, 0, 21 * pulse);
  ctx.stroke();
  ctx.fillStyle = PINK;
  ctx.fill();
  ctx.restore();
}

function drawBlanket(ctx, P) {
  if (R.blanket <= .02) return;
  ctx.save();
  ctx.globalAlpha = clamp(R.blanket * 1.4, 0, 1);
  ctx.lineJoin = "round";
  ctx.lineWidth = ART.ink * 1.7;
  ctx.strokeStyle = INK;
  ctx.beginPath();
  ctx.moveTo(6, 96);
  ctx.quadraticCurveTo(74, 74, 128, 116);
  ctx.quadraticCurveTo(96, 142, 44, 138);
  ctx.quadraticCurveTo(14, 132, 6, 96);
  ctx.closePath();
  ctx.stroke();
  ctx.fillStyle = "#E7A0C8";
  ctx.fill();
  ctx.restore();
}

function drawNotes(ctx, P) {
  if (R.notes <= .02) return;
  ctx.save();
  ctx.globalAlpha = clamp(R.notes * 1.5, 0, 1);
  ctx.fillStyle = PINK;
  ctx.strokeStyle = PINK;
  ctx.lineCap = "round";
  const float = Math.sin(P.t * 1.9) * 5;
  for (const [nx, ny, s, twin] of [[104, 34 + float, 1, false],
                                   [132, 62 - float, .85, true]]) {
    ctx.save();
    ctx.translate(nx, ny);
    ctx.scale(s, s);
    ctx.beginPath();
    ctx.ellipse(0, 0, 7, 5.4, -.35, 0, TAU);
    ctx.fill();
    ctx.lineWidth = 2.8;
    ctx.beginPath();
    ctx.moveTo(6.2, -1.6);
    ctx.lineTo(6.2, -22);
    if (twin) {
      ctx.moveTo(6.2, -22);
      ctx.lineTo(20, -18);
      ctx.moveTo(20, -18);
      ctx.lineTo(20, -3);
      ctx.stroke();
      ctx.beginPath();
      ctx.ellipse(13.8, -1.4, 7, 5.4, -.35, 0, TAU);
      ctx.fill();
    } else {
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(6.2, -22);
      ctx.quadraticCurveTo(15, -19, 13, -10);
      ctx.lineWidth = 3.2;
      ctx.stroke();
    }
    ctx.restore();
  }
  ctx.restore();
}

// On the floor rather than on the infant, so a lying pose does not tip them.
function drawGroundProps(ctx, P) {
  if (R.cup > .02) {
    ctx.save();
    ctx.globalAlpha = clamp(R.cup * 1.5, 0, 1);
    ctx.lineJoin = "round";
    ctx.lineWidth = 2.6;
    ctx.strokeStyle = INK;
    ctx.beginPath();
    roundRectPath(ctx, 54, 104, 30, 46, 3);
    ctx.fillStyle = "#fff";
    ctx.fill();
    ctx.stroke();
    ctx.lineWidth = 2;
    for (let i = 0; i < 4; i++) {
      ctx.beginPath();
      ctx.moveTo(58, 111 + i * 7);
      ctx.lineTo(80, 111 + i * 7);
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.arc(69, 141, 7, 0, TAU);
    ctx.fillStyle = "#1E8C4A";
    ctx.fill();
    ctx.restore();
  }
  const fallen = Math.max(R.groundHeart, R.blackHeart);
  if (fallen > .02) {
    ctx.save();
    ctx.globalAlpha = clamp(fallen * 1.5, 0, 1);
    ctx.translate(-88, 138);
    ctx.rotate(-.25);
    ctx.lineJoin = "round";
    ctx.lineWidth = ART.ink * 1.6;
    ctx.strokeStyle = INK;
    heartPath(ctx, 0, 0, 17);
    ctx.stroke();
    ctx.fillStyle = mix(PINK, "#100C0B", clamp(R.blackHeart / Math.max(fallen, .001), 0, 1));
    ctx.fill();
    ctx.restore();
  }
}

function drawParticles(ctx) {
  for (const bit of bits) {
    const life = bit.age / bit.life;
    ctx.save();
    ctx.globalAlpha = clamp((1 - life) * 3.4, 0, 1) * bit.alpha;
    ctx.translate(bit.x, bit.y);
    if (bit.kind === "tear") {
      ctx.lineWidth = 2.2;
      ctx.strokeStyle = INK;
      ctx.fillStyle = "#9BDCF7";
      tearPath(ctx, 0, 0, bit.size);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "rgba(255,255,255,.85)";
      ctx.beginPath();
      ctx.arc(-bit.size * .28, -bit.size * .1, bit.size * .28, 0, TAU);
      ctx.fill();
    } else if (bit.kind === "heart") {
      ctx.rotate(bit.rot);
      ctx.fillStyle = bit.tint || PINK;
      heartPath(ctx, 0, 0, bit.size);
      ctx.fill();
    } else {
      ctx.rotate(bit.rot);
      ctx.font = `700 ${Math.round(bit.size)}px "Helvetica Neue",Helvetica,Arial,sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = INK;
      ctx.fillText("z", 0, 0);
    }
    ctx.restore();
  }
}

function drawWordmark(ctx) {
  ctx.save();
  ctx.translate(0, ART.torsoTop + 43);
  ctx.transform(1, 0, -.17, 1, 0, 0);           // the wordmark's oblique
  ctx.font = '700 17px "Helvetica Neue",Helvetica,Arial,sans-serif';
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const width = ctx.measureText("KAIST").width || 44;
  ctx.scale(44 / width, 1);
  ctx.fillStyle = "#fff";
  ctx.fillText("KAIST", 0, 0);
  ctx.restore();
}

function drawNubzuki(ctx, P) {
  ctx.save();
  ctx.translate(0, ART.groundY);
  ctx.rotate(R.lean * DEG);
  ctx.translate(0, -ART.groundY + P.bounce);

  if (R.rage > .02) {
    const glow = ctx.createRadialGradient(0, 40, 10, 0, 40, 200);
    glow.addColorStop(0, `rgba(255,138,20,${(.62 * R.rage).toFixed(3)})`);
    glow.addColorStop(.55, `rgba(255,96,10,${(.30 * R.rage).toFixed(3)})`);
    glow.addColorStop(1, "rgba(255,96,10,0)");
    ctx.fillStyle = glow;
    ctx.fillRect(-220, -180, 440, 400);
  }

  const contact = clamp(1 + P.bounce / 6, .55, 1);
  const puddle = mix(mix("#EDEBEA", "#E2007E", clamp(R.flush * 1.1, 0, 1)),
                     "#100C0B", R.rage);
  ctx.globalAlpha = clamp(.16 + .74 * Math.max(R.rage, R.flush * .5), 0, .92);
  ctx.fillStyle = puddle;
  ctx.beginPath();
  ctx.ellipse(P.lieShift - 46 * R.lie, ART.groundY - P.hunchDrop * .5,
              ART.shadowRx * contact * (1 + .55 * R.lie),
              ART.shadowRy * contact, 0, 0, TAU);
  ctx.fill();
  ctx.globalAlpha = 1;

  drawGroundProps(ctx, P);

  ctx.save();
  // Lying tips the figure about its floor contact; the head counter-rotates.
  ctx.translate(P.lieShift, ART.groundY);
  ctx.rotate(-P.lieRot * DEG);
  ctx.translate(0, -ART.groundY + P.bodyY);

  inkFill(ctx, bodyTrace(P), P.skin);

  if (R.brownLegs > .02) {
    ctx.save();
    ctx.globalAlpha = clamp(R.brownLegs * 1.3, 0, 1);
    const hem = ctx.createLinearGradient(0, P.legTop - 22, 0, P.legTop + 12);
    hem.addColorStop(0, "rgba(122,75,36,0)");
    hem.addColorStop(1, BROWN);
    ctx.fillStyle = hem;
    ctx.beginPath();
    roundRectPath(ctx, -ART.torsoHw, P.legTop - 24, ART.torsoHw * 2,
                  P.torsoBot - P.legTop + 24, 14);
    legsTrace(P)(ctx);
    ctx.fill();
    ctx.restore();
  }

  ctx.save();
  applyHead(ctx, P);
  inkFill(ctx, headTrace, P.skin);
  drawFace(ctx, P);
  drawHeadProps(ctx, P);
  ctx.restore();

  // A raised arm passes in *front* of the head, spliced onto the arm behind.
  const front = [-1, 1].map(side => frontArm(side, P)).filter(Boolean);
  if (front.length) {
    ctx.save();
    ctx.save();
    applyHead(ctx, P);
    ctx.beginPath();
    ctx.ellipse(0, 0, ART.headRx + CLIP_PAD, ART.headRy + CLIP_PAD, 0, 0, TAU);
    ctx.restore();
    ctx.clip();
    inkFill(ctx, c => front.forEach(trace => trace(c)), P.skin);
    ctx.restore();
  }

  ctx.globalAlpha = .38;
  ctx.strokeStyle = INK;
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  ctx.ellipse(0, ART.torsoTop + 1, 26, 9, 0, .12 * Math.PI, .88 * Math.PI);
  ctx.stroke();
  ctx.globalAlpha = 1;

  drawWordmark(ctx);
  drawBlanket(ctx, P);
  drawHeldHeart(ctx, P);
  ctx.restore();

  drawNotes(ctx, P);
  drawParticles(ctx);
  ctx.restore();
}

/* --- the feelings wheel -------------------------------------------------- */

// The sheet's circumplex as readout and control: the dot is where the infant
// sits now, the tail its last 45 s (WHEEL_KEEP / WHEEL_HZ), a finger poses it.
const WHEEL_HZ = 2;
const WHEEL_KEEP = 90;
const wheelTrail = [];
let wheelNext = 0;
let wheelSeeded = false;

// Roughly the inverse of the pose blend, for the caption's number only.  Sleep
// is gated on positive valence so quiet-but-miserable stays awake and fussing.
function feltToState(x, y) {
  const arousal = -y;
  return {
    level: clamp(.55 * (1 - x) / 2 + .55 * (arousal + .32) / 1.27, 0, 1),
    asleep: smooth01((-arousal - .5) / .4) * smooth01((x + .2) / .5)
  };
}

function moodRgb() {
  if (R.rainbow > .4) return "232,150,32";
  if (R.rage > .35) return "196,22,28";
  if (R.dusk > .4) return "92,62,164";
  if (R.flush > .3 || R.tear > .3) return "226,0,126";
  return "60,169,225";
}

function wheelGeom(w, h, dpr) {
  const r = clamp(Math.min(w, h) * .12, 44 * dpr, 96 * dpr);
  const pad = 20 * dpr;
  return {cx: w - r - pad, cy: h - r - pad, r, reach: r * .84};
}

// The tap target that hands the wheel back to the Jetson, shown only while held.
function livePill(geom, dpr) {
  const pw = 58 * dpr, ph = 22 * dpr;
  return {x: geom.cx - pw / 2, y: geom.cy - geom.r - ph - 8 * dpr, w: pw, h: ph};
}

function drawWheel(ctx, w, h, dpr, rgb, held) {
  const {cx, cy, r} = wheelGeom(w, h, dpr);

  ctx.save();
  ctx.translate(cx, cy);

  ctx.fillStyle = held ? "rgba(255,255,255,.90)" : "rgba(255,255,255,.74)";
  ctx.beginPath();
  ctx.arc(0, 0, r, 0, TAU);
  ctx.fill();
  ctx.strokeStyle = held ? `rgba(${rgb},.85)` : "rgba(51,37,29,.28)";
  ctx.lineWidth = Math.max(1, (held ? 2 : 1.1) * dpr);
  ctx.stroke();

  ctx.strokeStyle = "rgba(51,37,29,.16)";
  ctx.setLineDash([3 * dpr, 4 * dpr]);
  ctx.beginPath();
  ctx.moveTo(-r, 0); ctx.lineTo(r, 0);
  ctx.moveTo(0, -r); ctx.lineTo(0, r);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = "rgba(51,37,29,.56)";
  ctx.font = `600 ${Math.round(8.5 * dpr)}px system-ui,-apple-system,sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  if ("letterSpacing" in ctx) ctx.letterSpacing = `${.09 * dpr}px`;
  const inset = r - 10 * dpr;
  ctx.fillText("ACTIVE", 0, -inset);
  ctx.fillText("CALM", 0, inset);
  for (const [label, side] of [["NEGATIVE", -1], ["POSITIVE", 1]]) {
    ctx.save();
    ctx.translate(side * inset, 0);
    ctx.rotate(side * Math.PI / 2);
    ctx.fillText(label, 0, 0);
    ctx.restore();
  }
  if ("letterSpacing" in ctx) ctx.letterSpacing = "0px";

  const reach = r * .84;
  // Every pose the sheet draws, marked where it sits.
  ctx.fillStyle = "rgba(51,37,29,.20)";
  for (const pose of POSES) {
    ctx.beginPath();
    ctx.arc(pose.at[0] * reach, pose.at[1] * reach, 1.6 * dpr, 0, TAU);
    ctx.fill();
  }

  ctx.lineCap = "round";
  ctx.lineWidth = 2 * dpr;
  for (let i = 1; i < wheelTrail.length; i++) {
    const a = wheelTrail[i - 1], b = wheelTrail[i];
    ctx.strokeStyle = `rgba(${rgb},${(.42 * i / wheelTrail.length).toFixed(3)})`;
    ctx.beginPath();
    ctx.moveTo(a.x * reach, a.y * reach);
    ctx.lineTo(b.x * reach, b.y * reach);
    ctx.stroke();
  }

  const dx = R.feltX * reach, dy = R.feltY * reach;
  const halo = ctx.createRadialGradient(dx, dy, 0, dx, dy, 9 * dpr);
  halo.addColorStop(0, `rgba(${rgb},.34)`);
  halo.addColorStop(1, `rgba(${rgb},0)`);
  ctx.fillStyle = halo;
  ctx.beginPath();
  ctx.arc(dx, dy, 9 * dpr, 0, TAU);
  ctx.fill();
  ctx.fillStyle = `rgb(${rgb})`;
  ctx.beginPath();
  ctx.arc(dx, dy, 4 * dpr, 0, TAU);
  ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,.92)";
  ctx.lineWidth = 1.6 * dpr;
  ctx.stroke();

  ctx.restore();

  if (held) {
    const pill = livePill({cx, cy, r}, dpr);
    ctx.save();
    ctx.fillStyle = "rgba(51,37,29,.88)";
    ctx.beginPath();
    roundRectPath(ctx, pill.x, pill.y, pill.w, pill.h, pill.h / 2);
    ctx.fill();
    ctx.fillStyle = "#fffaf2";
    ctx.font = `600 ${Math.round(9.5 * dpr)}px system-ui,-apple-system,sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("GO LIVE", pill.x + pill.w / 2, pill.y + pill.h / 2);
    ctx.restore();
  }
}

/* --- driving the wheel by hand ------------------------------------------- */

// Overrides what the face shows.  Local: it never reaches the judge.
let manual = null;
let wheelDrag = false;

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * (canvas.width / Math.max(1, rect.width)),
    y: (event.clientY - rect.top) * (canvas.height / Math.max(1, rect.height))
  };
}

function holdWheel(px, py) {
  const dpr = clamp(window.devicePixelRatio || 1, 1, 3);
  const geom = wheelGeom(canvas.width, canvas.height, dpr);
  let x = (px - geom.cx) / geom.reach, y = (py - geom.cy) / geom.reach;
  const d = Math.hypot(x, y);
  if (d > 1) { x /= d; y /= d; }
  manual = Object.assign({x, y}, feltToState(x, y));
}

canvas.addEventListener("pointerdown", event => {
  const dpr = clamp(window.devicePixelRatio || 1, 1, 3);
  const geom = wheelGeom(canvas.width, canvas.height, dpr);
  const p = canvasPoint(event);
  if (manual) {
    const pill = livePill(geom, dpr);
    if (p.x >= pill.x && p.x <= pill.x + pill.w &&
        p.y >= pill.y && p.y <= pill.y + pill.h) {
      manual = null;
      event.preventDefault();
      return;
    }
  }
  if (Math.hypot(p.x - geom.cx, p.y - geom.cy) > geom.r) return;
  wheelDrag = true;
  // Capture keeps the drag alive past the wheel's edge, but throws on a dead id.
  try { canvas.setPointerCapture(event.pointerId); } catch (_) {}
  holdWheel(p.x, p.y);
  event.preventDefault();
});

canvas.addEventListener("pointermove", event => {
  if (!wheelDrag) return;
  const p = canvasPoint(event);
  holdWheel(p.x, p.y);
  event.preventDefault();
});

for (const type of ["pointerup", "pointercancel", "pointerleave"]) {
  canvas.addEventListener(type, () => { wheelDrag = false; });
}

/* --- the animation loop ------------------------------------------------- */

// The cradle's own motion in degrees: this device's IMU if any, else the server.
function cradleTilt(t) {
  if (samples.length >= 4) {
    const recent = samples.slice(-6);
    const ax = recent.reduce((s, p) => s + p.x, 0) / recent.length;
    const ay = recent.reduce((s, p) => s + p.y, 0) / recent.length;
    return {lean: clamp(-ax * 7, -10, 10), bob: clamp(ay * 3, -6, 6)};
  }
  const ipad = live?.ipad;
  if (ipad?.connected && ipad.strength > .02) {
    const hz = clamp(ipad.dominant_hz || .5, .1, 3);
    return {lean: 9 * ipad.strength * Math.sin(TAU * hz * t),
            bob: 3 * ipad.strength * Math.sin(TAU * hz * t * 2 + 1.1)};
  }
  return {lean: Math.sin(t / 5.2) * .9 + Math.sin(t / 3.1) * .6, bob: 0};
}

function spawn(kind, x, y, tint) {
  if (bits.length > 26) return;
  seeded = (seeded * 1664525 + 1013904223) % 4294967296;
  const rand = seeded / 4294967296;
  if (kind === "tear") {
    bits.push({kind, x, y, vx: (rand - .5) * 12, vy: 30 + rand * 20,
               ay: 240, life: .78, age: 0, size: 4.4 + rand * 1.8, rot: 0, alpha: 1});
  } else if (kind === "heart") {
    bits.push({kind, x, y, vx: -8 - rand * 10, vy: -34 - rand * 14,
               ay: 0, life: 2.0, age: 0, size: 6 + rand * 4,
               rot: (rand - .5) * .8, alpha: .85, tint});
  } else {
    bits.push({kind, x, y, vx: 14 + rand * 10, vy: -22 - rand * 8,
               ay: 0, life: 2.6, age: 0, size: 20 + rand * 14,
               rot: -.25 + rand * .3, alpha: .55});
  }
}

// Where a point on the figure ends up once a lying pose has tipped it.
function lieXform(x, y, P) {
  const a = -P.lieRot * DEG;
  const dy = y + P.bodyY - ART.groundY;
  return {
    x: P.lieShift + x * Math.cos(a) - dy * Math.sin(a),
    y: ART.groundY + x * Math.sin(a) + dy * Math.cos(a)
  };
}

let lastFrame = 0;

function frame(nowMs) {
  const t = nowMs / 1000;
  const dt = clamp(lastFrame ? t - lastFrame : .016, .001, .05);
  lastFrame = t;

  const dpr = clamp(window.devicePixelRatio || 1, 1, 3);
  // visualViewport, not innerWidth/innerHeight: the layout viewport extends past
  // the glass, and a figure clipped there can never match a whole template.
  const vv = window.visualViewport;
  const vw = vv ? vv.width : innerWidth;
  const vh = vv ? vv.height : innerHeight;
  const w = Math.round(vw * dpr), h = Math.round(vh * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }

  // A hand on the wheel outranks the preview hook, which outranks the Jetson.
  const held = manual;
  // rough handling outranks the Jetson, but never a hand on the wheel
  const rough = shakeUpdate();
  // live.face is the plant's ground truth; the DRAWN infant is what the camera
  // reads back to close the loop.
  const jetLevel = live?.face?.level ?? live?.tag?.level ?? 0;
  const jetState = live?.face?.state ?? live?.tag?.emotion ?? "CALM";
  const level = held ? held.level
              : hasPreview ? clamp(previewLevel, 0, 1)
              : clamp(Math.max(jetLevel, rough), 0, 1);
  const serverState = hasPreview
    ? (previewState || (level < .12 ? "CALM" : level < .45 ? "FUSS" : "CRY"))
    : jetState;
  const present = held || hasPreview || live?.tag?.present !== false;

  // Where on the wheel the infant is, and therefore which poses it is made of.
  let spot, want;
  if (held) {
    spot = {x: held.x, y: held.y};
    want = poseAt(spot.x, spot.y);
  } else if (pinnedPose) {
    // The anchor's own parameters, not poseAt(): a bench capture is the pose.
    spot = {x: pinnedPose.at[0], y: pinnedPose.at[1]};
    want = fullParams(pinnedPose);
  } else if (live?.face?.state && rough < .05) {
    // Sense mode draws PURE poses (the camera names whole poses) and rotates
    // inside the state's pool; serve.py's POSE_STATE mirrors the pools.  A pick
    // holds 8-14 s -- long enough for the camera's anti-flicker vote to lock on.
    const st = live.face.state;
    const pool = STATE_POOL[st] || ["neutral"];
    const nowS = performance.now() / 1000;
    if (!posePick || posePick.st !== st || nowS > posePick.until) {
      posePick = {st, key: pool[Math.floor(Math.random() * pool.length)],
                  until: nowS + 8 + Math.random() * 6};
    }
    const pure = POSE_BY_KEY[posePick.key];
    spot = {x: pure.at[0], y: pure.at[1]};
    want = fullParams(pure);
  } else {
    const look = liveLook(level, serverState === "SLEEP" ? 1 : 0);
    spot = look.at;
    want = look.params;
  }
  for (const key of POSE_KEYS) R[key] = ease(R[key], want[key], 2.4, dt);

  // Blinks and saccades: both quicken with distress, stop once the eyes shut.
  if (t > R.nextBlink && R.eyeOpen > .5) {
    R.blinkUntil = t + .17;
    R.nextBlink = t + 1.5 + Math.random() * 3.6 * (1 - .6 * R.energy);
  }
  R.blink = t < R.blinkUntil ? Math.sin((R.blinkUntil - t) / .17 * Math.PI) : 0;
  if (t > R.nextLook) {
    R.lookTx = (Math.random() - .5) * 1.7;
    R.lookTy = (Math.random() - .5) * 1.2 + R.energy * .5;
    R.nextLook = t + 1.1 + Math.random() * 2.4;
  }
  R.lookX = ease(R.lookX, clamp(R.lookTx, -1, 1), 7, dt);
  R.lookY = ease(R.lookY, clamp(R.lookTy, -1, 1), 7, dt);

  const cradle = cradleTilt(t);
  R.lean = ease(R.lean, cradle.lean, 6.5, dt);
  R.bob = ease(R.bob, cradle.bob, 6.5, dt);

  // The dot must not lag a finger, and starts on the real reading (no fake tail).
  if (!wheelSeeded || wheelDrag) {
    wheelSeeded = true;
    R.feltX = spot.x;
    R.feltY = spot.y;
  }
  R.feltX = ease(R.feltX, spot.x, 2.2, dt);
  R.feltY = ease(R.feltY, spot.y, 2.2, dt);
  if (t >= wheelNext) {
    wheelNext = t + 1 / WHEEL_HZ;
    wheelTrail.push({x: R.feltX, y: R.feltY});
    if (wheelTrail.length > WHEEL_KEEP) wheelTrail.shift();
  }

  // One clock drives squash, bob and arm swing, so the body moves as one.
  const period = lerp(3.8, 1.7, R.energy);
  R.breath += dt / period;
  const breath = Math.sin(R.breath * TAU);
  // Rates near 4 Hz, amplitude under two units: wider reads as a render fault.
  const tremble = R.energy * R.energy * 1.5;
  const shakeX = tremble * Math.sin(t * TAU * 3.7);
  const shakeY = tremble * Math.sin(t * TAU * 4.6 + 1.1);

  const hunchDrop = R.hunch * 15;
  const fold = Math.max(R.sit, R.lie);
  const wobbleL = Math.sin(R.breath * TAU + .4) * (2.2 + 6.5 * R.energy);
  const wobbleR = Math.sin(R.breath * TAU + 2.6) * (2.2 + 6.5 * R.energy);
  const liftL = Math.max(R.heart * .90, R.armUp * .85);
  const liftR = R.armUp * .85;

  const P = {
    t,
    // Red arrives decisively: a half-blended skin reads as a fault, not rage.
    skin: mix(mix(BLUE, RAGE_RED, smooth01((R.rage - .30) / .55)),
              PINK, R.rainbow * .85),
    bounce: R.bob - Math.abs(Math.sin(t * TAU * (.45 + .55 * R.energy)))
                    * 2.6 * R.energy,
    bodyY: hunchDrop * .45 + 20 * R.sit + 8 * R.lie,
    hunchDrop,
    lieRot: 60 * R.lie,
    lieShift: 62 * R.lie,
    torsoBot: ART.torsoBot - 6 * R.hunch - 8 * fold,
    legTop: ART.legTop - 6 * R.hunch - 10 * fold,
    legBot: ART.legBot - 15 * R.hunch - 26 * fold,
    kick: Math.sin(t * TAU * 1.1) * 4 * R.energy,
    headX: shakeX,
    headY: -hunchDrop * .55 + breath * 1.6 + shakeY,
    headTilt: R.lean * .55 + Math.sin(t / 4.3) * 2.2 + shakeX * .35
              + 44 * R.lie,
    headSx: 1 + .020 * breath,
    headSy: 1 - .026 * breath
  };
  P.handL = handOf(-1, liftL, wobbleL);
  P.handR = handOf(1, liftR, wobbleR);

  // Tears crying, hearts soothed, "z"s asleep -- spawned in the tipped frame.
  const headAt = lieXform(P.headX, P.headY, P);
  if (R.tear > .18 && Math.random() < R.tear * dt * 9) {
    const side = Math.random() < .5 ? -1 : 1;
    spawn("tear", headAt.x + side * (ART.eyeDx + 7),
          headAt.y + ART.eyeDy + ART.eyeR * 1.15);
  }
  const rising = Math.max(R.heart > .55 ? .55 : 0, R.heartTrail);
  if (rising > .3 && Math.random() < dt * 1.4 * rising) {
    const from = lieXform(-ART.headRx * .5, 30, P);
    spawn("heart", from.x - ART.headRx * .35 + (Math.random() - .5) * 16,
          from.y, R.heartTrail > .6 ? "#B07AD8" : PINK);
  }
  if (R.eyeOpen < .12 && R.lie > .4 && Math.random() < dt * .8) {
    spawn("z", headAt.x + ART.headRx * .1, headAt.y - ART.headRy - 14);
  }
  for (let i = bits.length - 1; i >= 0; i--) {
    const bit = bits[i];
    bit.age += dt;
    bit.vy += bit.ay * dt;
    bit.x += bit.vx * dt;
    bit.y += bit.vy * dt;
    bit.rot += (bit.kind === "z" ? .3 : .6) * dt * (bit.vx > 0 ? 1 : -1);
    if (bit.age >= bit.life) bits.splice(i, 1);
  }

  g.setTransform(1, 0, 0, 1, 0, 0);
  g.fillStyle = "#fff";
  g.fillRect(0, 0, w, h);

  // A page-wide wash the colour of the mood (moodRgb), readable across the room.
  const hue = moodRgb();
  const auraAlpha = lerp(.07, .14, clamp(Math.max(level, R.rage, R.dusk,
                                                  R.rainbow), 0, 1));
  const aura = g.createRadialGradient(w / 2, h * .46, 0, w / 2, h * .46,
                                      Math.max(w, h) * .58);
  aura.addColorStop(0, `rgba(${hue},${auraAlpha.toFixed(3)})`);
  aura.addColorStop(1, `rgba(${hue},0)`);
  g.fillStyle = aura;
  g.fillRect(0, 0, w, h);

  // The band *above* the setup card, measured rather than assumed: it changes
  // with orientation, safe-area inset and status-line length.
  const cardTop = setup ? setup.getBoundingClientRect().top * dpr : h;
  const band = clamp(cardTop - 8 * dpr, h * .45, h);
  // .96: the camera's accuracy is pixels on the figure; a live capture had 76.
  const scale = Math.min(w * .96 / BOX.w, band * .96 / BOX.h);
  // FACE_CENTRE_Y is a preference: at 96% of the band there is no room to bias.
  const half = BOX.h * scale / 2;
  const midY = clamp(band * FACE_CENTRE_Y, half, Math.max(half, band - half));
  g.translate(w / 2 - (BOX.x + BOX.w / 2) * scale,
              midY - (BOX.y + BOX.h / 2) * scale);
  g.scale(scale, scale);
  g.globalAlpha = present ? 1 : .12;
  drawNubzuki(g, P);
  g.globalAlpha = 1;

  g.setTransform(1, 0, 0, 1, 0, 0);
  if (!present) {
    g.fillStyle = "rgba(51,37,29,.55)";
    g.font = `${Math.round(28 * dpr)}px system-ui`;
    g.textAlign = "center";
    g.fillText("Face hidden", w / 2, h / 2);
  }
  drawWheel(g, w, h, dpr, hue, !!held);

  // One writer for the caption, so the next SSE frame cannot contradict it.
  if (held) {
    setCaption(dominantPose(held.x, held.y).label,
               level.toFixed(2) + " manual");
  } else if (pinnedPose) {
    setCaption(pinnedPose.label, pinnedPose.key + " pinned");
  } else if (hasPreview) {
    setCaption(stateWord(serverState, level), level.toFixed(2) + " preview");
  } else if (live) {
    setCaption(stateWord(serverState, level), level.toFixed(2));
  }
  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);

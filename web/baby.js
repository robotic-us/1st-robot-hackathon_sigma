"use strict";

const canvas = document.getElementById("face");
const g = canvas.getContext("2d");
const start = document.getElementById("start");
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

function stateWord(state, level) {
  if (state === "SLEEP") return "Sleeping";
  if (state === "CALM" || level < .12) return "Calm";
  if (state === "FUSS" || level < .45) return "Fussing";
  return level >= .72 ? "Crying hard" : "Crying";
}

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
  const state = live.tag?.emotion || "CALM";
  const level = live.tag?.level ?? 0;
  emotionText.textContent = stateWord(state, level);
  levelText.textContent = level.toFixed(2);
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
    await fetch("/motion-sensor", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(features()),
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

function drawFace(ms) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(innerWidth * dpr), h = Math.round(innerHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h;
  }
  const state = live?.tag?.emotion || "CALM";
  const level = clamp(live?.tag?.level ?? 0, 0, 1);
  const present = live?.tag?.present !== false;
  const t = ms / 1000;
  const cx = w * .5, cy = h * .48;
  const radius = Math.min(w, h) * (.30 + .012 * Math.sin(t * 2));
  g.clearRect(0, 0, w, h);
  g.fillStyle = level < .12 ? "#ead4b8" : level < .45 ? "#edc9aa" : "#eab6a2";
  g.fillRect(0, 0, w, h);
  if (!present) {
    g.fillStyle = "rgba(51,37,29,.55)";
    g.font = `${Math.round(28 * dpr)}px system-ui`;
    g.textAlign = "center";
    g.fillText("Face hidden", cx, cy);
    requestAnimationFrame(drawFace);
    return;
  }

  // A deliberately simple, high-contrast face. The emotional geometry is
  // continuous in distress rather than switching between prerecorded clips.
  g.fillStyle = "#efc6a4";
  g.beginPath(); g.arc(cx, cy, radius, 0, Math.PI * 2); g.fill();
  g.strokeStyle = "#553b31";
  g.lineCap = "round";
  const eyeY = cy - radius * .18;
  const blink = state === "SLEEP" || Math.sin(t * .72) > .985;
  for (const side of [-1, 1]) {
    const ex = cx + side * radius * .34;
    g.lineWidth = radius * .035;
    g.beginPath();
    if (blink) {
      g.moveTo(ex - radius * .10, eyeY);
      g.quadraticCurveTo(ex, eyeY + radius * .055, ex + radius * .10, eyeY);
    } else {
      g.ellipse(ex, eyeY, radius * .075, radius * (.095 + level * .02), 0, 0, Math.PI * 2);
    }
    g.stroke();
    // Brows tilt inward as distress rises.
    g.beginPath();
    const inner = ex - side * radius * .11;
    const outer = ex + side * radius * .11;
    g.moveTo(inner, eyeY - radius * (.16 + .08 * level));
    g.lineTo(outer, eyeY - radius * (.16 - .08 * level));
    g.stroke();
  }
  const mouthY = cy + radius * .30;
  g.lineWidth = radius * .045;
  g.beginPath();
  if (level < .12) {
    g.arc(cx, mouthY - radius * .08, radius * .18, .12 * Math.PI, .88 * Math.PI);
  } else if (level < .45) {
    g.moveTo(cx - radius * .17, mouthY);
    g.quadraticCurveTo(cx, mouthY - radius * .06, cx + radius * .17, mouthY);
  } else {
    const cry = radius * (.12 + .13 * level + .025 * Math.sin(t * 9));
    g.ellipse(cx, mouthY, cry * .72, cry, 0, 0, Math.PI * 2);
  }
  g.stroke();
  if (level >= .45) {
    g.fillStyle = "rgba(70,145,190,.72)";
    for (const side of [-1, 1]) {
      g.beginPath();
      g.ellipse(cx + side * radius * .34, eyeY + radius * .18,
                radius * .035, radius * (.09 + .07 * level), 0, 0, Math.PI * 2);
      g.fill();
    }
  }
  requestAnimationFrame(drawFace);
}
requestAnimationFrame(drawFace);

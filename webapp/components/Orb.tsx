"use client";

import { useEffect, useRef } from "react";
import type { CradleState } from "@/lib/types";
import { LEVER_MM, ROLE, SWAY_CAP_MM, clamp, sway } from "@/lib/words";

/**
 * The infant as something alive.
 *
 * The outline is a blob, not a circle -- a handful of sine lobes turning at
 * different speeds, which is enough to read as breathing tissue instead of a
 * spinning polygon.  Everything that animates is driven by state the machine
 * actually publishes: distress sets how far and how fast the outline wobbles,
 * the envelope sets how brightly it burns, a taper settles it, and the body
 * rides the real plate offset.  A lost face goes dashed and red.
 *
 * The one thing not measured is the breathing cadence: it is a visual pulse to
 * make the shape feel inhabited, NOT a respiration reading.  Do not let it
 * grow into one without a sensor behind it.
 *
 * When a DREAM slot is playing, a dashed ghost marks where the dream says the
 * plate should be.  Jam the cradle and the two part -- divergence made visible.
 *
 * It reads the live frame from a ref rather than props: this loop runs at 60
 * fps and must never be a reason for React to re-render the page.
 */

const LOBES = [
  { k: 2, a: 0.024, w: 0.31 },
  { k: 3, a: 0.030, w: -0.55 },
  { k: 5, a: 0.017, w: 0.80 },
  { k: 7, a: 0.010, w: -1.15 },
];

/** #rrggbb -> rgba(), for the glow gradients. */
function rgba(hex: string, a: number): string {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.replace(/./g, "$&$&") : h, 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

export default function Orb({
  latest,
}: {
  latest: React.RefObject<CradleState | null>;
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const g = cv.getContext("2d");
    if (!g) return;

    // The palette only changes when the colour scheme does, so read each token
    // once and drop the cache on a theme flip -- this loop asks for colours
    // sixty times a second and getComputedStyle is not free.
    let tokens: Record<string, string> = {};
    const cssv = (name: string) =>
      name in tokens
        ? tokens[name]
        : (tokens[name] = getComputedStyle(document.documentElement)
            .getPropertyValue(name)
            .trim());
    const stateColor = (st: string) => cssv(ROLE[st] ?? "--muted");
    const scheme = window.matchMedia("(prefers-color-scheme: dark)");
    const retint = () => { tokens = {}; };
    scheme.addEventListener("change", retint);

    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const view = { off: 0, ghost: 0, lvl: 0, env: 0 };
    let raf = 0;

    /** One closed outline: radius R, deformed by the lobes at time t. */
    const blob = (cx: number, cy: number, R: number, t: number,
                  amp: number, speed: number) => {
      const N = 96;
      g.beginPath();
      for (let i = 0; i <= N; i++) {
        const th = (i / N) * Math.PI * 2;
        let d = 0;
        for (const l of LOBES) d += l.a * Math.sin(l.k * th + l.w * speed * t);
        const rr = R * (1 + d * amp);
        const x = cx + rr * Math.cos(th);
        const y = cy + rr * Math.sin(th);
        if (i) g.lineTo(x, y); else g.moveTo(x, y);
      }
      g.closePath();
    };

    const draw = () => {
      raf = requestAnimationFrame(draw);
      const S = latest.current;
      if (!S || document.hidden) return;

      // back the canvas with real device pixels, so the edge stays crisp
      const dpr = window.devicePixelRatio || 1;
      const w = Math.round(cv.clientWidth * dpr);
      const h = Math.round(cv.clientHeight * dpr);
      if (!w || !h) return;
      if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }

      const c = S.cradle;
      const W = cv.width, H = cv.height, span = Math.min(W, H);
      const t = performance.now() / 1000;
      const mm = sway(c);
      view.lvl += ((S.tag.level ?? 0) - view.lvl) * 0.15;
      view.env += ((c.env || 0) - view.env) * 0.1;
      const lvl = view.lvl;

      // agitation: distress drives it, a taper calms it, lost face unsettles it
      const settle = c.tapering ? 0.45 : 1;
      const amp = still
        ? 0.2
        : (0.45 + 1.5 * lvl) * settle * (S.tag.present ? 1 : 1.6);
      const speed = (0.9 + 2.4 * lvl) * settle;
      const breath = still
        ? 0
        : 0.045 * Math.sin(2 * Math.PI * (0.8 + 1.0 * lvl) * t);
      const R = span * (0.15 + 0.18 * lvl) * (1 + breath);

      // Exaggerated like the RViz mirror -- true sway is ~10 mm and would be a
      // few pixels -- but never wide enough to push the body out of the panel.
      const pxmm = Math.min(
        span * 0.011,
        Math.max(0, W / 2 - R - 16 * dpr) / SWAY_CAP_MM,
      );
      view.off += (mm * pxmm - view.off) * 0.35;
      view.ghost +=
        ((S.playing ? S.playing.dream_theta * LEVER_MM : mm) * pxmm -
          view.ghost) * 0.35;

      const room = Math.max(0, W / 2 - R * 1.05 - 6 * dpr);
      const cy = H / 2;
      const cx = W / 2 + clamp(view.off, -room, room);
      const col = S.tag.present ? stateColor(S.tag.emotion) : cssv("--critical");
      g.clearRect(0, 0, W, H);

      // the dream's plate, drawn only once it visibly parts from the real one
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
      aura.addColorStop(0, rgba(col, 0.13 + 0.2 * view.env));
      aura.addColorStop(1, rgba(col, 0));
      g.fillStyle = aura;
      g.beginPath();
      g.arc(cx, cy, R * 1.8, 0, Math.PI * 2);
      g.fill();

      // body, an inner outline turning the other way for depth, then the edge
      blob(cx, cy, R, t, amp, speed);
      g.fillStyle = rgba(col, 0.07);
      g.fill();

      blob(cx, cy, R * 0.62, -t, amp * 1.35, speed);
      g.lineWidth = 1.5 * dpr;
      g.strokeStyle = rgba(col, 0.3);
      g.stroke();

      blob(cx, cy, R, t, amp, speed);
      g.lineWidth = 5 * dpr;
      g.strokeStyle = col;
      if (!S.tag.present) g.setLineDash([12 * dpr, 10 * dpr]);
      g.stroke();
      g.setLineDash([]);
    };

    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      scheme.removeEventListener("change", retint);
    };
  }, [latest]);

  return <canvas ref={ref} />;
}

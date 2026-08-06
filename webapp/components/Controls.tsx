"use client";

import { useEffect } from "react";
import type { CradleState } from "@/lib/types";
import { send } from "@/lib/useCradle";

/**
 * Manual override.  Each button says what it does; the M-number is the
 * footnote, not the label -- an operator should not have to hold the report
 * in their head to drive the cradle.  One line each, so the row stays a row.
 */
const MOTIONS = [
  { id: "M10", label: "Rock slowly" },
  { id: "M12", label: "Rock steadily" },
  { id: "M16", label: "Rock wider" },
  { id: "M05", label: "Slow to a stop" },
  { id: "M01", label: "Hold still" },
];

export function Camera() {
  return (
    <div>
      <h2>Camera</h2>
      {/* plain <img>: this is an MJPEG stream, not something to optimise */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img className="cam" src="/frame" alt="camera stream" />
    </div>
  );
}

export function Controls({ S }: { S: CradleState | null }) {
  // 'j' trips the gate from anywhere, same as the button
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "j") send("/jam"); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  const auto = S?.cradle.auto ?? false;
  const jam = S?.jam ?? false;

  return (
    <div>
      <h2>Take over by hand</h2>
      <div className="buttons">
        <button
          className={auto ? "autoon" : undefined}
          onClick={() => send(`/auto?set=${auto ? "off" : "on"}`)}
        >
          <b>Automatic care</b><i>{S ? (auto ? "on" : "off") : "–"}</i>
        </button>

        {MOTIONS.map((m) => (
          <button key={m.id} onClick={() => send(`/motion?id=${m.id}`)}>
            <b>{m.label}</b><i>{m.id}</i>
          </button>
        ))}

        <button className={jam ? "jam jammed" : "jam"} onClick={() => send("/jam")}>
          <b>{jam ? "Jammed — release" : "Simulate a jam"}</b><i>j</i>
        </button>

        <span className="note">
          Rocking always eases in gently, and the safety gate overrides anything
          pressed here.
        </span>
      </div>
    </div>
  );
}

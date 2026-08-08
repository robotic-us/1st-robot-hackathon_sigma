"use client";

import type { CradleState } from "@/lib/types";

/**
 * Everything a reader should not need.  The raw numbers live here so the
 * cards above can stay in plain English, and so nobody is tempted to put
 * engineering units back on the front page.  When serve.py runs the
 * verification scenario (--verify), the judge state and the scripted phase
 * show here too -- docs/VERIFICATION.md layer 1, watchable live.
 */
export default function Internals({ S }: { S: CradleState }) {
  const c = S.cradle;

  return (
    <details>
      <summary>Internals — machine state, and the scenario acting it</summary>
      <div className="internals">
        <section>
          <h2>Machine internals</h2>
          <div className="kv">
            <span>machine state</span>
            <b>
              {c.state}
              {c.auto ? "" : " · auto off"}
              {c.trial_s ? ` · ${c.trial_s}s` : ""}
            </b>
          </div>
          <div className="kv">
            <span>motion / grade</span>
            <b>{c.motion ?? "M01"} {c.name} · {c.grade}</b>
          </div>
          <div className="kv">
            <span>f_hz · a_mm · env</span>
            <b>{c.f_hz} Hz · {c.a_mm.toFixed(1)} mm · env {c.env.toFixed(3)}</b>
          </div>
          <div className="kv">
            <span>distress raw · ema</span>
            <b>{(S.tag.level ?? 0).toFixed(2)} · {(c.ema ?? 0).toFixed(2)}</b>
          </div>
          <div className="kv">
            <span>arm angle θ</span>
            <b>{S.pose[0].toFixed(4)} rad</b>
          </div>
          <div className="kv">
            <span>peak acceleration</span>
            <b>{c.a_peak_g.toFixed(4)} g</b>
          </div>
          <div className="kv">
            <span>judge state</span>
            <b>{S.tag.emotion || "–"}</b>
          </div>
          <div className="kv">
            <span>scenario phase</span>
            <b>{S.tag.phase || "live camera"}</b>
          </div>
        </section>
      </div>
    </details>
  );
}

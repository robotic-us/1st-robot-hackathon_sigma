"use client";

import type { CradleState, Slot } from "@/lib/types";
import { clamp } from "@/lib/words";
import { send } from "@/lib/useCradle";

/**
 * Everything a reader should not need.  The raw numbers live here so the
 * cards above can stay in plain English, and so nobody is tempted to put
 * "dob_a 3.4 A" back on the front page.
 */
export default function Internals({
  S,
  slots,
}: {
  S: CradleState;
  slots: Slot[];
}) {
  const c = S.cradle;
  const d = S.decision;
  const errFrac = clamp(S.monitor.err / (S.monitor.threshold * 2), 0, 1);

  return (
    <details>
      <summary>
        DREAM-Chunk internals — slot ranking, divergence monitor, manual slots
      </summary>
      <div className="internals">
        <section>
          <h2>Divergence monitor</h2>
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
            <span>|measured − dreamed|</span>
            <b>
              {S.monitor.err.toFixed(3)} rad
              {S.monitor.diverged ? " · DIVERGED" : ""}
            </b>
          </div>
          <div className="bar">
            <i
              style={{
                width: `${errFrac * 100}%`,
                background: S.monitor.diverged ? "var(--critical)" : "var(--accent)",
              }}
            />
          </div>
          <div className="kv">
            <span>external force dob_a</span>
            <b>{S.dob.toFixed(1)} A</b>
          </div>
          <div className="kv">
            <span>dream rms (ASAP residual)</span>
            <b>{S.monitor.rms.toFixed(3)} rad</b>
          </div>
          <div className="kv">
            <span>
              {S.playing
                ? `playing slot ${S.playing.slot} · ${S.playing.name}`
                : "idle"}
            </span>
            <b>{S.playing ? `${Math.round(S.playing.progress * 100)}%` : ""}</b>
          </div>
        </section>

        <section>
          <h2>Last decision</h2>
          <table>
            <thead>
              <tr>
                <th>slot</th><th>cost</th><th>consist</th><th>resist</th>
                <th>contin</th><th>task</th><th />
              </tr>
            </thead>
            <tbody>
              {d && d.ranked.length ? (
                d.ranked.map((r) => (
                  <tr
                    key={r.slot}
                    className={r.vetoed ? "veto" : r.slot === d.chosen ? "win" : ""}
                  >
                    <td>{r.slot}</td>
                    <td>{r.vetoed ? "—" : r.cost.toFixed(2)}</td>
                    <td>{r.consist.toFixed(2)}</td>
                    <td>{r.resist.toFixed(2)}</td>
                    <td>{r.contin.toFixed(2)}</td>
                    <td>{r.task.toFixed(2)}</td>
                    <td>{r.vetoed ? "veto" : r.slot === d.chosen ? "play" : ""}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={7} style={{ textAlign: "center", color: "var(--muted)" }}>
                    no decision yet
                  </td>
                </tr>
              )}
            </tbody>
          </table>

          <div className="slots">
            {slots.map((s) => (
              <div
                key={s.id}
                className={`slot${S.playing?.slot === s.id ? " play" : ""}`}
                onClick={() => send(`/play?slot=${s.id}`)}
              >
                <b>{s.id} · {s.name}</b>
                {s.target_deg}° · {s.duration}s
              </div>
            ))}
          </div>
        </section>
      </div>
    </details>
  );
}

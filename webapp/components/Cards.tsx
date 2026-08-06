"use client";

import type { CradleState } from "@/lib/types";
import {
  ACC_CAP_G, CALM_LEVEL, CRY_LEVEL, ROLE, SWAY_CAP_MM,
  babyWord, clamp, cradleWords, nextWords, sway,
} from "@/lib/words";

const v = (token: string) => `var(${token})`;

/** The colour that stands for the baby right now; red when nobody is visible. */
export const stateColor = (S: CradleState) =>
  S.tag.present ? v(ROLE[S.tag.emotion] ?? "--muted") : v("--critical");

/** Both limits, as fractions -- the safety chip and two bars share this. */
function safety(S: CradleState) {
  const mm = Math.abs(sway(S.cradle));
  const swayFrac = clamp(mm / SWAY_CAP_MM, 0, 1);
  const accFrac = clamp(S.cradle.a_peak_g / ACC_CAP_G, 0, 1);
  const worst = Math.max(swayFrac, accFrac);
  return { mm, swayFrac, accFrac, worst, col: worst > 0.8 ? v("--warn") : v("--good") };
}

/** Zone 1's overlay: the sentence that sits under the orb. */
export function Say({ S }: { S: CradleState }) {
  const c = S.cradle;
  const [doing, detail] = cradleWords(c);
  return (
    <div className="say">
      <div className="headline">
        <span
          className="dot"
          style={{ background: c.state === "gate_fail" ? v("--critical") : stateColor(S) }}
        />
        <span className="who">{babyWord(S.tag)}</span>
        <span className="arrow">→</span>
        <span className="doing">
          {c.state === "gate_fail" ? "stopping for safety" : doing}
        </span>
      </div>
      <div className="detail">{detail}</div>
      <div className="why">{nextWords(c)}</div>
    </div>
  );
}

export function SafeChip({ S }: { S: CradleState }) {
  const { worst, col } = safety(S);
  const gate = S.cradle.state === "gate_fail";
  return (
    <div className="safechip">
      <span className="dot" style={{ background: gate ? v("--critical") : col }} />
      <span>
        {gate
          ? "safety gate tripped — winding down"
          : worst > 0.8
            ? "close to the limit, still inside it"
            : "well within safe limits"}
      </span>
    </div>
  );
}

/** Zone 2: four numbers, each against the limit that makes it mean something. */
export function Strip({ S }: { S: CradleState }) {
  const c = S.cradle;
  const lvl = S.tag.level ?? 0;
  const { mm, swayFrac, accFrac, col } = safety(S);

  // With no face there is nothing to read, so the bar must not keep asserting
  // a level -- a stale confident-looking reading is worse than none.
  const lvlCol = !S.tag.present
    ? v("--baseline")
    : lvl >= CRY_LEVEL ? v("--serious")
    : lvl >= CALM_LEVEL ? v("--warn") : v("--good");

  return (
    <div className="strip">
      <div className="stat">
        <div className="k">Upset level</div>
        <div className="val">{lvl.toFixed(2)}</div>
        <div className="bar">
          <i style={{ width: `${clamp(lvl, 0, 1) * 100}%`, background: lvlCol }} />
          <span className="tick" style={{ left: `${CALM_LEVEL * 100}%` }} />
          <span className="tick" style={{ left: `${CRY_LEVEL * 100}%` }} />
        </div>
      </div>

      <div className="stat">
        <div className="k">Rocking strength</div>
        <div className="val">
          {c.env > 0.01
            ? <>{Math.round(c.env * 100)} <small>%</small></>
            : <>– <small>idle</small></>}
        </div>
        <div className="bar">
          <i style={{
            width: `${clamp(c.env, 0, 1) * 100}%`,
            background: c.tapering ? v("--warn") : v("--accent"),
          }} />
        </div>
      </div>

      <div className="stat">
        <div className="k">How far it swings</div>
        <div className="val">{mm.toFixed(1)} <small>of {SWAY_CAP_MM} mm</small></div>
        <div className="bar">
          <i style={{ width: `${swayFrac * 100}%`, background: col }} />
        </div>
      </div>

      <div className="stat">
        <div className="k">How hard it pushes</div>
        <div className="val">
          {c.a_peak_g.toFixed(3)} <small>of {ACC_CAP_G} g</small>
        </div>
        <div className="bar">
          <i style={{ width: `${accFrac * 100}%`, background: col }} />
        </div>
      </div>
    </div>
  );
}

export function Events({ S }: { S: CradleState }) {
  return (
    <div>
      <h2>What just happened</h2>
      <div className="log">
        {S.events.length
          ? S.events.map((line, i) => <div key={`${i}-${line}`}>{line}</div>)
          : "waiting…"}
      </div>
    </div>
  );
}

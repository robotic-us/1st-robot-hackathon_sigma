/**
 * Plain English for everything the machine publishes.
 *
 * The rule for this app: a reader should never have to know what "amplitude
 * envelope" or "0.018 g" means to understand what the cradle is doing and
 * why.  Engineering terms are allowed only behind the internals drawer.
 *
 * The thresholds below are mirrored from core/cradle.py.  They are duplicated,
 * not derived, so if the report's numbers move, they move here too.
 */
import type { Cradle, Tag } from "./types";

export const CALM_LEVEL = 0.12;      // below this, nobody is fussing
export const CRY_LEVEL = 0.45;       // above this a trial starts one rung up
export const NO_IMPROVE_S = 60;      // no improvement by here: taper + caregiver
export const CHECK_EVERY_S = 30;     // trial checkpoints
export const GATE_RECOVER_S = 2;     // face must be back this long
export const SWAY_CAP_MM = 30;       // the envelope cap every slot stays under
export const ACC_CAP_G = 0.05;       // the engineering ceiling
export const LEVER_MM = 227;         // plate offset per radian of arm angle

/** Emotion label -> a word a parent would use. */
const BABY_WORD: Record<string, string> = {
  SLEEP: "Sleeping", CALM: "Calm", HAPPY: "Happy", NEUTRAL: "Settled",
  FUSS: "Fussing", SAD: "Fussing", SURPRISE: "Startled",
  CRY: "Crying", ANGRY: "Crying",
  AWAKE: "Awake", EYES_CLOSED: "Eyes closed",
  SLEEP_CANDIDATE: "Drifting off", DISTRESS_FACE: "Looks upset",
  UNKNOWN: "Can't see the baby",
  QUIET_AWAKE: "Quiet and awake", STARTLE: "Startled",
  FUSS_WEAK: "Fussing",
  STRONG_DISTRESS: "Very upset", PAIN_SUSPECT: "Needs you now",
  DROWSY: "Getting sleepy", SLEEP_TENTATIVE: "Falling asleep",
  SLEEP_STABLE: "Sleeping", STATE_UNCLEAR: "Can't see the baby",
};

/** Status role per state, so a colour never travels without its label. */
export const ROLE: Record<string, string> = {
  /* ink-ramp rule: FAINT = AT REST, STRONG = NEEDS SOMEONE.  Sleep is the
     calmest state on the page and wears the faintest step -- --accent here is
     full ink and belongs to the machine's own action, never to the baby. */
  SLEEP: "--baseline", CALM: "--good", HAPPY: "--good", NEUTRAL: "--muted",
  FUSS: "--warn", SAD: "--warn", SURPRISE: "--warn",
  CRY: "--serious", ANGRY: "--serious",
  AWAKE: "--good", EYES_CLOSED: "--baseline", SLEEP_CANDIDATE: "--baseline",
  DISTRESS_FACE: "--warn", UNKNOWN: "--muted",
  QUIET_AWAKE: "--good", STARTLE: "--warn", FUSS_WEAK: "--warn",
  STRONG_DISTRESS: "--serious", PAIN_SUSPECT: "--critical",
  DROWSY: "--baseline", SLEEP_TENTATIVE: "--baseline",
  SLEEP_STABLE: "--baseline", STATE_UNCLEAR: "--muted",
};

export function babyWord(tag: Tag): string {
  if (!tag.present) return "Can't see the baby";
  if (tag.emotion) return BABY_WORD[tag.emotion] ?? tag.emotion;
  // state-card mode: no face model, just a level from the card's identity
  const lvl = tag.level ?? 0;
  return lvl >= CRY_LEVEL ? "Crying" : lvl >= CALM_LEVEL ? "Fussing" : "Calm";
}

/**
 * What the cradle is doing, as a headline and a detail line.
 *
 * A rate in hertz means nothing to most readers, so the detail line says how
 * long one sway takes instead.
 */
export function cradleWords(c: Cradle): [string, string] {
  if (c.tapering) return ["Slowing to a stop", "easing the rocking down to nothing"];
  switch (c.kind) {
    case "static": return ["Holding still", "not moving"];
    case "pause": return ["Pausing to watch", "stopped for a moment to see what happens"];
    case "taper": return ["Slowing to a stop", "easing the rocking down to nothing"];
    case "soft_start": return ["Starting gently", "easing the rocking up"];
    case "micro_resume": return ["Easing back in", "the same rocking at half strength"];
  }
  const word = c.f_hz < 0.35 ? "Slow" : c.f_hz < 0.55 ? "Steady"
             : c.f_hz < 0.70 ? "Brisk" : "Quick";
  const wide = c.a_mm >= 15 ? ", wide" : "";
  const every = c.f_hz > 0 ? (1 / c.f_hz).toFixed(1) : "–";
  return [`${word}${wide} rocking`,
          `one sway every ${every} s · ${c.a_mm.toFixed(0)} mm each way`];
}

/** What the state machine will do next, in the reader's terms. */
export function nextWords(c: Cradle): string {
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

export const clamp = (v: number, lo: number, hi: number) =>
  v < lo ? lo : v > hi ? hi : v;

export const sway = (c: Cradle) => c.offset_mm.ap + c.offset_mm.ml;

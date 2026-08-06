/**
 * The shape of one SSE frame from serve.py's build_state().
 *
 * This is a transcription of a Python dict, so it goes stale the moment
 * build_state() grows a field.  When something here disagrees with
 * serve.py, serve.py wins.
 */

export type MachineState = "quiet" | "trial" | "settling" | "gate_fail";

/** core/cradle.py: MotionEngine.snapshot() merged with CradleMachine.snapshot() */
export interface Cradle {
  motion: string | null;          // "M13", or null when parked
  name: string;                   // "ML_SINE_0.6HZ_A10"
  grade: string;                  // C0 | P1 | R
  kind: string;                   // sine | static | pause | taper | soft_start | micro_resume
  f_hz: number;
  a_mm: number;                   // current amplitude, envelope applied
  env: number;                    // 0..1 amplitude envelope
  tapering: boolean;
  offset_mm: { ap: number; ml: number; z: number };
  a_peak_g: number;
  research: boolean;              // --research: whether R entries are allowed
  state: MachineState;
  auto: boolean;
  ema: number;                    // smoothed distress -- what the ladder tests
  trial_s: number;
  alert: string;
}

export interface Tag {
  present: boolean;
  id: number;
  level: number;                  // raw distress, 0..1
  x: number;
  motion: number;
  emotion: string;                // "" in state-card mode
  name: string;                   // face identity, "" unless --sense recognises
}

export interface Playing {
  slot: number;
  name: string;
  progress: number;
  dream_theta: number;            // rad -- where the dream says the plate is
}

export interface Monitor {
  slot: number | null;
  err: number;
  rms: number;
  diverged: boolean;
  threshold: number;
}

export interface Ranked {
  slot: number;
  cost: number;
  consist: number;
  resist: number;
  contin: number;
  task: number;
  vetoed: boolean;
  arc: number[][];
}

export interface Decision {
  seq: number;
  t: number;
  chosen: number | null;
  ranked: Ranked[];
}

export interface CradleState {
  t: number;
  pose: number[];
  tag: Tag;
  jam: boolean;
  dob: number;
  playing: Playing | null;
  cradle: Cradle;
  monitor: Monitor;
  decision: Decision | null;
  events: string[];
}

/** GET /slots */
export interface Slot {
  id: number;
  name: string;
  target_deg: number;
  duration: number;
}

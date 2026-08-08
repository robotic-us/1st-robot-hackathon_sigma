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
  axis: string;                   // ML sways | Z lifts | AP tilts (see-saw) | APML both | ""
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
  level: number;                  // raw distress, 0..1
  x: number;
  emotion: string;                // judge state (QUIET_AWAKE...) or FER label
  name: string;
  alarm: boolean;                 // pain/posture: the gate is about to act
  phase: string;                  // --verify: which scripted scenario phase
}





export interface CradleState {
  t: number;
  pose: number[];
  tag: Tag;
  jam: boolean;
  cradle: Cradle;
  events: string[];
}


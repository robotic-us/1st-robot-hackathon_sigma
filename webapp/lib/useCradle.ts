"use client";

import { useEffect, useRef, useState } from "react";
import type { CradleState, Slot } from "./types";

export type Conn = "connecting" | "live" | "reconnecting";

/**
 * One SSE subscription, shared by the whole page.
 *
 * It hands back both a state value and a ref holding the same object.  The
 * value drives React; the ref exists for the orb, whose animation loop runs at
 * 60 fps and must not re-render anything to read the latest frame.
 */
export function useCradle() {
  const [state, setState] = useState<CradleState | null>(null);
  const [conn, setConn] = useState<Conn>("connecting");
  const latest = useRef<CradleState | null>(null);

  useEffect(() => {
    const es = new EventSource("/events");
    es.onopen = () => setConn("live");
    es.onerror = () => setConn("reconnecting");
    es.onmessage = (e) => {
      const next = JSON.parse(e.data) as CradleState;
      latest.current = next;
      setState(next);
    };
    return () => es.close();
  }, []);

  return { state, conn, latest };
}

/** The DREAM slot table, fetched once for the internals drawer. */
export function useSlots() {
  const [slots, setSlots] = useState<Slot[]>([]);
  useEffect(() => {
    let alive = true;
    fetch("/slots")
      .then((r) => r.json())
      .then((list: Slot[]) => { if (alive) setSlots(list); })
      .catch(() => { /* the drawer simply stays empty */ });
    return () => { alive = false; };
  }, []);
  return slots;
}

/** Fire-and-forget command; serve.py answers with JSON nobody needs here. */
export const send = (path: string) => { void fetch(path).catch(() => {}); };

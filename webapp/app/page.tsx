"use client";

import { Events, SafeChip, Say, Strip } from "@/components/Cards";
import { Camera, Controls } from "@/components/Controls";
import Internals from "@/components/Internals";
import Orb from "@/components/Orb";
import { useCradle, useSlots } from "@/lib/useCradle";

/**
 * Three zones, in order of what a reader needs:
 *
 *   hero   how is the baby, what is the cradle doing, and why
 *   strip  by how much, each number against the limit that bounds it
 *   foot   camera, manual override, recent events
 *
 * Engineering-grade numbers live in the drawer at the bottom and nowhere else.
 */
export default function Page() {
  const { state: S, conn, latest } = useCradle();
  const slots = useSlots();

  return (
    <>
      <header>
        <h1>SIGMA <span>· infant-responsive cradle</span></h1>
        <span className={`conn${conn === "live" ? " live" : ""}`}>
          {conn === "live" ? "live" : conn}
        </span>
      </header>

      {S?.cradle.alert ? <div className="alert">⚠ {S.cradle.alert}</div> : null}

      <section className="hero">
        <Orb latest={latest} />
        {S ? <SafeChip S={S} /> : null}
        {S ? (
          <Say S={S} />
        ) : (
          <div className="say">
            <div className="headline">
              <span className="dot" />
              <span className="who">Connecting</span>
            </div>
            <div className="why">Waiting for the first reading from the cradle.</div>
          </div>
        )}
      </section>

      {S ? <Strip S={S} /> : null}

      <div className="foot">
        <Camera />
        <Controls S={S} />
        {S ? <Events S={S} /> : <div><h2>What just happened</h2>
          <div className="log">waiting…</div></div>}
      </div>

      {S ? <Internals S={S} slots={slots} /> : null}
    </>
  );
}

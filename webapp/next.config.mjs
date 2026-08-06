/**
 * Everything the page needs already exists on serve.py:8080 -- the SSE stream,
 * the MJPEG camera and the command endpoints.  These rewrites proxy them
 * through Next's own origin, which keeps the browser same-origin and means
 * serve.py never has to hand out CORS headers.  That matters more than it
 * looks: /jam, /motion and /auto command a machine that moves, so widening
 * who may call them is not a decision to make for the sake of a dev server.
 *
 * Point SIGMA_API elsewhere when the cradle runs on another host:
 *   SIGMA_API=http://jetson.local:8080 npm run dev
 */
const API = process.env.SIGMA_API ?? "http://127.0.0.1:8080";

const PROXIED = ["events", "frame", "slots", "motions",
                 "jam", "auto", "motion", "play"];

/** @type {import('next').NextConfig} */
export default {
  async rewrites() {
    return PROXIED.map((path) => ({
      source: `/${path}`,
      destination: `${API}/${path}`,
    }));
  },
};

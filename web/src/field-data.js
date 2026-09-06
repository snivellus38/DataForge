// Pure logic behind THE FIELD. No DOM, no WebGL, no fetch -- so all of it runs under Node and
// is covered by web/test/field.mjs. field-gl.js is deliberately the thin half.

import { IGNITION, SIGMA_RAMP, COMMUNITY, COMMUNITY_OTHER } from "./palette.js";
import { frame } from "./bigdata.js";

const RAMPS = { ignition: IGNITION, sigma: SIGMA_RAMP };

/** Named ramp -> four Float32Array(3) stops for the shader. */
export function rampStops(name = "ignition") {
  const hex = RAMPS[name] || (Array.isArray(name) ? name : IGNITION);
  return hex.map((h) => new Float32Array([1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255)));
}

/** OKLab lightness, used to ASSERT a sequential ramp is monotonic rather than eyeball it. */
export function oklabL(hex) {
  const s = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  const lin = s.map((v) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
  const [r, g, b] = lin;
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const q = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * q;
}

/**
 * Advance the glow one animation frame.
 *
 * Firing is instantaneous but the eye is not: without persistence the field strobes and no one
 * can see WHICH neurons lit. Activation rises immediately to the new value and falls
 * exponentially, so a token leaves a visible trail for a few frames. `decay` is per-frame
 * retention; `next` may be null to let everything fade (used between tokens and while paused).
 *
 * Writes in place and returns the same array -- this runs every frame over 12,288 elements.
 */
export function decayInto(act, next, decay = 0.86) {
  for (let i = 0; i < act.length; i++) act[i] *= decay;
  if (next) {
    const { idx, val, n } = next;
    for (let k = 0; k < n; k++) {
      const i = idx[k];
      if (val[k] > act[i]) act[i] = val[k];       // attack is instant, release is the decay
    }
  }
  return act;
}

/**
 * Activations for one token, normalised to 0..1 against a stable reference rather than the
 * frame's own max -- per-frame normalisation would make a quiet token look exactly as bright as
 * a loud one and destroy the very comparison the animation exists to show.
 */
export const ACT_CURVE = 0.40;

export function tokenFrame(pack, sentence, layer, t, ref, curve = ACT_CURVE) {
  const f = frame(pack, sentence, layer, t);
  const val = new Float32Array(f.n);
  // Activations are heavy-tailed: a linear map leaves almost everything in the bottom of the ramp
  // and only a handful of neurons visible. The gamma is monotonic, so ORDER is preserved exactly
  // -- brighter still means more active -- it only redistributes where the ramp's steps fall.
  // Same reasoning as log-scaling the degree; tuned on real frames via research/preview_field.py.
  for (let i = 0; i < f.n; i++) val[i] = Math.min(1, f.val[i] / ref) ** curve;
  return { idx: f.idx, val, n: f.n };
}

/** A robust brightness reference: the p99 activation over a whole sentence/layer. */
export function referenceLevel(pack, sentence, layer, nTok, sample = 24) {
  const vals = [];
  const step = Math.max(1, Math.floor(nTok / sample));
  for (let t = 0; t < nTok; t += step) {
    const f = frame(pack, sentence, layer, t);
    for (let i = 0; i < f.n; i++) vals.push(f.val[i]);
  }
  if (!vals.length) return 1;
  vals.sort((a, b) => a - b);
  return vals[Math.min(vals.length - 1, Math.floor(vals.length * 0.99))] || 1;
}

/** Nearest node to a model-space point, or -1 beyond `maxDist`. Linear over 12,288 is ~0.1ms. */
export function nearestNode(xy, x, y, maxDist = Infinity) {
  let best = -1, bd = maxDist * maxDist;
  for (let i = 0; i < xy.length; i += 2) {
    const dx = xy[i] - x, dy = xy[i + 1] - y;
    if (!Number.isFinite(dx) || !Number.isFinite(dy)) continue;
    const d = dx * dx + dy * dy;
    if (d < bd) { bd = d; best = i >> 1; }
  }
  return best;
}

/**
 * The synapses of one neuron, as global ids.
 *
 * G* is exported PER HEAD with head-local indices, while the field is indexed globally
 * (id = head*N + n). Getting this wrong silently draws edges between unrelated neurons, which is
 * exactly the kind of plausible-looking lie the artifact must not contain -- so the conversion
 * lives here, in a tested function, rather than inline in a renderer.
 */
export function edgesFor(fieldPack, globalId, N, limit = 400) {
  const head = Math.floor(globalId / N), local = globalId % N;
  const ei = fieldPack.t[`h${head}.edge_i`], ej = fieldPack.t[`h${head}.edge_j`];
  const ew = fieldPack.t[`h${head}.edge_w`];
  if (!ei) return { i: new Uint16Array(0), j: new Uint16Array(0), w: new Float32Array(0) };
  const oi = [], oj = [], ow = [];
  let peak = 0;
  for (let k = 0; k < ei.length && oi.length < limit; k++) {
    if (ei[k] !== local && ej[k] !== local) continue;
    oi.push(head * N + ei[k]); oj.push(head * N + ej[k]); ow.push(ew[k]);
    peak = Math.max(peak, Math.abs(ew[k]));
  }
  const w = new Float32Array(ow.length);
  for (let k = 0; k < ow.length; k++) w[k] = ow[k] / (peak || 1);
  return { i: Uint16Array.from(oi), j: Uint16Array.from(oj), w };
}

/**
 * The causal attention scores reaching INTO token t, strongest first.
 *
 * The export stores np.tril_indices(T, -1), which is row-major: row t occupies a contiguous run
 * of exactly t entries starting at t*(t-1)/2. So this is a slice, not a scan -- worth knowing,
 * because the naive filter over all 15,051 scores every frame is what makes an arc diagram
 * stutter.
 *
 * Scores are SIGNED and the negatives are the teaching point: activations are non-negative by
 * construction, so a negative score can only come from RoPE rotating the keys before they meet.
 * 38.5% of them are negative on this model (measured, layer 3 head 0).
 */
export function arcsInto(pack, sentence, t, limit = 140) {
  const w = pack.t[`s${sentence}.attn_w`];
  if (!w || t < 1) return { from: new Int32Array(0), w: new Float32Array(0), peak: 0, negShare: 0 };
  const base = (t * (t - 1)) / 2;
  const idx = Array.from({ length: t }, (_, s) => s);
  let neg = 0, peak = 0;
  for (let s = 0; s < t; s++) {
    const v = w[base + s];
    if (v < 0) neg++;
    if (Math.abs(v) > peak) peak = Math.abs(v);
  }
  idx.sort((a, b) => Math.abs(w[base + b]) - Math.abs(w[base + a]));
  const keep = idx.slice(0, limit);
  const out = new Float32Array(keep.length);
  for (let k = 0; k < keep.length; k++) out[k] = w[base + keep[k]];
  return { from: Int32Array.from(keep), w: out, peak, negShare: t ? neg / t : 0 };
}

/** Colour for a node under the community mode: top-3 get a hue, everything else is "other". */
export function communityColor(cluster) {
  return cluster < COMMUNITY.length ? COMMUNITY[cluster] : COMMUNITY_OTHER;
}

/** Degree -> 0..1, log-scaled. Degrees run 0..~876 with a heavy tail, so linear would show ~nothing. */
export function degreeLevel(deg, maxDeg) {
  return maxDeg > 0 ? Math.log1p(deg) / Math.log1p(maxDeg) : 0;
}

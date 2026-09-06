// THE FLOW MODEL — one token's whole trip through one BDH iteration, as columns and edges.
//
// Pure. No DOM, no canvas, no fetch, so all of it runs under Node and is covered by
// web/test/flow.mjs. flow.js is the thin half that paints what this returns.
//
// WHY EVERY EDGE IS A REAL NUMBER
// -------------------------------
// A layered network picture is only worth drawing if the lines mean something. Each hop here is
// a weighted sum, so an edge's value is exactly `weight x source activation` — the term that
// source contributes to that target — and nothing on screen is a decorative spline:
//
//   v*  -> x_pre    v[d] * dx_head[d, n]        decoder_x, shipped for the walked head
//   x   -> a*       qr[n] * sigma[n, d]         the Hebbian read; sigma accumulated here
//   a*  -> y_pre    a[d] * flow.dy[p, d]        decoder_y, shipped for the pooled neurons
//   y   -> dv       y[g] * flow.enc[p, d]       encoder, same pool
//   v'  -> byte     vfinal[d] * lm_head[d, b]
//
// WHY THERE IS A FIXED CAST
// -------------------------
// 12,288 neurons cannot each be a dot. The export commits a POOL (`flow.ids`, 256 neurons that
// actually lead tokens, ordered by corpus firing frequency) and ships the weight slices for
// exactly those, which is what lets the edges above be real. This module draws a stable prefix
// of that pool — so a dot keeps its identity and its row as the animation runs — plus whichever
// pool members lead the current token. Both counts, and the pool's measured coverage of the true
// per-token leaders, come back in `stats` so the page can state them instead of implying it
// draws everything.

import { f16, sparseAt, rope } from "./walkdata.js";

export const STABLE = 18;      // dots held across every token, by corpus firing frequency
export const LEADERS_X = 5;    // extra dots: this token's strongest neurons before the gate
export const LEADERS_Y = 3;    // ...and after it
export const LEADERS = LEADERS_X + LEADERS_Y;
export const FAN = 3;          // edges kept per target node: its strongest contributors
export const TICKS = 192;      // the D-wide columns are drawn in full, one tick per dimension

/** Column identities. `unit` drives how the renderer lays a column out. */
export const COLUMNS = [
  { id: "byte",  name: "Input byte",     sub: "1 of 256",        unit: "byte",  op: "raw UTF-8, no tokeniser" },
  { id: "v",     name: "Residual v*",    sub: "192",             unit: "dim",   op: "LayerNorm'd, into this iteration" },
  { id: "xpre",  name: "Dₓ",             sub: "192 → 3,072",     unit: "neuron", op: "one matrix multiply per head" },
  { id: "x",     name: "ReLU → x",       sub: "12,288",          unit: "neuron", op: "everything negative becomes exactly 0" },
  { id: "a",     name: "σ read-out a*",  sub: "192",             unit: "dim",   op: "RoPE, Q·K, then read the state" },
  { id: "y",     name: "Dᵧ ⊙ x → y",     sub: "12,288",          unit: "neuron", op: "an AND of two sparse conditions" },
  { id: "dv",    name: "E + → v*′",      sub: "192",             unit: "dim",   op: "back to the residual, then again" },
  { id: "byteout", name: "lm_head",      sub: "256",             unit: "byte",  op: "one distribution over bytes" },
];

/**
 * Sigma rows for the whole flow pool, accumulated the recurrent way.
 *
 * The full state is N x D = 589,824 floats and rebuilding it every token is wasteful when the
 * diagram reads back at most the pool's 256 rows. Those are 49,152 floats and cost 49K
 * multiply-adds per token, so they can be carried the whole way — and the pool, not the cast, is
 * the right granularity: a neuron that only becomes a leader at token 30 still needs the row it
 * has been accumulating since token 0.
 *
 * Call `advance(t)` for each token in order. A row is only valid for reading at token t once the
 * accumulator has been advanced through t-1, which is the model's tril(-1) mask: a token reads
 * the state BEFORE its own write lands.
 */
export function sigmaRows(p, w, layer, head) {
  preflight(p);
  const { N, D } = p;
  const ids = p.flowIds;
  const rows = new Float32Array(ids.length * D);
  const v = f16(p, `w${w}.L${layer}.v`);
  return {
    t: 0, rows, ids, w, layer, head,
    /** Fold token `t`'s rank-one write, Δσ = rope(x)_t ⊗ v_t, into the pooled rows. */
    advance(t) {
      const sp = sparseAt(p, `w${w}.L${layer}.x`, t);
      const qr = rope(p, sp, t, head);
      const vo = t * D;
      for (let k = 0; k < ids.length; k++) {
        const n = ids[k] - head * N;
        if (n < 0 || n >= N) continue;                 // not in the traced head: its row stays 0
        const q = qr[n];
        if (q === 0) continue;
        const off = k * D;
        for (let d = 0; d < D; d++) rows[off + d] += q * v[vo + d];
      }
      this.t = t + 1;
    },
  };
}

/** Top-k indices of `score` by absolute value, descending. Allocation-light: k is tiny. */
export function topK(score, k) {
  const out = [];
  for (let i = 0; i < score.length; i++) {
    const a = Math.abs(score[i]);
    if (a === 0) continue;
    if (out.length < k) {
      out.push([i, a]);
      if (out.length === k) out.sort((p1, p2) => p2[1] - p1[1]);
    } else if (a > out[k - 1][1]) {
      out[k - 1] = [i, a];
      out.sort((p1, p2) => p2[1] - p1[1]);
    }
  }
  if (out.length < k) out.sort((p1, p2) => p2[1] - p1[1]);
  return out.map((e) => e[0]);
}

/** x_pre for the traced head, derived rather than shipped: x_pre[n] = Σ_d v[d]·dx_head[d,n]. */
export function xPre(p, v, tOff) {
  const { N, D } = p;
  const dx = p.flowDx;                                  // dense (D, N), f32, cached by preflight
  const out = new Float32Array(N);
  for (let d = 0; d < D; d++) {
    const vd = v[tOff + d];
    if (vd === 0) continue;
    const row = d * N;
    for (let n = 0; n < N; n++) out[n] += vd * dx[row + n];
  }
  return out;
}

/**
 * One-time decode of everything the flow view reuses across tokens. Kept off `buildFrame` so a
 * 1.18 MB f16 matrix is not re-expanded 45 times a second.
 */
export function preflight(p) {
  if (p.flowDx) return p;
  p.flowDx = f16(p, "dx_head");
  p.flowIds = p.t.get("flow.ids").view;                 // global neuron ids, fire-count order
  p.flowDy = f16(p, "flow.dy");                         // (P, D)
  p.flowEnc = f16(p, "flow.enc");                       // (P, D)
  p.lmHead = f16(p, "lm_head");                         // (D, 256)
  p.fire = p.t.get("fire_count").view;
  p.poolAt = new Map();
  for (let i = 0; i < p.flowIds.length; i++) p.poolAt.set(p.flowIds[i], i);
  return p;
}

/**
 * The drawn cast for this token: a stable prefix of the pool, then this token's own leaders taken
 * from BOTH sides of the gate.
 *
 * Picking the visitors by `x` alone looked right and was not: the gate kills 98% of what fires,
 * so on most tokens none of the drawn neurons survived it and the last third of the diagram had
 * nothing running through it. That is a true fact about the model and a useless picture of it.
 * Reserving a few slots for gate survivors puts the AND on screen instead of implying the update
 * comes from nowhere -- 98.1% of (sentence, iteration, token) steps have at least one pooled
 * survivor to show, measured over all 900 of them.
 */
export function castFor(p, xDense, yDense = new Map(),
                        { stable = STABLE, lx = LEADERS_X, ly = LEADERS_Y } = {}) {
  const ids = p.flowIds;
  const cast = [];
  for (let i = 0; i < Math.min(stable, ids.length); i++) {
    cast.push({ pool: i, id: ids[i], stable: true });
  }
  const taken = new Set(cast.map((c) => c.pool));
  const pick = (dense, k, why) => {
    const score = new Float32Array(ids.length);
    for (let i = 0; i < ids.length; i++) score[i] = taken.has(i) ? 0 : (dense.get(ids[i]) || 0);
    for (const i of topK(score, k)) {
      taken.add(i);
      cast.push({ pool: i, id: ids[i], stable: false, why });
    }
  };
  pick(yDense, ly, "gate");     // survivors first, so a busy x column cannot crowd them out
  pick(xDense, lx, "fires");
  return cast;
}

/**
 * Build every column and every edge for one (walk, iteration, token).
 *
 * `sig` is a sigmaRows accumulator already advanced through t-1; pass null before the state has
 * anything in it and the a* edges come back empty, which is the correct picture at token 0.
 */
export function buildFrame(p, { w, layer, head, t, sig = null, cast: fixedCast = null }) {
  preflight(p);
  const { N, D } = p;
  const meta = p.manifest.walk.find((x) => x.i === w);
  const T = meta.T;

  const v = f16(p, `w${w}.L${layer}.v`);
  const a = f16(p, `w${w}.L${layer}.H${head}.a`);
  const dvAll = f16(p, `w${w}.L${layer}.dv`);
  const logits = f16(p, `w${w}.logits`);
  const vfinal = f16(p, `w${w}.vfinal`);

  const xs = sparseAt(p, `w${w}.L${layer}.x`, t);
  const ys = sparseAt(p, `w${w}.L${layer}.y`, t);
  const xDense = new Map(), yDense = new Map();
  for (let i = 0; i < xs.n; i++) xDense.set(xs.ids[i], xs.val[i]);
  for (let i = 0; i < ys.n; i++) yDense.set(ys.ids[i], ys.val[i]);

  const cast = fixedCast || castFor(p, xDense, yDense);
  const pre = xPre(p, v, t * D);
  const qr = rope(p, xs, t, head);

  // ── columns ────────────────────────────────────────────────────────────────────────────────
  const vSlice = v.subarray(t * D, t * D + D);
  const aSlice = a.subarray(t * D, t * D + D);
  const dvSlice = dvAll.subarray(t * D, t * D + D);
  const vfSlice = vfinal.subarray(t * D, t * D + D);
  const lgSlice = logits.subarray(t * 256, t * 256 + 256);

  const nodes = { byte: [], v: [], xpre: [], x: [], a: [], y: [], dv: [], byteout: [] };

  nodes.byte.push({ i: 0, value: meta.bytes[t], label: glyph(meta.bytes[t]),
                    detail: `byte ${meta.bytes[t]} · position ${t + 1} of ${T}` });

  for (let d = 0; d < D; d++) {
    nodes.v.push({ i: d, value: vSlice[d] });
    nodes.a.push({ i: d, value: aSlice[d] });
    nodes.dv.push({ i: d, value: dvSlice[d], after: vfSlice[d] });
  }

  for (const c of cast) {
    const local = c.id - head * N;
    const inHead = local >= 0 && local < N;
    const xv = xDense.get(c.id) || 0;
    const yv = yDense.get(c.id) || 0;
    nodes.xpre.push({ i: c.id, pool: c.pool, stable: c.stable, why: c.why, inHead,
                      value: inHead ? pre[local] : NaN });
    nodes.x.push({ i: c.id, pool: c.pool, stable: c.stable, why: c.why, inHead, value: xv,
                   roped: inHead ? qr[local] : NaN, fire: p.fire[c.id] });
    nodes.y.push({ i: c.id, pool: c.pool, stable: c.stable, why: c.why, inHead, value: yv,
                   gate: xv, fire: p.fire[c.id] });
  }

  // topK is by |value|, which is right for every signed column and WRONG here: it returned the
  // bytes with the most extreme logits, i.e. the ones the model most strongly ruled out, all at
  // p = 0.00%. A distribution wants the largest logits.
  const topBytes = topSigned(lgSlice, 8);
  const truth = t + 1 < T ? meta.bytes[t + 1] : -1;
  if (truth >= 0 && !topBytes.includes(truth)) topBytes.push(truth);
  const mx = Math.max(...topBytes.map((b) => lgSlice[b]));
  let z = 0;
  for (let b = 0; b < 256; b++) z += Math.exp(lgSlice[b] - mx);
  for (const b of topBytes) {
    nodes.byteout.push({ i: b, value: lgSlice[b], p: Math.exp(lgSlice[b] - mx) / z,
                         label: glyph(b), truth: b === truth, chosen: b === topBytes[0] });
  }

  // ── edges: every one is weight x source activation ─────────────────────────────────────────
  const edges = [];
  const push = (from, to, si, ti, val) => edges.push({ from, to, si, ti, w: val });

  // byte -> v*: the residual this token is carrying. One link into its largest components.
  for (const d of topK(vSlice, FAN)) push("byte", "v", 0, d, vSlice[d]);

  // v* -> x_pre: v[d] * dx_head[d, n]
  for (let k = 0; k < nodes.xpre.length; k++) {
    const nd = nodes.xpre[k];
    if (!nd.inHead) continue;
    const local = nd.i - head * N;
    const contrib = new Float32Array(D);
    for (let d = 0; d < D; d++) contrib[d] = vSlice[d] * p.flowDx[d * N + local];
    for (const d of topK(contrib, FAN)) push("v", "xpre", d, k, contrib[d]);
  }

  // x_pre -> x: the ReLU itself. One link per drawn neuron, and it is either alive or it is not.
  for (let k = 0; k < nodes.x.length; k++) push("xpre", "x", k, k, nodes.x[k].value);

  // x -> a*: qr[n] * sigma[n, d]. This is the Hebbian read, and it is why sigma is a column of
  // edges rather than a column of dots -- the state IS the connection between these two.
  if (sig) {
    for (let k = 0; k < nodes.x.length; k++) {
      const nd = nodes.x[k];
      if (!nd.inHead || !nd.value) continue;
      const row = nd.pool * D;
      const contrib = new Float32Array(D);
      for (let d = 0; d < D; d++) contrib[d] = nd.roped * sig.rows[row + d];
      for (const d of topK(contrib, FAN)) push("x", "a", k, d, contrib[d]);
    }
  }

  // a* -> y: a[d] * decoder_y[d, n], then gated by x
  for (let k = 0; k < nodes.y.length; k++) {
    const nd = nodes.y[k];
    const off = nd.pool * D;
    const contrib = new Float32Array(D);
    for (let d = 0; d < D; d++) contrib[d] = aSlice[d] * p.flowDy[off + d];
    for (const d of topK(contrib, FAN)) push("a", "y", d, k, contrib[d]);
  }

  // y -> dv: y[g] * encoder[g, d].
  //
  // The share reported below is the L1 of the update the DRAWN neurons actually build, against
  // the L1 of the real update the model applied. Summing |contribution| instead would double
  // count cancellation -- opposite-signed terms vanish in dv but not in a sum of magnitudes --
  // and could report more than 100%.
  const drawnDv = new Float32Array(D);
  let survivors = 0, dvTotal = 0, drawnTotal = 0;
  for (let d = 0; d < D; d++) dvTotal += Math.abs(dvSlice[d]);
  for (let k = 0; k < nodes.y.length; k++) {
    const nd = nodes.y[k];
    if (!nd.value) continue;
    survivors++;
    const off = nd.pool * D;
    const contrib = new Float32Array(D);
    for (let d = 0; d < D; d++) {
      contrib[d] = nd.value * p.flowEnc[off + d];
      drawnDv[d] += contrib[d];
    }
    for (const d of topK(contrib, FAN)) push("y", "dv", k, d, contrib[d]);
  }
  for (let d = 0; d < D; d++) drawnTotal += Math.abs(drawnDv[d]);

  // v*' -> byte: vfinal[d] * lm_head[d, b]
  for (let k = 0; k < nodes.byteout.length; k++) {
    const b = nodes.byteout[k].i;
    const contrib = new Float32Array(D);
    for (let d = 0; d < D; d++) contrib[d] = vfSlice[d] * p.lmHead[d * 256 + b];
    for (const d of topK(contrib, FAN)) push("dv", "byteout", d, k, contrib[d]);
  }

  // ── the numbers the page states rather than implies ────────────────────────────────────────
  let preNeg = 0;
  for (let n = 0; n < N; n++) if (pre[n] < 0) preNeg++;
  let inHeadFiring = 0;
  for (let i = 0; i < xs.n; i++) if (xs.ids[i] >= head * N && xs.ids[i] < (head + 1) * N) inHeadFiring++;

  const stats = {
    T, t, byte: meta.bytes[t],
    xFiring: xs.n, yFiring: ys.n, total: p.H * N,
    xFrac: xs.n / (p.H * N), yFrac: ys.n / (p.H * N),
    preNeg, preNegFrac: preNeg / N, inHeadFiring, headN: N,
    drawn: cast.length, stable: cast.filter((c) => c.stable).length,
    pool: p.flowIds.length,
    poolCoverage: p.manifest.flow.leader_coverage,
    leaderK: p.manifest.flow.leader_k,
    // What the drawn neurons contribute to this iteration's actual residual update. Deliberately
    // NOT called a share: drawnDv is a partial sum, and omitted neurons cancel some of the drawn
    // ones, so the ratio of magnitudes can exceed 1 (measured max 101.6% over all 450 steps).
    // The page says "the size of" rather than "of", which is what this number means.
    ySurvivors: survivors,
    dvRatio: dvTotal ? drawnTotal / dvTotal : 0,
    top: nodes.byteout[0], truth,
    edges: edges.length,
    // fire_count counts (token, iteration) steps on which a neuron was non-zero, so the
    // denominator is tokens x iterations -- not neuron-token slots, which is 12,288x larger.
    corpusSteps: p.manifest.corpus.n_neuron_token_steps / (p.H * N),
  };

  return { nodes, edges, stats, cast, columns: COLUMNS };
}

/** Indices of the k LARGEST values, descending. Not by magnitude -- see the byte column. */
export function topSigned(v, k) {
  const idx = Array.from({ length: v.length }, (_, i) => i);
  idx.sort((a, b) => v[b] - v[a]);
  return idx.slice(0, k);
}

/** Printable form of a byte. The model reads raw UTF-8, so most bytes are not characters. */
export function glyph(b) {
  if (b === 32) return "␣";
  if (b === 10) return "⏎";
  if (b >= 33 && b < 127) return String.fromCharCode(b);
  return `·${b}`;
}

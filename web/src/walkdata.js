// WALK PACK — pure decode + the two derivations the payload deliberately leaves out.
//
// No DOM, no WebGL, no canvas. Everything here is arrays in, arrays out, so the gate
// (web/test/walk.mjs) can run the real code under node and so a missing GPU context can never
// take the numbers down with it. Same contract as bigdata.js, which this deliberately mirrors.
//
// TWO THINGS ARE COMPUTED HERE RATHER THAN SHIPPED, and both are checked by the gate:
//
//   rope()      RoPE is a closed form in (neuron index, position). export_walk.py verified the
//               model's own q_roped is reproduced at 0.0 max abs error, so shipping it would be
//               ~140 KB per sentence of redundancy.
//
//   sigmaRun()  every write into the state is rank-one -- sigma_t = sigma_{t-1} + rope(x)_t (x) v_t
//               -- so the browser can accumulate the whole state itself and then check its own
//               answer against the shipped a_ast. That check is the point: it is what makes the
//               recurrence something the reader watches happen rather than reads about.

const CTOR = {
  "|u1": Uint8Array, "<u2": Uint16Array, "<u4": Uint32Array,
  "<f4": Float32Array, "<f2": Uint16Array,          // f16 arrives raw, widened on read
};

/** Build typed-array views over one ArrayBuffer. No copying. */
export function loadPack(manifest, buffer) {
  const t = new Map();
  for (const e of manifest.tensors) {
    const C = CTOR[e.dtype];
    if (!C) throw new Error(`walk pack: unknown dtype ${e.dtype} for ${e.name}`);
    // export_walk.py pads every tensor to 8 bytes so this stays on the zero-copy path; the
    // fallback matches bigdata.js and exists so a hand-built pack cannot silently read garbage.
    const view = (e.offset % C.BYTES_PER_ELEMENT)
      ? new C(buffer.slice(e.offset, e.offset + e.bytes))
      : new C(buffer, e.offset, e.bytes / C.BYTES_PER_ELEMENT);
    t.set(e.name, { view, meta: e });
  }
  return { manifest, t, cfg: manifest.config, N: manifest.config.N, D: manifest.config.n_embd,
           H: manifest.config.n_head, L: manifest.config.n_layer };
}

const get = (p, name) => {
  const e = p.t.get(name);
  if (!e) throw new Error(`walk pack: missing tensor ${name}`);
  return e.view;
};
export const has = (p, name) => p.t.has(name);

/** IEEE half -> double. The pack stores f16 as raw u16; this is the only place that matters. */
function h2f(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -14) * (m / 1024);
  if (e === 31) return m ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + m / 1024);
}
export function f16(p, name) {
  const raw = get(p, name), out = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) out[i] = h2f(raw[i]);
  return out;
}

/** One token's slice of a sparse (T, M) tensor: parallel index/value arrays, values dequantised. */
export function sparseAt(p, base, t) {
  const idx = get(p, `${base}.idx`), val = get(p, `${base}.val`);
  const off = get(p, `${base}.off`), scale = get(p, `${base}.scale`);
  const a = off[t], b = off[t + 1], n = b - a;
  const ids = new Uint16Array(n), v = new Float32Array(n), s = scale[t] / 255;
  for (let i = 0; i < n; i++) { ids[i] = idx[a + i]; v[i] = val[a + i] * s; }
  return { ids, val: v, n, scale: scale[t] };
}

/**
 * Dequantise a signed dense (T, M) tensor, given a per-token (min, max) pair.
 *
 * Nothing ships in this shape any more: x_pre was the only user, and it is now derived from
 * `v` and `dx_head` instead (1.18 MB of matrix against 2.76 MB of its outputs, and more accurate
 * -- see research/export_walk.py). Kept because the shape is a reasonable thing for the pack to
 * carry again, and web/test/walk.mjs would catch it going stale the moment something did.
 */
/**
 * RoPE, exactly as bdh_big._causal_attention applies it, on a SPARSE token vector.
 *
 * The rotation pairs neuron 2k with 2k+1, so a sparse input has to be rotated in the dense
 * neuron space of its own head: a neuron whose partner is silent still rotates into a non-zero
 * value. Returns a dense Float32Array(N) for one head.
 */
export function rope(p, { ids, val }, t, head = 0) {
  const { N } = p, freqs = get(p, "rope_freqs");
  const dense = new Float32Array(N), lo = head * N, hi = lo + N;
  for (let i = 0; i < ids.length; i++) {
    const g = ids[i];
    if (g >= lo && g < hi) dense[g - lo] = val[i];
  }
  const out = new Float32Array(N), TWO_PI = Math.PI * 2;
  for (let n = 0; n < N; n += 2) {
    const ph = ((t * freqs[n]) % 1) * TWO_PI, c = Math.cos(ph), s = Math.sin(ph);
    const a = dense[n], b = dense[n + 1];
    // bdh_big builds the rotated partner as (-x[1::2], x[::2]) interleaved, so the even lane
    // takes -b and the odd lane takes +a. Getting this backwards silently corrupts every score.
    out[n]     = a * c - b * s;
    out[n + 1] = b * c + a * s;
  }
  return out;
}

/**
 * Accumulate sigma token by token and read it back, the recurrent way.
 *
 * Returns the read-outs a_t = sigma_t^T rope(x)_t for t < upTo, computed BEFORE that token's own
 * write (the model's mask is tril(-1): a token cannot attend to itself). `sigma` comes back so a
 * caller can draw the state or a single write.
 */
export function sigmaRun(p, w, layer, head, upTo = Infinity) {
  const { N, D } = p;
  const v = f16(p, `w${w}.L${layer}.v`);
  const base = `w${w}.L${layer}.x`;
  const T = Math.min(get(p, `${base}.off`).length - 1, upTo);
  const sigma = new Float32Array(N * D);
  const out = new Float32Array(T * D);
  for (let t = 0; t < T; t++) {
    const sp = sparseAt(p, base, t);
    const k = rope(p, sp, t, head);
    // read before writing
    for (let n = 0; n < N; n++) {
      const kn = k[n];
      if (kn === 0) continue;
      const r = n * D, o = t * D;
      for (let d = 0; d < D; d++) out[o + d] += kn * sigma[r + d];
    }
    // write: one rank-one update, the whole mechanism
    const vo = t * D;
    for (let n = 0; n < N; n++) {
      const kn = k[n];
      if (kn === 0) continue;
      const r = n * D;
      for (let d = 0; d < D; d++) sigma[r + d] += kn * v[vo + d];
    }
  }
  return { out, sigma, T, D, N };
}

/** delta-sigma for one token: the rank-one write on its own. Genuinely structured, unlike sigma. */
export function deltaSigma(p, w, layer, head, t) {
  const { N, D } = p;
  const v = f16(p, `w${w}.L${layer}.v`);
  const k = rope(p, sparseAt(p, `w${w}.L${layer}.x`, t), t, head);
  return { k, v: v.subarray(t * D, t * D + D), N, D };
}

/** ||sigma_n|| per neuron -- which neurons carry the binding. */
export function rowEnergy(sigma, N, D) {
  const e = new Float32Array(N);
  for (let n = 0; n < N; n++) {
    let s = 0; const r = n * D;
    for (let d = 0; d < D; d++) s += sigma[r + d] * sigma[r + d];
    e[n] = Math.sqrt(s);
  }
  return e;
}

/** Causal scores as a dense T x T lower triangle (upper triangle and diagonal are 0 by mask). */
export function attnMatrix(p, w, layer, head, T) {
  const flat = get(p, `w${w}.L${layer}.H${head}.attn`);
  const M = new Float32Array(T * T);
  let i = 0;
  for (let r = 1; r < T; r++) for (let c = 0; c < r; c++) M[r * T + c] = flat[i++];
  return M;
}

/** Share of causal scores below zero. RoPE, not a bug -- and it varies a lot by sentence/layer. */
export function negShare(p, w, layer, head) {
  const f = get(p, `w${w}.L${layer}.H${head}.attn`);
  let n = 0;
  for (let i = 0; i < f.length; i++) if (f[i] < 0) n++;
  return { negative: n, total: f.length, frac: f.length ? n / f.length : 0 };
}

/**
 * Softmax one row of the scores -- the "break it" control on the score-matrix stop.
 *
 * This is the single change that makes a fixed-size state impossible: softmax normalises each
 * score against every other score in the row, so no running sum over past keys can produce it
 * without keeping all of them. Q != K and dropping the ReLU do NOT break the recurrence
 * (CLAUDE.md item 4) -- only this does.
 */
export function softmaxRow(M, T, row) {
  const out = new Float32Array(T);
  let mx = -Infinity;
  for (let c = 0; c < row; c++) if (M[row * T + c] > mx) mx = M[row * T + c];
  if (mx === -Infinity) return out;
  let s = 0;
  for (let c = 0; c < row; c++) { out[c] = Math.exp(M[row * T + c] - mx); s += out[c]; }
  for (let c = 0; c < row; c++) out[c] /= s;
  return out;
}

/** Per-token gate accounting: |x>0|, |y>0|, and the survival rate through ReLU(y_pre) (x) x. */
export function gateAt(p, w, layer, t) {
  const x = sparseAt(p, `w${w}.L${layer}.x`, t);
  const y = sparseAt(p, `w${w}.L${layer}.y`, t);
  return { fired: x.n, survived: y.n, survival: x.n ? y.n / x.n : 0, x, y };
}

/** Top-k next-byte prediction from the shipped logits. Truth beside estimate. */
export function topBytes(p, w, t, k = 8) {
  const lg = f16(p, `w${w}.logits`), V = 256, o = t * V;
  let mx = -Infinity;
  for (let i = 0; i < V; i++) if (lg[o + i] > mx) mx = lg[o + i];
  const pr = new Float32Array(V);
  let s = 0;
  for (let i = 0; i < V; i++) { pr[i] = Math.exp(lg[o + i] - mx); s += pr[i]; }
  const idx = Array.from({ length: V }, (_, i) => i).sort((a, b) => pr[b] - pr[a]).slice(0, k);
  return idx.map((byte) => ({ byte, p: pr[byte] / s }));
}

/**
 * Matthews correlation between "isolated in G*" and "never fires in the corpus".
 *
 * G* = D_x^T E^T is computed from the WEIGHTS ALONE, with no data. If it predicts which neurons
 * do nothing on real French, structure predicts function. Computed here rather than quoted so
 * the page re-derives it in front of the reader from the shipped bytes.
 */
export function silenceMCC(fireCount, outDeg, inDeg) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (let i = 0; i < fireCount.length; i++) {
    const isolated = (outDeg[i] + inDeg[i]) === 0;
    const silent = fireCount[i] === 0;
    if (isolated && silent) tp++;
    else if (isolated && !silent) fp++;
    else if (!isolated && !silent) tn++;
    else fn++;
  }
  const num = tp * tn - fp * fn;
  const den = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
  return {
    mcc: den ? num / den : 0, tp, fp, tn, fn,
    silentFrac: (tp + fn) / fireCount.length,
    isolatedFrac: (tp + fp) / fireCount.length,
    pSilentGivenIsolated: (tp + fp) ? tp / (tp + fp) : 0,
  };
}

/** Mean total degree among silent vs firing neurons -- 2.1 against 97.3 when last measured. */
export function degreeBySilence(fireCount, outDeg, inDeg) {
  let ds = 0, ns = 0, df = 0, nf = 0;
  for (let i = 0; i < fireCount.length; i++) {
    const d = outDeg[i] + inDeg[i];
    if (fireCount[i] === 0) { ds += d; ns++; } else { df += d; nf++; }
  }
  return { silent: ns ? ds / ns : 0, firing: nf ? df / nf : 0, nSilent: ns, nFiring: nf };
}

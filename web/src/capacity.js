// Live associative-capacity measurement -- the falsification mechanism for the claim's second half.
//
// The claim says capacity is set by how much the keys OVERLAP, not by the state's size. That is
// testable in one click: flip the keys from non-negative (what BDH's ReLU produces) to signed.
// If capacity does not improve, the claim is wrong.
//
// This is a JS port of research/sigma_capacity.py and is validated against it in
// web/test/capacity.mjs. No training is involved: writing k bindings into an N x D state and
// reading them back is a property of the state and the key geometry alone, which is exactly why
// it is the honest way to measure capacity. (Measuring it through a trained model conflates
// capacity with demonstration coverage -- see CLAUDE.md item 9.)

/** Deterministic RNG so the page shows the same curve on every load and in tests. */
export function rng(seed = 1) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const gauss = (r) => {
  const u = Math.max(r(), 1e-12), v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};

/** bdh.py's frequency table: dims 2i and 2i+1 share a frequency. */
export function freqTable(N, theta = 65536) {
  const f = new Float64Array(N);
  for (let i = 0; i < N; i++) f[i] = 1 / Math.pow(theta, (Math.floor(i / 2) * 2) / N) / (2 * Math.PI);
  return f;
}

function ropeRow(v, N, pos, freqs) {
  const o = new Float64Array(N);
  for (let i = 0; i < N; i += 2) {
    let ph = (pos * freqs[i]) % 1; if (ph < 0) ph += 1;
    const a = ph * 2 * Math.PI, c = Math.cos(a), s = Math.sin(a);
    o[i] = v[i] * c - v[i + 1] * s;
    o[i + 1] = v[i + 1] * c + v[i] * s;
  }
  return o;
}

/**
 * Write k bindings into an N x D state and read every key back.
 * Retrieval is correct when the true value wins on cosine among all k stored values.
 * @returns fraction of the k bindings retrieved correctly
 */
export function capacityTrial(k, { N = 256, D = 64, nonneg = true, seed = 1 } = {}) {
  const r = rng(seed), freqs = freqTable(N);
  const keys = [], vals = [];
  for (let i = 0; i < k; i++) {
    const key = new Float64Array(N);
    for (let n = 0; n < N; n++) {
      const g = gauss(r);
      key[n] = nonneg ? (g > 0 ? g : 0) : g;      // ReLU'd == what BDH's x_sparse looks like
    }
    keys.push(ropeRow(key, N, i, freqs));
    const val = new Float64Array(D);
    for (let d = 0; d < D; d++) val[d] = gauss(r);
    vals.push(val);
  }

  const sigma = new Float64Array(N * D);
  for (let i = 0; i < k; i++) {
    const kr = keys[i], v = vals[i];
    for (let n = 0; n < N; n++) {
      const kn = kr[n]; if (kn === 0) continue;
      for (let d = 0; d < D; d++) sigma[n * D + d] += kn * v[d];
    }
  }

  const norm = (a) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s) || 1e-12; };
  const vn = vals.map((v) => norm(v));
  let hits = 0;
  const read = new Float64Array(D);
  for (let j = 0; j < k; j++) {
    read.fill(0);
    const kr = keys[j];
    for (let n = 0; n < N; n++) {
      const kn = kr[n]; if (kn === 0) continue;
      for (let d = 0; d < D; d++) read[d] += kn * sigma[n * D + d];
    }
    const rn = norm(read);
    let best = 0, bestSim = -Infinity;
    for (let i = 0; i < k; i++) {
      let dot = 0;
      for (let d = 0; d < D; d++) dot += read[d] * vals[i][d];
      const sim = dot / (rn * vn[i]);
      if (sim > bestSim) { bestSim = sim; best = i; }
    }
    if (best === j) hits++;
  }
  return hits / k;
}

/** Mean pairwise cosine between keys -- the mechanism. ReLU'd Gaussians give 1/pi analytically. */
export function meanCosine({ N = 256, nonneg = true, pairs = 400, seed = 7 } = {}) {
  const r = rng(seed);
  let acc = 0;
  for (let p = 0; p < pairs; p++) {
    const a = new Float64Array(N), b = new Float64Array(N);
    for (let n = 0; n < N; n++) {
      const g1 = gauss(r), g2 = gauss(r);
      a[n] = nonneg ? (g1 > 0 ? g1 : 0) : g1;
      b[n] = nonneg ? (g2 > 0 ? g2 : 0) : g2;
    }
    let dot = 0, na = 0, nb = 0;
    for (let n = 0; n < N; n++) { dot += a[n] * b[n]; na += a[n] * a[n]; nb += b[n] * b[n]; }
    acc += dot / (Math.sqrt(na) * Math.sqrt(nb) || 1e-12);
  }
  return acc / pairs;
}

export const ANALYTIC_COSINE = 1 / Math.PI;   // ReLU'd Gaussians: E[x]^2/E[x^2] = (1/2pi)/(1/2)

/** Curve over k, averaged over `trials` seeds. Cheap: a few hundred ms for the default sweep. */
export function capacityCurve(ks, { trials = 6, ...opts } = {}) {
  return ks.map((k) => {
    let s = 0;
    for (let t = 0; t < trials; t++) s += capacityTrial(k, { ...opts, seed: 1000 + t * 17 });
    return { k, acc: s / trials };
  });
}

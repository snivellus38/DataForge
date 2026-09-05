// Browser-side BDH forward pass. Port of the vendored, unmodified pathwaycom/bdh (MIT).
//
// Written by hand rather than run through ONNX for one reason: the artifact needs the synaptic
// state sigma exposed while it is being written, and an opaque runtime cannot give us that.
//
// Shapes, with D = n_embd, N = mult*D/nh neurons per head:
//   x        (T, D)          residual stream, shared across heads (bdh.py keeps it at (B,1,T,D))
//   x_sparse (nh, T, N)      ReLU'd, non-negative -- this is BOTH query and key (K is Q)
//   sigma    (nh, N, D)      the fixed-size synaptic state
//
// bdh.py is WEIGHT-TIED across layers: encoder/encoder_v/decoder are reused every iteration,
// so n_layer is an iteration count of one shared operator, not depth.

const EPS = 1e-5; // torch.nn.LayerNorm default

export function layerNorm(v, off, D) { // in-place over one D-vector; no affine (elementwise_affine=False)
  let m = 0; for (let i = 0; i < D; i++) m += v[off + i];
  m /= D;
  let s = 0; for (let i = 0; i < D; i++) { const d = v[off + i] - m; s += d * d; }
  const inv = 1 / Math.sqrt(s / D + EPS);
  for (let i = 0; i < D; i++) v[off + i] = (v[off + i] - m) * inv;
}

// out(T,B) = a(T,A) @ b(A,B)
function matmul(a, b, T, A, B, out) {
  out.fill(0);
  for (let t = 0; t < T; t++) {
    const ao = t * A, oo = t * B;
    for (let i = 0; i < A; i++) {
      const av = a[ao + i]; if (av === 0) continue;   // x_sparse is ~half zeros; this pays
      const bo = i * B;
      for (let j = 0; j < B; j++) out[oo + j] += av * b[bo + j];
    }
  }
}

// RoPE, matching bdh.py: dims 2i and 2i+1 share a frequency (quantize(t,2) = floor(t/2)*2),
// rotating each pair by phase = (t * freq mod 1) * 2*pi.
function ropeInPlace(v, T, N, freqs) {
  for (let t = 0; t < T; t++) {
    const o = t * N;
    for (let i = 0; i < N; i += 2) {
      let ph = (t * freqs[i]) % 1; if (ph < 0) ph += 1;
      const a = ph * 2 * Math.PI, c = Math.cos(a), s = Math.sin(a);
      const v0 = v[o + i], v1 = v[o + i + 1];
      v[o + i]     = v0 * c - v1 * s;
      v[o + i + 1] = v1 * c + v0 * s;
    }
  }
}

export class BDHModel {
  constructor(manifest, buffer) {
    this.m = manifest;
    const c = manifest.config;
    this.D = c.n_embd; this.nh = c.n_head; this.L = c.n_layer;
    this.V = c.vocab_size; this.N = manifest.N;
    this.w = {};
    for (const t of manifest.tensors) {
      const n = t.bytes / (t.dtype === "float32" ? 4 : 2);
      const raw = t.dtype === "float32"
        ? new Float32Array(buffer, t.offset, n)
        : f16ToF32(new Uint16Array(buffer, t.offset, n));
      this.w[t.name] = raw;
    }
    this.freqs = this.w["attn.freqs"];
  }

  /** Runs the forward pass. Returns {logits, sigma, xSparse} -- sigma is (nh, N, D) per layer. */
  forward(tokens, { keepSigma = true } = {}) {
    const { D, nh, N, L, V } = this;
    const T = tokens.length;
    const x = new Float32Array(T * D);
    for (let t = 0; t < T; t++)
      for (let d = 0; d < D; d++) x[t * D + d] = this.w["embed.weight"][tokens[t] * D + d];
    for (let t = 0; t < T; t++) layerNorm(x, t * D, D);

    // xs holds the UN-ROPED x_sparse. bdh.py applies rope inside Attention.forward, which
    // returns a new tensor, so the x_sparse used by the multiplicative gate below is never
    // rotated. Roping in place here silently corrupts the gate (caught by the parity gate).
    const xs = new Float32Array(nh * T * N), qr = new Float32Array(T * N);
    const yKV = new Float32Array(nh * T * D), xy = new Float32Array(T * nh * N);
    const yMLP = new Float32Array(T * D), tmp = new Float32Array(T * N);
    const sigmas = [];

    for (let l = 0; l < L; l++) {
      for (let h = 0; h < nh; h++) {
        // x_sparse = relu(x @ encoder[h]) ; then rope. K is Q.
        matmul(x, this.w["encoder"].subarray(h * D * N, (h + 1) * D * N), T, D, N, tmp);
        for (let i = 0; i < T * N; i++) tmp[i] = tmp[i] > 0 ? tmp[i] : 0;
        xs.set(tmp, h * T * N);          // un-roped: feeds the gate
        qr.set(tmp); ropeInPlace(qr, T, N, this.freqs);   // roped: Q and K for attention

        // sigma_t = sum_{s<t} QR_s (outer) V_s ; out_t = sigma_t^T QR_t ; V = x.
        // Strictly causal (bdh.py uses tril(diagonal=-1)), so position 0 reads an empty state.
        const sig = new Float32Array(N * D);
        const ho = h * T * D;
        for (let t = 0; t < T; t++) {
          const qo = t * N, oo = ho + t * D;
          for (let n = 0; n < N; n++) {
            const q = qr[qo + n]; if (q === 0) continue;
            const so = n * D;
            for (let d = 0; d < D; d++) yKV[oo + d] += q * sig[so + d];
          }
          for (let n = 0; n < N; n++) {           // write AFTER reading -> s < t
            const q = qr[qo + n]; if (q === 0) continue;
            const so = n * D, xo = t * D;
            for (let d = 0; d < D; d++) sig[so + d] += q * x[xo + d];
          }
        }
        if (keepSigma) sigmas.push({ layer: l, head: h, sigma: sig });
        for (let t = 0; t < T; t++) layerNorm(yKV, ho + t * D, D);

        // y_sparse = relu(ln(yKV) @ encoder_v[h]) ; xy = x_sparse * y_sparse
        matmul(yKV.subarray(ho, ho + T * D),
               this.w["encoder_v"].subarray(h * D * N, (h + 1) * D * N), T, D, N, tmp);
        for (let i = 0; i < T * N; i++) tmp[i] = tmp[i] > 0 ? tmp[i] : 0;
        for (let t = 0; t < T; t++)
          for (let n = 0; n < N; n++)
            xy[t * nh * N + h * N + n] = xs[h * T * N + t * N + n] * tmp[t * N + n];
      }
      matmul(xy, this.w["decoder"], T, nh * N, D, yMLP);   // (T, nh*N) @ (nh*N, D)
      for (let t = 0; t < T; t++) layerNorm(yMLP, t * D, D);
      for (let i = 0; i < T * D; i++) x[i] += yMLP[i];
      for (let t = 0; t < T; t++) layerNorm(x, t * D, D);
      yKV.fill(0);
    }

    const logits = new Float32Array(T * V);
    matmul(x, this.w["lm_head"], T, D, V, logits);
    return { logits, T, sigmas, xSparse: xs };
  }
}

function f16ToF32(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i], s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
    out[i] = e === 0 ? (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024)
      : e === 0x1f ? (f ? NaN : (s ? -Infinity : Infinity))
      : (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return out;
}

/**
 * The artifact's centrepiece: compute one attention layer BOTH ways and return the residual.
 *
 *   parallel   scores = (rope(Q) @ rope(K)^T).tril(-1) ; out = scores @ V     <- bdh.py as written
 *   recurrent  sigma += rope(K)_t (outer) V_t ; out_t = sigma_t^T rope(Q)_t   <- the Hebbian form
 *
 * These agree to float precision, which is what makes the substrate demonstrably real rather
 * than animated. Verified against PyTorch at float64: 2.8e-14 (research/verify_equivalence.py).
 *
 * The break-it options exist to show WHICH property buys what -- measured, not assumed:
 *   softmax:true     equivalence DIES. Normalisation couples every score to every other, so the
 *                    state can no longer be a fixed-size running sum. This is why a Transformer
 *                    needs a KV cache that grows with context.
 *   decoupleQK:true  equivalence SURVIVES. Q=K is not what buys constant memory; it is what makes
 *                    sigma readable as a Hebbian synapse matrix over ONE neuron basis.
 *   noRelu:true      equivalence SURVIVES. Non-negativity buys sparsity and interpretability
 *                    (and costs associative capacity at a rate of 1/pi), not constant memory.
 */
export function equivalenceCheck(model, tokens, opts = {}) {
  const { softmax = false, decoupleQK = false, noRelu = false, layer = 0, head = 0 } = opts;
  const { D, nh, N } = model;
  const T = tokens.length;

  const x = new Float32Array(T * D);
  for (let t = 0; t < T; t++)
    for (let d = 0; d < D; d++) x[t * D + d] = model.w["embed.weight"][tokens[t] * D + d];
  for (let t = 0; t < T; t++) layerNorm(x, t * D, D);

  const proj = (W) => {
    const o = new Float32Array(T * N);
    for (let t = 0; t < T; t++)
      for (let i = 0; i < D; i++) {
        const v = x[t * D + i]; if (v === 0) continue;
        for (let j = 0; j < N; j++) o[t * N + j] += v * W[i * N + j];
      }
    if (!noRelu) for (let i = 0; i < T * N; i++) o[i] = o[i] > 0 ? o[i] : 0;
    return o;
  };
  const encH = model.w["encoder"].subarray(head * D * N, (head + 1) * D * N);
  const encV = model.w["encoder_v"].subarray(head * D * N, (head + 1) * D * N);
  const q = proj(encH);
  const k = decoupleQK ? proj(encV) : q;          // a real learned projection, not noise
  const qr = Float32Array.from(q), kr = Float32Array.from(k);
  ropeInPlace(qr, T, N, model.freqs);
  ropeInPlace(kr, T, N, model.freqs);

  // --- parallel (bdh.py as written) ---
  const scores = new Float32Array(T * T);
  for (let t = 0; t < T; t++)
    for (let s = 0; s < t; s++) {                 // tril(diagonal=-1): strictly causal
      let acc = 0;
      for (let n = 0; n < N; n++) acc += qr[t * N + n] * kr[s * N + n];
      scores[t * T + s] = acc;
    }
  if (softmax) {
    for (let t = 0; t < T; t++) {
      let mx = -Infinity;
      for (let s = 0; s < t; s++) mx = Math.max(mx, scores[t * T + s]);
      let sum = 0;
      for (let s = 0; s < t; s++) { const e = Math.exp(scores[t * T + s] - mx); scores[t * T + s] = e; sum += e; }
      if (sum > 0) for (let s = 0; s < t; s++) scores[t * T + s] /= sum;
    }
  }
  const par = new Float32Array(T * D);
  for (let t = 0; t < T; t++)
    for (let s = 0; s < t; s++) {
      const w = scores[t * T + s]; if (w === 0) continue;
      for (let d = 0; d < D; d++) par[t * D + d] += w * x[s * D + d];
    }

  // --- recurrent (materialised sigma) ---
  const rec = new Float32Array(T * D), sigma = new Float32Array(N * D);
  const snapshots = [];
  for (let t = 0; t < T; t++) {
    for (let n = 0; n < N; n++) {
      const qq = qr[t * N + n]; if (qq === 0) continue;
      for (let d = 0; d < D; d++) rec[t * D + d] += qq * sigma[n * D + d];
    }
    for (let n = 0; n < N; n++) {
      const kk = kr[t * N + n]; if (kk === 0) continue;
      for (let d = 0; d < D; d++) sigma[n * D + d] += kk * x[t * D + d];
    }
    snapshots.push(sigma.slice());               // sigma after each token, for the write animation
  }

  let maxAbs = 0, scale = 0;
  for (let i = 0; i < T * D; i++) {
    maxAbs = Math.max(maxAbs, Math.abs(par[i] - rec[i]));
    scale = Math.max(scale, Math.abs(par[i]));
  }
  return { maxAbs, relative: maxAbs / (scale || 1), parallel: par, recurrent: rec,
           sigma, snapshots, T, N, D, opts: { softmax, decoupleQK, noRelu } };
}

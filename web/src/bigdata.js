// Reader for the 8M-model packs (field.bin / traces.bin) written by research/export_field.py
// and research/export_traces.py.
//
// PURE FUNCTIONS ONLY -- no DOM, no WebGL, no fetch. The renderer sits on top of this and stays
// thin, so everything that can actually be wrong (offsets, quantisation, sparse indexing) is
// reachable from Node and is covered by web/test/bigdata.mjs. CLAUDE.md item 15: a suite that
// never executes the real decode path is not testing the product.
//
// Both packs share one layout: a JSON manifest listing {name, dtype, shape, offset, bytes} and
// a single little-endian blob. Same idea as model.json/model.bin, so there is one thing to learn.

const CTOR = {
  "<u2": Uint16Array, "|u1": Uint8Array, "<u4": Uint32Array,
  "<f4": Float32Array, "<i4": Int32Array,
};

/** Map a manifest + ArrayBuffer to { name -> typed array view }. Views, not copies. */
export function loadPack(manifest, buffer) {
  const t = {};
  for (const e of manifest.tensors) {
    const C = CTOR[e.dtype];
    if (!C) throw new Error(`bigdata: unhandled dtype ${e.dtype} for ${e.name}`);
    if (e.offset % C.BYTES_PER_ELEMENT !== 0) {
      // A misaligned offset would make the typed-array view throw or silently read garbage.
      t[e.name] = new C(buffer.slice(e.offset, e.offset + e.bytes));
    } else {
      t[e.name] = new C(buffer, e.offset, e.bytes / C.BYTES_PER_ELEMENT);
    }
  }
  return { manifest, t };
}

/** Un-quantise a uint16 position array back to model coordinates using the manifest's box. */
export function positions(pack, name) {
  const e = pack.manifest.tensors.find((x) => x.name === name);
  if (!e) throw new Error(`bigdata: no tensor ${name}`);
  if (!e.box) throw new Error(`bigdata: ${name} has no box; it is not a position array`);
  const q = pack.t[name];
  const [lo, hi] = e.box;
  const out = new Float32Array(q.length);
  const sx = (hi[0] - lo[0]) / 65535, sy = (hi[1] - lo[1]) / 65535;
  for (let i = 0; i < q.length; i += 2) {
    out[i] = lo[0] + q[i] * sx;
    out[i + 1] = lo[1] + q[i + 1] * sy;
  }
  return out;
}

/**
 * The active neurons at one token of one sentence/layer.
 * Returns { idx, val, n } where idx are GLOBAL neuron ids (h*N + n, matching global.xy) and val
 * are activations rescaled out of uint8. Zero-copy on idx; val is allocated per call.
 */
export function frame(pack, sentence, layer, t) {
  const p = `s${sentence}.L${layer}.`;
  const off = pack.t[p + "off"], idx = pack.t[p + "idx"], raw = pack.t[p + "val"];
  if (!off) throw new Error(`bigdata: sentence ${sentence} layer ${layer} not in this pack`);
  if (t < 0 || t >= off.length - 1) throw new Error(`bigdata: token ${t} out of range`);
  const a = off[t], b = off[t + 1], scale = pack.t[p + "scale"][t] / 255;
  const val = new Float32Array(b - a);
  for (let i = a; i < b; i++) val[i - a] = raw[i] * scale;
  return { idx: idx.subarray(a, b), val, n: b - a };
}

/** Number of tokens available for a sentence/layer. */
export function nTokens(pack, sentence, layer) {
  const off = pack.t[`s${sentence}.L${layer}.off`];
  return off ? off.length - 1 : 0;
}

/** Per-neuron ||sigma_n|| at token t, rescaled. Hero sentence only. */
export function sigmaEnergy(pack, sentence, t) {
  const e = pack.manifest.tensors.find((x) => x.name === `s${sentence}.sigma_energy`);
  if (!e) throw new Error(`bigdata: sentence ${sentence} has no sigma energy`);
  const N = e.shape[1];
  const raw = pack.t[`s${sentence}.sigma_energy`];
  const scale = pack.t[`s${sentence}.sigma_energy_scale`][t] / 255;
  const out = new Float32Array(N);
  for (let i = 0; i < N; i++) out[i] = raw[t * N + i] * scale;
  return out;
}

/** Causal attention scores as {i, j, w} triples. w is signed -- the negatives are the point. */
export function attention(pack, sentence) {
  const i = pack.t[`s${sentence}.attn_i`];
  if (!i) throw new Error(`bigdata: sentence ${sentence} has no attention`);
  return { i, j: pack.t[`s${sentence}.attn_j`], w: pack.t[`s${sentence}.attn_w`] };
}

/** Fraction of causal attention scores below zero -- RoPE breaking score non-negativity. */
export function negativeShare(pack, sentence) {
  const { w } = attention(pack, sentence);
  let n = 0;
  for (let i = 0; i < w.length; i++) if (w[i] < 0) n++;
  return n / w.length;
}

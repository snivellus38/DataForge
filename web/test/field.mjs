// PHASE B GATE: the field's logic, tested where there is no GPU.
//
// field-gl.js is buffer plumbing; everything that can be quietly wrong lives in field-data.js.
// The edge test is the important one: G* is exported per head with head-local indices while the
// field is indexed globally, so an off-by-N there would draw confident, plausible, wrong synapses.
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { loadPack, positions } from "../src/bigdata.js";
import {
  rampStops, oklabL, decayInto, tokenFrame, referenceLevel,
  nearestNode, edgesFor, communityColor, degreeLevel, arcsInto,
} from "../src/field-data.js";
import { IGNITION, SIGMA_RAMP, COMMUNITY, COMMUNITY_OTHER } from "../src/palette.js";
import { measure, meanPairwiseCosine, sampleKeys, mulberry32, INV_PI } from "../src/orthant.js";

const rd = (p) => {
  let b = readFileSync(p);
  if (p.endsWith(".gz")) b = gunzipSync(b);          // traces ships gzipped; the layout is unchanged
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};
const field = loadPack(JSON.parse(readFileSync("web/public/big/field.json", "utf8")), rd("web/public/big/field.bin"));
const traces = loadPack(JSON.parse(readFileSync("web/public/big/traces.json", "utf8")), rd("web/public/big/traces.bin.gz"));

let fail = 0;
const check = (ok, msg) => { if (!ok) { fail++; console.log(`  FAIL ${msg}`); } };

// ── 1. sequential ramps must be monotonic in lightness ──────────────────────────────────────
// "Sequential = one hue, light->dark" is a hard rule; a non-monotonic ramp encodes magnitude as
// something the eye cannot order. Computed, not eyeballed.
for (const [name, ramp] of [["ignition", IGNITION], ["sigma", SIGMA_RAMP]]) {
  const L = ramp.map(oklabL);
  const mono = L.every((v, i) => i === 0 || v > L[i - 1]);
  check(mono, `${name} ramp not monotonic in OKLab L: ${L.map((v) => v.toFixed(3)).join(" ")}`);
  console.log(`ramp ${name.padEnd(9)} L = ${L.map((v) => v.toFixed(3)).join(" -> ")}  ${mono ? "monotonic" : "NOT MONOTONIC"}`);
}
check(rampStops("ignition").length === 4, "rampStops must yield 4 shader stops");
check(rampStops("ignition")[0].length === 3, "each stop must be rgb");

// ── 2. decay: attack instant, release exponential ───────────────────────────────────────────
const act = new Float32Array(8);
decayInto(act, { idx: Uint16Array.from([2]), val: Float32Array.from([1]), n: 1 }, 0.5);
check(act[2] === 1, `attack should be instant, got ${act[2]}`);
decayInto(act, null, 0.5);
check(Math.abs(act[2] - 0.5) < 1e-9, `release should halve at decay 0.5, got ${act[2]}`);
decayInto(act, { idx: Uint16Array.from([2]), val: Float32Array.from([0.1]), n: 1 }, 0.5);
check(Math.abs(act[2] - 0.25) < 1e-9, `a weaker re-fire must not brighten a decaying trail, got ${act[2]}`);
check(act[0] === 0, "untouched neurons must stay dark");
console.log("decay      attack instant, release exponential, weak re-fire does not brighten");

// ── 3. token frames are normalised against a stable reference, not per-frame ─────────────────
const hero = traces.manifest.hero, layer = traces.manifest.default_layer;
const T = traces.manifest.sentences.find((s) => s.i === hero).T;
const ref = referenceLevel(traces, hero, layer, T);
check(ref > 0, "reference level must be positive");
let maxSeen = 0, anyBelow = false;
for (let t = 0; t < T; t += 7) {
  const f = tokenFrame(traces, hero, layer, t, ref);
  for (let i = 0; i < f.n; i++) { maxSeen = Math.max(maxSeen, f.val[i]); if (f.val[i] < 0.5) anyBelow = true; }
}
check(maxSeen <= 1.0000001, `normalised activation exceeded 1 (${maxSeen})`);
check(anyBelow, "every activation saturated -- reference level is too low to discriminate");
console.log(`frames     p99 reference ${ref.toFixed(3)}, normalised max ${maxSeen.toFixed(3)}, dynamic range preserved`);

// ── 4. hit testing ──────────────────────────────────────────────────────────────────────────
const xy = positions(field, "global.xy");
const pick = 5000;
const hit = nearestNode(xy, xy[pick * 2], xy[pick * 2 + 1]);
check(hit === pick, `nearestNode returned ${hit}, expected ${pick}`);
check(nearestNode(xy, 1e9, 1e9, 1) === -1, "nearestNode must respect maxDist");
console.log(`hit test   exact node recovered, maxDist respected`);

// ── 5. THE ONE THAT MATTERS: per-head edge indices -> global field ids ───────────────────────
const N = field.manifest.N_per_head, nNodes = field.manifest.n_neurons_total;
let checked = 0;
for (const head of [0, 3]) {
  const ei = field.t[`h${head}.edge_i`];
  const gid = head * N + ei[0];                       // a neuron known to have synapses
  const e = edgesFor(field, gid, N);
  check(e.i.length > 0, `no edges found for a neuron that has them (head ${head})`);
  for (let k = 0; k < e.i.length; k++) {
    check(e.i[k] < nNodes && e.j[k] < nNodes, `edge id out of field range (head ${head})`);
    check(Math.floor(e.i[k] / N) === head && Math.floor(e.j[k] / N) === head,
          `edge crossed heads: ${e.i[k]}->${e.j[k]} is not in head ${head}`);
    check(e.i[k] === gid || e.j[k] === gid, `edge ${e.i[k]}->${e.j[k]} does not touch ${gid}`);
    check(Math.abs(e.w[k]) <= 1.0000001, `edge weight not normalised: ${e.w[k]}`);
  }
  checked += e.i.length;
}
// a neuron with no synapses must yield nothing, not throw and not invent
const placed = field.t["h0.xy_graph_placed"];
const isolated = placed.indexOf(0);
check(isolated >= 0, "expected at least one isolated neuron in head 0");
check(edgesFor(field, isolated, N).i.length === 0, `isolated neuron ${isolated} returned edges`);
console.log(`edges      ${checked} edges checked, all within head, all touching the source, normalised`);

// ── 5b. arcsInto: the contiguous-row shortcut must equal the honest scan ─────────────────────
// arcsInto assumes np.tril_indices(T,-1) is row-major, so row t lives at offset t*(t-1)/2 with
// exactly t entries. If that is off by one the arcs are drawn from the wrong bytes -- confidently,
// and with no visible symptom. So verify against the exported attn_i / attn_j directly.
{
  // Every sentence, not just the hero: attention is exported for all seven now, and the row
  // offset t*(t-1)/2 depends on T, so a sentence of a different length is a genuinely different
  // case. The hero happened to be the only one this ever ran on.
  let checkedRows = 0, worstDiff = 0, checkedSentences = 0;
  for (const meta of traces.manifest.sentences) {
    const si = meta.i;
    const ai = traces.t[`s${si}.attn_i`], aj = traces.t[`s${si}.attn_j`];
    const aw = traces.t[`s${si}.attn_w`];
    check(!!aw, `sentence ${si} has no attention scores to check`);
    if (!aw) continue;
    checkedSentences++;
    for (const t of [1, 2, 17, Math.floor(meta.T / 2), meta.T - 1]) {
      if (t < 1 || t >= meta.T) continue;
      // ground truth: scan every stored score for those whose destination row is t
      const truth = new Map();
      for (let k = 0; k < ai.length; k++) if (ai[k] === t) truth.set(aj[k], aw[k]);
      const got = arcsInto(traces, si, t, 1e9);
      check(got.from.length === truth.size,
            `arcsInto(s${si}, t=${t}) returned ${got.from.length} arcs, scan found ${truth.size}`);
      for (let k = 0; k < got.from.length; k++) {
        const want = truth.get(got.from[k]);
        check(want !== undefined, `arcsInto(s${si}, t=${t}) invented a source byte ${got.from[k]}`);
        worstDiff = Math.max(worstDiff, Math.abs(want - got.w[k]));
      }
      // negShare must be computed over the whole row, not just the kept subset
      let neg = 0;
      for (const v of truth.values()) if (v < 0) neg++;
      check(Math.abs(got.negShare - neg / truth.size) < 1e-9,
            `arcsInto(s${si}, t=${t}) negShare ${got.negShare} != ${neg / truth.size}`);
      checkedRows++;
    }
  }
  check(worstDiff === 0, `arcsInto returned different weights than the scan (max ${worstDiff})`);
  // and the limit must keep the STRONGEST, not the first n
  const heroS = traces.manifest.hero;
  const lim = arcsInto(traces, heroS, 173, 10);
  const all = arcsInto(traces, heroS, 173, 1e9);
  const top = [...all.w].map(Math.abs).sort((a, b) => b - a).slice(0, 10);
  check(lim.from.length === 10, `limit ignored (${lim.from.length} arcs)`);
  check(Math.abs(Math.min(...[...lim.w].map(Math.abs)) - top[9]) < 1e-9,
        "limit kept the first arcs, not the strongest");
  console.log(`arcs       ${checkedRows} rows across ${checkedSentences} sentences match an ` +
              `exhaustive scan exactly; limit keeps the strongest; ` +
              `hero t=173 negShare ${(all.negShare * 100).toFixed(1)}%`);
}

// ── 5c. every sentence animates: six iterations, each with a usable brightness reference ────
// tokenFrame divides by referenceLevel, so a sentence/layer that is absent throws inside frame()
// and a sentence whose reference came out 0 renders a field of NaN. Both were reachable on six
// of the seven sentences before the export carried them.
{
  let worstFrac = 1, iters = 0;
  for (const meta of traces.manifest.sentences) {
    for (const L of meta.layers) {
      const r = referenceLevel(traces, meta.i, L.layer, meta.T);
      check(Number.isFinite(r) && r > 0, `s${meta.i} L${L.layer}: reference level ${r}`);
      const f = tokenFrame(traces, meta.i, L.layer, Math.floor(meta.T / 2), r);
      check(f.n > 0, `s${meta.i} L${L.layer}: nothing fires mid-sentence`);
      check([...f.val].every((v) => Number.isFinite(v) && v >= 0 && v <= 1),
            `s${meta.i} L${L.layer}: activations outside 0..1 after the gamma`);
      worstFrac = Math.min(worstFrac, f.n / field.manifest.n_neurons_total);
      iters++;
    }
  }
  console.log(`frames     ${iters} sentence-iterations decode; sparsest mid-sentence frame ` +
              `${(worstFrac * 100).toFixed(2)}% active`);
}

// ── 6. community colours obey the 3-slot cap ────────────────────────────────────────────────
check(communityColor(0) === COMMUNITY[0] && communityColor(2) === COMMUNITY[2], "top-3 must get hues");
check(communityColor(3) === COMMUNITY_OTHER && communityColor(255) === COMMUNITY_OTHER,
      "4th community onward must fold to 'other', never a generated hue");
const cl = field.t["h0.cluster"];
let hued = 0; for (let i = 0; i < cl.length; i++) if (cl[i] < 3) hued++;
console.log(`community  3 hues + other; ${hued}/${cl.length} of head 0 falls in the top 3`);

// ── 7. degree scaling ───────────────────────────────────────────────────────────────────────
const deg = field.t["h0.out_deg"];
let maxDeg = 0; for (const d of deg) maxDeg = Math.max(maxDeg, d);
check(degreeLevel(0, maxDeg) === 0, "degree 0 must map to 0");
check(Math.abs(degreeLevel(maxDeg, maxDeg) - 1) < 1e-9, "max degree must map to 1");
check(degreeLevel(maxDeg / 2, maxDeg) > 0.5, "log scaling must lift the middle of a heavy tail");
console.log(`degree     max ${maxDeg}, log-scaled so the tail is visible`);

// ── 8. the orthant: 1/pi is the claim's second half, so measure it ──────────────────────────
{
  // ReLU'd keys in the model's own dimension must land on arccos(1/pi); signed keys on 90.
  const hi = measure(160, 3072, true, 7), hiS = measure(160, 3072, false, 7);
  check(Math.abs(hi.cos - INV_PI) < 0.02,
        `ReLU'd cos ${hi.cos.toFixed(4)} is not 1/pi = ${INV_PI.toFixed(4)}`);
  check(Math.abs(hiS.cos) < 0.02, `signed cos ${hiS.cos.toFixed(4)} is not ~0`);
  check(hiS.deg - hi.deg > 12, "the ReLU'd/signed angle gap collapsed -- the claim depends on it");
  console.log(`orthant    d=3072  ReLU'd cos ${hi.cos.toFixed(4)} (${hi.deg.toFixed(1)} deg)  ` +
              `signed ${hiS.cos.toFixed(4)} (${hiS.deg.toFixed(1)} deg)  analytic 1/pi ${INV_PI.toFixed(4)}`);

  // 1/pi is the LARGE-d limit. At d=3 the crowding is STRONGER, and an earlier draft of
  // orthant.js claimed the constant was dimension-free -- it is not. Pin the direction of the
  // discrepancy so the on-screen note cannot silently become wrong.
  const lo = measure(160, 3, true, 7);
  check(lo.cos > hi.cos + 0.05,
        `d=3 cos ${lo.cos.toFixed(3)} should exceed the high-d limit ${hi.cos.toFixed(3)}; ` +
        `if this flips, the small-d note in orthant.js is wrong`);
  console.log(`           d=3     ReLU'd cos ${lo.cos.toFixed(4)} (${lo.deg.toFixed(1)} deg) ` +
              `-- stronger than the limit, as the panel says`);

  // determinism: same seed, same picture, or the figure moves between reloads
  check(measure(32, 3, true, 5).cos === measure(32, 3, true, 5).cos, "sampling is not deterministic");

  // the pair subsample must agree with the exhaustive mean it stands in for
  const V = sampleKeys(96, 256, true, mulberry32(3));
  const full = meanPairwiseCosine(V, 96, 256);
  const samp = meanPairwiseCosine(V, 96, 256, 1200, 4);
  check(Math.abs(full - samp) < 0.01,
        `pair subsampling drifted: exhaustive ${full.toFixed(4)} vs sampled ${samp.toFixed(4)}`);

  // every key must be a unit vector, and non-negative when ReLU'd
  const U = sampleKeys(64, 3, true, mulberry32(11));
  let badNorm = 0, negative = 0;
  for (let i = 0; i < 64; i++) {
    let n = 0;
    for (let c = 0; c < 3; c++) { const v = U[i * 3 + c]; n += v * v; if (v < 0) negative++; }
    if (Math.abs(Math.sqrt(n) - 1) > 1e-5) badNorm++;
  }
  check(badNorm === 0, `${badNorm} keys are not unit length`);
  check(negative === 0, `${negative} ReLU'd coordinates are negative`);
  console.log(`           64 keys unit-length, none negative; 1200-pair estimate within ` +
              `${Math.abs(full - samp).toExponential(1)} of exhaustive`);
}

console.log(fail === 0 ? "\nFIELD LOGIC PASSED" : `\nFIELD LOGIC FAILED (${fail} check(s))`);
process.exit(fail === 0 ? 0 : 1);

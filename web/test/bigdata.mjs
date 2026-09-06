// PHASE A GATE: the browser must decode the 8M-model packs to the numbers the model produced.
//
// The fixture carries activations as PyTorch computed them, before quantisation, so a pass here
// covers the whole chain: model -> sparse pack -> binary -> JS decode. It also re-derives the
// two headline numbers (active fraction, negative-score share) from the shipped bytes rather
// than trusting the manifest, because those numbers go on screen.
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { loadPack, positions, frame, nTokens, sigmaEnergy, negativeShare } from "../src/bigdata.js";

const rd = (p) => {
  let b = readFileSync(p);
  if (p.endsWith(".gz")) b = gunzipSync(b);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};
const traces = loadPack(JSON.parse(readFileSync("web/public/big/traces.json", "utf8")),
                        rd("web/public/big/traces.bin.gz"));
const field = loadPack(JSON.parse(readFileSync("web/public/big/field.json", "utf8")),
                       rd("web/public/big/field.bin"));
const fx = JSON.parse(readFileSync("web/test/big_fixture.json", "utf8"));

let fail = 0;
const check = (ok, msg) => { if (!ok) { fail++; console.log(`  FAIL ${msg}`); } };

// ── 1. activations round-trip within one quantisation step ──────────────────────────────────
let worst = 0;
for (const s of fx.samples) {
  const f = frame(traces, s.s, s.L, s.t);
  const at = f.idx.indexOf(s.neuron);
  if (at < 0) { fail++; console.log(`  FAIL neuron ${s.neuron} missing at s${s.s} L${s.L} t${s.t}`); continue; }
  if (f.n !== s.n_active) { fail++; console.log(`  FAIL active count ${f.n} != ${s.n_active}`); continue; }
  // one uint8 step is scale/255; allow half a step of rounding plus fp32 slack
  const tol = s.scale / 255 * 0.5 + 1e-6;
  const err = Math.abs(f.val[at] - s.value);
  worst = Math.max(worst, err / (s.scale / 255));
  check(err <= tol, `s${s.s} L${s.L} t${s.t} n${s.neuron}: ${f.val[at]} vs ${s.value} (tol ${tol})`);
}
console.log(`activations: ${fx.samples.length} spot checks, worst error ` +
            `${worst.toFixed(3)} quantisation steps (limit 0.5)`);

// ── 2. the sparsity claim, recomputed from the shipped bytes ─────────────────────────────────
const cfg = traces.manifest.config;
let active = 0, slots = 0;
for (const s of traces.manifest.sentences) {
  for (const L of s.layers) {
    const T = nTokens(traces, s.i, L.layer);
    check(T === s.T, `sentence ${s.i} layer ${L.layer}: ${T} tokens, manifest says ${s.T}`);
    for (let t = 0; t < T; t++) active += frame(traces, s.i, L.layer, t).n;
    slots += T * cfg.N_total;
  }
}
const frac = active / slots;
console.log(`sparsity:    ${(frac * 100).toFixed(2)}% of ${cfg.N_total.toLocaleString("en-US")} ` +
            `neurons active per token (${active.toLocaleString("en-US")} / ${slots.toLocaleString("en-US")})`);
check(Math.abs(frac - fx.measured_active_frac) < 1e-4,
      `recomputed ${frac} != manifest ${fx.measured_active_frac}`);
check(frac > 0.02 && frac < 0.10, `active fraction ${frac} outside the ~5% regime we claim`);

// ── 3. EVERY sentence carries everything the page needs ─────────────────────────
//
// This is the regression that shipped: the pack gave the hero sentence all six iterations,
// attention scores and sigma row energy, and gave the other six sentences one layer and nothing
// else -- so the iteration selector, the small-multiples view, the arc diagram and the sigma
// colour mode were dead on 6 of the 7 sentences. Nothing caught it, because every assertion in
// this file only ever looked at the hero. The coverage claim is asserted per sentence now.
const nLayers = traces.manifest.config.n_layer;
for (const s of traces.manifest.sentences) {
  check(s.layers.length === nLayers,
        `sentence ${s.i} has ${s.layers.length} iterations exported, needs all ${nLayers}`);
  for (let L = 0; L < nLayers; L++) {
    check(s.layers.some((x) => x.layer === L), `sentence ${s.i} is missing iteration ${L}`);
    check(nTokens(traces, s.i, L) === s.T, `sentence ${s.i} iteration ${L} has the wrong token count`);
  }
  check(!!traces.t[`s${s.i}.attn_w`], `sentence ${s.i} has no attention scores`);
  check(!!traces.t[`s${s.i}.sigma_energy`], `sentence ${s.i} has no sigma row energy`);
}
console.log(`coverage:    all ${traces.manifest.sentences.length} sentences carry ` +
            `${nLayers} iterations + attention + sigma energy`);

// ── 4. RoPE really does drive attention scores negative -- on every sentence ───────────
const hero = traces.manifest.hero;
let negHits = 0, negTotal = 0;
const perSentence = [];
for (const s of traces.manifest.sentences) {
  const w = traces.t[`s${s.i}.attn_w`];
  // strict-causal tril(-1) is T*(T-1)/2 scores; a mismatch means the row-major layout that
  // arcsInto() slices into is wrong, which would draw arcs from the wrong tokens
  check(w.length === (s.T * (s.T - 1)) / 2,
        `sentence ${s.i}: ${w.length} scores, expected ${(s.T * (s.T - 1)) / 2} for T=${s.T}`);
  const n = negativeShare(traces, s.i);
  perSentence.push(n);
  check(n > 0.01, `sentence ${s.i}: only ${n} of its causal scores are negative`);
  check(Math.abs(n - s.neg_share) < 1e-4,
        `sentence ${s.i}: manifest says ${s.neg_share}, bytes say ${n}`);
  for (let k = 0; k < w.length; k++) if (w[k] < 0) negHits++;
  negTotal += w.length;
}
const negAll = negHits / negTotal;
console.log(`attention:   ${(negAll * 100).toFixed(1)}% of all causal scores are NEGATIVE ` +
            `(per sentence ${(Math.min(...perSentence) * 100).toFixed(1)}% -- ` +
            `${(Math.max(...perSentence) * 100).toFixed(1)}%)`);
// The page prints the pooled figure under "corpus / all causal scores", so it has to BE the
// pooled figure. CLAUDE.md item 24: 38.5% is the hero sentence alone, not a corpus number.
check(Math.abs(negAll - traces.manifest.corpus_neg_share) < 1e-4,
      `manifest corpus_neg_share ${traces.manifest.corpus_neg_share} != recomputed ${negAll}`);
check(Math.max(...perSentence) - Math.min(...perSentence) > 0.05,
      "negative share is identical across sentences -- attention is probably not per-sentence");

// ── 5. sigma energy: non-negative and growing, on every sentence ──────────────────
const sum = (a) => a.reduce((x, y) => x + y, 0);
for (const s of traces.manifest.sentences) {
  const Ts = nTokens(traces, s.i, traces.manifest.default_layer);
  const a0 = sigmaEnergy(traces, s.i, 1), aN = sigmaEnergy(traces, s.i, Ts - 1);
  check(a0.every((v) => v >= 0), `sentence ${s.i}: sigma energy has negative entries`);
  check(sum(aN) > sum(a0), `sentence ${s.i}: sigma energy did not grow across the sequence`);
  // the recurrence the export accumulated must still reproduce the parallel attention it ran
  check(s.sigma_residual < 1e-3,
        `sentence ${s.i}: recurrent sigma vs parallel attention residual ${s.sigma_residual}`);
}
const Tn = nTokens(traces, hero, traces.manifest.default_layer);
const e0 = sigmaEnergy(traces, hero, 1), eN = sigmaEnergy(traces, hero, Tn - 1);
console.log(`sigma:       hero row energy ${sum(e0).toFixed(1)} at t=1 -> ${sum(eN).toFixed(1)} ` +
            `at t=${Tn - 1}; all sentences grow, worst recurrence residual ` +
            `${Math.max(...traces.manifest.sentences.map((s) => s.sigma_residual)).toExponential(1)}`);

// ── 6. the field: positions decode, and every trace index addresses a real node ──────────────
const xy = positions(field, "global.xy");
const nNodes = xy.length / 2;
check(nNodes === field.manifest.n_neurons_total, `${nNodes} positions != ${field.manifest.n_neurons_total}`);
check(xy.every(Number.isFinite), "global.xy contains non-finite coordinates");
let maxIdx = 0;
for (const s of traces.manifest.sentences) {
  for (const L of s.layers) {
    const f = frame(traces, s.i, L.layer, Math.floor(s.T / 2));
    for (const v of f.idx) maxIdx = Math.max(maxIdx, v);
  }
}
check(maxIdx < nNodes, `trace neuron id ${maxIdx} is outside the ${nNodes}-node field`);
console.log(`field:       ${nNodes.toLocaleString("en-US")} nodes, max trace id ${maxIdx} -- ids align`);

// the graph layout deliberately omits isolated neurons; the mask must say so
const placed = field.t["h0.xy_graph_placed"];
const nPlaced = placed.reduce((a, b) => a + b, 0);
check(nPlaced > 0 && nPlaced < placed.length,
      `h0.xy_graph_placed marks ${nPlaced}/${placed.length} -- expected a strict subset`);
console.log(`             h0 graph layout places ${nPlaced}/${placed.length} ` +
            `(${placed.length - nPlaced} isolated, by design)`);

// ── payload budget ──────────────────────────────────────────────────────────────────────────
// Raised from 5.00 MB when every sentence gained its six iterations, attention and sigma energy
// -- about 3x the raw content, 11.3 MB uncompressed. gzip holds the SHIPPED size to ~6.3 MB, so
// the page carries roughly three times the data for about half again the bytes. The budget is on
// the wire size, because that is what a viewer waits for.
const BUDGET = 6.5;
const mb = (p) => readFileSync(p).byteLength / 1e6;
const rawMB = gunzipSync(readFileSync("web/public/big/traces.bin.gz")).byteLength / 1e6;
const total = mb("web/public/big/traces.bin.gz") + mb("web/public/big/field.bin");
console.log(`payload:     ${total.toFixed(2)} MB shipped (budget ${BUDGET.toFixed(2)} MB) -- ` +
            `traces ${rawMB.toFixed(2)} MB raw, gzip ratio ` +
            `${(mb("web/public/big/traces.bin.gz") / rawMB).toFixed(2)}`);
check(total < BUDGET, `payload ${total.toFixed(2)} MB exceeds the ${BUDGET} MB budget`);
check(rawMB > total, "traces.bin.gz is not actually compressed");

console.log(fail === 0 ? "\nGATE PASSED" : `\nGATE FAILED (${fail} check(s))`);
process.exit(fail === 0 ? 0 : 1);

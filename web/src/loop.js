// THE LOOP — controller.
//
// Every number this page shows comes out of walk.bin, which came out of the 8M model via
// research/export_walk.py. Two of them are not read from the pack at all but recomputed here in
// the browser: the RoPE rotation (stop 2) and the sigma recurrence (stop 4). That is deliberate
// -- those are the two claims the page makes about the mechanism, so the page had better be able
// to do them rather than quote them.
//
// Structure: STOPS is the narration, one entry per stage; each has a render() that draws into
// its own pane. The diagram and the transport persist across all of them.

import { createDiagram } from "./diagram.js";
import { createFlow } from "./flow.js";
import { buildFrame, sigmaRows, preflight, xPre, FAN } from "./flow-data.js";
import {
  loadPack, sparseAt, rope, sigmaRun, deltaSigma, attnMatrix, negShare,
  softmaxRow, gateAt, topBytes, f16,
} from "./walkdata.js";
import { drawStrip, drawSparseMarks, drawScores, drawRotation, drawSigma, drawBars, drawLine, fit }
  from "./stages.js";

const $ = (id) => document.getElementById(id);
const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;

const S = {
  pack: null, man: null, diagram: null,
  stop: 0, w: 0, layer: 3, head: 0, t: 0, T: 0, flow: null, sig: null, vCache: null,
  playing: false, hover: null, sigmaCache: new Map(),
};

// ── narration ────────────────────────────────────────────────────────────────────────────────
const STOPS = [
  {
    id: "whole", title: "One block, run six times",
    h: "The whole block",
    body: `<p>This is an <b>8-million-parameter Dragon Hatchling</b> translating English into
      French, one byte at a time. Everything below is the model's own arithmetic, replayed.</p>
      <p>The diagram at the bottom is the entire architecture. Not a simplification of it &mdash;
      <em>all of it</em>. What makes BDH strange is the dashed arrow underneath: <b>encoder</b>,
      <b>D<sub>x</sub></b> and <b>D<sub>y</sub></b> are built once and reused, so
      <code>n_layer</code> is not depth. It is a count of how many times <em>one</em> operator
      runs.</p>
      <p>Walk the eight stops, or click any box.</p>
      <span class="src">Architecture transcribed in <code>research/bdh_big.py</code>; the
      checkpoint is the author's own prior work, disclosed in the README. Two deviations from
      reference BDH: a learned positional embedding on top of RoPE, and no decay term.</span>`,
  },
  {
    id: "encode", title: "192 numbers become 12,288, then almost all of them die",
    h: "Into neuron space — Dₓ, then the ReLU",
    body: `<p>The residual carrying this token is <b>192 numbers</b>. One matrix multiply fans it
      out to <b>3,072 per head</b> &mdash; 12,288 in total. Then a ReLU deletes everything
      negative.</p>
      <p>Almost everything is negative. Across the corpus <b>94.9%</b> of pre-activations are
      below zero, so only <em>5.1%</em> of neurons survive into the next stage.</p>
      <p>That sparsity is not a trick applied afterwards; it falls out of a rectifier on a wide
      layer. It is also what makes the rest of this page possible &mdash; a state you can read is
      a state that is mostly empty.</p>
      <span class="src">Measured over 7 sentences, 55,959,552 neuron-token slots, all six
      iterations. Our measurement, our model.</span>`,
  },
  {
    id: "rotate", title: "Two positive vectors, rotated apart",
    h: "RoPE — where the negative numbers come from",
    body: `<p>The surviving activations are used as <b>both</b> the query and the key.
      <code>Q = K = x</code>. That is what lets the state be read later as synapses over one
      basis of neurons rather than two unrelated ones.</p>
      <p>But <b>every value in x is non-negative</b>, and the dot product of two non-negative
      vectors can never be negative. A model built this way would have no way to say
      <em>"not this one"</em>.</p>
      <p>RoPE fixes that by accident. It rotates neuron pairs by an angle that depends on the
      position, so two keys from different positions no longer live in the same quadrant &mdash;
      and their dot product can go below zero.</p>
      <span class="src">The rotation drawn here is recomputed in your browser from
      <code>rope_freqs</code>, then checked against the model's own scores in
      <code>web/test/walk.mjs</code>.</span>`,
  },
  {
    id: "score", title: "Attention with nothing normalising it",
    h: "Q·K — the score matrix",
    body: `<p>Every token scores itself against every earlier token. There is <b>no softmax</b>,
      and the diagonal is empty: the mask is <code>tril(-1)</code>, so a token cannot attend to
      itself.</p>
      <p>Notice the blue cells. Those are <b>negative</b> scores &mdash; destructive
      interference, standing in for the decay term this model does not have.</p>
      <p>Now tick the softmax box. The row turns into a probability distribution, and in doing so
      it becomes <em>impossible</em> to compute from a fixed-size running sum: every score is
      normalised against every other, so you must keep them all. <b>That is why a Transformer
      needs a KV cache that grows with context, and this does not.</b></p>
      <span class="src">Q&ne;K and dropping the ReLU do <em>not</em> break the recurrence &mdash;
      only softmax does. Verified in <code>research/verify_equivalence.py</code>.</span>`,
  },
  {
    id: "state", title: "The state, written one rank-one update at a time",
    h: "σ — write, then read",
    body: `<p>Instead of keeping every past key, the model keeps one matrix. Each token adds a
      single <b>rank-one</b> update to it, and each token answers by reading it back.</p>
      <p>The panel on the left is <b>&sigma; accumulated in your browser</b>, from the shipped
      activations, by the recurrence in the box &mdash; not a picture of &sigma; that was shipped
      ready-made. The residual underneath is our answer checked against what the model's parallel
      attention actually produced.</p>
      <p>The raw state reads as texture, and that is honest: <b>&sigma; is dense within a few
      tokens</b>. The structure lives in the individual writes, so tick the box to see one.</p>
      <span class="src">This is BDH-CQ's equation (1) in its named special case,
      <span class="mono">S<sub>t</sub> = S<sub>t&minus;1</sub> + U<sub>&theta;</sub>(D<sub>t</sub>)</span>
      &mdash; arXiv:2608.09888 §3.2. Their general update rule is proprietary; this one is what
      the public code computes.</span>`,
  },
  {
    id: "gate", title: "A neuron only speaks if it was already speaking",
    h: "The gate — ReLU(Dᵧa) ⊙ x",
    body: `<p>The read-out comes back, gets projected into neuron space again, rectified again,
      and then <b>multiplied element-wise by the activations from the start of the block</b>.</p>
      <p>So it is an <b>AND</b>. A neuron contributes to the residual only if the attention
      read-out wants it <em>and</em> it was already firing. Two sparse conditions, intersected.</p>
      <p>The result is dramatic and, as far as we can tell, unreported: where <code>x</code> is
      <b>5.13%</b> active, <code>y</code> is <b>0.94%</b>. <em>Fewer than one neuron in a hundred
      reaches the residual.</em></p>
      <span class="src">Our measurement, same corpus. The middle strip is reconstructed exactly:
      on the support of x, ReLU(D<sub>y</sub>a) = y / x.</span>`,
  },
  {
    id: "iterate", title: "The same weights, six times, recruiting as they go",
    h: "E, the residual, and ×6",
    body: `<p>What survives the gate is projected back down by <b>E</b> and added to the residual.
      Then the whole block runs again &mdash; <b>with the same weights</b>.</p>
      <p>The iterations are not redundant. The number of neurons firing climbs steadily, but the
      <em>overlap</em> with the first iteration collapses: the shared operator keeps recruiting
      new neurons for several passes and then settles, with the last two iterations agreeing far
      more than the first two.</p>
      <p>This is the shape of BDH-CQ's equation (3),
      <span class="mono">H<sub>r+1</sub> = F<sub>&theta;</sub>(H<sub>r</sub>, S<sub>K</sub>)</span>
      &mdash; the same &theta;, applied R times.</p>
      <span class="src"><b>Caveat that has to stay on screen:</b> the BDH-CQ report never states a
      value for R, and never says what changes between its effort levels. This is our model's
      behaviour, not a claim about theirs.</span>`,
  },
  {
    id: "out", title: "And then a byte",
    h: "lm_head — the next byte",
    body: `<p>After six iterations the residual meets one more matrix and becomes a distribution
      over <b>256 bytes</b>. No tokeniser: this model reads and writes raw UTF-8.</p>
      <p>The bar the model chose is lit; the byte that actually came next is marked. Scrub through
      the sentence and watch the confidence move &mdash; it is high inside a word, and drops at
      the boundary where a translation could legitimately go several ways.</p>
      <span class="src">Logits shipped as f16 from the same forward pass as every other stage on
      this page.</span>`,
  },
];

// ── boot ─────────────────────────────────────────────────────────────────────────────────────
/**
 * Inflate a gzip ArrayBuffer with DecompressionStream. An explicit reader loop rather than
 * `new Response(stream).arrayBuffer()`, so it depends on nothing but the stream itself and runs
 * unchanged in Node under the smoke gate's stubbed fetch.
 */
async function inflate(buf) {
  const ds = new DecompressionStream("gzip");
  const wtr = ds.writable.getWriter();
  wtr.write(new Uint8Array(buf));
  wtr.close();
  const rdr = ds.readable.getReader();
  const chunks = [];
  let n = 0;
  for (;;) {
    const { value, done } = await rdr.read();
    if (done) break;
    chunks.push(value);
    n += value.length;
  }
  const out = new Uint8Array(n);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out.buffer;
}

async function boot() {
  // The manifest declares its own compression, so the loader does not hard-code which packs are
  // gzipped. walk.bin.gz is 4.87 MB shipped from 6.02 MB raw.
  const man = await fetch("public/walk.json").then((r) => r.json());
  const raw = await fetch(`public/walk.bin${man.compression === "gzip" ? ".gz" : ""}`)
    .then((r) => r.arrayBuffer());
  const bin = man.compression === "gzip" ? await inflate(raw) : raw;
  S.man = man;
  S.pack = loadPack(man, bin);
  S.layer = man.default_layer;
  S.head = man.default_head;

  buildSelectors();
  S.diagram = createDiagram($("diagram"), { onPick: go });
  S.flow = createFlow($("p0-flow"));
  // the smoke gate pins a node through the real click handler; this is only so it can read back
  // what got selected without re-deriving the layout
  if (typeof window !== "undefined") window.__flow = S.flow;
  addEventListener("resize", () => { if (S.stop === 0) S.flow.resize(); });
  setSentence(0);
  bindControls();
  go(0);
  $("boot").hidden = true;
  requestAnimationFrame(loop);
}

function buildSelectors() {
  $("s-sent").innerHTML = S.man.walk
    .map((w, i) => `<option value="${i}">${label(w)}</option>`).join("");
  $("s-layer").innerHTML = Array.from({ length: S.pack.L }, (_, i) =>
    `<option value="${i}"${i === S.layer ? " selected" : ""}>${i + 1} of ${S.pack.L}</option>`).join("");
  $("s-head").innerHTML = Array.from({ length: S.pack.H }, (_, i) =>
    `<option value="${i}">head ${i}</option>`).join("");
}

const label = (w) => {
  const s = w.text.replace(/<[^>]*>/g, " ").trim();
  return `${s.slice(0, 34)}${s.length > 34 ? "…" : ""}  (${w.T} bytes)`;
};

function setSentence(i) {
  S.w = i;
  S.T = S.man.walk[i].T;
  S.t = Math.min(S.t, S.T - 1);
  S.sigmaCache.clear();
  $("scrub").max = String(S.T - 1);
  buildStrip();
  // the long sentence only carries the default iteration; do not offer the others
  const limited = !S.man.walk[i].full_resolution;
  if (limited) S.layer = S.man.default_layer;
  [...$("s-layer").options].forEach((o) => {
    o.disabled = limited && Number(o.value) !== S.man.default_layer;
  });
  $("s-layer").value = String(S.layer);
}

function buildStrip() {
  const w = S.man.walk[S.w];
  const host = $("strip");
  host.innerHTML = "";
  const txt = w.bytes.map((b) => String.fromCharCode(b)).join("");
  let inTag = false;
  w.bytes.forEach((b, i) => {
    const ch = txt[i];
    if (ch === "<") inTag = true;
    const el = document.createElement("b");
    el.textContent = ch === " " ? "·" : ch;
    el.dataset.t = String(i);
    if (inTag) el.classList.add("tag");
    if (ch === ">") inTag = false;
    el.addEventListener("click", () => { S.t = i; S.playing = false; syncPlay(); render(); });
    host.appendChild(el);
  });
  paintStrip();
}

function paintStrip() {
  for (const el of $("strip").children) {
    const t = Number(el.dataset.t);
    el.classList.toggle("now", t === S.t);
    el.classList.toggle("seen", t < S.t);
  }
}

// ── stop routing ─────────────────────────────────────────────────────────────────────────────
function go(stop) {
  S.stop = Math.max(0, Math.min(STOPS.length - 1, stop));
  const st = STOPS[S.stop];
  $("r-no").textContent = String(S.stop);
  $("r-title").textContent = st.title;
  $("r-body").innerHTML = st.body;
  $("s-h").textContent = st.h;
  $("r-prev").disabled = S.stop === 0;
  $("r-next").disabled = S.stop === STOPS.length - 1;
  for (const p of document.querySelectorAll(".pane")) {
    p.hidden = Number(p.dataset.stop) !== S.stop;
  }
  // head only means something where a head is actually being drawn
  $("tool-head").hidden = ![2, 3, 4, 5].includes(S.stop);
  // the flow diagram's disclosure and counters live in the rail, so they belong to its stop
  $("r-live").hidden = S.stop !== 0;
  S.diagram.highlight(S.stop);
  render();
}

// ── render ───────────────────────────────────────────────────────────────────────────────────
/** The residual entering the current iteration, cached per (sentence, iteration). */
function residualAt(t) {
  const k = `${S.w}.${S.layer}`;
  if (!S.vCache || S.vCache.key !== k) {
    S.vCache = { key: k, v: f16(S.pack, `w${S.w}.L${S.layer}.v`) };
  }
  return S.vCache.v.subarray(t * S.pack.D, (t + 1) * S.pack.D);
}

function render() {
  paintStrip();
  $("scrub").value = String(S.t);
  // v* is live on every stop, not just the one that opens it
  S.diagram.setResidual(residualAt(S.t));
  const fns = [renderWhole, renderEncode, renderRotate, renderScore,
               renderState, renderGate, renderIterate, renderOut];
  try { fns[S.stop](); } catch (e) { console.error("stop", S.stop, e); }
}

const stats = (host, rows) => {
  $(host).innerHTML = rows.map(([dt, b, sub]) =>
    `<div><dt>${dt}</dt><dd><b>${b}</b><span>${sub || ""}</span></dd></div>`).join("");
};

/**
 * The sigma accumulator behind the flow diagram's x -> a* edges.
 *
 * sigma is a running sum, so it can only be advanced forwards. Scrubbing backwards, or changing
 * sentence, iteration or head, means the state that existed at this token is a different state --
 * so rebuild from token 0 rather than carry a wrong one. Rebuilding 45-105 tokens x 256 pooled
 * rows is ~1 ms; carrying a stale state would be a picture of a state the model never had.
 */
function sigmaUpTo(t) {
  const k = `${S.w}.${S.layer}.${S.head}`;
  if (!S.sig || S.sig.key !== k || S.sig.acc.t > t) {
    S.sig = { key: k, acc: sigmaRows(S.pack, S.w, S.layer, S.head) };
  }
  while (S.sig.acc.t < t) S.sig.acc.advance(S.sig.acc.t);
  return S.sig.acc;
}

function renderWhole() {
  const c = S.man.corpus;
  const f = buildFrame(S.pack, {
    w: S.w, layer: S.layer, head: S.head, t: S.t, sig: sigmaUpTo(S.t),
  });
  S.flow.setFrame(f);
  const st = f.stats;
  const pc = (v, n = 1) => `${(v * 100).toFixed(n)}%`;
  $("p0-note").innerHTML =
    `The drawn neurons are all from <b>head ${S.man.flow.head}</b>. That is the traced head &mdash; ` +
    `the only one whose <span class="mono">x_pre</span> this page can derive and whose &sigma; row ` +
    `it accumulates &mdash; so it is the only one a path can be followed all the way through. The ` +
    `counts above the columns are all four heads. ` +
    `Every line is <b>weight &times; source activation</b> &mdash; the term that node contributes ` +
    `to the next one &mdash; keeping each target's <b>${FAN} strongest</b> inputs. The 192-wide ` +
    `columns are drawn in full. The two 12,288-neuron columns cannot be: they show ` +
    `<b>${st.drawn}</b> of them &mdash; ${st.stable} held fixed by how often they fire across the ` +
    `corpus, so a dot keeps its row, plus this token's strongest on <em>both sides of the gate</em> ` +
    `&mdash; a hollow ring marks one of those visitors. ` +
    `That cast is drawn from a pool of ${st.pool} which covers <b>${pc(st.poolCoverage)}</b> of ` +
    `every step's true top-${st.leaderK}. At this token <b>${st.ySurvivors}</b> of the drawn ` +
    `neurons survive the gate, and the update they build is <b>${pc(st.dvRatio)}</b> the size of ` +
    `the one the model actually applied. Hover any node for its numbers; <b>click one to trace its ` +
    `route</b> &mdash; everything it draws on, and everything it feeds, all the way out to the byte.`;
  stats("p0-stats", [
    ["neurons", (S.pack.N * S.pack.H).toLocaleString(), "3,072 per head × 4"],
    ["parameters", "7.96 M", "no per-layer weights"],
    ["firing at once", pct(c.x_active_frac, 2), "measured, our corpus"],
    ["reaching the residual", pct(c.y_active_frac, 2), `${c.gate_sparser_by}× sparser still`],
  ]);
}

function renderEncode() {
  const N = S.pack.N;
  // x_pre is derived, not shipped: x_pre[n] = sum_d v[t,d] * decoder_x[head,d,n]. The matrix is
  // 1.18 MB against 2.76 MB of outputs, and the derived values are f16-weight accurate rather
  // than uint8-quantised against a per-token range -- max relative error 2.1e-4, measured by
  // research/export_walk.py and re-checked in web/test/walk.mjs.
  const pre = xPre(preflight(S.pack), f16(S.pack, `w${S.w}.L${S.layer}.v`), S.t * S.pack.D);
  const dense = new Float32Array(N);
  const sp = sparseAt(S.pack, `w${S.w}.L${S.layer}.x`, S.t);
  let inHead = 0;
  for (let i = 0; i < sp.ids.length; i++) {
    const g = sp.ids[i];
    if (g >= S.head * N && g < (S.head + 1) * N) { dense[g - S.head * N] = sp.val[i]; inHead++; }
  }
  // The count comes from the ReLU's own survivors rather than from the sign of the derived
  // pre-activations: x's support is exact, so this cannot drift by a neuron sitting a rounding
  // error either side of zero.
  const neg = N - inHead;

  drawStrip($("p1-pre"), pre, { signed: true, gamma: 0.5, height: 58 });
  drawStrip($("p1-post"), dense, { gamma: 0.45, height: 58 });
  $("p1-pre-n").textContent = N.toLocaleString();
  $("p1-post-n").textContent = inHead.toLocaleString();

  stats("p1-stats", [
    ["below zero", pct(neg / N), "deleted by the ReLU"],
    ["survivors", inHead.toLocaleString(), `of ${N.toLocaleString()} in this head`],
    ["this token", pct(inHead / N, 2), "sparsity varies per token"],
  ]);
  drawLine($("p1-curve"), [{
    values: S.man.corpus.per_layer.map((p) => p.x_frac * 100), colour: "#e0872a",
  }], { height: 108, yMax: 7 });
}

function renderRotate() {
  const N = S.pack.N;
  const sp = sparseAt(S.pack, `w${S.w}.L${S.layer}.x`, S.t);
  const roped = rope(S.pack, sp, S.t, S.head);
  const dense = new Float32Array(N);
  for (let i = 0; i < sp.ids.length; i++) {
    const g = sp.ids[i];
    if (g >= S.head * N && g < (S.head + 1) * N) dense[g - S.head * N] = sp.val[i];
  }
  // sample pairs that are actually alive, so the wheel is not mostly origin dots
  const pairs = [];
  let norm = 0;
  for (let n = 0; n < N && pairs.length < 90; n += 2) {
    if (dense[n] === 0 && dense[n + 1] === 0) continue;
    norm = Math.max(norm, Math.hypot(dense[n], dense[n + 1]));
    pairs.push({ x0: dense[n], y0: dense[n + 1], x1: roped[n], y1: roped[n + 1] });
  }
  for (const p of pairs) p.norm = norm || 1;
  drawRotation($("p2-wheel"), pairs, { size: 300 });

  let below = 0;
  for (let n = 0; n < N; n++) if (roped[n] < 0) below++;
  $("p2-explain").innerHTML =
    `<p>The shaded quadrant is where <b>every un-rotated key lives</b>: both components
     non-negative, because they just came out of a ReLU.</p>
     <p>Amber lines are the pairs before rotation &mdash; all of them inside that quadrant.
     Blue lines are the same pairs after RoPE, at this token's position. <b>${below.toLocaleString()}
     of ${N.toLocaleString()}</b> rotated components are now below zero.</p>
     <p>Scrub the sentence: the angle grows with position, and each neuron pair turns at its own
     rate. Two keys far apart in the sentence end up pointing in genuinely different directions,
     and their dot product can be negative.</p>`;
  stats("p2-stats", [
    ["position", String(S.t), "drives the angle"],
    ["negative after rotation", pct(below / N), "was exactly 0% before"],
    ["distinct rates", (N / 2).toLocaleString(), "one per neuron pair"],
  ]);
}

function scoresNow() {
  return attnMatrix(S.pack, S.w, S.layer, S.head, S.T);
}

function renderScore() {
  const M = scoresNow();
  const useSm = $("p3-softmax").checked;
  const row = useSm ? softmaxRow(M, S.T, S.t) : null;
  const hover = S.hover || { r: S.t, c: Math.max(0, S.t - 1) };
  const r = drawScores($("p3-mx"), M, S.T, { size: paneSize(), hover, softmaxRow: row });
  $("p3-mx").dataset.cell = String(r ? r.cell : 0);

  const v = M[hover.r * S.T + hover.c];
  const bytes = S.man.walk[S.w].bytes;
  const chr = (i) => JSON.stringify(String.fromCharCode(bytes[i])).slice(1, -1);
  $("p3-read").innerHTML = hover.c < hover.r
    ? `<span style="color:var(--ink3)">token ${hover.r} “${chr(hover.r)}” &nbsp;&larr;&nbsp;
         token ${hover.c} “${chr(hover.c)}”</span>
       <span class="big ${v < 0 ? "neg" : "pos"}">${v >= 0 ? "+" : ""}${v.toFixed(3)}</span>
       ${useSm ? `<span style="color:var(--cool)">softmax → ${(row[hover.c] * 100).toFixed(2)}%
         of this row</span>` : `<span style="color:var(--ink3)">raw score, nothing normalised it</span>`}`
    : `<span style="color:var(--ink3)">masked &mdash; token ${hover.r} cannot see token
        ${hover.c}. The mask is tril(&minus;1), so not even itself.</span>`;

  const ns = negShare(S.pack, S.w, S.layer, S.head);
  let sum = 0;
  for (let c = 0; c < hover.r; c++) sum += M[hover.r * S.T + c];
  $("p3-note").innerHTML = useSm
    ? `Every value in the row now depends on every other value in the row. No running sum over
       past keys can produce that &mdash; you have to keep them all.`
    : `This row sums to <b class="mono">${sum.toFixed(2)}</b>, not 1. Nothing normalises it, and
       that is exactly what a fixed-size state needs.`;
  stats("p3-stats", [
    ["negative here", pct(ns.frac), `${ns.negative.toLocaleString()} of ${ns.total.toLocaleString()}`],
    ["pooled, corpus", pct(S.man.corpus.neg_share_pooled[`L${S.layer}H${S.head}`]),
     "same layer and head"],
  ]);
}

const paneSize = () => Math.max(240, Math.min(460, ($("panes").clientHeight || 460) - 120));

function sigmaAt(t) {
  const key = `${S.w}.${S.layer}.${S.head}.${t}`;
  if (!S.sigmaCache.has(key)) {
    S.sigmaCache.set(key, sigmaRun(S.pack, S.w, S.layer, S.head, t + 1));
    if (S.sigmaCache.size > 8) S.sigmaCache.delete(S.sigmaCache.keys().next().value);
  }
  return S.sigmaCache.get(key);
}

function renderState() {
  const { N, D } = S.pack;
  const showDelta = $("p4-delta").checked;
  const run = sigmaAt(S.t);

  if (showDelta) {
    const d = deltaSigma(S.pack, S.w, S.layer, S.head, S.t);
    const flat = new Float32Array(N * D);
    for (let n = 0; n < N; n++) {
      if (d.k[n] === 0) continue;
      for (let j = 0; j < D; j++) flat[n * D + j] = d.k[n] * d.v[j];
    }
    drawSigma($("p4-sigma"), flat, N, D, { height: paneSize() });
  } else {
    drawSigma($("p4-sigma"), run.sigma, N, D, { height: paneSize() });
  }
  $("p4-t").textContent = String(S.t);
  $("p4-cap").firstChild.textContent = showDelta ? "one token’s write into σ — " : "σ after ";

  // our answer vs the model's: the check that makes the recurrence a demonstration
  const ref = f16(S.pack, `w${S.w}.L${S.layer}.H${S.head}.a`);
  let err = 0, sq = 0;
  for (let d = 0; d < D; d++) {
    err = Math.max(err, Math.abs(run.out[S.t * D + d] - ref[S.t * D + d]));
    sq += ref[S.t * D + d] ** 2;
  }
  const rms = Math.sqrt(sq / D) || 1;
  let live = 0;
  for (let n = 0; n < N; n++) {
    let any = 0;
    for (let d = 0; d < D; d += 16) if (run.sigma[n * D + d] !== 0) { any = 1; break; }
    live += any;
  }
  stats("p4-stats", [
    ["rows written", live.toLocaleString(), `of ${N.toLocaleString()}`],
    ["our σ vs the model", (err / rms).toExponential(1), "relative, computed here"],
  ]);
  $("p4-note").innerHTML = showDelta
    ? `A single write is <b>rank-one</b>: one neuron pattern times one 192-dim value. Structure is
       visible because there is only one of them.`
    : `The accumulated state is <b>dense</b> within a few tokens &mdash; every write lands on
       ~43% of the rows, so they overlap almost immediately. Reading the raw matrix is not how
       you understand it; the writes are.`;
}

function renderGate() {
  const N = S.pack.N;
  const g = gateAt(S.pack, S.w, S.layer, S.t);
  const inHead = (sp) => {
    const d = new Float32Array(N);
    let n = 0;
    for (let i = 0; i < sp.ids.length; i++) {
      const id = sp.ids[i];
      if (id >= S.head * N && id < (S.head + 1) * N) { d[id - S.head * N] = sp.val[i]; n++; }
    }
    return { d, n };
  };
  const X = inHead(g.x), Y = inHead(g.y);

  // ReLU(D_y a) restricted to x's support is exactly y / x; off that support it is multiplied
  // by zero and provably cannot reach the residual, so nothing is being hidden by not shipping it.
  const want = new Float32Array(N);
  for (let i = 0; i < N; i++) if (X.d[i] > 0 && Y.d[i] > 0) want[i] = Y.d[i] / X.d[i];

  drawStrip($("p5-want"), want, { stops: ["#191d24", "#1c5cab", "#3987e5", "#cde2fb"], height: 46 });
  drawStrip($("p5-fire"), X.d, { height: 46 });
  drawSparseMarks($("p5-and"), Y.d.reduce((a, v, i) => (v > 0 && a.push(i), a), []),
                  Y.d.filter((v) => v > 0), N, { height: 46 });
  $("p5-n").textContent = Y.n.toLocaleString();

  stats("p5-stats", [
    ["fired", X.n.toLocaleString(), "x > 0 in this head"],
    ["survived the gate", Y.n.toLocaleString(), "y > 0"],
    ["survival", pct(X.n ? Y.n / X.n : 0), "this token"],
    ["corpus", pct(S.man.corpus.y_active_frac, 2), `x is ${S.man.corpus.gate_sparser_by}× denser`],
  ]);
}

function renderIterate() {
  const w = S.man.walk[S.w];
  const jac = w.layers.map((l) => l.jaccard_vs_iter1);
  const act = w.layers.map((l) => l.mean_active_per_token);
  drawLine($("p6-jac"), [{ values: jac, colour: "#3987e5" }], { height: 150, yMax: 1 });
  drawLine($("p6-act"), [{ values: act, colour: "#e0872a" }], { height: 150 });
  const last = jac[jac.length - 1];
  stats("p6-stats", [
    ["iterations", String(S.pack.L), "one shared operator"],
    ["overlap with the first", last.toFixed(2), "Jaccard, by the last"],
    ["firing, iteration 1", Math.round(act[0]).toLocaleString(), "per token"],
    ["firing, iteration 6", Math.round(act[act.length - 1]).toLocaleString(), "still climbing"],
  ]);
}

function renderOut() {
  const top = topBytes(S.pack, S.w, S.t, 12);
  const bytes = S.man.walk[S.w].bytes;
  const actual = S.t + 1 < bytes.length ? bytes[S.t + 1] : -1;
  const show = (b) => (b === 32 ? "␣" : b >= 33 && b < 127 ? String.fromCharCode(b) : `·${b}`);
  const hi = top.findIndex((x) => x.byte === actual);
  drawBars($("p7-bars"), top.map((x) => ({ v: x.p, label: show(x.byte) })),
           { height: 190, highlight: hi });
  const pAct = top.find((x) => x.byte === actual);
  stats("p7-stats", [
    ["model's pick", show(top[0].byte), `p = ${top[0].p.toFixed(3)}`],
    ["what came next", actual < 0 ? "—" : show(actual), "the truth"],
    ["probability of it", pAct ? pAct.p.toFixed(3) : "<0.001", pAct ? "" : "outside the top 12"],
    ["vocabulary", "256", "raw bytes, no tokeniser"],
  ]);
  $("p7-note").innerHTML = hi === 0
    ? `The model's first choice was right here.`
    : `The model's first choice was <b>${show(top[0].byte)}</b>; the byte that came next was
       <b>${actual < 0 ? "—" : show(actual)}</b>. Disagreement is not failure &mdash; it is where
       more than one translation was legitimate.`;
}

// ── interaction ──────────────────────────────────────────────────────────────────────────────
function bindControls() {
  $("r-prev").onclick = () => go(S.stop - 1);
  $("r-next").onclick = () => go(S.stop + 1);
  $("s-sent").onchange = (e) => { setSentence(Number(e.target.value)); render(); };
  $("s-layer").onchange = (e) => { S.layer = Number(e.target.value); S.sigmaCache.clear(); render(); };
  $("s-head").onchange = (e) => { S.head = Number(e.target.value); S.sigmaCache.clear(); render(); };
  $("p3-softmax").onchange = render;
  $("p4-delta").onchange = render;
  $("scrub").oninput = (e) => { S.t = Number(e.target.value); S.playing = false; syncPlay(); render(); };
  $("play").onclick = () => { S.playing = !S.playing; syncPlay(); };

  $("p3-mx").addEventListener("mousemove", (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const cell = (r.width || 1) / S.T;
    const c = Math.floor((e.clientX - r.left) / cell), row = Math.floor((e.clientY - r.top) / cell);
    if (row >= 0 && row < S.T && c >= 0 && c < S.T) { S.hover = { r: row, c }; renderScore(); }
  });
  $("p3-mx").addEventListener("mouseleave", () => { S.hover = null; renderScore(); });

  addEventListener("keydown", (e) => {
    if (e.target.matches("input,select,textarea")) return;
    if (e.key === "ArrowRight" && e.shiftKey) { go(S.stop + 1); e.preventDefault(); }
    else if (e.key === "ArrowLeft" && e.shiftKey) { go(S.stop - 1); e.preventDefault(); }
    else if (e.key === "ArrowRight") { S.t = Math.min(S.T - 1, S.t + 1); render(); }
    else if (e.key === "ArrowLeft") { S.t = Math.max(0, S.t - 1); render(); }
    else if (e.key === " ") { S.playing = !S.playing; syncPlay(); e.preventDefault(); }
    else if (e.key >= "0" && e.key <= "7") go(Number(e.key));
  });
  addEventListener("resize", () => render());
}

function syncPlay() {
  $("play").innerHTML = S.playing ? "&#10073;&#10073;" : "&#9654;";
  $("play").setAttribute("aria-label", S.playing ? "Pause" : "Play");
}

let lastStep = 0;
function loop(now) {
  if (S.diagram) S.diagram.tick(now);
  if (S.playing && now - lastStep > 220) {
    lastStep = now;
    S.t = (S.t + 1) % S.T;
    render();
  }
  requestAnimationFrame(loop);
}

boot().catch((e) => {
  console.error(e);
  const b = $("boot");
  b.hidden = false;
  b.classList.add("failed");
  b.innerHTML = `<p>Could not load the stage tensors.<br>
    <span class="mini">${String(e).slice(0, 200)}</span></p>`;
});

// THE PRICE — controller for the twelve-slide deck.
//
// Three kinds of thing appear on this page and each is labelled on screen:
//   LIVE       a 131,072-parameter BDH we trained runs in this tab. Slides 1, 2, 3, 7, 8.
//   OURS       measurements from our 8M model, replayed from exported tensors. Slides 4, 5, 6, 9.
//   REPLAYED   BDH-CQ numbers, quoted with a section locator. Slides 10, 11. Never computed here,
//              because BDH-CQ has no public weights or API and pretending otherwise would be a
//              fabrication (CLAUDE.md item 1).
//
// The 8M model is NOT run live: 8M fp16 weights are ~16 MB and a forward pass is ~2.3 GFLOP, so
// it is replayed from tensors the exporters produced. Its numbers are still ours, and the two
// derived quantities on slide 6 (the MCC, the degree split) are recomputed in the browser from
// the shipped bytes rather than read out of a JSON field.

import { BDHModel, equivalenceCheck, trace } from "./bdh.js";
import { makeTokenMap, chip, tgtHue } from "./tokens.js";
import { drawMatrix, drawDiff } from "./sigma-view.js";
import { capacityCurve, capacityTrial, meanCosine, ANALYTIC_COSINE } from "./capacity.js";
import { drawCapacity } from "./chart.js";
import { silenceMCC, degreeBySilence } from "./walkdata.js";

const $ = (id) => document.getElementById(id);
const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
const N_SLIDES = 12;

const S = {
  model: null, man: null, tok: null, presets: null,
  pairs: [], removed: new Set(), qIdx: 1, hash0: "",
  sigView: "write", trace: null, toks: null, ms: 0, hashOk: true,
  answer: 0, answerP: 0, truth: 0, heroT: 0, heroDone: false, heroHold: 0,
  concepts: null, concept: "currency",
  syn: null, synSent: 0, synT: 0, synPlaying: false,
  fire: null, outDeg: null, inDeg: null,
  merge: null, capCache: null, slide: 0,
};

const LEDGER = [
  ["computed", "<b>$0.00070 per task</b> is not a price. It is 0.85 H200-GPU-seconds costed at an assumed $3/GPU-hour. Change the assumption and the headline moves proportionally. §5, p.4."],
  ["unverified", "The report gives <b>two different costs for the same 118/400 result</b> — $0.00070 in §5 and $0.00265246 in §6.6 — and never reconciles them. At the higher figure, “57× cheaper” becomes roughly 15×."],
  ["reported", "<b>29.5% pass@2</b> is developer-reported. BDH-CQ appears on neither the official ARC Prize leaderboard nor the community one (checked 2026-09-04). HRM and TRM do."],
  ["reported", "The <b>independent audit</b> was run by two co-authors of the paper. The report says so plainly; the press release does not."],
  ["unverified", "<b>ConceptARC (59.38%)</b> is listed in the training mixture in §4.2 and then used as an evaluation set in §6.1. §6.5 concedes the control does not rule out training exposure."],
  ["absent", "There are <b>no ARC-AGI-2 results</b> anywhere — not in the report, not in the blogs. It is named as future work."],
  ["absent", "What physically changes between the <b>effort levels</b> is not disclosed, and no value of R is ever given."],
];

const RECALL = [
  ["Where does a demonstration go?",
   "Nowhere in the weights. It goes into σ, a state built fresh each forward pass and thrown away after. That is why removing the line removed the rule, and why the weight hash never moved."],
  ["Which single change forces a cache that grows with context?",
   "Softmax. It normalises every score against every other, so no fixed-size running sum reproduces it — you must keep all past keys. Q=K and the ReLU are <em>not</em> what buys constant memory."],
  ["What actually limits how much it remembers?",
   "Overlap between keys, not the size of the state. Non-negative keys overlap by 1/π on average and interfere; signed keys are orthogonal on average and hold bindings past the state's own rank."],
  ["And what does that non-negativity buy?",
   "Inspectability: 5.1% sparse activations, neurons with measurable concept preferences, a synapse graph that predicts which neurons stay silent. The capacity cost is the price of that."],
  ["What is the one thing on this page that is <em>not</em> ours?",
   "Every BDH-CQ number. It has no public weights or API, so those are replayed from arXiv:2608.09888 with a section locator, never computed here."],
];

const PROV = [
  ["live, in this tab", "the 131K model on slides 1–3, the capacity curves on 6–7"],
  ["ours, replayed", "the 8M model on slides 4, 5, 6, 8 — exported by <code>research/export_*.py</code>"],
  ["replayed, cited", "every BDH-CQ figure on slides 9–10, with §-locators"],
  ["never animated", "no figure on this site is a scripted animation of a computation"],
  ["not an official model", "neither model is BDH or BDH-CQ; both are ours, at toy and small scale"],
];

// ── boot ─────────────────────────────────────────────────────────────────────────────────────
async function boot() {
  const [man, bin, presets] = await Promise.all([
    fetch("public/model.json").then((r) => r.json()),
    fetch("public/model.bin").then((r) => r.arrayBuffer()),
    fetch("public/presets.json").then((r) => r.json()),
  ]);
  S.man = man; S.presets = presets;
  S.model = new BDHModel(man, bin);
  S.tok = makeTokenMap(man);

  const P = presets.hook;
  S.pairs = P.pairs.map(([s, t]) => ({ src: s.charCodeAt(0), tgt: t.charCodeAt(0) }));
  S.qIdx = S.pairs.findIndex((p) => p.src === P.query.charCodeAt(0));
  S.hash0 = await weightsHash(S.model);
  $("h-hash").textContent = S.hash0;

  buildNav();
  buildLedger();
  buildRecall();
  buildProvenance();
  bindHook();
  bindSigma();
  bindEquiv();
  bindCapacity();

  recompute();
  replay();
  renderEquiv();
  renderCapacity();
  renderDegradation();

  $("boot").hidden = true;

  // the 8M-derived slides load after the live model is on screen, so nothing waits on them
  loadBig().catch((e) => console.error("big data:", e));
}

async function loadBig() {
  // fire.json rather than the walk pack: this page needs the 12,288 firing counters and nothing
  // else from that run, and the pack is 4.87 MB of per-stage tensors for the loop walkthrough.
  const [cman, syn, merge, fire, fman, fbin] = await Promise.all([
    fetch("public/concepts.json").then((r) => r.json()),
    fetch("public/synapses.json").then((r) => r.json()),
    fetch("public/merge.json").then((r) => r.json()),
    fetch("public/fire.json").then((r) => r.json()),
    fetch("public/big/field.json").then((r) => r.json()),
    fetch("public/big/field.bin").then((r) => r.arrayBuffer()),
  ]);
  S.concepts = cman; S.syn = syn; S.merge = merge;

  S.fire = Uint16Array.from(fire.counts);
  const N = fire.config.N, H = fire.config.n_head;
  S.outDeg = new Uint16Array(N * H);
  S.inDeg = new Uint16Array(N * H);
  const tOf = (n) => {
    const e = fman.tensors.find((x) => x.name === n);
    return new Uint16Array(fbin, e.offset, e.bytes / 2);
  };
  for (let h = 0; h < H; h++) {
    S.outDeg.set(tOf(`h${h}.out_deg`), h * N);
    S.inDeg.set(tOf(`h${h}.in_deg`), h * N);
  }

  buildConcepts();
  buildSynapses();
  renderConcepts();
  renderSynapses();
  renderSilence();
  renderMerge();
}

async function weightsHash(model) {
  const parts = Object.keys(model.w).sort().map((k) => new Uint8Array(model.w[k].buffer,
    model.w[k].byteOffset, model.w[k].byteLength));
  const total = parts.reduce((n, p) => n + p.length, 0);
  const all = new Uint8Array(total);
  let o = 0;
  for (const p of parts) { all.set(p, o); o += p.length; }
  const d = await crypto.subtle.digest("SHA-256", all);
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, "0")).join("").slice(0, 32);
}

// ── slide 1: the sixty-second test, played out ───────────────────────────────────────────────
//
// The static version of this slide was correct and dull: three chips and an answer, nothing
// moving. The claim is about a state being WRITTEN, so the slide now plays the forward pass --
// tokens revealed one at a time, sigma accumulating beside them from trace()'s real per-token
// snapshots, the answer resolving at the end. Everything on screen is the model's own arithmetic;
// only the pacing is ours.
const active = () => S.pairs.filter((_, i) => !S.removed.has(i));

function buildTokens() {
  // Order is (source, target, SEP) per demonstration, then (QUERY, source). This exact sequence
  // is asserted against the checkpoint by web/test/app_logic.mjs and recorded in presets.json
  // as `hook.bytes` -- getting it wrong yields a model that is confidently wrong even WITH the
  // demonstration present, which looks like a broken claim rather than a broken sequence.
  const t = [];
  for (const p of active()) t.push(p.src, p.tgt, S.man.tokens.SEP);
  t.push(S.man.tokens.QRY, S.pairs[S.qIdx].src);
  return t;
}

function bindHook() {
  $("h-toggle").onclick = () => {
    const b = $("h-toggle");
    if (S.removed.has(S.qIdx)) {
      S.removed.delete(S.qIdx);
      b.textContent = "Remove the demonstration it needs →";
      b.classList.remove("done");
    } else {
      S.removed.add(S.qIdx);
      b.textContent = "← Put it back";
      b.classList.add("done");
    }
    recompute();
    replay();
  };
  $("h-replay").onclick = replay;
}

function replay() {
  S.heroT = 0;
  S.heroDone = false;
  S.heroHold = 0;
  renderHero();
}

/** Run the model and keep the per-token trace. Called once per change of the demonstrations. */
function recompute() {
  const toks = buildTokens();
  const t0 = performance.now();
  const out = S.model.forward(toks, { keepSigma: false });
  S.ms = performance.now() - t0;
  S.toks = toks;

  // trace() gives per-token sigma snapshots and the rank-one write at each token. forward()'s
  // `sigmas` is one entry PER LAYER/HEAD -- {layer, head, sigma} OBJECTS, not per token -- so
  // indexing it as a flat array yielded NaN for every cell and rendered a black rectangle for
  // all three views at once. That is what this slide shipped with; do not undo it.
  S.trace = trace(S.model, toks, { layer: 0, head: 0 });

  const V = S.model.V, last = out.T - 1;
  let best = 0, sum = 0, mx = -Infinity;
  for (let v = 0; v < V; v++) mx = Math.max(mx, out.logits[last * V + v]);
  for (let v = 0; v < V; v++) sum += Math.exp(out.logits[last * V + v] - mx);
  for (let v = 1; v < V; v++) {
    if (out.logits[last * V + v] > out.logits[last * V + best]) best = v;
  }
  S.answer = best;
  S.answerP = Math.exp(out.logits[last * V + best] - mx) / sum;
  S.truth = S.pairs[S.qIdx].tgt;

  $("sig-t").max = String(S.trace.T - 1);
  if (Number($("sig-t").value) > S.trace.T - 1) $("sig-t").value = String(S.trace.T - 1);
  renderSigma();
  verifyHash();
}

const kindOf = (b) => S.tok.get(b) || { kind: "tgt", i: 0, label: "?", byte: b };

/** One frame of the hero animation. Draws the tape up to token t and sigma as it stands there. */
function renderHero() {
  if (!S.trace) return;
  const T = S.trace.T;
  const t = Math.min(S.heroT, T - 1);

  const tape = $("h-tape");
  if (tape.children.length !== T) {
    tape.innerHTML = "";
    S.toks.forEach((b) => {
      const w = document.createElement("span");
      w.className = "tk";
      w.appendChild(chip(kindOf(b)));
      tape.appendChild(w);
    });
  }
  for (let i = 0; i < T; i++) {
    tape.children[i].classList.toggle("in", i <= t);
    tape.children[i].classList.toggle("now", i === t && !S.heroDone);
  }

  drawMatrix($("h-sigma"), S.trace.snaps[t], S.model.N, S.model.D, { cellW: 8, cellH: 2 });
  $("h-tcount").textContent = "token " + (t + 1) + " of " + T;
  $("h-phase").textContent = S.heroDone ? "answered"
    : S.toks[t] === S.man.tokens.QRY || t === T - 1 ? "asking the question"
    : "reading a demonstration";

  if (S.heroDone) {
    $("h-model").innerHTML = "";
    $("h-model").appendChild(chip(kindOf(S.answer), { big: true }));
    $("h-truth").innerHTML = "";
    $("h-truth").appendChild(chip(kindOf(S.truth), { big: true }));
    $("h-p").textContent = S.answerP.toFixed(3);
    const correct = S.answer === S.truth;
    const v = $("h-verdict");
    v.className = "verdict " + (correct ? "good" : "bad");
    v.innerHTML = correct
      ? "The rule was never in the weights. It was in those three lines — and it went into the " +
        "state on the right, which is thrown away the moment this pass ends."
      : "<b>Wrong, and confidently so.</b> The query bytes are byte-identical to a moment ago; " +
        "only the demonstration is gone. It did not hedge — it answered with another target " +
        "symbol at p=" + S.answerP.toFixed(3) + ": structurally plausible, simply incorrect.";
  } else {
    $("h-model").innerHTML = "";
    $("h-truth").innerHTML = "";
    $("h-p").textContent = "—";
    $("h-verdict").className = "verdict";
    $("h-verdict").textContent = "";
  }

  const sig = S.trace.snaps[t];
  let nz = 0;
  for (let i = 0; i < sig.length; i++) if (sig[i] !== 0) nz++;
  $("h-meters").innerHTML =
    "<div><dt>parameter updates</dt><dd><b>0</b><span>nothing trained or tuned</span></dd></div>" +
    "<div><dt>σ cells written</dt><dd><b>" + pct(nz / sig.length, 0) + "</b><span>of " +
    S.model.N + "×" + S.model.D + ", built this pass</span></dd></div>" +
    "<div><dt>forward pass</dt><dd><b>" + S.ms.toFixed(0) + " ms</b><span>" +
    (S.hashOk ? "weights unchanged ✓" : "live, in your browser") + "</span></dd></div>";
}

async function verifyHash() {
  // Recomputed after every change to the demonstrations. If this ever differed, the whole claim
  // would be false -- so it is checked rather than asserted in prose.
  const h = await weightsHash(S.model);
  S.hashOk = h === S.hash0;
  $("h-hashnote").textContent = S.hashOk
    ? "unchanged since page load ✓"
    : "CHANGED — that would be a bug";
}

// ── slide 2: sigma, three ways ───────────────────────────────────────────────────────────────
function bindSigma() {
  for (const b of $("sig-tabs").querySelectorAll("button")) {
    b.onclick = () => {
      S.sigView = b.dataset.v;
      for (const o of $("sig-tabs").querySelectorAll("button")) o.classList.toggle("on", o === b);
      renderSigma();
    };
  }
  $("sig-t").oninput = renderSigma;
}

function renderSigma() {
  if (!S.trace) return;
  const { N, D } = S.model;
  const c = $("sig-canvas");
  const T = S.trace.T;
  const t = Math.min(Number($("sig-t").value) || 0, T - 1);
  $("sig-tval").textContent = String(t);

  if (S.sigView === "write") {
    drawMatrix(c, S.trace.writes[t], N, D, { cellW: 8, cellH: 2 });
  } else if (S.sigView === "state") {
    drawMatrix(c, S.trace.snaps[t], N, D, { cellW: 8, cellH: 2 });
  } else {
    // What one line was worth: the same state, with and without the removed demonstration.
    // Traced fresh rather than cached, because the sequence length differs between the two.
    const keep = S.removed;
    S.removed = keep.has(S.qIdx) ? new Set() : new Set([S.qIdx]);
    const alt = trace(S.model, buildTokens(), { layer: 0, head: 0 });
    S.removed = keep;
    drawDiff(c, S.trace.snaps[S.trace.T - 1], alt.snaps[alt.T - 1], N, D, { cellW: 8, cellH: 2 });
  }

  $("sig-rows").textContent = N + " neurons";
  $("sig-cols").textContent = D + " dims";

  const cur = S.trace.snaps[t], w = S.trace.writes[t];
  let nz = 0, wnz = 0;
  for (let i = 0; i < cur.length; i++) {
    if (cur[i] !== 0) nz++;
    if (w[i] !== 0) wnz++;
  }
  const isWrite = S.sigView === "write";
  $("sig-stats").innerHTML =
    "<div><dt>state shape</dt><dd><b>" + N + " × " + D +
    "</b><span>fixed, whatever the context</span></dd></div>" +
    "<div><dt>" + (isWrite ? "cells this write touched" : "cells non-zero") + "</dt><dd><b>" +
    pct((isWrite ? wnz : nz) / cur.length, 0) + "</b><span>" +
    (isWrite ? "one token, rank-one" : "dense within a few tokens") + "</span></dd></div>";

  $("sig-note").innerHTML = isWrite
    ? "One token's write, alone: <b>rank-one</b> — a single neuron pattern times a single value " +
      "vector. Drag the token slider and watch the pattern move. This is the only view where the " +
      "structure is visible."
    : S.sigView === "state"
      ? "The accumulated state. It reads as texture because it <b>is</b> dense — each write lands " +
        "on ~43% of rows, so they overlap almost immediately. That is honest, not a rendering " +
        "failure, and it is why the write and the difference are the views worth having."
      : "The difference between the finished state <em>with</em> the demonstration and " +
        "<em>without</em> it. Everything visible here is what that one line was worth — and it " +
        "is the entire reason the answer changed.";
}

// ── slide 3: the equivalence, and breaking it ────────────────────────────────────────────────
function bindEquiv() {
  for (const id of ["t-softmax", "t-qk", "t-relu"]) $(id).onchange = renderEquiv;
}

function renderEquiv() {
  const toks = buildTokens();
  const r = equivalenceCheck(S.model, toks, {
    softmax: $("t-softmax").checked,
    decoupleQK: $("t-qk").checked,
    noRelu: $("t-relu").checked,
  });
  const broken = r.relative > 1e-3;
  $("eq-resid").textContent = r.maxAbs.toExponential(1);
  $("eq-resid").parentElement.className = "resid " + (broken ? "broken" : "ok");
  $("eq-note").textContent = broken
    ? `relative ${r.relative.toExponential(1)} — these are different computations`
    : `relative ${r.relative.toExponential(1)} — the same computation, two ways`;

  const on = [$("t-softmax").checked && "softmax", $("t-qk").checked && "Q≠K",
              $("t-relu").checked && "no ReLU"].filter(Boolean);
  $("eq-lesson").innerHTML = !on.length
    ? `Identical to float precision. The recurrent state is not a metaphor for what the parallel
       form does — it <b>is</b> what it does.`
    : broken
      ? `Broken by <b>softmax</b>. Normalising each score against every other means no fixed-size
         running sum can reproduce it: you must keep every past key. <b>That is exactly why a
         Transformer needs a KV cache that grows with context, and this does not.</b>`
      : `<b>Still identical</b>, with ${on.join(" and ")} on. Tying Q to K and clamping activations
         non-negative are not what buys constant memory. They buy something else: they make σ
         <em>readable</em> as synapses over one basis of neurons. Legibility, not memory —
         and the next slides are the bill for it.`;
}

// ── slide 4: concept selectivity ─────────────────────────────────────────────────────────────
function buildConcepts() {
  const host = $("c-pills");
  host.innerHTML = "";
  for (const name of Object.keys(S.concepts.concepts)) {
    const b = document.createElement("button");
    b.textContent = name.replace("_", " ");
    b.className = name === S.concept ? "on" : "";
    b.onclick = () => {
      S.concept = name;
      for (const o of host.children) o.classList.toggle("on", o === b);
      renderConcepts();
    };
    host.appendChild(b);
  }
  $("c-draws").textContent = S.concepts.draws.toLocaleString();
}

function renderConcepts() {
  if (!S.concepts) return;
  const sum = S.concepts.summary;
  const mine = S.concepts.top.filter((t) => t.concept === S.concept);

  // observed selectivity against each neuron's own null: everything above the diagonal beat it
  const c = $("c-scatter");
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const w = c.clientWidth || 520, h = 250;
  c.width = w * dpr; c.height = h * dpr;
  c.style.height = h + "px";
  const ctx = c.getContext("2d");
  if (ctx) {
    ctx.clearRect(0, 0, c.width, c.height);
    const X = (v) => 40 * dpr + v * (c.width - 56 * dpr);
    const Y = (v) => c.height - 30 * dpr - v * (c.height - 44 * dpr);
    ctx.strokeStyle = "rgba(255,255,255,.12)";
    ctx.lineWidth = dpr;
    ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(1), Y(1)); ctx.stroke();
    ctx.fillStyle = "#6f6b64";
    ctx.font = `${10 * dpr}px ui-sans-serif, system-ui`;
    ctx.fillText("chance line", X(0.62), Y(0.55));
    for (const t of S.concepts.top) {
      const on = t.concept === S.concept;
      ctx.fillStyle = on ? "#3987e5" : "rgba(255,255,255,.10)";
      ctx.beginPath();
      ctx.arc(X(t.null_p95), Y(t.selectivity), (on ? 3.4 : 2) * dpr, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  $("c-stats").innerHTML =
    `<div><dt>fire at all</dt><dd><b>${sum.n_alive.toLocaleString()}</b><span>of ${sum.n_neurons.toLocaleString()} neurons</span></dd></div>
     <div><dt>beat their own null</dt><dd><b>${sum.n_beating_own_null.toLocaleString()}</b><span>chance: ~${sum.expected_by_chance}</span></dd></div>
     <div><dt>enrichment</dt><dd><b>${sum.enrichment}×</b><span>over chance</span></dd></div>
     <div><dt>survive FDR 5%</dt><dd><b>${sum.n_significant_fdr5.toLocaleString()}</b><span>Benjamini–Hochberg</span></dd></div>`;

  $("c-list").innerHTML = mine.slice(0, 24).map((t) =>
    `<b title="selectivity ${t.selectivity} vs null p95 ${t.null_p95}, p=${t.p}, fires on ${t.fires_on} labelled bytes">#${t.global}</b>`).join("");
}

// ── slide 5: synapses over time ──────────────────────────────────────────────────────────────
function buildSynapses() {
  $("y-sent").innerHTML = S.syn.sentences.map((s, i) => {
    const t = s.text.replace(/<[^>]*>/g, " ").trim();
    return `<option value="${i}">${t.slice(0, 32)}… (${s.T})</option>`;
  }).join("");
  $("y-sent").onchange = (e) => { S.synSent = Number(e.target.value); S.synT = 0; renderSynapses(); };
  $("y-play").onclick = () => {
    S.synPlaying = !S.synPlaying;
    $("y-play").innerHTML = S.synPlaying ? "&#10073;&#10073;" : "&#9654;";
  };
}

function renderSynapses() {
  if (!S.syn) return;
  const s = S.syn.sentences[S.synSent];
  const P = S.syn.pairs;
  const c = $("y-timeline");
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const w = c.clientWidth || 520, h = 230;
  c.width = w * dpr; c.height = h * dpr;
  c.style.height = h + "px";
  const ctx = c.getContext("2d");
  if (ctx) {
    ctx.clearRect(0, 0, c.width, c.height);
    const rows = s.vals.length, rh = c.height / rows;
    for (let r = 0; r < rows; r++) {
      for (let t = 0; t < s.T; t++) {
        const v = s.vals[r][t];
        if (v <= 0) continue;
        const u = Math.pow(v, 0.45);
        ctx.fillStyle = `rgba(224,135,42,${0.12 + 0.88 * u})`;
        ctx.fillRect((t / s.T) * c.width, r * rh, Math.ceil(c.width / s.T), Math.ceil(rh) - 1);
      }
    }
    ctx.strokeStyle = "#f3f1ec";
    ctx.lineWidth = 1.5 * dpr;
    ctx.beginPath();
    ctx.moveTo((S.synT / s.T) * c.width, 0);
    ctx.lineTo((S.synT / s.T) * c.width, c.height);
    ctx.stroke();
  }

  const strip = $("y-strip");
  if (strip.children.length !== s.bytes.length) {
    strip.innerHTML = "";
    s.bytes.forEach((b, i) => {
      const el = document.createElement("b");
      el.textContent = String.fromCharCode(b) === " " ? "·" : String.fromCharCode(b);
      el.onclick = () => { S.synT = i; renderSynapses(); };
      strip.appendChild(el);
    });
  }
  for (let i = 0; i < strip.children.length; i++) {
    strip.children[i].classList.toggle("now", i === S.synT);
    strip.children[i].classList.toggle("seen", i < S.synT);
  }

  let live = 0, top = null;
  s.vals.forEach((row, r) => {
    if (row[S.synT] > 0.02) live++;
    if (!top || row[S.synT] > s.vals[top][S.synT]) top = r;
  });
  const p = P[top] || P[0];
  $("y-stats").innerHTML =
    `<div><dt>tracked synapses</dt><dd><b>${P.length}</b><span>wired and co-firing</span></dd></div>
     <div><dt>active now</dt><dd><b>${live}</b><span>at this byte</span></dd></div>
     <div><dt>strongest</dt><dd><b>${p.i}→${p.j}</b><span>G* = ${p.w.toFixed(3)}</span></dd></div>`;
}

// ── slide 6: the wiring predicts the silence ─────────────────────────────────────────────────
function renderSilence() {
  if (!S.fire) return;
  const m = silenceMCC(S.fire, S.outDeg, S.inDeg);
  const d = degreeBySilence(S.fire, S.outDeg, S.inDeg);

  const c = $("g-scatter");
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const w = c.clientWidth || 520, h = 250;
  c.width = w * dpr; c.height = h * dpr;
  c.style.height = h + "px";
  const ctx = c.getContext("2d");
  if (ctx) {
    ctx.clearRect(0, 0, c.width, c.height);
    let maxD = 0, maxF = 0;
    for (let i = 0; i < S.fire.length; i++) {
      maxD = Math.max(maxD, S.outDeg[i] + S.inDeg[i]);
      maxF = Math.max(maxF, S.fire[i]);
    }
    const X = (v) => 34 * dpr + Math.pow(v / maxD, 0.45) * (c.width - 48 * dpr);
    const Y = (v) => c.height - 26 * dpr - Math.pow(v / maxF, 0.45) * (c.height - 40 * dpr);
    for (let i = 0; i < S.fire.length; i += 1) {
      const deg = S.outDeg[i] + S.inDeg[i];
      const silent = S.fire[i] === 0;
      ctx.fillStyle = silent ? "rgba(217,89,38,.30)" : "rgba(57,135,229,.22)";
      ctx.fillRect(X(deg), Y(S.fire[i]), 1.6 * dpr, 1.6 * dpr);
    }
    ctx.fillStyle = "#6f6b64";
    ctx.font = `${10 * dpr}px ui-sans-serif, system-ui`;
    ctx.fillText("orange = never fires", 40 * dpr, 18 * dpr);
  }

  $("g-stats").innerHTML =
    `<div><dt>Matthews corr.</dt><dd><b>+${m.mcc.toFixed(3)}</b><span>isolated vs silent</span></dd></div>
     <div><dt>P(silent | isolated)</dt><dd><b>${pct(m.pSilentGivenIsolated)}</b><span>base rate ${pct(m.silentFrac)}</span></dd></div>
     <div><dt>never fire</dt><dd><b>${pct(m.silentFrac)}</b><span>${(m.tp + m.fn).toLocaleString()} neurons</span></dd></div>
     <div><dt>mean degree</dt><dd><b>${d.silent.toFixed(1)} / ${d.firing.toFixed(1)}</b><span>silent / firing</span></dd></div>`;

  $("g-matrix").innerHTML =
    `<div class="hd"></div><div class="hd">never fires</div><div class="hd">fires</div>
     <div class="hd">isolated in G*</div><div class="hit">${m.tp.toLocaleString()}</div><div>${m.fp.toLocaleString()}</div>
     <div class="hd">has synapses</div><div>${m.fn.toLocaleString()}</div><div class="hit">${m.tn.toLocaleString()}</div>`;
}

// ── slide 7: capacity ────────────────────────────────────────────────────────────────────────
const KS = [2, 4, 8, 16, 32, 64, 128, 192, 256];

function bindCapacity() {
  $("k-slider").oninput = () => { $("k-val").textContent = $("k-slider").value; renderCapacityStats(); };
  $("k-signed").onchange = renderCapacityStats;
}

function renderCapacity() {
  if (!S.capCache) {
    S.capCache = [
      { label: "ReLU'd keys (BDH)", points: capacityCurve(KS, { nonneg: true, trials: 5 }) },
      { label: "signed keys", points: capacityCurve(KS, { nonneg: false, trials: 5 }) },
    ].map((s) => ({ ...s, points: s.points.map((p) => ({ k: p.k, acc: p.acc })) }));
  }
  drawCapacity($("k-chart"), S.capCache);
  renderCapacityStats();
}

function renderCapacityStats() {
  const k = Number($("k-slider").value);
  const nonneg = !$("k-signed").checked;
  const acc = capacityTrial(k, { nonneg }).acc;
  const cos = meanCosine({ nonneg });
  $("k-stats").innerHTML =
    `<div><dt>retrieval at k=${k}</dt><dd><b>${pct(acc, 0)}</b><span>${nonneg ? "non-negative keys" : "signed keys"}</span></dd></div>
     <div><dt>mean pairwise overlap</dt><dd><b>${cos.toFixed(3)}</b><span>${nonneg ? `analytic 1/π = ${ANALYTIC_COSINE.toFixed(3)}` : "orthogonal on average"}</span></dd></div>
     <div><dt>state rank</dt><dd><b>64</b><span>signed keys pass it; ReLU'd do not reach it</span></dd></div>`;
}

// ── slide 8: two curves ──────────────────────────────────────────────────────────────────────
function renderDegradation() {
  const acc = S.presets.model_acc_by_k;
  const ks = Object.keys(acc).map(Number).sort((a, b) => a - b);
  const model = { label: "the trained model", points: ks.map((k) => ({ k, acc: acc[k] })) };
  const state = { label: "the same state, measured directly",
                  points: capacityCurve(ks.filter((k) => k >= 2), { nonneg: true, trials: 4 })
                    .map((p) => ({ k: p.k, acc: p.acc })) };
  drawCapacity($("d-chart"), [model, state]);
  const at16 = state.points.find((p) => p.k === 16);
  $("d-stats").innerHTML =
    `<div><dt>model at k=16</dt><dd><b>${pct(acc[16], 0)}</b><span>trained on k ≤ 10</span></dd></div>
     <div><dt>state at k=16</dt><dd><b>${pct(at16 ? at16.acc : 1, 0)}</b><span>same size, measured directly</span></dd></div>`;
  $("d-note").innerHTML = `The gap between these two lines is <b>not</b> the state running out of
    room. It is the model never having been shown a sequence that long. Two different failures
    that produce the same-shaped curve.`;
}

// ── slide 9: the merge ───────────────────────────────────────────────────────────────────────
function renderMerge() {
  if (!S.merge) return;
  const r = S.merge.results;
  const rows = Object.entries(r).map(([name, v]) => {
    if (v.error) return `<tr><th>${name}</th><td colspan="2" class="src">would not load</td></tr>`;
    const bad = v.fr > 10;
    return `<tr><th>${name}</th><td class="${bad ? "" : "good"}">${v.fr.toFixed(2)}</td>
            <td class="${bad ? "" : "good"}">${v.pt.toFixed(2)}</td></tr>`;
  }).join("");
  $("m-table").innerHTML =
    `<table class="dt"><tr><th></th><th>French probe</th><th>Portuguese probe</th></tr>${rows}
     </table>
     <p class="mini src">Mean next-byte cross-entropy over 4 short probe sentences per language.
     Chance is ln(256) = 5.55. <b>Not</b> a held-out validation corpus, and not comparable in
     absolute terms to the paper's Table 2. <span class="tag ours">ours</span></p>`;

  const key = Object.keys(S.merge.samples)[0];
  $("m-samples").innerHTML = (S.merge.samples[key] || []).map((s) =>
    `<code>${s.replace(/(i{6,})/g, "<b>$1</b>").slice(0, 120)}</code>`).join("");
  $("m-note").innerHTML = `Reproduce with <code>python research/merge_replicate.py</code> — it runs
    all four variants, including the two controls, in one command.`;
}

// ── static content ───────────────────────────────────────────────────────────────────────────
function buildLedger() {
  $("ledger").innerHTML = LEDGER.map(([k, t]) =>
    `<li><span class="k ${k}">${k}</span><span>${t}</span></li>`).join("");
}

function buildRecall() {
  $("recall").innerHTML = RECALL.map(([q, a]) =>
    `<div class="q"><span class="more">show</span><b>${q}</b><div class="a">${a}</div></div>`).join("");
  for (const q of $("recall").children) {
    q.onclick = () => {
      q.classList.toggle("open");
      q.querySelector(".more").textContent = q.classList.contains("open") ? "hide" : "show";
    };
  }
}

function buildProvenance() {
  $("prov").innerHTML = PROV.map(([d, t]) => `<dt>${d}</dt><dd>${t}</dd>`).join("");
}

// ── deck navigation ──────────────────────────────────────────────────────────────────────────
function buildNav() {
  $("dots").innerHTML = Array.from({ length: N_SLIDES }, (_, i) =>
    `<button role="tab" aria-label="Slide ${i + 1}" data-i="${i}"></button>`).join("");
  for (const b of $("dots").children) b.onclick = () => goto(Number(b.dataset.i));
  $("prev").onclick = () => goto(S.slide - 1);
  $("next").onclick = () => goto(S.slide + 1);
  addEventListener("keydown", (e) => {
    if (e.target.matches("input,select,textarea,button")) return;
    if (e.key === "ArrowRight" || e.key === "PageDown") { goto(S.slide + 1); e.preventDefault(); }
    if (e.key === "ArrowLeft" || e.key === "PageUp") { goto(S.slide - 1); e.preventDefault(); }
    if (e.key === "Home") goto(0);
    if (e.key === "End") goto(N_SLIDES - 1);
  });
  $("deck").addEventListener("scroll", () => {
    const i = Math.round($("deck").scrollLeft / (window.innerWidth || 1));
    if (i !== S.slide) { S.slide = i; syncNav(); }
  }, { passive: true });
  syncNav();
}

function goto(i) {
  S.slide = Math.max(0, Math.min(N_SLIDES - 1, i));
  const el = document.querySelector(`.slide[data-i="${S.slide}"]`);
  if (el) el.scrollIntoView({ behavior: "smooth", inline: "start", block: "nearest" });
  syncNav();
}

function syncNav() {
  for (const b of $("dots").children) b.classList.toggle("on", Number(b.dataset.i) === S.slide);
  $("prev").disabled = S.slide === 0;
  $("next").disabled = S.slide === N_SLIDES - 1;
}

let lastHero = 0;
function heroTick(now) {
  if (!S.trace) return;
  const T = S.trace.T;
  if (S.heroDone) {
    // hold the finished answer for a beat, then replay -- the slide should never sit static
    if (now - S.heroHold > 4200) { S.heroT = 0; S.heroDone = false; renderHero(); }
    return;
  }
  if (now - lastHero < 320) return;
  lastHero = now;
  if (S.heroT >= T - 1) { S.heroDone = true; S.heroHold = now; }
  else S.heroT++;
  renderHero();
}

let lastTick = 0;
function tick(now) {
  heroTick(now);
  if (S.synPlaying && S.syn && now - lastTick > 130) {
    lastTick = now;
    S.synT = (S.synT + 1) % S.syn.sentences[S.synSent].T;
    renderSynapses();
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

boot().catch((e) => {
  console.error(e);
  const b = $("boot");
  b.hidden = false;
  b.classList.add("failed");
  b.innerHTML = `<p>Could not start.<br><span class="mini">${String(e).slice(0, 200)}</span></p>`;
});

// The sixty-second moment: remove one demonstration, with the query byte-identical, and watch
// the answer break while `parameter updates` never moves.
//
// Everything here is live. The model is a 131,072-parameter BDH we trained (vendored, unmodified
// pathwaycom/bdh architecture) running in the browser at ~20-85ms per forward pass.
import { BDHModel, equivalenceCheck } from "./bdh.js";
import { makeTokenMap, chip } from "./tokens.js";
import { drawMatrix, drawDiff, drawRowEnergy, scaleOf } from "./sigma-view.js";

const $ = (s) => document.querySelector(s);

const state = {
  model: null, manifest: null, tokMap: null,
  pairs: [],            // [{src, tgt}] byte pairs, the bijection defined in-prompt
  removed: new Set(),   // indices removed by the learner
  queryIdx: 1,          // which pair is being queried
  view: "state",
  prevSigma: null,      // sigma from the previous run, for the difference map
};

async function boot() {
  const [manifest, binRes, presets] = await Promise.all([
    fetch("public/model.json").then((r) => r.json()),
    fetch("public/model.bin").then((r) => r.arrayBuffer()),
    fetch("public/presets.json").then((r) => r.json()),
  ]);
  state.manifest = manifest;
  state.model = new BDHModel(manifest, binRes);
  state.tokMap = makeTokenMap(manifest);

  // Open with the preset already running -- no blank canvas, no Run button.
  const P = presets.hook;
  state.pairs = P.pairs.map(([s, t]) => ({ src: s.charCodeAt(0), tgt: t.charCodeAt(0) }));
  state.queryIdx = state.pairs.findIndex((p) => p.src === P.query.charCodeAt(0));

  $("#sigma-dims").textContent = `${state.model.N} x ${state.model.D}`;
  $("#ax-rows").textContent = `${state.model.N} neurons ↓`;
  $("#ax-cols").textContent = `${state.model.D} dims →`;
  $("#whash").textContent = await weightsHash(binRes);

  $("#add-demo").onclick = addDemo;
  $("#sigma-tabs").onclick = (e) => {
    const b = e.target.closest("button"); if (!b) return;
    state.view = b.dataset.view;
    [...$("#sigma-tabs").children].forEach((c) => c.classList.toggle("on", c === b));
    render();
  };
  for (const id of ["#t-softmax", "#t-qk", "#t-relu"]) $(id).onchange = renderEquivalence;
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);

  render();
}

/** Weights never change during adaptation. Hashing them is the claim's second attack surface. */
async function weightsHash(buf) {
  const d = await crypto.subtle.digest("SHA-256", buf);
  return [...new Uint8Array(d).slice(0, 6)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

const active = () => state.pairs.filter((_, i) => !state.removed.has(i));

/** Byte sequence: k demonstrations, then the query. Identical construction to cipher_task.py. */
function buildTokens() {
  const seq = [];
  for (const p of active()) seq.push(p.src, p.tgt, state.manifest.tokens.SEP);
  seq.push(state.manifest.tokens.QRY, state.pairs[state.queryIdx].src);
  return seq;
}

function addDemo() {
  const usedS = new Set(state.pairs.map((p) => p.src));
  const usedT = new Set(state.pairs.map((p) => p.tgt));
  const s = state.manifest.tokens.SRC.find((b) => !usedS.has(b));
  const t = state.manifest.tokens.TGT.find((b) => !usedT.has(b));
  if (s === undefined || t === undefined) return;
  state.pairs.push({ src: s, tgt: t });
  render();
}

function renderDemos() {
  const list = $("#demo-list");
  list.textContent = "";
  state.pairs.forEach((p, i) => {
    const row = document.createElement("div");
    row.className = "demo" + (state.removed.has(i) ? " removed" : "");
    row.append(chip(state.tokMap.get(p.src)));
    const a = document.createElement("span"); a.className = "arrow"; a.textContent = "→";
    row.append(a, chip(state.tokMap.get(p.tgt)));
    const btn = document.createElement("button");
    btn.textContent = state.removed.has(i) ? "↺" : "×";
    btn.title = state.removed.has(i) ? "restore" : "remove this demonstration";
    btn.onclick = () => {
      state.removed.has(i) ? state.removed.delete(i) : state.removed.add(i);
      render();
    };
    row.append(btn);
    list.append(row);
  });
  $("#add-demo").disabled = state.pairs.length >= 12;
}

function render() {
  renderDemos();
  const tokens = buildTokens();
  const qPair = state.pairs[state.queryIdx];

  // query row
  const qr = $("#query-row");
  qr.textContent = "";
  qr.append(chip(state.tokMap.get(qPair.src), { big: true }));
  const a = document.createElement("span"); a.className = "arrow"; a.textContent = "→";
  const ghost = document.createElement("span");
  ghost.className = "chip chip-big chip-ghost"; ghost.textContent = "?";
  qr.append(a, ghost);

  // forward pass
  const t0 = performance.now();
  const out = state.model.forward(tokens);
  const ms = performance.now() - t0;

  const V = state.model.V, last = tokens.length - 1;
  let best = 0;
  for (let i = 1; i < V; i++) if (out.logits[last * V + i] > out.logits[last * V + best]) best = i;

  // The oracle: the mapping is defined ONLY by the demonstrations still present.
  const shown = active().find((p) => p.src === qPair.src);
  const truth = shown ? shown.tgt : null;

  $("#model-out").textContent = "";
  $("#model-out").append(chip(state.tokMap.get(best) ?? { kind: "sep", label: "?", byte: best },
                              { big: true }));
  $("#oracle-out").textContent = "";
  if (truth !== null) {
    $("#oracle-out").append(chip(state.tokMap.get(truth), { big: true }));
  } else {
    const u = document.createElement("span");
    u.className = "chip chip-big chip-ghost"; u.textContent = "—";
    $("#oracle-out").append(u);
  }

  const note = $("#verdict-note");
  if (truth === null) {
    note.innerHTML = `<span class="no">unreachable</span> — that binding is no longer in the
      prompt, so nothing defines the answer. The model still answers, confidently. ${ms.toFixed(0)}ms`;
  } else if (best === truth) {
    note.innerHTML = `<span class="ok">correct</span> — read out of σ, with zero
      parameter updates. ${ms.toFixed(0)}ms`;
  } else {
    note.innerHTML = `<span class="no">wrong</span> — same query bytes, different
      demonstrations. ${ms.toFixed(0)}ms`;
  }
  $("#param-updates").textContent = "0";

  renderSigma(out);
  renderEquivalence();
}

function renderSigma(out) {
  const { N, D } = state.model;
  const cur = out.sigmas[0].sigma;            // layer 0, head 0
  const cv = $("#sigma-canvas");
  const noteEl = $("#sigma-note");

  if (state.view === "state") {
    drawMatrix(cv, cur, N, D);
    noteEl.textContent = "σ is dense: every cell carries some value within a few tokens, " +
      "because the keys are ~43% dense. The structure lives in the writes and the differences.";
    drawRowEnergy($("#energy-canvas"), cur, N, D);
  } else if (state.view === "write") {
    // The last write: rank-one by construction, and the one genuinely legible picture.
    const s = out.sigmas[0];
    const snaps = equivalenceCheck(state.model, buildTokens(), {}).snapshots;
    const n = snaps.length;
    const d = new Float32Array(N * D);
    if (n >= 2) for (let i = 0; i < d.length; i++) d[i] = snaps[n - 1][i] - snaps[n - 2][i];
    drawMatrix(cv, d, N, D);
    noteEl.textContent = "Δσ for the final token = rope(K)ₜ ⊗ Vₜ. " +
      "A rank-one outer product — one row pattern times one column pattern.";
    drawRowEnergy($("#energy-canvas"), d, N, D);
  } else {
    if (state.prevSigma) {
      drawDiff(cv, cur, state.prevSigma, N, D);
      noteEl.textContent = "Change in σ since the previous configuration. " +
        "Remove a demonstration and its contribution disappears from the state.";
      const d = new Float32Array(N * D);
      for (let i = 0; i < d.length; i++) d[i] = cur[i] - state.prevSigma[i];
      drawRowEnergy($("#energy-canvas"), d, N, D);
    } else {
      drawMatrix(cv, cur, N, D);
      noteEl.textContent = "Change a demonstration to see the difference map.";
      drawRowEnergy($("#energy-canvas"), cur, N, D);
    }
  }
  state.prevSigma = Float32Array.from(cur);
}

function renderEquivalence() {
  const opts = {
    softmax: $("#t-softmax").checked,
    decoupleQK: $("#t-qk").checked,
    noRelu: $("#t-relu").checked,
  };
  const r = equivalenceCheck(state.model, buildTokens(), opts);
  const broken = r.relative > 1e-5;
  $("#resid").textContent = r.maxAbs.toExponential(2);
  $(".residual").classList.toggle("broken", broken);

  const note = $("#equiv-note");
  if (opts.softmax) {
    note.innerHTML = `<span class="no">Equivalence dies.</span> Softmax normalises across every
      past position, so the state can no longer be a fixed-size running sum — you must keep
      all past keys. <strong>This is why a Transformer needs a KV cache that grows.</strong>`;
  } else if (opts.decoupleQK || opts.noRelu) {
    note.innerHTML = `<span class="ok">Still equivalent.</span> These do <em>not</em> buy constant
      memory. Q=K and the ReLU buy something else: they make σ readable as a Hebbian synapse
      matrix over one non-negative neuron basis — and that non-negativity is what costs
      capacity.`;
  } else {
    note.innerHTML = `Parallel and recurrent agree to float precision. Verified against PyTorch
      at float64: <code>2.8e-14</code>.`;
  }
}

boot().catch((e) => {
  document.body.innerHTML = `<pre style="padding:24px;color:#e34948">${e.stack}</pre>`;
});

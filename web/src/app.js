// Demonstrations Are Weights — orchestration.
//
// Everything on the page except the boxed table in Act Four is computed live from a
// 131,072-parameter BDH we trained (architecture: vendored, unmodified pathwaycom/bdh).
// No step is animated or scripted.
import { BDHModel, equivalenceCheck, trace } from "./bdh.js";
import { makeTokenMap, chip } from "./tokens.js";
import { drawMatrix, drawDiff, drawRowEnergy } from "./sigma-view.js";
import { capacityCurve, capacityTrial, meanCosine } from "./capacity.js";
import { drawCapacity, capacityTable } from "./chart.js";
import { MachineRoom } from "./machine.js";
import { tgtHue } from "./tokens.js";

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

const S = {
  model: null, manifest: null, tok: null, presets: null,
  pairs: [], removed: new Set(), queryIdx: 1,
  view: "write", nLayer: 4, prevSigma: null, capCurves: null,
};

/* Every BDH-CQ number, labelled by what KIND of claim it is. This is the evidence-discipline
   beat: a computed cost is not a price, a developer-reported score is not a verified one. */
const LEDGER = [
  ["computed", "<b>$0.00070 per task</b> is not a price. It is 0.85 H200-GPU-seconds costed at an assumed $3/GPU-hour. Change the assumption and the headline moves proportionally. §5, p.4."],
  ["unverified", "The report gives <b>two different costs for the same 118/400 result</b> — $0.00070 in §5 and $0.00265246 in §6.6 — and never reconciles them. At the higher figure, “57× cheaper” becomes roughly 15×."],
  ["reported", "<b>29.5% pass@2</b> is developer-reported. BDH-CQ appears on neither the official ARC Prize leaderboard nor the community one (checked 2026-09-04). HRM and TRM do."],
  ["reported", "The <b>independent audit</b> was run by two co-authors of the paper. The report says so plainly; the press release does not."],
  ["unverified", "<b>ConceptARC (59.38%)</b> is listed in the training mixture in §4.2 and then used as an evaluation set in §6.1. §6.5 concedes the control does not rule out training exposure."],
  ["absent", "There are <b>no ARC-AGI-2 results</b> anywhere — not in the report, not in the blogs. It is named as future work."],
  ["absent", "What physically changes between the <b>effort levels</b> is not disclosed, and no value of R is ever given."],
];

async function boot() {
  const [manifest, bin, presets] = await Promise.all([
    fetch("public/model.json").then((r) => r.json()),
    fetch("public/model.bin").then((r) => r.arrayBuffer()),
    fetch("public/presets.json").then((r) => r.json()),
  ]);
  S.manifest = manifest; S.presets = presets;
  S.model = new BDHModel(manifest, bin);
  S.tok = makeTokenMap(manifest);

  const P = presets.hook;
  S.pairs = P.pairs.map(([s, t]) => ({ src: s.charCodeAt(0), tgt: t.charCodeAt(0) }));
  S.queryIdx = S.pairs.findIndex((p) => p.src === P.query.charCodeAt(0));

  $("#sigma-dims").textContent = `${S.model.N} × ${S.model.D}`;
  $("#ax-rows").textContent = `${S.model.N} neurons`;
  $("#ax-cols").textContent = `${S.model.D} dims`;
  S.hash0 = await liveWeightsHash(S.model);
  $("#whash").textContent = S.hash0;

  $("#try-remove").onclick = () => {
    const b = $("#try-remove");
    if (S.removed.has(S.queryIdx)) {
      S.removed.delete(S.queryIdx);
      b.textContent = "Remove the demonstration it needs →"; b.classList.remove("done");
    } else {
      S.removed.add(S.queryIdx);
      b.textContent = "Put it back ↺"; b.classList.add("done");
    }
    render();
  };
  $("#sigma-tabs").onclick = (e) => {
    const b = e.target.closest("button"); if (!b) return;
    S.view = b.dataset.view;
    $$("#sigma-tabs button").forEach((c) => c.classList.toggle("on", c === b));
    render();
  };
  for (const id of ["#t-softmax", "#t-qk", "#t-relu"]) $(id).onchange = renderEquiv;
  $("#k-slider").oninput = onK;
  $("#t-signed").onchange = onK;   // both curves are always drawn; this moves the readout only
  $("#cap-table-toggle").onclick = () => {
    const t = $("#cap-table"), on = t.hidden;
    t.hidden = !on; $("#cap-chart").hidden = on;
    $("#cap-table-toggle").textContent = on ? "chart" : "table";
  };
  $("#add-demo").onclick = addDemo;
  $("#l-slider").oninput = () => {
    S.nLayer = +$("#l-slider").value; $("#l-val").textContent = S.nLayer; render();
  };
  $$(".recall-q").forEach((q) => q.onclick = () => {
    if (q.classList.contains("open")) return;
    q.classList.add("open");
    const a = document.createElement("span");
    a.className = "ans"; a.innerHTML = q.dataset.a; q.append(a);
  });
  matchMedia("(prefers-color-scheme: dark)")
    .addEventListener("change", () => { render(); renderCapacity(); });

  $("#ledger").innerHTML = LEDGER.map(([tier, html]) =>
    `<li><span class="tier tier-${tier}">${tier}</span><span>${html}</span></li>`).join("");

  S.machine = new MachineRoom($("#machine-room"));
  spy();
  render();
  renderCapacity();
}

/**
 * Hash the weight arrays the forward pass ACTUALLY reads -- not the source file.
 * The fp16 tensors are decoded into fresh Float32Arrays at load, so digesting the downloaded
 * blob would prove the wrong thing. This digests the live arrays, and is re-run after every
 * interaction so "parameter updates: 0" is a measurement rather than a caption.
 */
async function liveWeightsHash(model) {
  const parts = Object.keys(model.w).sort().map((k) => model.w[k]);
  const total = parts.reduce((n, a) => n + a.byteLength, 0);
  const flat = new Uint8Array(total);
  let o = 0;
  for (const a of parts) { flat.set(new Uint8Array(a.buffer, a.byteOffset, a.byteLength), o); o += a.byteLength; }
  const d = await crypto.subtle.digest("SHA-256", flat);
  return [...new Uint8Array(d)].slice(0, 6).map((b) => b.toString(16).padStart(2, "0")).join("");
}

const active = () => S.pairs.filter((_, i) => !S.removed.has(i));

function buildTokens() {
  const seq = [];
  for (const p of active()) seq.push(p.src, p.tgt, S.manifest.tokens.SEP);
  seq.push(S.manifest.tokens.QRY, S.pairs[S.queryIdx].src);
  return seq;
}

function addDemo() {
  const uS = new Set(S.pairs.map((p) => p.src)), uT = new Set(S.pairs.map((p) => p.tgt));
  const s = S.manifest.tokens.SRC.find((b) => !uS.has(b));
  const t = S.manifest.tokens.TGT.find((b) => !uT.has(b));
  if (s === undefined || t === undefined) return;
  S.pairs.push({ src: s, tgt: t });
  render();
}

function demoRow(p, i) {
  const row = document.createElement("div");
  row.className = "demo" + (S.removed.has(i) ? " removed" : "");
  row.append(chip(S.tok.get(p.src)));
  const a = document.createElement("span"); a.className = "arrow"; a.textContent = "→";
  row.append(a, chip(S.tok.get(p.tgt)));
  const b = document.createElement("button");
  b.textContent = S.removed.has(i) ? "↺" : "×";
  b.title = S.removed.has(i) ? "restore this demonstration" : "remove this demonstration";
  b.onclick = (e) => {
    e.stopPropagation();
    S.removed.has(i) ? S.removed.delete(i) : S.removed.add(i);
    render();
  };
  row.append(b);
  return row;
}

function render() {
  const list = $("#demo-list"); list.textContent = "";
  S.pairs.forEach((p, i) => list.append(demoRow(p, i)));
  const sb = $("#sb-demos"); sb.textContent = "";
  S.pairs.forEach((p, i) => sb.append(demoRow(p, i)));
  $("#add-demo").disabled = S.pairs.length >= 12;

  const pick = $("#sb-query-pick"); pick.textContent = "";
  S.pairs.forEach((p, i) => {
    const c = chip(S.tok.get(p.src));
    if (i === S.queryIdx) c.classList.add("sel");
    c.style.cursor = "pointer";
    c.onclick = () => { S.queryIdx = i; render(); };
    pick.append(c);
  });

  const tokens = buildTokens();
  const qp = S.pairs[S.queryIdx];

  const qr = $("#query-row"); qr.textContent = "";
  qr.append(chip(S.tok.get(qp.src), { big: true }));
  const ar = document.createElement("span"); ar.className = "arrow"; ar.textContent = "→";
  const gh = document.createElement("span");
  gh.className = "chip chip-big chip-ghost"; gh.textContent = "?";
  qr.append(ar, gh);

  const t0 = performance.now();
  const out = S.model.forward(tokens, { layers: S.nLayer });
  $("#fwd-ms").textContent = `${(performance.now() - t0).toFixed(0)} ms`;

  const V = S.model.V, last = tokens.length - 1;
  let best = 0;
  for (let i = 1; i < V; i++) if (out.logits[last * V + i] > out.logits[last * V + best]) best = i;
  const shown = active().find((p) => p.src === qp.src);
  const truth = shown ? shown.tgt : null;

  $("#model-out").textContent = "";
  $("#model-out").append(chip(S.tok.get(best) ?? { kind: "sep", label: "?", byte: best }, { big: true }));
  $("#oracle-out").textContent = "";
  if (truth !== null) $("#oracle-out").append(chip(S.tok.get(truth), { big: true }));
  else {
    const u = document.createElement("span");
    u.className = "chip chip-big chip-ghost"; u.textContent = "—";
    $("#oracle-out").append(u);
  }

  const n = $("#verdict-note");
  if (truth === null)
    n.innerHTML = `<b>Unreachable.</b> Nothing in the prompt defines this answer any more, so
      there is no right answer to give. The model answers anyway, and confidently. The weights
      never moved.`;
  else if (best === truth)
    n.innerHTML = `<b>Correct</b> — and nothing was trained. The rule was read out of a state
      built during this forward pass, then discarded.`;
  else
    n.innerHTML = `<b>Wrong.</b> Identical query bytes, different demonstrations.`;

  $("#sb-tokens").textContent =
    tokens.map((b) => {
      const t = S.tok.get(b);
      return t ? (t.kind === "sep" ? "·" : t.kind === "qry" ? "|" : t.label) : "?";
    }).join(" ") + `\n\n${tokens.length} tokens · bytes [${tokens.join(", ")}]`;

  $("#l-note").textContent = S.nLayer === 4
    ? "4 is what this model was trained with."
    : `Trained at 4. At L=${S.nLayer} the same weights run ${S.nLayer > 4 ? "more" : "fewer"} times than training ever used, so behaviour here is off-distribution.`;

  renderSigma(out);
  renderEquiv();
  if (S.machine) {
    const tr = trace(S.model, tokens);
    S.machine.load(tr, tokens.map((b) => {
      const k = S.tok.get(b);
      return { ...k, hue: k.kind === "tgt" ? tgtHue(k.i) : null };
    }));
  }
  verifyWeightsUnchanged();
}

/** Re-hash after every interaction. If adaptation ever touched a parameter, this would move. */
async function verifyWeightsUnchanged() {
  const h = await liveWeightsHash(S.model);
  const same = h === S.hash0;
  $("#whash").textContent = h;
  const note = $("#whash-note");
  note.textContent = same
    ? `re-checked after this interaction, unchanged`
    : `CHANGED — this should be impossible`;
  note.style.color = same ? "" : "var(--bad)";
  $("#param-updates").textContent = same ? "0" : "?";
}

function renderSigma(out) {
  const { N, D } = S.model;
  const cur = out.sigmas[0].sigma;
  const cv = $("#sigma-canvas"), note = $("#sigma-note");
  let nz = 0; for (let i = 0; i < cur.length; i++) if (cur[i] !== 0) nz++;
  $("#sigma-density").textContent = `${(100 * nz / cur.length).toFixed(0)}%`;

  if (S.view === "write") {
    const snaps = equivalenceCheck(S.model, buildTokens(), {}).snapshots;
    const d = new Float32Array(N * D);
    if (snaps.length >= 2)
      for (let i = 0; i < d.length; i++)
        d[i] = snaps[snaps.length - 1][i] - snaps[snaps.length - 2][i];
    drawMatrix(cv, d, N, D);
    note.textContent = "One token's contribution: rope(K)ₜ ⊗ Vₜ. A rank-one outer product — one row pattern times one column pattern, which is why it bands.";
    drawRowEnergy($("#energy-canvas"), d, N, D);
  } else if (S.view === "state") {
    drawMatrix(cv, cur, N, D);
    note.textContent = "The accumulated state. Dense within a few tokens, because the keys are themselves about 43% dense — so the raw matrix is not where the structure shows.";
    drawRowEnergy($("#energy-canvas"), cur, N, D);
  } else {
    if (S.prevSigma && S.prevSigma.length === cur.length) {
      drawDiff(cv, cur, S.prevSigma, N, D);
      const d = new Float32Array(N * D);
      for (let i = 0; i < d.length; i++) d[i] = cur[i] - S.prevSigma[i];
      drawRowEnergy($("#energy-canvas"), d, N, D);
      note.textContent = "Difference from the previous configuration. Remove a demonstration and its whole contribution leaves the state.";
    } else {
      drawMatrix(cv, cur, N, D);
      note.textContent = "Change something — remove a demonstration, or query a different symbol — to see what moves.";
      drawRowEnergy($("#energy-canvas"), cur, N, D);
    }
  }
  S.prevSigma = Float32Array.from(cur);
}

function renderEquiv() {
  const o = {
    softmax: $("#t-softmax").checked,
    decoupleQK: $("#t-qk").checked,
    noRelu: $("#t-relu").checked,
  };
  const r = equivalenceCheck(S.model, buildTokens(), o);
  $("#resid").textContent = r.maxAbs.toExponential(2);
  $(".residual-card").classList.toggle("broken", r.relative > 1e-5);
  const n = $("#equiv-note");
  if (o.softmax)
    n.innerHTML = `<b>Broken.</b> Softmax normalises each score against every other, so no
      fixed-size running sum can reproduce it. You have to keep every past key — which is
      exactly what a KV cache is.`;
  else if (o.decoupleQK || o.noRelu)
    n.innerHTML = `<b>Still equivalent.</b> These do not buy constant memory. Tying Q to K and
      clamping activations non-negative are what make σ <em>readable</em> as synapses over one
      basis of neurons.`;
  else
    n.innerHTML = `Parallel and recurrent agree to floating-point noise — the same layer computed
      two entirely different ways. Verified against PyTorch at float64: <code>2.8e-14</code>.`;
}

/* ── capacity ────────────────────────────────────────────────────────────── */
const KS = [2, 4, 8, 16, 32, 64, 128, 192, 256];

function renderCapacity() {
  S.capCurves = [
    { name: "ReLU'd keys", points: capacityCurve(KS, { nonneg: true, trials: 4 }) },
    { name: "signed keys", points: capacityCurve(KS, { nonneg: false, trials: 4 }) },
  ];
  drawCapacity($("#cap-chart"), S.capCurves, { marker: +$("#k-slider").value });
  capacityTable($("#cap-table"), S.capCurves);
  $("#cos-nn").textContent = meanCosine({ nonneg: true }).toFixed(3);
  $("#cos-s").textContent = meanCosine({ nonneg: false }).toFixed(3);
  onK();
}

function onK() {
  const k = +$("#k-slider").value, signed = $("#t-signed").checked;
  $("#k-val").textContent = k; $("#k-echo").textContent = k;
  $("#cap-at-k").textContent =
    `${Math.round(capacityTrial(k, { nonneg: !signed, seed: 1000 }) * 100)}%`;
  $("#cap-at-k-note").textContent = signed ? "signed keys" : "ReLU'd keys, as BDH has";
  if (S.capCurves) drawCapacity($("#cap-chart"), S.capCurves, { marker: k });
  $("#cap-note").innerHTML = signed
    ? `<b>The same state now holds far more.</b> Identical shape, identical rank — only the key
       geometry changed. Signed keys are orthogonal on average, so they barely interfere, and they
       keep working past k = N = 256, beyond the state's own rank.`
    : `Non-negative keys cannot be near-orthogonal. Two ReLU'd Gaussians overlap by 1/π ≈ 0.318
       on average, and that overlap is what corrupts reads as k grows. The ReLU is what makes
       BDH's activations sparse, positive and inspectable
       (<a href="https://arxiv.org/abs/2509.26507">arXiv:2509.26507</a>) — this is its price.`;
}

/* ── scroll rail ─────────────────────────────────────────────────────────── */
function spy() {
  const io = new IntersectionObserver((es) => {
    for (const e of es)
      if (e.isIntersecting)
        $$("#rail a").forEach((a) => a.classList.toggle("on", a.dataset.act === e.target.id));
  }, { rootMargin: "-45% 0px -45% 0px" });
  $$("section.act").forEach((s) => io.observe(s));
}

boot().catch((e) => {
  document.body.innerHTML =
    `<pre style="padding:32px;font:13px/1.6 ui-monospace,monospace;color:#e34948">${e.stack}</pre>`;
});

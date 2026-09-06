// THE FIELD, driven. Loads the 8M model's exported field + traces, runs the playback loop, and
// wires the overlays. All maths lives in field-data.js / bigdata.js; this is orchestration.

import { loadPack, positions, frame, nTokens, sigmaEnergy } from "./bigdata.js";
import {
  decayInto, tokenFrame, referenceLevel, nearestNode, edgesFor,
  communityColor, degreeLevel, arcsInto,
} from "./field-data.js";
import { createArcs } from "./arcs.js";
import { createOrthant, measure } from "./orthant.js";
import { createField } from "./field-gl.js";
import { FIELD_SURFACE, IGNITION, SIGMA_RAMP, COMMUNITY, COMMUNITY_OTHER } from "./palette.js";

const $ = (id) => document.getElementById(id);
const canvas = $("field");

/**
 * Inflate a gzip ArrayBuffer with DecompressionStream.
 *
 * Written as an explicit reader loop rather than `new Response(stream).arrayBuffer()` so it
 * depends on nothing but the stream itself -- this runs in the browser, and in Node under the
 * smoke gate against a stubbed fetch that has no Response body.
 */
async function inflate(buf) {
  const ds = new DecompressionStream("gzip");
  const w = ds.writable.getWriter();
  w.write(new Uint8Array(buf));
  w.close();
  const r = ds.readable.getReader();
  const chunks = [];
  let n = 0;
  for (;;) {
    const { value, done } = await r.read();
    if (done) break;
    chunks.push(value);
    n += value.length;
  }
  const out = new Uint8Array(n);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out.buffer;
}

/**
 * The manifest declares its own compression, so the loader does not have to know which pack is
 * which and there is no 404-then-retry. traces is gzipped (11.3 MB raw, 5.9 MB shipped -- every
 * sentence carries all six iterations, attention and sigma); field is small and is not.
 */
const fetchPack = async (stem) => {
  const man = await fetch(`${stem}.json`).then((r) => r.json());
  const gz = man.compression === "gzip";
  const bin = await fetch(`${stem}.bin${gz ? ".gz" : ""}`).then((r) => r.arrayBuffer());
  return loadPack(man, gz ? await inflate(bin) : bin);
};

const S = {                                  // everything mutable, in one place
  sent: 0, layer: 0, tok: 0, playing: false, mode: "act", layout: "global.xy",
  ref: 1, nTok: 1, hover: -1, pinned: -1, dragging: false, last: 0, view: "field",
};

let field, fieldPack, tracePack, xy, N, nNodes, maxDeg;
let staticLevel = null;                      // for the non-animated modes
let arcs = null;                             // canvas-2D attention view
let tileAct = [], tileRef = [];              // one activation buffer per iteration
let orth = null, orthData = null;            // the 1/pi geometry view

init().catch((e) => {
  console.error(e);
  $("nogl").hidden = false;
  $("nogl").querySelector("p").innerHTML = `<b>Could not load the field.</b> ${e.message}`;
});

async function init() {
  [fieldPack, tracePack] = await Promise.all([
    fetchPack("public/big/field"), fetchPack("public/big/traces"),
  ]);
  N = fieldPack.manifest.N_per_head;
  nNodes = fieldPack.manifest.n_neurons_total;
  xy = positions(fieldPack, "global.xy");
  $("n-neurons").textContent = nNodes.toLocaleString("en-US");
  // Was hard-coded to 38.5%, which is the hero sentence at this layer/head, not the corpus.
  // Attention is exported for all seven sentences now, so the label can be literally true.
  if (tracePack.manifest.corpus_neg_share != null) {
    $("a-negall").textContent = `${(tracePack.manifest.corpus_neg_share * 100).toFixed(1)}%`;
  }

  maxDeg = 0;
  for (let h = 0; h < 4; h++) {
    for (const d of fieldPack.t[`h${h}.out_deg`]) if (d > maxDeg) maxDeg = d;
  }

  buildSelectors();

  field = createField(canvas, { xy, edges: true }, { surface: FIELD_SURFACE });
  if (!field.ok) return showFallback();

  arcs = createArcs($("arcs"), { pos: "#e0872a", neg: "#3987e5", ground: FIELD_SURFACE });
  orth = createOrthant($("orthant"), { ground: FIELD_SURFACE });
  bindPointer();
  bindControls();
  bindPanels();
  setSentence(tracePack.manifest.hero);
  S.playing = true;
  $("play").innerHTML = "&#10073;&#10073;";
  requestAnimationFrame(loop);
}

// ── selectors ────────────────────────────────────────────────────────────────────────────────
function buildSelectors() {
  const ss = $("sel-sent");
  for (const s of tracePack.manifest.sentences) {
    const o = document.createElement("option");
    o.value = s.i;
    const src = s.text.replace(/^<F:en>/, "").split("<T:fr>")[0];
    o.textContent = (s.hero ? "★ " : "") + src.slice(0, 42);
    ss.append(o);
  }
  ss.value = tracePack.manifest.hero;
}

function layerOptions() {
  const s = tracePack.manifest.sentences.find((x) => x.i === S.sent);
  const sel = $("sel-layer");
  sel.innerHTML = "";
  for (const L of s.layers) {
    const o = document.createElement("option");
    o.value = L.layer;
    o.textContent = `${L.layer + 1} of 6` + (s.layers.length === 1 ? "  (only this one exported)" : "");
    sel.append(o);
  }
  sel.disabled = s.layers.length === 1;
  S.layer = s.layers[Math.min(s.layers.length - 1, s.layers.findIndex((l) => l.layer === S.layer) >= 0
    ? s.layers.findIndex((l) => l.layer === S.layer) : 0)].layer;
  sel.value = S.layer;
}

function setSentence(i) {
  S.sent = +i;
  layerOptions();
  S.nTok = nTokens(tracePack, S.sent, S.layer);
  S.ref = referenceLevel(tracePack, S.sent, S.layer, S.nTok);
  S.tok = 0;
  $("scrub").max = String(S.nTok - 1);
  buildStrip();
  paintedTok = -1;
  field.activation.fill(0);

  const s = tracePack.manifest.sentences.find((x) => x.i === S.sent);
  arcs.setSentence(s.text, s.bytes);
  tileAct = s.layers.map(() => new Float32Array(nNodes));
  tileRef = s.layers.map((L) => referenceLevel(tracePack, S.sent, L.layer, S.nTok));
  refreshMode();
  refreshView();
}

/**
 * What THIS sentence actually has in the pack.
 *
 * This used to read `arcs: !!s.hero`, and the export only gave the hero sentence its six
 * iterations, attention and sigma energy. So on the other six sentences the iteration selector
 * was disabled, the small-multiples view refused to draw, the arc diagram refused to draw, and
 * the sigma colour mode silently painted an empty field. Every sentence now carries all of it
 * -- but the check stays keyed on the presence of the tensors rather than on a flag, because a
 * flag is what let a whole page go dead without a single gate noticing.
 */
function capsFor(sent = S.sent) {
  const s = tracePack.manifest.sentences.find((x) => x.i === sent);
  return {
    tiles: s.layers.length > 1,
    arcs: !!tracePack.t[`s${sent}.attn_w`],
    sigma: !!tracePack.t[`s${sent}.sigma_energy`],
    s,
  };
}

function refreshView() {
  const cap = capsFor();
  const v = S.view;
  const showArcs = v === "arcs" && cap.arcs;
  const showOrth = v === "orthant";
  $("arcs").hidden = !showArcs;
  $("orthant").hidden = !showOrth;
  $("field").hidden = showArcs || showOrth;
  $("tile-labels").hidden = !(v === "tiles" && cap.tiles);
  $("arc-stats").hidden = !showArcs;
  $("orth-panel").hidden = !showOrth;
  $("card").hidden = true;
  // The orthant is pure geometry -- no model, no sentence -- so the controls that select one are
  // meaningless there and the transport would imply a timeline it does not have.
  for (const id of ["sel-sent", "sel-layout"]) $(id).disabled = showOrth;
  $("sel-layer").disabled = showOrth || showArcs;
  document.querySelector(".transport").hidden = showOrth;

  if (showOrth) {
    if (!orthData) setOrthantK(+$("o-slider").value);   // lazy: never on the boot path
    $("mode-note").textContent =
      "No model here: these are random unit vectors, ReLU'd on the left and signed on the right. " +
      "The gap is the whole cost of non-negativity.";
  } else if (v === "arcs" && !cap.arcs) {
    $("mode-note").textContent =
      "No attention scores in the pack for this sentence — re-run npm run export:big.";
  } else if (v === "tiles" && !cap.tiles) {
    $("mode-note").textContent =
      "Only one iteration is in the pack for this sentence — re-run npm run export:big.";
  } else if (showArcs) {
    // The arc diagram is one exported layer/head for every sentence, so the iteration selector
    // does not drive it. Say so, rather than leaving a live-looking control that does nothing.
    $("mode-note").textContent =
      `Causal attention scores, ${tracePack.manifest.attention_scope || "one layer and head"}. ` +
      "Blue arcs are NEGATIVE: activations are non-negative by construction, so a negative score " +
      "can only come from RoPE rotating the keys before they meet. The iteration selector does " +
      "not apply here — one layer per sentence is exported.";
  } else {
    $("mode-note").textContent = MODE_NOTE[S.mode];
  }
  // colour-by only applies to the field views; arcs carry their own sign encoding
  $("sel-mode").disabled = showArcs || showOrth;
}

/**
 * Resample both key sets. The 3-D pair is what the spheres draw; the 3,072-D pair is the model's
 * actual key dimension and is what the headline number must come from. Both are real draws --
 * nothing here is a hard-coded constant standing in for a measurement.
 *
 * 3,072 dims x 256 keys is ~100M multiply-adds for the all-pairs cosine, so it runs on release
 * of the slider, not on every input event.
 */
function setOrthantK(k) {
  const D_MODEL = fieldPack.manifest.N_per_head;
  orthData = {
    k, d: 3,
    relu: measure(k, 3, true, 7),
    signed: measure(k, 3, false, 7),
    // 128 keys is plenty for a MEAN at this dimension and halves the Box-Muller cost; the
    // estimate is tight to ~1e-3 either way, against an effect of 0.318 vs 0.000.
    hiK: Math.min(k, 128),
    hi: measure(Math.min(k, 128), D_MODEL, true, 7),
    hiSigned: measure(Math.min(k, 128), D_MODEL, false, 7),
  };
  $("o-k").textContent = String(k);
  $("o-hi").textContent = `${orthData.hi.deg.toFixed(1)}°`;
  $("o-hisign").textContent = `${orthData.hiSigned.deg.toFixed(1)}°`;
}

/** Grid for the six small multiples, kept clear of the overlay panels where there is room. */
function tileRects(count) {
  const W = window.innerWidth, H = window.innerHeight;
  const wide = W - 380 - 290 > 560;
  const x0 = wide ? 368 : 16, x1 = wide ? W - 286 : W - 16;
  const y0 = wide ? 24 : 250, y1 = H - 132;
  const cols = count <= 3 ? count : 3, rows = Math.ceil(count / cols);
  const gw = (x1 - x0) / cols, gh = (y1 - y0) / rows;
  return Array.from({ length: count }, (_, i) => ({
    x: x0 + (i % cols) * gw, y: y0 + Math.floor(i / cols) * gh,
    w: gw - 8, h: gh - 8,
  }));
}

// ── the byte strip ───────────────────────────────────────────────────────────────────────────
function buildStrip() {
  const s = tracePack.manifest.sentences.find((x) => x.i === S.sent);
  const strip = $("strip");
  strip.innerHTML = "";
  const text = s.text;
  // Model tokens are raw BYTES (CLAUDE.md item 10). For this model they are UTF-8 of readable
  // Latin text, so bytes map to characters -- but the tags are structure, not language, and are
  // marked as such rather than being shown as if they were words the model translated.
  let bi = 0;
  for (let k = 0; k < text.length; k++) {
    const ch = text[k];
    const len = new TextEncoder().encode(ch).length;
    const b = document.createElement("b");
    b.textContent = ch === " " ? "·" : ch;
    b.dataset.t = String(bi);
    if (/[<>:]/.test(ch) || (k > 0 && text.slice(0, k + 1).match(/<[FT]:[a-z]{2}>$/))) b.classList.add("tag");
    b.title = `byte ${bi}`;
    b.onclick = () => { S.tok = Math.min(S.nTok - 1, bi); S.playing = false; syncPlay(); };
    strip.append(b);
    bi += len;
  }
}

let paintedTok = -1;

function paintStrip() {
  if (paintedTok === S.tok) return;        // the strip only changes when the token does
  paintedTok = S.tok;
  const strip = $("strip");
  for (const b of strip.children) {
    const t = +b.dataset.t;
    b.classList.toggle("now", t === S.tok);
    b.classList.toggle("seen", t < S.tok);
  }
  const now = strip.querySelector(".now");
  // Optional-call: jsdom has no scrollIntoView, and neither do some embedded webviews. Scrolling
  // is a nicety; the highlight is the information, and it must not be able to kill the loop.
  if (now && S.playing) now.scrollIntoView?.({ block: "nearest", inline: "nearest" });
}

// ── modes ────────────────────────────────────────────────────────────────────────────────────
const MODE_NOTE = {
  act: "Amber is a magnitude: brighter means a larger activation at this token. About 5% of neurons are non-zero at any moment — that is the ReLU, not a display threshold.",
  sigma: "‖σₙ‖ — how much has been written onto each neuron's row of the state so far. It only grows: this is the memory filling up.",
  degree: "Synapses per neuron in G* = DₓᵀEᵀ, log-scaled. 39% of neurons have none; 13% carry half the graph.",
  community: "Louvain communities. Only the three largest get a hue — modularity is 0.08 and they have no spatial separation, so more colours would draw structure that is not there.",
};

function refreshMode() {
  const m = S.mode;
  $("mode-note").textContent = MODE_NOTE[m];
  field.setRamp(m === "sigma" ? "sigma" : "ignition");
  staticLevel = null;

  if (m === "degree") {
    staticLevel = new Float32Array(nNodes);
    for (let h = 0; h < 4; h++) {
      const d = fieldPack.t[`h${h}.out_deg`];
      for (let i = 0; i < d.length; i++) staticLevel[h * N + i] = degreeLevel(d[i], maxDeg);
    }
  } else if (m === "community") {
    staticLevel = new Float32Array(nNodes);
    const rgb = new Float32Array(nNodes * 3);
    for (let h = 0; h < 4; h++) {
      const c = fieldPack.t[`h${h}.cluster`];
      for (let i = 0; i < c.length; i++) {
        const g = h * N + i;
        // Members of the top 3 are lit; everything else is dim grey. The hue is carried by the
        // tint buffer, so the legend's three swatches are what actually appears on screen.
        staticLevel[g] = c[i] < 3 ? 1 : 0.14;
        const hex = communityColor(c[i]);
        for (let k = 0; k < 3; k++) rgb[g * 3 + k] = parseInt(hex.slice(1 + k * 2, 3 + k * 2), 16) / 255;
      }
    }
    field.setTint(rgb);
  }
  legend();
}

function legend() {
  const bar = (stops) => `<span class="bar" style="background:linear-gradient(90deg,${stops.join(",")})"></span>`;
  const dot = (c) => `<span class="sw" style="background:${c}"></span>`;
  const L = $("legend");
  if (S.mode === "act") L.innerHTML = `<i>${bar(IGNITION)}</i><i>silent</i><i style="margin-left:auto">firing hard</i>`;
  else if (S.mode === "sigma") L.innerHTML = `<i>${bar(SIGMA_RAMP)}</i><i>nothing written</i><i style="margin-left:auto">saturated</i>`;
  else if (S.mode === "degree") L.innerHTML = `<i>${bar(IGNITION)}</i><i>no synapses</i><i style="margin-left:auto">${maxDeg}</i>`;
  else L.innerHTML = COMMUNITY.map((c, i) => `<i>${dot(c)}#${i + 1}</i>`).join("") +
                     `<i>${dot(COMMUNITY_OTHER)}other</i>`;
}

// ── the loop ─────────────────────────────────────────────────────────────────────────────────
const MS_PER_TOKEN = 70;

function loop(now) {
  const dt = now - S.last;
  if (S.playing && dt > MS_PER_TOKEN) {
    S.last = now;
    S.tok = (S.tok + 1) % S.nTok;
    $("scrub").value = String(S.tok);
  }

  const cap = capsFor();
  if (S.view === "orthant") {
    orth.spin(16);
    orth.draw(orthData);
    return requestAnimationFrame(loop);
  }
  if (S.view === "arcs" && cap.arcs) {
    const a = arcsInto(tracePack, S.sent, S.tok);
    arcs.draw(S.tok, a);
    $("a-neg").textContent = S.tok ? `${(a.negShare * 100).toFixed(0)}%` : "—";
    hud(); paintStrip();
    return requestAnimationFrame(loop);
  }
  if (S.view === "tiles" && cap.tiles) {
    const layers = cap.s.layers;
    const frames = [];
    for (let k = 0; k < layers.length; k++) {
      const f = tokenFrame(tracePack, S.sent, layers[k].layer, S.tok, tileRef[k]);
      frames.push(f);
      decayInto(tileAct[k], f);
    }
    const rects = tileRects(layers.length);
    field.renderTiles(rects.map((r, k) => ({ ...r, act: tileAct[k] })));
    placeTileLabels(rects, layers, frames);
    hud(); paintStrip();
    return requestAnimationFrame(loop);
  }

  const act = field.activation;
  if (staticLevel) {
    act.set(staticLevel);
  } else if (S.mode === "sigma") {
    act.fill(0);
    if (cap.sigma) {
      const e = sigmaEnergy(tracePack, S.sent, S.tok);
      let mx = 0;
      for (const v of e) if (v > mx) mx = v;
      // sigma energy is exported for ONE head at one layer, so only that head's slice lights
      const base = tracePack.manifest.attn_head * N;
      for (let i = 0; i < e.length; i++) act[base + i] = mx ? Math.sqrt(e[i] / mx) : 0;
    }
  } else {
    decayInto(act, S.playing || dt < 400 ? tokenFrame(tracePack, S.sent, S.layer, S.tok, S.ref) : null);
  }

  field.render();
  hud();
  paintStrip();
  requestAnimationFrame(loop);
}

/**
 * Labels carry the number that makes the small multiples worth comparing: how much each
 * iteration's active set OVERLAPS the first one.
 *
 * Measured on the hero sentence: the count climbs monotonically (591 -> 759 at token 40) while
 * Jaccard against iteration 1 falls to ~0.31 by iteration 6 -- yet 5-vs-6 is ~0.78. So the shared
 * operator recruits neurons for a few passes and then settles, with only ~20% of the union firing
 * in all six. Without the overlap number the six tiles just look like six similar clouds.
 */
let overlapMask = null;

function placeTileLabels(rects, layers, frames) {
  const box = $("tile-labels");
  if (box.children.length !== rects.length) {
    box.innerHTML = rects.map(() => "<b></b>").join("");
  }
  if (!overlapMask) overlapMask = new Uint8Array(nNodes);
  overlapMask.fill(0);
  for (const i of frames[0].idx) overlapMask[i] = 1;
  const base = frames[0].n;

  for (let k = 0; k < rects.length; k++) {
    let shared = 0;
    for (const i of frames[k].idx) if (overlapMask[i]) shared++;
    const union = base + frames[k].n - shared;
    const j = union ? shared / union : 1;
    box.children[k].innerHTML = k === 0
      ? `iteration <em>1</em> — ${frames[0].n.toLocaleString("en-US")} lit`
      : `iteration <em>${layers[k].layer + 1}</em> — ${frames[k].n.toLocaleString("en-US")} lit` +
        ` · <em>${(j * 100).toFixed(0)}%</em> shared with 1`;
    box.children[k].style.left = `${rects[k].x + 6}px`;
    box.children[k].style.top = `${rects[k].y + 4}px`;
  }
}

function hud() {
  const f = frame(tracePack, S.sent, S.layer, S.tok);
  $("c-active").textContent = f.n.toLocaleString("en-US");
  $("c-pct").textContent = `${(f.n / nNodes * 100).toFixed(1)}% of ${nNodes.toLocaleString("en-US")}`;
  $("c-tok").textContent = `${S.tok + 1} / ${S.nTok}`;
  const s = tracePack.manifest.sentences.find((x) => x.i === S.sent);
  const by = s.bytes[S.tok];
  $("c-byte").textContent = `byte ${by} · ${by >= 32 && by < 127 ? `"${String.fromCharCode(by)}"` : "—"}`;
  $("c-layer").textContent = `${S.layer + 1} / 6`;

  const id = S.pinned >= 0 ? S.pinned : S.hover;
  const card = $("card");
  if (id < 0) { card.hidden = true; return; }
  card.hidden = false;
  const h = Math.floor(id / N), local = id % N;
  $("k-id").textContent = id;
  $("k-head").textContent = h;
  $("k-local").textContent = local;
  $("k-deg").textContent = `${fieldPack.t[`h${h}.out_deg`][local]} / ${fieldPack.t[`h${h}.in_deg`][local]}`;
  const cl = fieldPack.t[`h${h}.cluster`][local];
  $("k-cl").textContent = cl === 255 ? "none (isolated)" : `#${cl + 1}`;
  $("k-act").textContent = field.activation[id].toFixed(3);
  $("k-note").textContent = S.pinned >= 0
    ? "Pinned. Its synapses are drawn; click empty space to release."
    : "Click to pin and draw its synapses.";
}

// ── interaction ──────────────────────────────────────────────────────────────────────────────
function bindPointer() {
  let dragStart = null, viewStart = null;

  canvas.addEventListener("pointermove", (e) => {
    if (dragStart) {
      const v = field.getView();
      const b = field.bounds();
      const k = b.span / Math.min(canvas.clientWidth, canvas.clientHeight) / v.zoom * 1.1;
      field.setView({ cx: viewStart.cx - (e.clientX - dragStart.x) * k,
                      cy: viewStart.cy + (e.clientY - dragStart.y) * k });
      return;
    }
    const [mx, my] = field.toModel(e.clientX, e.clientY);
    const tol = field.bounds().span * 0.012 / field.getView().zoom;
    S.hover = nearestNode(xy, mx, my, tol);
  });

  canvas.addEventListener("pointerdown", (e) => {
    dragStart = { x: e.clientX, y: e.clientY };
    viewStart = field.getView();
    canvas.setPointerCapture(e.pointerId);
    canvas.classList.add("grabbing");
  });

  canvas.addEventListener("pointerup", (e) => {
    const moved = dragStart && Math.hypot(e.clientX - dragStart.x, e.clientY - dragStart.y) > 4;
    dragStart = null;
    canvas.classList.remove("grabbing");
    if (moved) return;
    S.pinned = S.hover >= 0 && S.hover !== S.pinned ? S.hover : -1;
    field.setEdges(S.pinned >= 0 ? edgesFor(fieldPack, S.pinned, N) : null);
  });

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const v = field.getView();
    field.setView({ zoom: Math.min(40, Math.max(0.6, v.zoom * (e.deltaY < 0 ? 1.12 : 1 / 1.12))) });
  }, { passive: false });
}

function syncPlay() {
  $("play").innerHTML = S.playing ? "&#10073;&#10073;" : "&#9654;";
  $("play").setAttribute("aria-label", S.playing ? "Pause" : "Play");
}

/**
 * Collapse a side panel to its handle.
 *
 * The two HUDs float over a full-bleed canvas and cover a real fraction of it on a laptop, so
 * getting an unobstructed view has to be one click. Contents are hidden rather than translated
 * away, which keeps a collapsed panel out of the tab order instead of leaving five selects
 * focusable somewhere off screen.
 */
function setPanel(side, open) {
  const host = side === "l" ? $("hud-l") : $("panel");
  const body = side === "l" ? $("hud-l-body") : $("panel-body");
  const btn = side === "l" ? $("tgl-l") : $("tgl-r");
  body.hidden = !open;
  host.classList.toggle("is-collapsed", !open);
  btn.setAttribute("aria-expanded", String(open));
  // the chevron points where the panel will go
  btn.firstElementChild.innerHTML = (side === "l") === open ? "&lsaquo;" : "&rsaquo;";
  btn.title = open ? "Hide this panel — H hides both" : "Show this panel — H shows both";
  btn.setAttribute("aria-label", `${open ? "Hide" : "Show"} the ${side === "l" ? "info" : "controls"} panel`);
  // Nothing to re-measure: the canvases are full-bleed and field-gl re-reads the drawing buffer
  // on every render, and tileRects() is recomputed each frame from window.innerWidth.
}

function bindPanels() {
  const open = { l: true, r: true };
  const toggle = (side) => { open[side] = !open[side]; setPanel(side, open[side]); };
  $("tgl-l").onclick = () => toggle("l");
  $("tgl-r").onclick = () => toggle("r");
  addEventListener("keydown", (e) => {
    if (e.key !== "h" && e.key !== "H") return;
    if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target?.tagName || "")) return;
    e.preventDefault();
    const next = !(open.l && open.r);          // H is "show everything" unless everything is shown
    open.l = open.r = next;
    setPanel("l", next);
    setPanel("r", next);
  });
}

function bindControls() {
  $("play").onclick = () => { S.playing = !S.playing; syncPlay(); };
  $("scrub").oninput = (e) => { S.tok = +e.target.value; S.playing = false; syncPlay(); };
  $("sel-sent").onchange = (e) => setSentence(e.target.value);
  $("sel-layer").onchange = (e) => {
    S.layer = +e.target.value;
    S.nTok = nTokens(tracePack, S.sent, S.layer);
    S.ref = referenceLevel(tracePack, S.sent, S.layer, S.nTok);
  };
  $("sel-mode").onchange = (e) => { S.mode = e.target.value; refreshMode(); refreshView(); };
  $("sel-view").onchange = (e) => { S.view = e.target.value; refreshView(); };
  $("o-slider").oninput = (e) => { $("o-k").textContent = e.target.value; };
  $("o-slider").onchange = (e) => setOrthantK(+e.target.value);
  {
    const c = $("orthant");
    let last = null;
    c.addEventListener("pointerdown", (e) => {
      last = { x: e.clientX, y: e.clientY };
      c.setPointerCapture(e.pointerId);
      c.classList.add("grabbing");
    });
    c.addEventListener("pointermove", (e) => {
      if (!last) return;
      orth.drag(e.clientX - last.x, e.clientY - last.y);
      last = { x: e.clientX, y: e.clientY };
    });
    c.addEventListener("pointerup", () => { last = null; c.classList.remove("grabbing"); });
  }
  $("sel-layout").onchange = (e) => {
    // Switching layout means new positions for the same nodes; simplest correct thing is to
    // rebuild the field rather than mutate a static buffer behind the renderer's back.
    S.layout = e.target.value;
    const nxy = S.layout === "graph" ? headGraphXY(0) : positions(fieldPack, "global.xy");
    field.destroy();
    xy = nxy;
    field = createField(canvas, { xy, edges: true }, { surface: FIELD_SURFACE });
    refreshMode();
    refreshView();      // refreshMode resets the note to the colour-mode text; restore the view's
  };
  addEventListener("keydown", (e) => {
    // space / arrows drive the transport; H is in bindPanels and hides the chrome
    if (e.key === " ") { e.preventDefault(); S.playing = !S.playing; syncPlay(); }
    if (e.key === "ArrowRight") { S.tok = Math.min(S.nTok - 1, S.tok + 1); S.playing = false; syncPlay(); }
    if (e.key === "ArrowLeft") { S.tok = Math.max(0, S.tok - 1); S.playing = false; syncPlay(); }
  });
}

/** Head h's force-directed layout, expanded to the global index space. Isolated neurons -- 39%
 *  of them -- have no graph position, so they are parked off-canvas rather than drawn at (0,0),
 *  which would invent a dense cluster at the origin that does not exist. */
function headGraphXY(h) {
  const p = positions(fieldPack, `h${h}.xy_graph`);
  const placed = fieldPack.t[`h${h}.xy_graph_placed`];
  const out = new Float32Array(nNodes * 2).fill(NaN);
  for (let i = 0; i < placed.length; i++) {
    if (!placed[i]) continue;
    out[(h * N + i) * 2] = p[i * 2];
    out[(h * N + i) * 2 + 1] = p[i * 2 + 1];
  }
  return out;
}

function showFallback() {
  $("nogl").hidden = false;
  const dl = $("nogl-stats");
  const rows = [
    ["neurons", nNodes.toLocaleString("en-US")],
    ["active per token", `${(tracePack.manifest.measured_active_frac * 100).toFixed(2)}%`],
    ["synapses per head", fieldPack.manifest.heads[0].n_edges_total.toLocaleString("en-US")],
    ["modularity", fieldPack.manifest.heads[0].modularity.toFixed(3)],
    ["max out-degree", String(maxDeg)],
  ];
  dl.innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}

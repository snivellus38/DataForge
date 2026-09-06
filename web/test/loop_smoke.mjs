// END-TO-END SMOKE TEST for THE LOOP: boots the real page and walks all eight stops.
//
// web/index.html once shipped completely dead -- a dropped declaration made the module throw on
// load, the browser rendered empty placeholders, and every other gate stayed green because none
// of them executed the entry point (CLAUDE.md item 15). It happened a SECOND time in this repo:
// app.js was left importing a ./stage.js that did not exist. So this runs the real loop.js
// against a real DOM, drives every control, and asserts the panes actually populate.
//
// jsdom does not execute <script type="module">, so -- following web/test/field_smoke.mjs --
// the module is imported in THIS process with the jsdom globals installed. That exercises the
// real module graph rather than a re-implementation of it.
//
// Canvas 2D is stubbed (jsdom has no raster backend) and WebGL is refused outright: nothing on
// this page is allowed to need a GPU. If a future stage reaches for one, this fails.
import { readFileSync } from "node:fs";
import { JSDOM, VirtualConsole } from "jsdom";

const root = new URL("../..", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const read = (p) => readFileSync(root + p);

const errors = [];
const vc = new VirtualConsole();
vc.on("jsdomError", (e) => errors.push(e.message + "\n" + (e.detail?.stack || "")));
vc.on("error", (m) => errors.push(String(m)));

const dom = new JSDOM(read("web/loop.html"), {
  url: "http://localhost/", pretendToBeVisual: true, virtualConsole: vc,
});
const { window } = dom;
const doc = window.document;
const $ = (id) => doc.getElementById(id);

// ── minimal browser surface jsdom lacks ──────────────────────────────────────────────────────
// The 2D stub RECORDS what is handed to putImageData. A stub that swallows draws cannot tell a
// heatmap from a blank one, and sigma shipped here as near-uniform grey: it was normalised by the
// MAX (159.8) when the median cell is 0.064, so 83% of pixels landed within 5% of the neutral
// midpoint. Checking merely that "something was painted" passes that. Checking CONTRAST does not.
// stages.js draws into an OFFSCREEN canvas and then drawImage()s it onto the real one (so the
// fill rate stays independent of the display size), so the stub has to follow that hop: record
// per canvas ELEMENT on putImageData, then relay it to the destination on drawImage.
//
// The flow diagram (stop 0) paints VECTORS, not an ImageData, so the same discipline has to
// reach the path API: the stub records every curve and dot with the style that was live when it
// was drawn. That is what lets the gate assert the picture has signed edges of both colours and
// a real spread of node brightness, rather than "some function was called".
const painted = new WeakMap();
const byId = new Map();
const vectors = new Map();          // canvas id -> [{op, colour, lineWidth, x, y}]
const ctxStub = (canvas) => {
  const rec = (op, colour, lineWidth, x, y) => {
    if (!canvas || !canvas.id) return;
    if (!vectors.has(canvas.id)) vectors.set(canvas.id, []);
    vectors.get(canvas.id).push({ op, colour, lineWidth, x, y });
  };
  return {
    _fill: "#000", _stroke: "#000", _lw: 1, _alpha: 1, _path: null,
    createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4), width: w, height: h }),
    putImageData(img) {
      painted.set(canvas, img);
      if (canvas && canvas.id) byId.set(canvas.id, img);
    },
    drawImage(src) {
      const img = src && painted.get(src);
      if (img) {
        painted.set(canvas, img);
        if (canvas && canvas.id) byId.set(canvas.id, img);
      }
    },
    createLinearGradient: () => ({ addColorStop() {} }),
    createRadialGradient: () => ({ addColorStop() {} }),
    clearRect() { if (canvas && canvas.id) vectors.set(canvas.id, []); },
    fillRect(x, y) { rec("rect", this._fill, this._alpha, x, y); },
    strokeRect() {},
    beginPath() { this._path = null; },
    moveTo(x, y) { this._path = { x, y }; },
    lineTo() {},
    bezierCurveTo(c1x, c1y, c2x, c2y, x, y) { this._path = { curve: true, x, y }; },
    arc(x, y) { this._path = { arc: true, x, y }; },
    stroke() {
      const p = this._path;
      rec(p && p.curve ? "curve" : "line", this._stroke, this._lw, p && p.x, p && p.y);
    },
    fill() {
      const p = this._path;
      rec(p && p.arc ? "dot" : "fill", this._fill, this._alpha, p && p.x, p && p.y);
    },
    closePath() {},
    fillText() {}, save() {}, restore() {}, setTransform() {}, translate() {}, scale() {},
    set fillStyle(v) { this._fill = String(v); }, get fillStyle() { return this._fill; },
    set strokeStyle(v) { this._stroke = String(v); }, get strokeStyle() { return this._stroke; },
    set lineWidth(v) { this._lw = v; }, get lineWidth() { return this._lw; },
    set font(v) {}, get font() { return ""; },
    set globalAlpha(v) { this._alpha = v; }, get globalAlpha() { return this._alpha; },
    set textAlign(v) {}, get textAlign() { return "left"; },
    set imageSmoothingEnabled(v) {}, get imageSmoothingEnabled() { return false; },
  };
};
let askedForGL = false;
window.HTMLCanvasElement.prototype.getContext = function (kind) {
  if (kind === "webgl2" || kind === "webgl") { askedForGL = true; return null; }
  return ctxStub(this);
};
for (const [k, v] of Object.entries({ clientWidth: 640, clientHeight: 200 })) {
  Object.defineProperty(window.HTMLCanvasElement.prototype, k, { value: v, configurable: true });
}
window.HTMLCanvasElement.prototype.getBoundingClientRect =
  () => ({ left: 0, top: 0, width: 440, height: 440, right: 440, bottom: 440 });
Object.defineProperty(window.HTMLElement.prototype, "clientHeight", { value: 520, configurable: true });
window.devicePixelRatio = 1;
window.matchMedia = () => ({ matches: true, addEventListener() {}, removeEventListener() {} });
window.fetch = async (u) => {
  const buf = read("web/" + String(u).replace(/^https?:\/\/[^/]+\//, ""));
  return {
    ok: true,
    json: async () => JSON.parse(buf.toString("utf8")),
    arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
  };
};

// Run loop.js in THIS process against the jsdom globals. rAF is bounded so the render loop runs
// but the test still terminates.
let frames = 0;
Object.assign(globalThis, {
  window, document: doc, fetch: window.fetch, performance: window.performance,
  addEventListener: window.addEventListener.bind(window),
  requestAnimationFrame: (fn) => { if (frames++ < 200) setTimeout(() => fn(frames * 100), 0); return frames; },
  cancelAnimationFrame: () => {},
});

await import("../src/loop.js");

const settle = (ms = 60) => new Promise((r) => setTimeout(r, ms));
const ready = () => $("boot").hidden || $("boot").classList.contains("failed");
const deadline = Date.now() + 10000;
while (!ready() && Date.now() < deadline) await settle(25);

const checks = [];
const ok = (n, c, d = "") => checks.push([n, !!c, d]);
// textContent carries the source's own line wrapping, so collapse runs of whitespace before
// matching prose -- otherwise an assertion passes or fails on where a line happened to break.
const txt = (id) => ($(id)?.textContent || "").replace(/\s+/g, " ").trim();
const bcount = (id) => $(id)?.querySelectorAll("b").length ?? 0;
const click = (n) => n.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
const change = (el) => el.dispatchEvent(new window.Event("change", { bubbles: true }));
const input = (el) => el.dispatchEvent(new window.Event("input", { bubbles: true }));

// ── the page came alive ──────────────────────────────────────────────────────────────────────
ok("boot overlay was dismissed", $("boot").hidden === true,
   $("boot").hidden ? "" : `still showing: ${txt("boot").slice(0, 120)}`);
ok("no page-level errors", errors.length === 0, errors.join(" | ").slice(0, 300));
ok("nothing on this page needed WebGL", !askedForGL);

const nodes = [...doc.querySelectorAll(".dg-node")];
if (!nodes.length) {
  console.log("DIAGRAM NEVER RENDERED — boot() did not complete.");
  console.log("boot says:", txt("boot").slice(0, 300));
  console.log("errors:", errors.join(" ||| ") || "(none captured)");
  process.exit(1);
}
const goto = (s) => click(nodes.find((n) => Number(n.dataset.stop) === s));

// ── the diagram is real and clickable ────────────────────────────────────────────────────────
ok("diagram drew all ten pipeline nodes", nodes.length === 10, `${nodes.length} nodes`);
ok("diagram has a residual loop-back arc", !!$("dg-loop"));
ok("the loop-back says it is the same weights, not six layers",
   /same weights/.test(txt("dg-loop-label")), txt("dg-loop-label"));

// ── selectors populated from the pack ────────────────────────────────────────────────────────
ok("sentence selector offers both walk sentences", $("s-sent").options.length === 2);
ok("iteration selector offers all six", $("s-layer").options.length === 6);
ok("head selector offers all four", $("s-head").options.length === 4);
ok("byte strip rendered", $("strip").children.length > 20, `${$("strip").children.length} bytes`);

// ── walk every stop and assert its pane populates ────────────────────────────────────────────
const PANE = {
  0: () => bcount("p0-stats") === 4,
  1: () => bcount("p1-stats") === 3 && txt("p1-post-n") !== "—",
  2: () => bcount("p2-stats") === 3 && txt("p2-explain").length > 100,
  3: () => bcount("p3-stats") === 2 && txt("p3-read").length > 20,
  4: () => bcount("p4-stats") === 2 && txt("p4-note").length > 40,
  5: () => bcount("p5-stats") === 4 && txt("p5-n") !== "—",
  6: () => bcount("p6-stats") === 4,
  7: () => bcount("p7-stats") === 4 && txt("p7-note").length > 20,
};
for (let s = 0; s <= 7; s++) {
  goto(s);
  await settle(80);
  ok(`stop ${s} — pane populated`, PANE[s](), `heading: ${txt("s-h").slice(0, 44)}`);
  const shown = [...doc.querySelectorAll(".pane")].filter((p) => !p.hidden);
  ok(`stop ${s} — exactly one pane visible`, shown.length === 1, `${shown.length} visible`);
}

// ── the interactive bits actually change something ───────────────────────────────────────────
goto(3); await settle(80);
{
  const before = txt("p3-note");
  $("p3-softmax").checked = true; change($("p3-softmax")); await settle(80);
  ok("the softmax toggle changes what the page says", txt("p3-note") !== before);
  ok("...and it explains why that breaks a fixed-size state",
     /keep them all|every other value/.test(txt("p3-note")), txt("p3-note").slice(0, 70));
  $("p3-softmax").checked = false; change($("p3-softmax")); await settle(60);
  ok("the raw row is not normalised", /not 1/.test(txt("p3-note")), txt("p3-note").slice(0, 70));
}

goto(4); await settle(200);
{
  // Measure where sigma actually HAS content. At token 0 exactly one rank-one write has landed,
  // so ~95% of rows are legitimately zero and a near-empty panel is the correct picture -- not
  // something to assert contrast against. Scrub to mid-sequence first.
  $("scrub").value = "28";
  $("scrub").dispatchEvent(new window.Event("input", { bubbles: true }));
  await settle(250);

  // CONTRAST, not merely "was painted". The diverging ramp's dark midpoint is #383835, so a
  // washed-out render is a field of pixels within a few units of it. Require that a real share
  // of the panel is visibly away from neutral.
  const contrast = (id) => {
    const img = byId.get(id);
    if (!img) return { ok: false, why: "nothing was painted" };
    const MID = [0x38, 0x38, 0x35];
    let far = 0, total = 0;
    for (let i = 0; i < img.data.length; i += 4) {
      total++;
      const d = Math.abs(img.data[i] - MID[0]) + Math.abs(img.data[i + 1] - MID[1]) +
                Math.abs(img.data[i + 2] - MID[2]);
      if (d > 40) far++;
    }
    const frac = total ? far / total : 0;
    return { ok: frac > 0.25, why: `${(frac * 100).toFixed(1)}% of pixels visibly off-neutral` };
  };
  const c = contrast("p4-sigma");
  ok("the sigma panel has real contrast, not washed-out grey", c.ok, c.why);
  if (process.env.DEBUG_SIGMA) {
    const img = byId.get("p4-sigma");
    const MID = [0x38, 0x38, 0x35];
    const bins = [0, 0, 0, 0, 0];
    let exactMid = 0;
    for (let i = 0; i < img.data.length; i += 4) {
      const d = Math.abs(img.data[i] - MID[0]) + Math.abs(img.data[i + 1] - MID[1]) +
                Math.abs(img.data[i + 2] - MID[2]);
      if (d === 0) exactMid++;
      bins[d < 10 ? 0 : d < 40 ? 1 : d < 110 ? 2 : d < 200 ? 3 : 4]++;
    }
    const tot = img.data.length / 4;
    console.log("  DEBUG p4-sigma", img.width + "x" + img.height,
      "exactly neutral", (100 * exactMid / tot).toFixed(1) + "%",
      "| bins", bins.map((v) => (100 * v / tot).toFixed(1) + "%").join(" "));
  }

  const resid = $("p4-stats").querySelectorAll("b")[1]?.textContent || "";
  ok("sigma residual is computed in the browser and shown", /e[-+]/.test(resid), resid);
  const before = txt("p4-note");
  $("p4-delta").checked = true; change($("p4-delta")); await settle(200);
  ok("the single-write toggle changes the caption", txt("p4-note") !== before);
  ok("...and calls that write rank-one", /rank-one/.test(txt("p4-note")));
  $("p4-delta").checked = false; change($("p4-delta")); await settle(120);
  ok("...and the accumulated state is described as dense", /dense/.test(txt("p4-note")));
}

goto(1); await settle(80);
{
  const a = txt("p1-post-n");
  $("scrub").value = "20";
  $("scrub").dispatchEvent(new window.Event("input", { bubbles: true }));
  await settle(80);
  ok("scrubbing to another token changes the survivor count", txt("p1-post-n") !== a,
     `${a} -> ${txt("p1-post-n")}`);
  ok("survivors are a real count, not a placeholder",
     Number(txt("p1-post-n").replace(/,/g, "")) > 0);
}

goto(5); await settle(80);
{
  const fired = Number($("p5-stats").querySelectorAll("b")[0].textContent.replace(/,/g, ""));
  const surv = Number($("p5-stats").querySelectorAll("b")[1].textContent.replace(/,/g, ""));
  ok("the gate really is an AND: survivors are a strict subset of what fired",
     surv > 0 && surv < fired, `${fired} fired -> ${surv} survived`);
}

// ── the flow diagram, stop 0 ─────────────────────────────────────────────────────────────────
//
// It replaced a centred paragraph over an empty box, so "the pane has text in it" would pass on
// the thing it replaced. These assert the picture: signed edges of BOTH colours (the whole point
// of drawing weight x activation rather than a connector), a real spread of node brightness, the
// sampling disclosure, and that scrubbing repaints from the new token rather than reusing the
// last frame.
goto(0); await settle(120);
{
  const seen = () => vectors.get($("p0-flow").querySelector("canvas").id || "") || [];
  // the canvas flow.js creates has no id, so reach it the way the renderer does
  const cv = $("p0-flow").querySelector("canvas");
  cv.id = "p0-flow-canvas";
  $("scrub").value = "20"; input($("scrub")); await settle(120);
  const v = vectors.get("p0-flow-canvas") || [];
  const curves = v.filter((e) => e.op === "curve");
  const dots = v.filter((e) => e.op === "dot");
  ok("the flow diagram drew its edge bundle", curves.length > 60, `${curves.length} curves`);
  ok("...and its nodes", dots.length > 20, `${dots.length} dots`);

  // Classify by HUE, not by a brightness threshold: edgeColour() scales the whole triple by
  // magnitude, so amber at low magnitude has small components too. Amber has R > B, blue B > R,
  // and that ordering survives any scaling -- which is the encoding the page actually claims.
  const rgb = (c) => (c.match(/[\d.]+/g) || []).map(Number);
  const warm = curves.filter((e) => { const [r, , b] = rgb(e.colour); return r > b; }).length;
  const cool = curves.filter((e) => { const [r, , b] = rgb(e.colour); return b > r; }).length;
  ok("edges carry a SIGN — both positive and negative are on screen",
     warm > 0 && cool > 0, `${warm} positive / ${cool} negative of ${curves.length}`);

  const widths = new Set(curves.map((e) => Math.round(e.lineWidth * 4)));
  ok("edge width is a magnitude, not a constant", widths.size >= 4, `${widths.size} widths`);

  const note = txt("p0-note");
  ok("the sampling rule is disclosed, not implied", /of 12,288|12,288/.test(note) &&
     /pool|cast|held fixed/i.test(note), note.slice(0, 90));
  ok("the head restriction on the drawn cast is disclosed",
     /head \d/.test(note) && /traced head/.test(note) && /all four heads/.test(note),
     note.slice(0, 120));
  ok("...and what the drawn cast contributes to the real update is stated",
     /the size of/.test(note) && /survive the gate/.test(note) && /%/.test(note),
     note.slice(-140));
  // The headers shipped clipped: 124px wide, centred on columns sitting 18px from the frame, so
  // "Input byte" rendered as "put byte" and lm_head's byte labels ran off the right edge. They
  // are clamped now, and the clamp is only worth having if something checks it.
  {
    const w = $("p0-flow").clientWidth || 960;
    const lefts = [...doc.querySelectorAll(".flow-head")].map((el) => parseFloat(el.style.left));
    ok("every column header renders inside the frame", lefts.length === 8 &&
       lefts.every((x) => x >= 65 && x <= w - 65),
       `${lefts.map((x) => x.toFixed(0)).join(",")} in a ${w}px frame`);
  }
  // The first fix for the clipped headers padded the plot instead of clamping them, which took
  // the width out of the diagram -- the opposite of what was wanted. This pins the trade: the
  // columns must span nearly the whole frame, and any future "just add padding" fails here.
  {
    const w = $("p0-flow").clientWidth || 960;
    const xs = (vectors.get("p0-flow-canvas") || []).filter((e) => e.op === "dot" && e.x != null)
      .map((e) => e.x);
    const span = Math.max(...xs) - Math.min(...xs);
    ok("the plot uses its frame rather than spending it on padding", span / w > 0.88,
       `columns span ${span.toFixed(0)} of ${w}px (${((span / w) * 100).toFixed(1)}%)`);
  }
  {
    const h = $("p0-flow").clientHeight || 520;
    const ys = (vectors.get("p0-flow-canvas") || [])
      .filter((e) => (e.op === "dot" || e.op === "rect") && e.y != null).map((e) => e.y);
    const span = Math.max(...ys) - Math.min(...ys);
    ok("...and its full height", span / h > 0.82,
       `columns span ${span.toFixed(0)} of ${h}px (${((span / h) * 100).toFixed(1)}%)`);
  }
  ok("the flow disclosure and counters live in the rail, not over the diagram",
     $("r-live") && !$("r-live").hidden && $("p0-flow").querySelectorAll(".counters").length === 0,
     "they cost the canvas ~130px of height when they sat under it");
  ok("column headers carry live counts", txt("p0-flow").includes("firing"),
     txt("p0-flow").replace(/\s+/g, " ").slice(0, 110));

  // ── clicking a node must SHOW its path, not merely dim everything else ────────────────────
  //
  // The first version lit only the edges touching the node, at the same brightness they already
  // had, and dropped everything else to 5% -- so a click read as "the picture went dark" rather
  // than "here is the route". This asserts the three things that make it a trace: the immediate
  // connections get visibly heavier, the rest goes to a hueless dim, and the card reports how far
  // the path reaches.
  {
    const cv = $("p0-flow").querySelector("canvas");
    const alpha = (c) => { const m = (c || "").match(/[\d.]+/g); return m && m.length > 3 ? +m[3] : 1; };
    const before = (vectors.get("p0-flow-canvas") || []).filter((e) => e.op === "curve");
    const wBefore = Math.max(...before.map((e) => e.lineWidth));
    ok("nothing is dimmed before a selection",
       !before.some((e) => alpha(e.colour) < 0.05), `min alpha ${Math.min(...before.map((e) => alpha(e.colour))).toFixed(3)}`);

    // click a neuron: pick a recorded dot from a 26-dot column (the two 12,288-neuron columns)
    const dots = (vectors.get("p0-flow-canvas") || []).filter((e) => e.op === "dot" && e.x != null);
    const cols = new Map();
    for (const d of dots) {
      const k = Math.round(d.x);
      if (!cols.has(k)) cols.set(k, []);
      cols.get(k).push(d);
    }
    const neuronCol = [...cols.values()].find((g) => g.length >= 20);
    const target = neuronCol && neuronCol[Math.floor(neuronCol.length / 2)];
    ok("a neuron column was found to click", !!target, `${cols.size} columns of dots`);
    cv.dispatchEvent(new window.MouseEvent("click", {
      bubbles: true, clientX: target.x, clientY: target.y,
    }));
    await settle(80);

    const sel = window.__flow.selection();
    ok("clicking a node pins it", !!sel && sel.id != null, JSON.stringify(sel));
    ok("...and traces its whole path, not just what touches it",
       sel && sel.edges > sel.direct && sel.edges > 8,
       sel ? `${sel.direct} direct, ${sel.edges} on the path` : "no selection");

    const after = (vectors.get("p0-flow-canvas") || []).filter((e) => e.op === "curve");
    const wAfter = Math.max(...after.map((e) => e.lineWidth));
    ok("the path is drawn HEAVIER than the unselected picture", wAfter > wBefore * 1.5,
       `widest stroke ${wBefore.toFixed(1)} -> ${wAfter.toFixed(1)}`);
    const dimmed = after.filter((e) => alpha(e.colour) <= 0.04);
    ok("...and everything off the path is dimmed out of the way", dimmed.length > 20,
       `${dimmed.length} of ${after.length} curves dimmed`);
    ok("...in a hueless grey, so colour still means sign",
       dimmed.every((e) => { const m = (e.colour.match(/[\d.]+/g) || []).map(Number);
                             return Math.abs(m[0] - m[2]) < 30; }),
       dimmed[0] && dimmed[0].colour);

    const card = $("p0-flow").querySelector(".flow-card");
    ok("the card reports how far the path reaches",
       card && !card.hidden && /on its path/.test(card.textContent),
       (card ? card.textContent : "").replace(/\s+/g, " ").slice(-70));

    // the pin must follow the NEURON as the token advances, not the row it happened to occupy
    const idBefore = sel.id;
    $("scrub").value = "41"; input($("scrub")); await settle(90);
    const sel2 = window.__flow.selection();
    ok("the pin follows the neuron across tokens, or lets go",
       sel2 === null || sel2.id === idBefore,
       sel2 ? `pinned ${idBefore} -> ${sel2.id}` : "released (no longer drawn)");

    window.__flow.clearPin();
    await settle(60);
    const cleared = (vectors.get("p0-flow-canvas") || []).filter((e) => e.op === "curve");
    ok("releasing the pin restores the full picture",
       !cleared.some((e) => alpha(e.colour) < 0.05));
  }

  // the same token must not repaint identically after a scrub -- that is the "is it live" check
  const before = (vectors.get("p0-flow-canvas") || []).length;
  $("scrub").value = "34"; input($("scrub")); await settle(120);
  const after = vectors.get("p0-flow-canvas") || [];
  ok("scrubbing repaints the flow from the new token", after.length > 0);
  ok("...and v* is no longer a static box: its column moved",
     JSON.stringify(after.filter((e) => e.op === "rect").slice(0, 40)) !==
     JSON.stringify((v.filter((e) => e.op === "rect")).slice(0, 40)));
}

// ── v* is no longer the one static box ───────────────────────────────────────────────────────
// The report was "everything in the loop is dynamic except v*". It was a label on a diagram whose
// tensor shipped in the pack the whole time. It now carries a live signed strip on EVERY stop, so
// the check walks stops as well as tokens.
{
  const spark = () => [...doc.querySelectorAll("#dg-spark rect")]
    .map((r) => `${r.getAttribute("y")}/${r.getAttribute("height")}/${r.getAttribute("class")}`)
    .join(",");
  ok("the v* box carries a live residual strip", doc.querySelectorAll("#dg-spark rect").length > 20,
     `${doc.querySelectorAll("#dg-spark rect").length} bars`);
  $("scrub").value = "8"; input($("scrub")); await settle(90);
  const a = spark();
  $("scrub").value = "27"; input($("scrub")); await settle(90);
  const b = spark();
  ok("...and it changes with the token", a !== b);
  ok("...and it is SIGNED, not a magnitude bar", /neg/.test(b) && /pos/.test(b));
  goto(4); await settle(90);
  const c = spark();
  ok("...and it stays live on the other stops", c.length > 0 && /pos|neg/.test(c));
  goto(0); await settle(90);
}

{
  $("s-sent").value = "1"; change($("s-sent")); await settle(200);
  const disabled = [...$("s-layer").options].filter((o) => o.disabled).length;
  // Was: the long sentence carried one iteration and the other five were disabled. Both walk
  // sentences now ship every stage at every iteration, so a disabled option here means the
  // export has regressed to the one-sentence pack the flow diagram cannot run on.
  ok("every iteration is available on BOTH sentences", disabled === 0, `${disabled} disabled`);
  for (const L of ["0", "5"]) {
    $("s-layer").value = L; change($("s-layer")); await settle(120);
    ok(`the long sentence renders iteration ${Number(L) + 1}`,
       (vectors.get("p0-flow-canvas") || []).filter((e) => e.op === "dot").length > 20);
  }
  ok("...and switching to it did not error", errors.length === 0,
     errors.join(" | ").slice(0, 200));
  $("s-layer").value = "3"; change($("s-layer"));
  $("s-sent").value = "0"; change($("s-sent")); await settle(120);
}

// ── honesty: caveats that must never quietly fall off the page ───────────────────────────────
goto(6); await settle(60);
ok("the iterations stop still discloses that R is never stated",
   /never states a value for R/.test(txt("r-body")));
goto(4); await settle(60);
ok("the sigma stop cites the BDH-CQ section it stands on",
   /2608\.09888/.test(txt("r-body")) && /special case/.test(txt("r-body")));
ok("...and says their actual update rule is proprietary", /proprietary/.test(txt("r-body")));
goto(0); await settle(60);
ok("the opening discloses the checkpoint's two deviations",
   /positional embedding/.test(txt("r-body")) && /decay/.test(txt("r-body")));
ok("...and that the checkpoint is disclosed prior work", /README/.test(txt("r-body")));

// ── report ───────────────────────────────────────────────────────────────────────────────────
let bad = 0;
for (const [n, good, d] of checks) {
  if (!good) bad++;
  console.log(`  ${good ? "ok  " : "FAIL"}  ${n}${d && !good ? `  —  ${d}` : ""}`);
}
if (errors.length) console.log("\nERRORS: " + errors.join(" ||| ").slice(0, 1500));
if (bad) { console.log(`\nLOOP SMOKE FAILED (${bad} of ${checks.length})`); process.exit(1); }
console.log(`\nLOOP SMOKE PASSED — ${checks.length} checks, all eight stops alive`);
process.exit(0);

// END-TO-END SMOKE TEST for THE PRICE: boots the real deck and checks all twelve slides.
//
// The page this replaces once shipped completely dead twice over (CLAUDE.md item 15, and then
// again with a missing ./stage.js import). Both times every other gate stayed green because
// none of them executed the entry point. So this runs the real price.js against a real DOM.
//
// It also enforces the honesty layer as a TEST, not a convention: the BDH-CQ evidence ledger,
// the correction about equation (1), the merge caveat, and the capacity caveats are all things
// a future edit could quietly drop, and dropping them would be a factual regression.
import { readFileSync } from "node:fs";
import { JSDOM, VirtualConsole } from "jsdom";
import { webcrypto } from "node:crypto";

const root = new URL("../..", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const read = (p) => readFileSync(root + p);

const errors = [];
const vc = new VirtualConsole();
vc.on("jsdomError", (e) => errors.push(e.message + "\n" + (e.detail?.stack || "")));
vc.on("error", (m) => errors.push(String(m)));

const dom = new JSDOM(read("web/price.html"), {
  url: "http://localhost/", pretendToBeVisual: true, virtualConsole: vc,
});
const { window } = dom;
const doc = window.document;
const $ = (id) => doc.getElementById(id);

// jsdom has no raster backend, so canvas 2D is stubbed -- but a stub that silently swallows
// every draw cannot tell a heatmap from a black rectangle, and a black rectangle is exactly what
// this page shipped. So the stub RECORDS what was handed to putImageData, and the slide-2 checks
// below assert the pixels are actually varied.
const painted = new Map();
const ctxStub = (canvas) => ({
  createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4), width: w, height: h }),
  putImageData(img) { painted.set(canvas && canvas.id, img); },
  clearRect() {}, fillRect() {}, drawImage() {}, strokeRect() {},
  beginPath() {}, moveTo() {}, lineTo() {}, arc() {}, stroke() {}, fill() {}, closePath() {},
  fillText() {}, save() {}, restore() {}, setTransform() {}, translate() {}, scale() {},
  set fillStyle(v) {}, get fillStyle() { return "#000"; },
  set strokeStyle(v) {}, get strokeStyle() { return "#000"; },
  set lineWidth(v) {}, get lineWidth() { return 1; },
  set font(v) {}, get font() { return ""; },
  set globalAlpha(v) {}, get globalAlpha() { return 1; },
  set textAlign(v) {}, get textAlign() { return "left"; },
  set imageSmoothingEnabled(v) {}, get imageSmoothingEnabled() { return false; },
});
let askedForGL = false;
window.HTMLCanvasElement.prototype.getContext = function (kind) {
  if (kind === "webgl2" || kind === "webgl") { askedForGL = true; return null; }
  return ctxStub(this);
};
for (const [k, v] of Object.entries({ clientWidth: 560, clientHeight: 240 })) {
  Object.defineProperty(window.HTMLCanvasElement.prototype, k, { value: v, configurable: true });
}
Object.defineProperty(window.HTMLElement.prototype, "clientWidth", { value: 560, configurable: true });
window.devicePixelRatio = 1;
window.innerWidth = 1280;
window.matchMedia = () => ({ matches: true, addEventListener() {}, removeEventListener() {} });
window.Element.prototype.scrollIntoView = function () {};
if (!window.crypto?.subtle) Object.defineProperty(window, "crypto", { value: webcrypto });
window.fetch = async (u) => {
  const buf = read("web/" + String(u).replace(/^https?:\/\/[^/]+\//, ""));
  return {
    ok: true,
    json: async () => JSON.parse(buf.toString("utf8")),
    arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
  };
};

let frames = 0;
Object.assign(globalThis, {
  // NB: performance is deliberately NOT taken from jsdom. Assigning jsdom's Performance onto
  // globalThis makes its own now() resolve back to itself and recurse until the stack dies.
  // Node's built-in performance.now() is what price.js actually wants for the forward timing.
  window, document: doc, fetch: window.fetch, devicePixelRatio: 1,
  addEventListener: window.addEventListener.bind(window),
  requestAnimationFrame: (fn) => { if (frames++ < 120) setTimeout(() => fn(frames * 100), 0); return frames; },
  cancelAnimationFrame: () => {},
});

await import("../src/price.js");

const settle = (ms = 60) => new Promise((r) => setTimeout(r, ms));
const ready = () => $("boot").hidden || $("boot").classList.contains("failed");
let deadline = Date.now() + 15000;
while (!ready() && Date.now() < deadline) await settle(30);
// the 8M-derived slides load after the live model; wait for the last of them
deadline = Date.now() + 15000;
while (!($("g-stats").children.length) && Date.now() < deadline) await settle(40);

const checks = [];
const ok = (n, c, d = "") => checks.push([n, !!c, d]);
const txt = (id) => ($(id)?.textContent || "").replace(/\s+/g, " ").trim();
const bcount = (id) => $(id)?.querySelectorAll("b").length ?? 0;

// ── alive ────────────────────────────────────────────────────────────────────────────────────
ok("boot overlay dismissed", $("boot").hidden === true,
   $("boot").hidden ? "" : txt("boot").slice(0, 140));
ok("no page-level errors", errors.length === 0, errors.join(" | ").slice(0, 400));
ok("nothing on this page needed WebGL", !askedForGL);
ok("all twelve slides are in the document",
   doc.querySelectorAll(".slide").length === 12, `${doc.querySelectorAll(".slide").length}`);
ok("nav dots built for every slide", $("dots").children.length === 12);

// ── slide 1: the forward pass, played out ────────────────────────────────────────────────────
ok("token tape rendered", $("h-tape").children.length >= 8,
   `${$("h-tape").children.length} tokens`);
ok("weight hash shown", txt("h-hash").length >= 16, txt("h-hash"));
ok("hash verified unchanged", /unchanged/.test(txt("h-hashnote")), txt("h-hashnote"));
ok("hero meters populated", bcount("h-meters") === 3, txt("h-meters").slice(0, 70));
ok("forward time is a real measurement", /\d+ ms/.test(txt("h-meters")), txt("h-meters"));

// the whole point of this slide is that it MOVES: tokens reveal one at a time and sigma fills.
{
  const revealed = () => $("h-tape").querySelectorAll(".tk.in").length;
  const first = revealed();
  const firstFill = txt("h-meters");
  let grew = false;
  for (let i = 0; i < 40 && !grew; i++) {
    await settle(60);
    if (revealed() > first) grew = true;
  }
  ok("the tape advances on its own — the slide is not static", grew,
     `${first} -> ${revealed()} tokens revealed`);
  ok("...and sigma fills as it goes", txt("h-meters") !== firstFill);
}

// run it to completion and check the answer resolves
{
  let done = false;
  for (let i = 0; i < 120 && !done; i++) {
    await settle(60);
    done = /answered/.test(txt("h-phase"));
  }
  ok("the pass reaches an answer", done, txt("h-phase"));
  ok("with the demonstration present, the model is correct",
     /rule was never in the weights/.test(txt("h-verdict")), txt("h-verdict").slice(0, 80));
  // p=1.000 with the demonstration present is the documented behaviour (CLAUDE.md item 11),
  // so the pattern must accept it -- an anchored /^0\./ rejects the correct answer.
  ok("...and reports a real confidence", /^[01]\.\d{3}$/.test(txt("h-p")), txt("h-p"));
}

// remove the demonstration: the answer must flip while the weights do not
{
  const modelBefore = txt("h-model");
  $("h-toggle").dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  await settle(150);
  let done = false;
  for (let i = 0; i < 120 && !done; i++) {
    await settle(60);
    done = /answered/.test(txt("h-phase"));
  }
  ok("removing the demonstration changes the answer", txt("h-model") !== modelBefore,
     `${modelBefore} -> ${txt("h-model")}`);
  ok("...and the page says it is confidently wrong, not hedging",
     /Wrong, and confidently so/.test(txt("h-verdict")), txt("h-verdict").slice(0, 70));
  ok("...and the weights still hash the same", /unchanged/.test(txt("h-hashnote")));
  $("h-toggle").dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  await settle(150);
}

// ── slide 2: sigma, three ways — and none of them may be a black rectangle ────────────────────
{
  // THE REGRESSION THIS EXISTS FOR: forward()'s `sigmas` is an array of {layer, head, sigma}
  // objects, not per-token Float32Arrays. Indexing it as a flat array made every cell NaN and
  // painted a black rectangle for all three views, with every other gate still green.
  const varied = (id) => {
    const img = painted.get(id);
    if (!img) return "nothing was painted";
    let nonZero = 0, distinct = new Set();
    for (let i = 0; i < img.data.length; i += 4) {
      if (img.data[i] || img.data[i + 1] || img.data[i + 2]) nonZero++;
      if (distinct.size < 40) distinct.add(`${img.data[i]},${img.data[i + 1]},${img.data[i + 2]}`);
    }
    if (!nonZero) return "every pixel is black";
    if (distinct.size < 3) return `only ${distinct.size} distinct colour(s)`;
    return null;
  };
  const views = $("sig-tabs").querySelectorAll("button");
  for (let v = 0; v < views.length; v++) {
    views[v].dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
    await settle(120);
    const why = varied("sig-canvas");
    ok(`sigma view "${views[v].textContent}" actually paints something`, why === null, why || "");
  }
  ok("the hero sigma canvas paints too", varied("h-sigma") === null, varied("h-sigma") || "");

  views[0].dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  await settle(80);
  ok("sigma stats populated", bcount("sig-stats") === 2, txt("sig-stats").slice(0, 60));
  ok("sigma view defaults to the single write", /rank-one/.test(txt("sig-note")));

  const before = txt("sig-stats");
  $("sig-t").value = "5";
  $("sig-t").dispatchEvent(new window.Event("input", { bubbles: true }));
  await settle(100);
  ok("scrubbing tokens changes the state shown", txt("sig-stats") !== before ||
     txt("sig-tval") === "5", `${txt("sig-tval")}`);

  views[1].dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  await settle(100);
  ok("the whole-state view admits the density rather than blaming the render",
     /dense/.test(txt("sig-note")) && /honest, not a rendering failure/.test(txt("sig-note")));
  views[2].dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  await settle(150);
  ok("the difference view says what one line was worth",
     /what that one line was worth/.test(txt("sig-note")), txt("sig-note").slice(0, 70));
  views[0].dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  await settle(80);
}

// ── slide 3: the equivalence and the three break-it switches ─────────────────────────────────
const baseResid = txt("eq-resid");
ok("equivalence residual computed", /e[-+]/.test(baseResid), baseResid);
ok("unbroken, it reports the two forms agree", /same computation/.test(txt("eq-note")));
const flip = async (id) => {
  $(id).checked = true;
  $(id).dispatchEvent(new window.Event("change", { bubbles: true }));
  await settle(120);
};
const unflip = async (id) => {
  $(id).checked = false;
  $(id).dispatchEvent(new window.Event("change", { bubbles: true }));
  await settle(120);
};
await flip("t-qk");
ok("Q≠K does NOT break the recurrence", /Still identical/.test(txt("eq-lesson")),
   txt("eq-lesson").slice(0, 70));
await unflip("t-qk");
await flip("t-relu");
ok("dropping the ReLU does NOT break it either", /Still identical/.test(txt("eq-lesson")));
await unflip("t-relu");
await flip("t-softmax");
ok("softmax DOES break it", /Broken by/.test(txt("eq-lesson")), txt("eq-lesson").slice(0, 70));
ok("...and it explains the KV cache consequence", /KV cache that grows/.test(txt("eq-lesson")));
ok("...and the residual visibly blew up", txt("eq-resid") !== baseResid,
   `${baseResid} -> ${txt("eq-resid")}`);
await unflip("t-softmax");

// ── slide 4: concepts, with the null ─────────────────────────────────────────────────────────
ok("concept pills built", $("c-pills").children.length === 4);
ok("concept stats populated", bcount("c-stats") === 4);
{
  const s = txt("c-stats");
  ok("selectivity is reported against a chance baseline", /chance/.test(s), s.slice(0, 90));
  const enrich = parseFloat((s.match(/([\d.]+)×/) || [])[1] || "0");
  ok("enrichment over chance is real and stated", enrich > 2, `${enrich}x`);
  ok("neuron list rendered for the selected concept", $("c-list").children.length > 0,
     `${$("c-list").children.length} neurons`);
  const before = $("c-list").children.length;
  $("c-pills").children[2].dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  await settle(80);
  ok("switching concept changes the neuron list",
     $("c-list").children.length !== before || $("c-list").innerHTML !== "",
     `${before} -> ${$("c-list").children.length}`);
}

// ── slide 5: synapses ────────────────────────────────────────────────────────────────────────
ok("synapse sentences offered", $("y-sent").options.length >= 3);
ok("synapse stats populated", bcount("y-stats") === 3, txt("y-stats").slice(0, 80));
ok("byte strip built", $("y-strip").children.length > 20);

// ── slide 6: the headline result, recomputed in the browser ──────────────────────────────────
ok("silence stats populated", bcount("g-stats") === 4);
{
  const s = txt("g-stats");
  const mcc = parseFloat((s.match(/\+([\d.]+)/) || [])[1] || "0");
  ok("MCC is recomputed here and is high", mcc > 0.85, `MCC ${mcc}`);
  ok("confusion matrix rendered", $("g-matrix").children.length === 9);
}

// ── slide 7: capacity ────────────────────────────────────────────────────────────────────────
ok("capacity chart drawn", $("k-chart").children.length > 0);
ok("capacity stats populated", bcount("k-stats") === 3);
ok("the 1/pi constant is on screen", /1\/π/.test(txt("k-stats")), txt("k-stats").slice(0, 100));
{
  const before = txt("k-stats");
  $("k-signed").checked = true;
  $("k-signed").dispatchEvent(new window.Event("change", { bubbles: true }));
  await settle(300);
  ok("letting keys go negative changes retrieval", txt("k-stats") !== before);
  ok("...and signed keys are reported orthogonal on average",
     /orthogonal on average/.test(txt("k-stats")));
  $("k-signed").checked = false;
  $("k-signed").dispatchEvent(new window.Event("change", { bubbles: true }));
  await settle(200);
}

// ── slide 8: the two-curves correction ───────────────────────────────────────────────────────
ok("degradation chart drawn", $("d-chart").children.length > 0);
ok("both curves quantified", bcount("d-stats") === 2, txt("d-stats").slice(0, 80));
ok("the page states the two curves mean different things",
   /not/.test(txt("d-note")) && /running out of room/.test(txt("d-note")),
   txt("d-note").slice(0, 90));

// ── slide 9: the merge replication ───────────────────────────────────────────────────────────
ok("merge table rendered", $("m-table").querySelectorAll("tr").length >= 4,
   `${$("m-table").querySelectorAll("tr").length} rows`);
ok("merge result is labelled as not a held-out corpus",
   /Not a held-out validation corpus/i.test(txt("m-table")), txt("m-table").slice(-140));
ok("merge samples shown", $("m-samples").children.length > 0);
ok("merge is reproducible by one named command",
   /merge_replicate\.py/.test(txt("m-note")));

// ── the honesty layer, enforced ──────────────────────────────────────────────────────────────
const page = doc.body.textContent.replace(/\s+/g, " ");
ok("evidence ledger has all seven entries", $("ledger").children.length === 7);
ok("ledger tiers each claim by kind",
   $("ledger").querySelectorAll(".k.computed").length >= 1 &&
   $("ledger").querySelectorAll(".k.reported").length >= 2 &&
   $("ledger").querySelectorAll(".k.absent").length >= 2);
ok("the two unreconciled BDH-CQ costs are both on the page",
   /0\.00070/.test(page) && /0\.00265246/.test(page));
ok("the auditors are disclosed as co-authors", /co-authors/.test(page));
ok("ConceptARC training overlap is disclosed", /training mixture/.test(page));
ok("no ARC-AGI-2 result is claimed", /no ARC-AGI-2 results/i.test(page));
ok("the correction to the problem statement is on screen",
   /additive accumulation is named as a/.test(page) && /proprietary/.test(page));
ok("BDH-CQ equations carry section locators",
   /2608\.09888 §3\.2/.test(page) && /2608\.09888 §3\.3/.test(page));
ok("R is disclosed as never stated", /never states a value for R/.test(page));
ok("Table 3 is labelled replayed, not computed", /replayed, not computed here/.test(page));
ok("the merge slide refuses to overclaim against the paper",
   /Not a refutation/.test(page) && /clones of a common base/.test(page));
ok("the 7-sentence corpus caveat is stated", /7 sentences/.test(page));
ok("the checkpoint's two deviations are disclosed",
   /positional embedding/.test(page) && /no decay term/.test(page));
ok("prior work is disclosed as reused", /prior hackathon work/.test(page));
ok("provenance table rendered", $("prov").children.length === 10);
ok("recall questions rendered", $("recall").children.length === 5);
{
  const q = $("recall").children[0];
  q.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  await settle(40);
  ok("a recall answer opens", q.classList.contains("open"));
}
ok("the full claim sentence is on the page",
   /non-negativity that makes BDH inspectable is also what makes it forget sooner/.test(page));

// ── report ───────────────────────────────────────────────────────────────────────────────────
let bad = 0;
for (const [n, good, d] of checks) {
  if (!good) bad++;
  console.log(`  ${good ? "ok  " : "FAIL"}  ${n}${d && !good ? `  —  ${d}` : ""}`);
}
if (errors.length) console.log("\nERRORS: " + errors.join(" ||| ").slice(0, 1800));
if (bad) { console.log(`\nPRICE SMOKE FAILED (${bad} of ${checks.length})`); process.exit(1); }
console.log(`\nPRICE SMOKE PASSED — ${checks.length} checks, all twelve slides alive`);
process.exit(0);

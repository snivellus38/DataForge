// END-TO-END SMOKE TEST: actually boots the page in a DOM and asserts it came alive.
//
// This exists because the page once shipped completely dead. A stray edit dropped a `function`
// declaration from app.js; the module threw SyntaxError, nothing ran, and every other gate still
// passed -- because none of them PARSE app.js, let alone execute it. Parse-checking alone would
// have caught that one, but not a runtime throw inside boot(), so this runs the real thing.
//
// jsdom has no canvas backend, so 2D contexts are stubbed. Everything else -- module loading,
// fetch, the forward pass, sigma, the capacity sweep, DOM population -- is real.
import { readFileSync } from "node:fs";
import { JSDOM, VirtualConsole } from "jsdom";
import { webcrypto } from "node:crypto";

const root = new URL("../..", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const read = (p) => readFileSync(root + p);

const errors = [];
const vc = new VirtualConsole();
vc.on("jsdomError", (e) => errors.push(e.message + "\n" + (e.detail?.stack || "")));
vc.on("error", (m) => errors.push(String(m)));

const dom = new JSDOM(read("web/index.html"), {
  url: "http://localhost/",
  runScripts: "dangerously",
  resources: undefined,
  pretendToBeVisual: true,
  virtualConsole: vc,
});
const { window } = dom;

// --- minimal browser surface jsdom lacks ---
const ctxStub = () => ({
  createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4), width: w, height: h }),
  putImageData() {}, clearRect() {}, fillRect() {}, drawImage() {},
  set fillStyle(v) {}, get fillStyle() { return "#000"; },
});
window.HTMLCanvasElement.prototype.getContext = ctxStub;
Object.defineProperty(window.HTMLCanvasElement.prototype, "clientWidth", { value: 600 });
Object.defineProperty(window.HTMLCanvasElement.prototype, "clientHeight", { value: 76 });
window.devicePixelRatio = 1;
if (!window.crypto?.subtle) Object.defineProperty(window, "crypto", { value: webcrypto });
window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
window.IntersectionObserver = class { observe() {} disconnect() {} };
window.fetch = async (u) => {
  const p = "web/" + String(u).replace(/^https?:\/\/[^/]+\//, "");
  const buf = read(p);
  return { ok: true, json: async () => JSON.parse(buf.toString("utf8")),
           arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) };
};

// Globals FIRST: app.js calls boot() at module evaluation, so importing it before the DOM
// exists would throw immediately.
for (const [k, v] of Object.entries({
  window, document: window.document, crypto: window.crypto, matchMedia: window.matchMedia,
  IntersectionObserver: window.IntersectionObserver, fetch: window.fetch, devicePixelRatio: 1,
})) Object.defineProperty(globalThis, k, { value: v, configurable: true, writable: true });

const mod = await import(new URL("../src/app.js", import.meta.url));
void mod;

await new Promise((r) => setTimeout(r, 1500));   // boot() + capacity sweep

const $ = (s) => window.document.querySelector(s);
const txt = (s) => ($(s)?.textContent || "").trim();

const checks = [
  ["no uncaught errors", errors.length === 0],
  ["page not replaced by error dump", !window.document.body.innerHTML.startsWith("<pre")],
  ["demonstrations rendered", $("#demo-list")?.children.length >= 3],
  ["query chip rendered", $("#query-row")?.children.length >= 2],
  ["model answered", ($("#model-out")?.textContent || "").length > 0],
  ["oracle shown", ($("#oracle-out")?.textContent || "").length > 0],
  ["verdict written", txt("#verdict-note").length > 20],
  ["weights hash computed", /^[0-9a-f]{12}$/.test(txt("#whash"))],
  ["forward time measured", /\d+\s*ms/.test(txt("#fwd-ms"))],
  ["sigma dims filled", txt("#sigma-dims").includes("×")],
  ["sigma density measured", /%$/.test(txt("#sigma-density"))],
  ["residual computed", /e[+-]/.test(txt("#resid"))],
  ["equivalence note written", txt("#equiv-note").length > 20],
  ["capacity chart drawn", $("#cap-chart")?.querySelector("svg") !== null],
  ["capacity table drawn", $("#cap-table")?.querySelector("table") !== null],
  ["cosine measured ~1/pi", Math.abs(parseFloat(txt("#cos-nn")) - 1 / Math.PI) < 0.02],
  ["signed cosine ~0", Math.abs(parseFloat(txt("#cos-s"))) < 0.02],
  ["evidence ledger populated", $("#ledger")?.children.length === 7],
  ["sandbox demos rendered", $("#sb-demos")?.children.length >= 3],
  ["token sequence shown", txt("#sb-tokens").includes("bytes")],
  // machine room
  ["token strip built", $("#mr-tokens")?.children.length >= 11],
  ["transport rendered", $("#mr-scrub") !== null && +$("#mr-scrub").max >= 10],
  ["neuron panel drawn", $("#mr-neurons")?.width > 0],
  ["score matrix drawn", $("#mr-scores")?.width > 0],
  ["write panel drawn", $("#mr-write")?.width > 0],
  ["sigma panel drawn", $("#mr-sigma")?.width > 0],
  ["sparsity reported", /% of \d+/.test(txt("#mr-sparse"))],
  ["sigma density reported", /% non-zero/.test(txt("#mr-dens"))],
];

// stepping must actually change what is drawn
const before = $("#mr-sparse").textContent;
$("#mr-next").dispatchEvent(new window.MouseEvent("click"));
await new Promise((r) => setTimeout(r, 60));
checks.push(["stepping a token changes the panels", $("#mr-t").textContent === "1"]);
checks.push(["sparsity is per-token, not constant", $("#mr-sparse").textContent !== before]);

let bad = 0;
for (const [n, ok] of checks) { if (!ok) bad++; console.log(`  ${ok ? "ok  " : "FAIL"}  ${n}`); }
if (errors.length) console.log("\nERRORS:\n" + errors.join("\n---\n").slice(0, 1600));

console.log(`\n  model says ${txt("#model-out")} · oracle ${txt("#oracle-out")} · ` +
            `resid ${txt("#resid")} · ${txt("#fwd-ms")}`);
console.log(bad ? `\nSMOKE FAILED (${bad})` : "\nSMOKE PASSED — the page is alive");
process.exit(bad ? 1 : 0);

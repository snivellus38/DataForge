// Structural checks that would otherwise only surface by opening a browser:
//  - every #id app.js reaches for exists in index.html
//  - every id in index.html is actually used (dead markup)
//  - the layers override really changes computation
import { readFileSync } from "node:fs";
import { BDHModel } from "../src/bdh.js";

const html = readFileSync("web/index.html", "utf8");
const js = readFileSync("web/src/app.js", "utf8");

const inHtml = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]));
const wanted = new Set([...js.matchAll(/\$\("#([A-Za-z0-9_-]+)"\)/g)].map((m) => m[1]));

const missing = [...wanted].filter((i) => !inHtml.has(i));
const unused = [...inHtml].filter((i) => !wanted.has(i) && !js.includes(`"#${i}`));

console.log(`ids in html: ${inHtml.size}   referenced by app.js: ${wanted.size}`);
if (missing.length) console.log("  MISSING in html: " + missing.join(", "));
if (unused.length) console.log("  note, not referenced by id: " + unused.join(", "));

// selectors used besides ids
const classSel = [...js.matchAll(/\$\("\.([A-Za-z0-9_-]+)"\)/g)].map((m) => m[1]);
const badClass = classSel.filter((c) => !html.includes(`class="${c}`) && !html.includes(` ${c}"`) && !html.includes(`${c} `));
if (badClass.length) console.log("  MISSING class targets: " + badClass.join(", "));

// layers override must actually change the computation
const man = JSON.parse(readFileSync("web/public/model.json", "utf8"));
const bin = readFileSync("web/public/model.bin");
const m = new BDHModel(man, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
const toks = JSON.parse(readFileSync("web/public/presets.json", "utf8")).hook.bytes;
const base = m.forward(toks, { keepSigma: false });
const l2 = m.forward(toks, { keepSigma: false, layers: 2 });
const l4 = m.forward(toks, { keepSigma: false, layers: 4 });
let dSame = 0, dDiff = 0;
for (let i = 0; i < base.logits.length; i++) {
  dSame = Math.max(dSame, Math.abs(base.logits[i] - l4.logits[i]));
  dDiff = Math.max(dDiff, Math.abs(base.logits[i] - l2.logits[i]));
}

const checks = [
  ["no missing element ids", missing.length === 0],
  ["no missing class targets", badClass.length === 0],
  ["layers:4 == default (model trained at 4)", dSame === 0],
  ["layers:2 changes the output", dDiff > 1e-3],
];
let bad = 0;
console.log();
for (const [n, ok] of checks) { if (!ok) bad++; console.log(`  ${ok ? "ok  " : "FAIL"}  ${n}`); }
console.log(`\n  L=4 delta ${dSame.toExponential(1)}   L=2 delta ${dDiff.toExponential(1)}`);
console.log(bad ? `\nWIRING FAILED (${bad})` : "\nWIRING OK");
process.exit(bad ? 1 : 0);

// Exercises the page's exact sequence construction and oracle rule, headlessly.
// If the sixty-second moment stops working, this fails before anyone opens a browser.
import { readFileSync } from "node:fs";
import { BDHModel } from "../src/bdh.js";

const man = JSON.parse(readFileSync("web/public/model.json", "utf8"));
const bin = readFileSync("web/public/model.bin");
const model = new BDHModel(man, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
const P = JSON.parse(readFileSync("web/public/presets.json", "utf8")).hook;

const pairs = P.pairs.map(([s, t]) => ({ src: s.charCodeAt(0), tgt: t.charCodeAt(0) }));
const qi = pairs.findIndex((p) => p.src === P.query.charCodeAt(0));

const build = (removed) => {                      // identical to app.js buildTokens()
  const seq = [];
  pairs.forEach((p, i) => { if (!removed.has(i)) seq.push(p.src, p.tgt, man.tokens.SEP); });
  seq.push(man.tokens.QRY, pairs[qi].src);
  return seq;
};
const answer = (toks) => {
  const { logits } = model.forward(toks, { keepSigma: false });
  const V = model.V, o = (toks.length - 1) * V;
  let b = 0; for (let i = 1; i < V; i++) if (logits[o + i] > logits[o + b]) b = i;
  return b;
};

const full = build(new Set()), ablated = build(new Set([qi]));
const aFull = answer(full), aAbl = answer(ablated);
const truth = pairs[qi].tgt;

const qBytesSame = full.slice(-2).join() === ablated.slice(-2).join();
const checks = [
  ["query bytes identical in both", qBytesSame],
  ["with demonstration -> correct", aFull === truth],
  ["without demonstration -> different answer", aAbl !== truth],
  ["ablated answer still a valid target token", man.tokens.TGT.includes(aAbl)],
  ["removing a demo shortens the sequence by 3", full.length - ablated.length === 3],
];
let bad = 0;
for (const [name, ok] of checks) { if (!ok) bad++; console.log(`  ${ok ? "ok  " : "FAIL"}  ${name}`); }
console.log(`\n  truth=${truth}  full=${aFull}  ablated=${aAbl}`);
console.log(bad ? `\nAPP LOGIC FAILED (${bad})` : "\nAPP LOGIC PASSED — sixty-second moment holds");
process.exit(bad ? 1 : 0);

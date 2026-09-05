// The JS twin of research/verify_equivalence.py. Asserts the POSITIVE result and the NEGATIVE
// ones -- if someone "improves" the model and quietly breaks the sigma story, this fails loudly.
import { readFileSync } from "node:fs";
import { BDHModel, equivalenceCheck } from "../src/bdh.js";

const manifest = JSON.parse(readFileSync("web/public/model.json", "utf8"));
const bin = readFileSync("web/public/model.bin");
const model = new BDHModel(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
const tokens = JSON.parse(readFileSync("web/test/fixture.json", "utf8")).cases[1].idx[0];

const cases = [
  ["bdh.py as written",  {},                    "survive"],
  ["Q != K",             { decoupleQK: true },  "survive"],
  ["no ReLU",            { noRelu: true },      "survive"],
  ["softmax ON",         { softmax: true },     "break"],
];
let fail = 0;
console.log(`T=${tokens.length}  sigma=${model.N}x${model.D}\n`);
for (const [name, opts, expect] of cases) {
  const r = equivalenceCheck(model, tokens, opts);
  const survived = r.relative < 1e-5;
  const ok = (expect === "survive") === survived;
  if (!ok) fail++;
  console.log(`  ${name.padEnd(20)} rel=${r.relative.toExponential(2)}  ` +
              `${survived ? "equivalent" : "BREAKS"}  expected ${expect}  ${ok ? "ok" : "MISMATCH"}`);
}
const s = equivalenceCheck(model, tokens, {});
console.log(`\nsigma snapshots: ${s.snapshots.length} (one per token, for the write animation)`);
console.log(fail === 0 ? "\nEQUIVALENCE TESTS PASSED" : `\nFAILED (${fail})`);
process.exit(fail ? 1 : 0);

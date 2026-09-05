// PHASE 2 GATE: the JS port must reproduce PyTorch logits. Nothing visual gets built until
// this passes -- every panel in the artifact depends on the browser computing the real thing.
import { readFileSync } from "node:fs";
import { BDHModel } from "../src/bdh.js";

const manifest = JSON.parse(readFileSync("web/public/model.json", "utf8"));
const bin = readFileSync("web/public/model.bin");
const buf = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
const fx = JSON.parse(readFileSync("web/test/fixture.json", "utf8"));

const model = new BDHModel(manifest, buf);
console.log(`model: N=${model.N} D=${model.D} nh=${model.nh} L=${model.L} ` +
            `params=${manifest.params.toLocaleString()}`);

let worst = 0, fail = 0;
for (const c of fx.cases) {
  const B = c.idx.length, V = model.V;
  for (let b = 0; b < B; b++) {
    const t0 = performance.now();
    const { logits, T } = model.forward(c.idx[b]);
    const ms = performance.now() - t0;

    const ref = c.logits_answer.slice(b * V, (b + 1) * V);
    const got = logits.subarray(c.answer_pos * V, (c.answer_pos + 1) * V);
    let md = 0, scale = 0;
    for (let i = 0; i < V; i++) {
      md = Math.max(md, Math.abs(ref[i] - got[i]));
      scale = Math.max(scale, Math.abs(ref[i]));
    }
    const rel = md / scale;
    let am = 0; for (let i = 1; i < V; i++) if (got[i] > got[am]) am = i;
    const agree = am === c.argmax_answer[b];
    worst = Math.max(worst, rel);
    if (rel > 1e-4 || !agree) fail++;
    console.log(`  k=${String(c.k).padStart(2)} b=${b} T=${String(T).padStart(2)}  ` +
      `max|d|=${md.toExponential(2)}  rel=${rel.toExponential(2)}  ` +
      `argmax ${am}${agree ? " == " : " != "}${c.argmax_answer[b]}  ${ms.toFixed(1)}ms`);
  }
}

// sigma must reproduce the attention output it is supposed to be equivalent to
const c0 = fx.cases[0];
const { sigmas } = model.forward(c0.idx[0]);
console.log(`\nsigma: ${sigmas.length} states (L*nh), each ${model.N}x${model.D} = ` +
            `${(model.N * model.D).toLocaleString()} floats`);

console.log(`\nworst relative logit error: ${worst.toExponential(2)}  (tolerance 1e-4)`);
console.log(fail === 0 ? "GATE PASSED" : `GATE FAILED (${fail} case(s))`);
process.exit(fail === 0 ? 0 : 1);

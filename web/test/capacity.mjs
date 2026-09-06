// The JS capacity measurement must agree with research/sigma_capacity.py, which is what the
// README quotes. Reference values are the seeded default run: python research/sigma_capacity.py
import { capacityCurve, meanCosine, ANALYTIC_COSINE } from "../src/capacity.js";

const PY = {                       // research/sigma_capacity.py, seeded run, 2026-09-06
  nonneg: { 8: 1.00, 16: 0.99, 32: 0.94, 64: 0.75, 128: 0.48, 192: 0.32, 256: 0.23 },
  signed: { 8: 1.00, 16: 1.00, 32: 1.00, 64: 1.00, 128: 1.00, 192: 1.00, 256: 1.00 },
};
const KS = [8, 16, 32, 64, 128, 192, 256];

const cNN = meanCosine({ nonneg: true }), cS = meanCosine({ nonneg: false });
console.log(`mean pairwise cosine  non-negative ${cNN.toFixed(3)}  (analytic 1/pi = ${ANALYTIC_COSINE.toFixed(3)})`);
console.log(`mean pairwise cosine  signed       ${cS.toFixed(3)}\n`);

let bad = 0;
const t0 = performance.now();
for (const [name, ref] of Object.entries(PY)) {
  const got = capacityCurve(KS, { nonneg: name === "nonneg", trials: 6 });
  const line = got.map(({ k, acc }) => {
    const d = Math.abs(acc - ref[k]);
    if (d > 0.10) bad++;                       // 10pp tolerance: both sides are stochastic
    return `k${k}=${(acc * 100).toFixed(0)}%${d > 0.10 ? "!" : ""}`;
  }).join("  ");
  console.log(`  ${name.padEnd(7)} ${line}`);
  console.log(`  ${"py".padEnd(7)} ${KS.map((k) => `k${k}=${(ref[k] * 100).toFixed(0)}%`).join("  ")}\n`);
}
console.log(`sweep took ${(performance.now() - t0).toFixed(0)}ms (both curves, 6 trials each)`);

const checks = [
  ["non-negative cosine matches 1/pi", Math.abs(cNN - ANALYTIC_COSINE) < 0.02],
  ["signed cosine ~ 0", Math.abs(cS) < 0.02],
  ["all k within 10pp of PyTorch", bad === 0],
];
let fail = 0;
console.log();
for (const [n, ok] of checks) { if (!ok) fail++; console.log(`  ${ok ? "ok  " : "FAIL"}  ${n}`); }
console.log(fail ? `\nCAPACITY PORT FAILED (${fail})` : "\nCAPACITY PORT MATCHES PYTORCH");
process.exit(fail ? 1 : 0);

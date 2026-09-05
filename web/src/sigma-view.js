// Renderers for the synaptic state.
//
// MEASURED (web/test/inspect_sigma.mjs): sigma is DENSE -- 100% of 16,384 cells nonzero within a
// few tokens, because the keys are ~43% dense. A heatmap of raw sigma alone therefore reads as
// noise. What is legible is the *structure of a write* and the *change* between two states, so
// those are first-class views here, not afterthoughts.
import { divergingRGB, INK, mode } from "./palette.js";

/** Robust symmetric scale: p99 of |v|, so a few outliers don't wash the map out. */
export function scaleOf(arr) {
  const n = arr.length, step = Math.max(1, (n / 4096) | 0), s = [];
  for (let i = 0; i < n; i += step) s.push(Math.abs(arr[i]));
  s.sort((a, b) => a - b);
  return s[Math.floor(s.length * 0.99)] || 1;
}

/** Heatmap of an (N x D) matrix. Diverging, gray at zero. */
export function drawMatrix(canvas, data, N, D, { scale = null, cellW = 3, cellH = 2 } = {}) {
  const s = scale ?? scaleOf(data);
  canvas.width = D * cellW; canvas.height = N * cellH;
  const ctx = canvas.getContext("2d", { alpha: false });
  const img = ctx.createImageData(D * cellW, N * cellH);
  const m = mode();
  for (let n = 0; n < N; n++)
    for (let d = 0; d < D; d++) {
      const [r, g, b] = divergingRGB(data[n * D + d] / s, m);
      for (let y = 0; y < cellH; y++) {
        let o = (((n * cellH + y) * D * cellW) + d * cellW) * 4;
        for (let x = 0; x < cellW; x++, o += 4) {
          img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
        }
      }
    }
  ctx.putImageData(img, 0, 0);
  return s;
}

/** Difference of two states. This is what makes "one demonstration removed" visible. */
export function drawDiff(canvas, a, b, N, D, opts = {}) {
  const d = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) d[i] = a[i] - b[i];
  return drawMatrix(canvas, d, N, D, opts);
}

/** Row-energy profile ||sigma_n||: which neurons actually hold the binding. */
export function drawRowEnergy(canvas, data, N, D, { highlight = null } = {}) {
  const ink = INK[mode()];
  const w = canvas.width = canvas.clientWidth * devicePixelRatio;
  const h = canvas.height = canvas.clientHeight * devicePixelRatio;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  const e = new Float64Array(N);
  let mx = 0;
  for (let n = 0; n < N; n++) {
    let s = 0;
    for (let j = 0; j < D; j++) { const v = data[n * D + j]; s += v * v; }
    e[n] = Math.sqrt(s); if (e[n] > mx) mx = e[n];
  }
  ctx.fillStyle = ink.grid;
  ctx.fillRect(0, h - 1, w, 1);
  const bw = w / N;
  for (let n = 0; n < N; n++) {
    const bh = (e[n] / (mx || 1)) * (h - 2);
    ctx.fillStyle = highlight && highlight.has(n) ? "#eb6834" : ink.secondary;
    ctx.fillRect(n * bw, h - 1 - bh, Math.max(1, bw - devicePixelRatio), bh);
  }
  return { max: mx, energy: e };
}

// STAGE RENDERERS — one per stop on THE LOOP.
//
// Every function here takes a canvas (or an SVG host) plus decoded arrays and draws the model's
// real numbers. Nothing is animated that is not a real quantity changing; where a value is
// interpolated for legibility (the neuron strips fold 3072 values into fewer pixel columns),
// the fold is a MAX, not a mean, and the caption says so -- taking a mean of a 95%-zero vector
// draws a black bar and teaches that nothing is happening.
//
// Canvas 2D, not WebGL: these are strips and matrices of a few thousand cells, they redraw in
// well under a frame, and canvas degrades to a caption in jsdom instead of failing to acquire a
// context. THE FIELD earns WebGL because it draws 12,288 additive points; nothing here does.

import { divergingRGB, IGNITION, SIGMA_RAMP } from "./palette.js";

const hex2rgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

/** Sample a multi-stop ramp at u in [0,1]. */
export function ramp(stops, u) {
  const k = Math.max(0, Math.min(0.9999, u)) * (stops.length - 1);
  const i = Math.floor(k), f = k - i;
  const a = hex2rgb(stops[i]), b = hex2rgb(stops[Math.min(i + 1, stops.length - 1)]);
  return [a[0] + (b[0] - a[0]) * f | 0, a[1] + (b[1] - a[1]) * f | 0, a[2] + (b[2] - a[2]) * f | 0];
}

/** Size a canvas to its box at device pixel ratio. Returns null when there is no 2D context. */
export function fit(canvas, hpx) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth || 800;
  const h = hpx || canvas.clientHeight || 200;
  canvas.width = Math.max(1, Math.round(w * dpr));
  canvas.height = Math.max(1, Math.round(h * dpr));
  canvas.style.height = h + "px";
  const ctx = canvas.getContext("2d");
  return ctx ? { ctx, w: canvas.width, h: canvas.height, dpr } : null;
}

/**
 * A 1-D neuron strip: `n` values folded into the canvas width.
 *
 * Folding is by MAX because activations are ~95% zero -- averaging a sparse vector into pixel
 * columns hides exactly the thing being shown. `signed` switches to the diverging ramp so
 * x_pre's negative half (96% of it) is visible as the thing the ReLU is about to delete.
 */
export function drawStrip(canvas, values, { signed = false, stops = IGNITION, gamma = 0.55,
                                            height = 54, peak = null } = {}) {
  const f = fit(canvas, height);
  if (!f) return null;
  const { ctx, w, h } = f;
  const n = values.length;
  let hi = peak;
  if (hi == null) { hi = 0; for (let i = 0; i < n; i++) hi = Math.max(hi, Math.abs(values[i])); }
  hi = hi || 1;
  const img = ctx.createImageData(w, 1);
  for (let px = 0; px < w; px++) {
    const a = Math.floor(px * n / w), b = Math.max(a + 1, Math.floor((px + 1) * n / w));
    let v = 0;
    for (let i = a; i < b && i < n; i++) if (Math.abs(values[i]) > Math.abs(v)) v = values[i];
    let rgb;
    if (signed) rgb = divergingRGB(Math.sign(v) * Math.pow(Math.abs(v) / hi, gamma), "dark");
    else rgb = ramp(stops, Math.pow(Math.max(0, v) / hi, gamma));
    const o = px * 4;
    img.data[o] = rgb[0]; img.data[o + 1] = rgb[1]; img.data[o + 2] = rgb[2]; img.data[o + 3] = 255;
  }
  // one scanline, stretched -- keeps the fill rate trivial no matter how tall the strip is
  const tmp = document.createElement("canvas");
  tmp.width = w; tmp.height = 1;
  const tctx = tmp.getContext("2d");
  if (tctx) { tctx.putImageData(img, 0, 0); ctx.imageSmoothingEnabled = false; ctx.drawImage(tmp, 0, 0, w, h); }
  return { peak: hi };
}

/** Scatter the same strip as discrete marks, so individual survivors stay countable. */
export function drawSparseMarks(canvas, ids, vals, n, { height = 54, stops = IGNITION } = {}) {
  const f = fit(canvas, height);
  if (!f) return null;
  const { ctx, w, h } = f;
  ctx.clearRect(0, 0, w, h);
  let hi = 0;
  for (let i = 0; i < vals.length; i++) hi = Math.max(hi, vals[i]);
  hi = hi || 1;
  for (let i = 0; i < ids.length; i++) {
    const x = Math.floor(ids[i] * w / n);
    const rgb = ramp(stops, Math.pow(vals[i] / hi, 0.45));
    ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
    ctx.fillRect(x, 0, Math.max(1, Math.round(w / n) + 1), h);
  }
  return { peak: hi };
}

/**
 * The causal score matrix. Strictly lower-triangular: a token cannot attend to itself, so the
 * diagonal is structurally empty and is drawn as such rather than left ambiguous.
 */
export function drawScores(canvas, M, T, { size = 420, hover = null, softmaxRow = null } = {}) {
  const f = fit(canvas, size);
  if (!f) return null;
  const { ctx, w, h } = f;
  ctx.clearRect(0, 0, w, h);
  const cell = Math.min(w, h) / T;
  // Percentile, not max -- same reason as drawSigma. On the 105-byte sentence the median causal
  // score is 1.2% of the peak, so max-normalising leaves only ~36% of the matrix visible; p99
  // with clamping lifts the bulk into range and lets the few extreme cells saturate.
  const mags = [];
  for (let i = 0; i < M.length; i++) if (M[i] !== 0) mags.push(Math.abs(M[i]));
  mags.sort((a, b) => a - b);
  const hi = (mags.length ? mags[Math.floor(mags.length * 0.99)] : 0) || 1;
  for (let r = 0; r < T; r++) {
    for (let c = 0; c < r; c++) {
      const v = M[r * T + c];
      const rgb = softmaxRow && r === hover?.r
        ? ramp(SIGMA_RAMP, Math.pow(softmaxRow[c], 0.4))
        : divergingRGB(Math.sign(v) * Math.pow(Math.min(1, Math.abs(v) / hi), 0.5), "dark");
      ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
      ctx.fillRect(c * cell, r * cell, Math.ceil(cell), Math.ceil(cell));
    }
  }
  // the forbidden half, marked rather than merely absent
  ctx.strokeStyle = "rgba(255,255,255,.10)";
  ctx.lineWidth = Math.max(1, f.dpr);
  ctx.beginPath();
  ctx.moveTo(0, 0); ctx.lineTo(T * cell, T * cell);
  ctx.stroke();
  if (hover) {
    ctx.strokeStyle = "#e0872a";
    ctx.lineWidth = Math.max(1.5, f.dpr * 1.5);
    ctx.strokeRect(hover.c * cell, hover.r * cell, cell, cell);
    ctx.globalAlpha = 0.22;
    ctx.fillStyle = "#e0872a";
    ctx.fillRect(0, hover.r * cell, hover.r * cell, cell);
    ctx.globalAlpha = 1;
  }
  return { cell: cell / f.dpr, hi };
}

/**
 * RoPE, shown as what it actually is: a rotation of (neuron 2k, neuron 2k+1) pairs by an angle
 * that depends on the position. Two non-negative vectors always have a non-negative dot product;
 * rotate them by different angles and that guarantee is gone. That is the whole reason a
 * softmax-free BDH has anything resembling decay (CLAUDE.md item 4).
 */
export function drawRotation(canvas, pairs, { size = 300 } = {}) {
  const f = fit(canvas, size);
  if (!f) return null;
  const { ctx, w, h, dpr } = f;
  ctx.clearRect(0, 0, w, h);
  const cx = w / 2, cy = h / 2, R = Math.min(w, h) * 0.40;

  ctx.strokeStyle = "rgba(255,255,255,.10)";
  ctx.lineWidth = dpr;
  ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();
  // the non-negative quadrant: where every un-rotated key lives, by construction
  ctx.fillStyle = "rgba(224,135,42,.10)";
  ctx.beginPath(); ctx.moveTo(cx, cy); ctx.arc(cx, cy, R, -Math.PI / 2, 0); ctx.closePath(); ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,.18)";
  ctx.beginPath();
  ctx.moveTo(cx - R, cy); ctx.lineTo(cx + R, cy);
  ctx.moveTo(cx, cy - R); ctx.lineTo(cx, cy + R);
  ctx.stroke();

  for (const p of pairs) {
    const draw = (a, b, col, alpha, wide) => {
      const m = Math.hypot(a, b) || 1;
      const s = (Math.min(1, m / (p.norm || 1))) * R;
      ctx.strokeStyle = col; ctx.globalAlpha = alpha; ctx.lineWidth = wide * dpr;
      ctx.beginPath(); ctx.moveTo(cx, cy);
      ctx.lineTo(cx + (a / m) * s, cy - (b / m) * s);
      ctx.stroke();
    };
    draw(p.x0, p.y0, "#e0872a", 0.55, 1.2);
    draw(p.x1, p.y1, "#3987e5", 0.95, 1.8);
  }
  ctx.globalAlpha = 1;
  return true;
}

/**
 * sigma as a heat map.
 *
 * TWO THINGS MAKE THIS READABLE, AND WITHOUT EITHER IT RENDERS AS BLANK GREY:
 *
 * 1. SCALE BY A PERCENTILE, NOT THE MAX. Measured on the shipped pack at T=28: max |sigma| is
 *    159.8 while p99 is 3.27 and the MEDIAN is 0.064 -- so the median cell is 4e-4 of the max.
 *    Normalising by the max puts almost every pixel within 3% of the neutral midpoint, which is
 *    exactly the washed-out grey this shipped with. p99 with clamping above it keeps the outliers
 *    saturated (which is honest -- they really are off the scale) and gives the bulk a range.
 *
 * 2. MAX-POOL THE ROWS. There are 3072 neuron rows and only a few hundred pixel rows, so a point
 *    sample throws away ~90% of them and can miss a strong row entirely. Pooling by largest |v|
 *    keeps the sparse structure -- only 1,819 of 3,072 rows carry any energy at all, and 717 of
 *    them carry 90% of it, so the bright rows ARE the picture.
 */
export function drawSigma(canvas, sigma, N, D, { height = 300, gamma = 0.38 } = {}) {
  const f = fit(canvas, height);
  if (!f) return null;
  const { ctx, w, h } = f;
  const rows = Math.max(1, Math.min(N, h));

  // robust scale: p99 of |v| over a strided sample, so one outlier cannot flatten everything
  const sample = [];
  for (let i = 0; i < sigma.length; i += 13) {
    const v = Math.abs(sigma[i]);
    if (v > 0) sample.push(v);
  }
  sample.sort((x, y) => x - y);
  const hi = sample.length ? sample[Math.floor(sample.length * 0.99)] || sample[sample.length - 1] : 1;
  const scale = hi || 1;

  const img = ctx.createImageData(w, rows);
  for (let ry = 0; ry < rows; ry++) {
    const n0 = Math.floor(ry * N / rows);
    const n1 = Math.max(n0 + 1, Math.floor((ry + 1) * N / rows));
    for (let px = 0; px < w; px++) {
      const d0 = Math.floor(px * D / w);
      const d1 = Math.max(d0 + 1, Math.floor((px + 1) * D / w));
      // pool by largest magnitude, keeping its sign -- a mean over a 40%-zero matrix draws grey
      let best = 0;
      for (let n = n0; n < n1; n++) {
        const base = n * D;
        for (let d = d0; d < d1; d++) {
          const v = sigma[base + d];
          if (Math.abs(v) > Math.abs(best)) best = v;
        }
      }
      const u = Math.min(1, Math.abs(best) / scale);
      const rgb = divergingRGB(Math.sign(best) * Math.pow(u, gamma), "dark");
      const o = (ry * w + px) * 4;
      img.data[o] = rgb[0]; img.data[o + 1] = rgb[1]; img.data[o + 2] = rgb[2]; img.data[o + 3] = 255;
    }
  }
  const tmp = document.createElement("canvas");
  tmp.width = w; tmp.height = rows;
  const tctx = tmp.getContext("2d");
  if (tctx) {
    tctx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, w, h);
  }
  return { peak: scale, max: sample.length ? sample[sample.length - 1] : 0, rows };
}

/** A simple bar chart on canvas: next-byte probabilities, gate survival, jaccard, whatever. */
export function drawBars(canvas, items, { height = 180, highlight = -1, colour = "#e0872a" } = {}) {
  const f = fit(canvas, height);
  if (!f) return null;
  const { ctx, w, h, dpr } = f;
  ctx.clearRect(0, 0, w, h);
  const n = items.length;
  if (!n) return null;
  const bw = w / n, pad = Math.max(2, bw * 0.18);
  let hi = 0;
  for (const it of items) hi = Math.max(hi, it.v);
  hi = hi || 1;
  ctx.font = `${11 * dpr}px ui-monospace, monospace`;
  ctx.textAlign = "center";
  for (let i = 0; i < n; i++) {
    const bh = (items[i].v / hi) * (h - 26 * dpr);
    ctx.fillStyle = i === highlight ? "#ffe4b0" : colour;
    ctx.globalAlpha = i === highlight ? 1 : 0.78;
    ctx.fillRect(i * bw + pad, h - bh - 20 * dpr, bw - pad * 2, bh);
    ctx.globalAlpha = 1;
    ctx.fillStyle = i === highlight ? "#f3f1ec" : "#6f6b64";
    ctx.fillText(items[i].label ?? "", i * bw + bw / 2, h - 6 * dpr);
  }
  return { hi };
}

/** Small line chart for the per-iteration curves (jaccard, active count, negative share). */
export function drawLine(canvas, series, { height = 160, yMax = null, yMin = 0 } = {}) {
  const f = fit(canvas, height);
  if (!f) return null;
  const { ctx, w, h, dpr } = f;
  ctx.clearRect(0, 0, w, h);
  const padL = 34 * dpr, padB = 20 * dpr, padT = 10 * dpr, padR = 8 * dpr;
  let hi = yMax;
  if (hi == null) { hi = 0; for (const s of series) for (const v of s.values) hi = Math.max(hi, v); }
  hi = hi || 1;
  ctx.strokeStyle = "rgba(255,255,255,.08)";
  ctx.lineWidth = dpr;
  for (let g = 0; g <= 4; g++) {
    const y = padT + (h - padT - padB) * (g / 4);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
  }
  for (const s of series) {
    const n = s.values.length;
    ctx.strokeStyle = s.colour || "#e0872a";
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    s.values.forEach((v, i) => {
      const x = padL + (w - padL - padR) * (n === 1 ? 0.5 : i / (n - 1));
      const y = padT + (h - padT - padB) * (1 - (v - yMin) / (hi - yMin));
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
    s.values.forEach((v, i) => {
      const x = padL + (w - padL - padR) * (n === 1 ? 0.5 : i / (n - 1));
      const y = padT + (h - padT - padB) * (1 - (v - yMin) / (hi - yMin));
      ctx.fillStyle = s.colour || "#e0872a";
      ctx.beginPath(); ctx.arc(x, y, 3 * dpr, 0, Math.PI * 2); ctx.fill();
    });
  }
  ctx.fillStyle = "#6f6b64";
  ctx.font = `${10 * dpr}px ui-monospace, monospace`;
  ctx.textAlign = "right";
  ctx.fillText(hi.toFixed(hi < 2 ? 2 : 0), padL - 5 * dpr, padT + 4 * dpr);
  ctx.fillText(String(yMin), padL - 5 * dpr, h - padB + 4 * dpr);
  return { hi };
}

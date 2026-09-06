// THE ORTHANT: why non-negativity caps what the state can hold.
//
// This is the other pan of the artifact's claim. The field shows what non-negativity BUYS -- a
// sparse, readable, graph-shaped state. This shows what it COSTS, and the cost is geometry, not
// engineering.
//
// BDH's keys are ReLU'd, so every key lives in the positive orthant: the 1/2^d corner of space
// where no coordinate is negative. Two random vectors drawn there cannot be near-orthogonal. For
// standard Gaussians x, y and a = ReLU(x), b = ReLU(y):
//
//     E[a.b]   = d * E[ReLU(z)]^2 = d / (2*pi)
//     E[|a|^2] = d * E[ReLU(z)^2] = d / 2
//     =>  cos  = (d / 2pi) / (d / 2) = 1 / pi = 0.3183...
//
// 1/pi IS THE LARGE-d LIMIT, NOT A DIMENSION-FREE CONSTANT. An earlier draft of this file said
// the d cancels so the number is the same at d=3 and d=3072. Measured, that is wrong: the
// derivation above is a ratio of expectations, but each key is normalised to unit length before
// the cosine is taken, and at small d that normalisation correlates with the dot product and
// biases it upward. Measured means over 256 keys, seed 7:
//
//     d=3     0.467  (62.2 deg)      d=192    0.3167  (71.5 deg)
//     d=16    0.319  (71.4 deg)      d=3072   0.3187  (71.4 deg)
//
// So the effect is if anything STRONGER in the 3 dimensions you can look at, and it has already
// converged by d=16. The panel therefore reports the drawn 3D value under each sphere and the
// model's own 3,072-D value separately -- never one labelled as the other.
//
// Signed keys average cos = 0 -- 90 degrees -- in every dimension tested, and they retrieve far
// past the rank bound (CLAUDE.md item 8). So
// the interference is not caused by the state being too small; it is caused by the keys being
// unable to spread out. Interpretability is paid for in memory, at a rate of 1/pi.

export const INV_PI = 1 / Math.PI;

/** Deterministic PRNG, so the picture and the test agree run to run. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const gauss = (rnd) => {
  // Box-Muller. u must avoid exactly 0 or log(0) is -Infinity.
  const u = Math.max(1e-12, rnd()), v = rnd();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};

/**
 * k unit vectors in d dimensions, optionally ReLU'd before normalising.
 * Returns a flat Float32Array of k*d. A ReLU'd row that came out all-zero is resampled rather
 * than normalised by zero -- it happens with probability 2^-d, which is not negligible at d=3.
 */
export function sampleKeys(k, d, relu, rnd) {
  const V = new Float32Array(k * d);
  for (let i = 0; i < k; i++) {
    let norm = 0, guard = 0;
    do {
      norm = 0;
      for (let j = 0; j < d; j++) {
        let z = gauss(rnd);
        if (relu && z < 0) z = 0;
        V[i * d + j] = z;
        norm += z * z;
      }
      guard++;
    } while (norm === 0 && guard < 64);
    const inv = norm > 0 ? 1 / Math.sqrt(norm) : 0;
    for (let j = 0; j < d; j++) V[i * d + j] *= inv;
  }
  return V;
}

/**
 * Mean cosine over pairs of unit rows.
 *
 * All k(k-1)/2 pairs when that is cheap; a deterministic random SAMPLE of `maxPairs` otherwise.
 * At k=256, d=3072 the exhaustive version is ~100M multiply-adds and blocked the page for over a
 * second before the first frame -- the smoke test caught it as an unpopulated HUD. It is a mean
 * either way, and a few thousand pairs pins it to ~1e-3, which is far finer than the effect
 * being shown (0.318 vs 0.000).
 */
export function meanPairwiseCosine(V, k, d, maxPairs = Infinity, seed = 99) {
  if (k < 2) return 0;
  const total = (k * (k - 1)) / 2;
  const dot = (i, j) => {
    let s = 0;
    for (let c = 0; c < d; c++) s += V[i * d + c] * V[j * d + c];
    return s;
  };
  if (total <= maxPairs) {
    let sum = 0;
    for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) sum += dot(i, j);
    return sum / total;
  }
  const rnd = mulberry32(seed);
  let sum = 0;
  for (let p = 0; p < maxPairs; p++) {
    const i = Math.floor(rnd() * k);
    let j = Math.floor(rnd() * (k - 1));
    if (j >= i) j++;                       // uniform over j != i
    sum += dot(i, j);
  }
  return sum / maxPairs;
}

/** Convenience: sample and measure in one call. */
export function measure(k, d, relu, seed = 1, maxPairs = 6000) {
  const V = sampleKeys(k, d, relu, mulberry32(seed));
  const cos = meanPairwiseCosine(V, k, d, maxPairs, seed + 1);
  return { cos, deg: (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI, V };
}

// ── renderer ─────────────────────────────────────────────────────────────────────────────────
const DPR = () => Math.min(window.devicePixelRatio || 1, 2);

export function createOrthant(canvas, opts = {}) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return { ok: false };

  const col = {
    relu: opts.relu || "#e0872a",
    signed: opts.signed || "#3987e5",
    ink: opts.ink || "#a8a49c",
    ink3: opts.ink3 || "#6f6b64",
    grid: opts.grid || "rgba(255,255,255,.10)",
    ground: opts.ground || "#100f0e",
  };
  let yaw = 0.7, pitch = -0.42;

  const rot = (p) => {
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    const x = p[0] * cy - p[2] * sy;
    const z = p[0] * sy + p[2] * cy;
    return [x, p[1] * cp - z * sp, p[1] * sp + z * cp];
  };

  function sphere(cx, cy, R, V, k, d, tint, label, stat) {
    // horizon
    ctx.strokeStyle = col.grid;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2); ctx.stroke();

    // three great circles, drawn back-half dim so the sphere reads as a solid
    for (const plane of [0, 1, 2]) {
      for (const half of [0, 1]) {
        ctx.beginPath();
        let started = false;
        for (let a = 0; a <= 128; a++) {
          const th = (a / 128) * Math.PI * 2;
          const c = Math.cos(th), s = Math.sin(th);
          const p = plane === 0 ? [c, s, 0] : plane === 1 ? [c, 0, s] : [0, c, s];
          const r = rot(p);
          const back = r[2] < 0;
          if (back !== !!half) { started = false; continue; }
          const X = cx + r[0] * R, Y = cy - r[1] * R;
          if (!started) { ctx.moveTo(X, Y); started = true; } else ctx.lineTo(X, Y);
        }
        ctx.strokeStyle = half ? "rgba(255,255,255,.05)" : col.grid;
        ctx.stroke();
      }
    }

    // the positive octant: three quarter-arcs in the xy, yz and zx planes
    if (tint === col.relu) {
      ctx.strokeStyle = tint;
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = 1.6;
      for (const plane of [0, 1, 2]) {
        ctx.beginPath();
        let started = false;
        for (let a = 0; a <= 48; a++) {
          const th = (a / 48) * (Math.PI / 2);
          const c = Math.cos(th), s = Math.sin(th);
          const p = plane === 0 ? [c, s, 0] : plane === 1 ? [c, 0, s] : [0, c, s];
          const r = rot(p);
          const X = cx + r[0] * R, Y = cy - r[1] * R;
          if (!started) { ctx.moveTo(X, Y); started = true; } else ctx.lineTo(X, Y);
        }
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    // the vectors, painter-sorted so near points land on top
    const pts = [];
    for (let i = 0; i < k; i++) {
      // project the d-dimensional key down to its first three coordinates for display; at d=3
      // this is the vector itself, which is the case the panel actually measures beside it
      const v = [V[i * d], V[i * d + 1], V[i * d + 2]];
      const n = Math.hypot(v[0], v[1], v[2]) || 1;
      pts.push(rot([v[0] / n, v[1] / n, v[2] / n]));
    }
    pts.sort((a, b) => a[2] - b[2]);
    for (const p of pts) {
      const near = (p[2] + 1) / 2;
      ctx.globalAlpha = 0.30 + 0.70 * near;
      ctx.fillStyle = tint;
      ctx.beginPath();
      ctx.arc(cx + p[0] * R, cy - p[1] * R, 1.4 + 2.0 * near, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    ctx.textAlign = "center";
    ctx.fillStyle = col.ink;
    ctx.font = "600 12px ui-sans-serif, system-ui, sans-serif";
    ctx.fillText(label, cx, cy + R + 30);
    ctx.fillStyle = tint;
    ctx.font = "600 22px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
    ctx.fillText(stat, cx, cy + R + 58);
  }

  return {
    ok: true,
    drag(dx, dy) { yaw += dx * 0.006; pitch = Math.max(-1.3, Math.min(1.3, pitch + dy * 0.006)); },
    spin(dt) { yaw += dt * 0.00013; },
    /** @param data { relu:{V,cos,deg}, signed:{V,cos,deg}, k, d } */
    draw(data) {
      const dpr = DPR();
      const r = canvas.getBoundingClientRect();
      const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = col.ground;
      ctx.fillRect(0, 0, r.width, r.height);

      const R = Math.min(r.width / 4.8, (r.height - 190) / 2.4);
      const cy = r.height / 2 - 26;
      sphere(r.width * 0.30, cy, R, data.relu.V, data.k, data.d, col.relu,
             "ReLU'd keys — confined to the positive orthant",
             `${data.relu.deg.toFixed(1)}°`);
      sphere(r.width * 0.70, cy, R, data.signed.V, data.k, data.d, col.signed,
             "signed keys — free to use the whole sphere",
             `${data.signed.deg.toFixed(1)}°`);

      // The two big numbers above are measured in the 3 dimensions actually drawn. The model's
      // keys live in 3,072, where the same constraint gives 71.4 deg, so say which is which
      // rather than letting the picture's number stand in for the model's.
      ctx.textAlign = "center";
      ctx.fillStyle = col.ink3;
      ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
      ctx.fillText(`mean angle between keys, measured in the ${data.d} dimensions drawn` +
                   ` · ${data.k} keys`, r.width / 2, cy + R + 82);
    },
  };
}

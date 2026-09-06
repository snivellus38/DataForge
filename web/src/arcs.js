// ATTENTION AS ARCS. Canvas 2D, over the actual bytes of the sentence.
//
// WHY NOT A HEATMAP. The usual picture of attention is a T x T matrix, which for T=174 is 30,276
// cells of which the causal half is meaningful and almost all are near zero. It reads as noise
// and it hides the one thing worth seeing here: the SIGN. BDH's activations are non-negative by
// construction, so a negative attention score cannot come from the activations -- it can only
// come from RoPE rotating the keys before they meet. 38.5% of causal scores on this model are
// negative (measured, layer 3 head 0), and that destructive interference is what stands in for
// the decay term the public bdh.py does not have (CLAUDE.md item 4).
//
// An arc diagram puts sign in the colour and magnitude in the weight, over readable text.

const DPR = () => Math.min(window.devicePixelRatio || 1, 2);

export function createArcs(canvas, opts = {}) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return { ok: false };

  const col = {
    pos: opts.pos || "#e0872a",
    neg: opts.neg || "#3987e5",
    ink: opts.ink || "#a8a49c",
    dim: opts.dim || "#4a4640",
    ground: opts.ground || "#100f0e",
  };

  let bytes = [], text = "";

  function layout() {
    const dpr = DPR();
    const r = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    return { dpr, w: r.width, h: r.height };
  }

  return {
    ok: true,
    setSentence(t, b) { text = t; bytes = b; },
    /**
     * @param t     current token index
     * @param arcs  { from, w, peak, negShare } from field-data.arcsInto
     */
    draw(t, arcs) {
      const { dpr, w, h } = layout();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = col.ground;
      ctx.fillRect(0, 0, w, h);

      const n = bytes.length;
      const pad = 18;
      const step = (w - pad * 2) / Math.max(1, n - 1);
      const baseY = h - 30;
      const xOf = (i) => pad + i * step;

      // the byte row
      const fs = Math.max(7, Math.min(13, step * 1.5));
      ctx.font = `${fs}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let i = 0; i < n; i++) {
        const b = bytes[i];
        const ch = b >= 32 && b < 127 ? String.fromCharCode(b) : "·";
        ctx.fillStyle = i === t ? col.pos : i < t ? col.ink : col.dim;
        ctx.fillText(ch === " " ? "·" : ch, xOf(i), baseY + 6);
      }

      if (!arcs || !arcs.from.length) return;

      // arcs: quadratic curves from each source byte up and over into the current one.
      // Height encodes DISTANCE (how far back it reaches), thickness and alpha encode magnitude,
      // colour encodes sign. Three channels, no legend needed beyond the two colours.
      const maxSpan = Math.max(1, t);
      ctx.lineCap = "round";
      // weakest first so the strong arcs land on top and stay readable
      const order = Array.from(arcs.from.keys()).sort(
        (a, b) => Math.abs(arcs.w[a]) - Math.abs(arcs.w[b]));
      for (const k of order) {
        const s = arcs.from[k], v = arcs.w[k];
        const mag = Math.abs(v) / (arcs.peak || 1);
        const x0 = xOf(s), x1 = xOf(t);
        const lift = (baseY - 14) * (0.22 + 0.78 * ((t - s) / maxSpan));
        ctx.strokeStyle = v >= 0 ? col.pos : col.neg;
        ctx.globalAlpha = 0.10 + 0.75 * mag;
        ctx.lineWidth = 0.4 + 2.2 * mag;
        ctx.beginPath();
        ctx.moveTo(x0, baseY - 4);
        ctx.quadraticCurveTo((x0 + x1) / 2, baseY - 4 - lift, x1, baseY - 4);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;

      // marker on the token being read
      ctx.fillStyle = col.pos;
      ctx.fillRect(xOf(t) - 0.75, baseY - 6, 1.5, 6);
    },
  };
}

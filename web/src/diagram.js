// THE PIPELINE DIAGRAM — one BDH iteration, drawn once and never taken off screen.
//
// This is the thing the page is built around. Transformer Explainer and bbycroft's LLM viz both
// work because the architecture stays visible while you dig into one piece of it; the reader
// never loses the map. Our earlier essay had no diagram at all, which is why it read as prose
// with figures rather than as an instrument.
//
// Inline SVG rather than canvas, because every node is a click target and a screen-reader label,
// and because the whole thing has to stay crisp at any zoom. jsdom builds SVG DOM fine, so the
// smoke gate can assert the nodes exist and that stepping a stop moves the highlight.
//
// The one non-obvious shape here is the RESIDUAL ARC. n_layer in BDH is not depth -- encoder,
// decoder_x and decoder_y are built once and reused every iteration (CLAUDE.md item 6), so the
// diagram must literally loop back on itself. A left-to-right stack of six blocks would draw a
// Transformer and teach the wrong thing.

const NS = "http://www.w3.org/2000/svg";
const el = (n, a = {}) => {
  const e = document.createElementNS(NS, n);
  for (const [k, v] of Object.entries(a)) e.setAttribute(k, v);
  return e;
};

// One column per stage. `stop` is which walkthrough stop lights this node; `w` its width.
// `kind` drives styling: op = a learned matrix, fn = a fixed nonlinearity, val = a tensor.
export const NODES = [
  { id: "v",     label: "v*",       sub: "192",      kind: "val", stop: 0, w: 104, live: true },
  { id: "ln",    label: "LN",       sub: "",         kind: "fn",  stop: 1, w: 44 },
  { id: "dx",    label: "Dₓ",  sub: "192×3072", kind: "op", stop: 1, w: 78 },
  { id: "relu",  label: "ReLU",     sub: "→5.1%", kind: "fn", stop: 1, w: 62 },
  { id: "rope",  label: "RoPE",     sub: "Q=K",      kind: "fn",  stop: 2, w: 62 },
  { id: "score", label: "Q·K", sub: "no softmax", kind: "op", stop: 3, w: 72 },
  { id: "sigma", label: "σ",   sub: "3072×192", kind: "val", stop: 4, w: 72 },
  { id: "dy",    label: "Dᵧ ⊙ x", sub: "→0.9%", kind: "op", stop: 5, w: 78 },
  { id: "enc",   label: "E +",      sub: "back to v*", kind: "op", stop: 6, w: 68 },
  { id: "head",  label: "lm_head",  sub: "256",      kind: "op",  stop: 7, w: 72 },
];

const GAP = 26, PAD = 14, ROW_Y = 46, H = 42;
const SPARK = 44;          // bars in the live v* strip; 192 dims max-pool onto these

export function createDiagram(host, { onPick } = {}) {
  host.innerHTML = "";
  const svg = el("svg", { class: "diagram", role: "img",
    "aria-label": "One iteration of the Dragon Hatchling: v* through LayerNorm, D_x, ReLU, RoPE, " +
                  "scores, sigma, the gate and the encoder, looping back six times before lm_head" });
  host.appendChild(svg);

  // lay out left to right, measuring as we go
  let x = PAD;
  const boxes = new Map();
  for (const n of NODES) { boxes.set(n.id, { ...n, x, y: ROW_Y }); x += n.w + GAP; }
  const W = x - GAP + PAD;
  const VH = 118;
  svg.setAttribute("viewBox", `0 0 ${W} ${VH}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");

  const defs = el("defs");
  const mk = (id, cls) => {
    const m = el("marker", { id, viewBox: "0 0 8 8", refX: 7, refY: 4,
      markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" });
    m.appendChild(el("path", { d: "M0,0 L8,4 L0,8 z", class: cls }));
    return m;
  };
  defs.appendChild(mk("dg-arrow", "dg-arrowhead"));
  defs.appendChild(mk("dg-arrow-on", "dg-arrowhead on"));
  svg.appendChild(defs);

  // ── the residual loop-back: E + returns into v*, and the WHOLE block runs six times ────────
  const enc = boxes.get("enc"), v0 = boxes.get("v");
  const loop = el("path", {
    class: "dg-loop", id: "dg-loop",
    "marker-end": "url(#dg-arrow)",
    d: `M ${enc.x + enc.w / 2} ${ROW_Y + H} V ${ROW_Y + H + 24} H ${v0.x + v0.w / 2} V ${ROW_Y + H + 4}`,
  });
  svg.appendChild(loop);
  const loopLabel = el("text", {
    class: "dg-loop-label", id: "dg-loop-label",
    x: (enc.x + v0.x) / 2, y: ROW_Y + H + 38, "text-anchor": "middle",
  });
  loopLabel.textContent = "×6 — the same weights, not six layers";
  svg.appendChild(loopLabel);

  // ── edges ──────────────────────────────────────────────────────────────────────────────────
  const edges = [];
  for (let i = 0; i < NODES.length - 1; i++) {
    const a = boxes.get(NODES[i].id), b = boxes.get(NODES[i + 1].id);
    const p = el("path", {
      class: "dg-edge", "marker-end": "url(#dg-arrow)",
      "data-from": a.id, "data-to": b.id,
      d: `M ${a.x + a.w} ${ROW_Y + H / 2} H ${b.x - 5}`,
    });
    svg.appendChild(p);
    edges.push(p);
  }

  // a flowing dot per edge, for the "watch it move" state on stop 0
  const flows = edges.map((p) => {
    const c = el("circle", { class: "dg-flow", r: 3, cx: -99, cy: ROW_Y + H / 2 });
    svg.appendChild(c);
    return c;
  });

  // ── nodes ──────────────────────────────────────────────────────────────────────────────────
  const gs = new Map();
  let sparkBars = [], sparkMid = 0;
  for (const n of NODES) {
    const b = boxes.get(n.id);
    const g = el("g", { class: `dg-node kind-${n.kind}`, "data-id": n.id, "data-stop": n.stop,
                        tabindex: "0", role: "button",
                        "aria-label": `${n.label} — go to stop ${n.stop}` });
    g.appendChild(el("rect", { x: b.x, y: b.y, width: b.w, height: H, rx: 7, class: "dg-box" }));
    const t = el("text", { x: b.x + b.w / 2, y: b.y + (n.sub ? 19 : 25), "text-anchor": "middle",
                           class: "dg-label" });
    t.textContent = n.label;
    g.appendChild(t);
    if (n.sub) {
      const s = el("text", { x: b.x + b.w / 2, y: b.y + 32, "text-anchor": "middle",
                             class: "dg-sub" });
      s.textContent = n.sub;
      g.appendChild(s);
    }
    // v* is the one box whose CONTENTS change every token, so it carries them. 192 signed
    // dimensions max-pooled down to SPARK bars -- max-pooling, not point sampling, for the same
    // reason drawSigma downsamples that way: taking every k-th value can miss the one dimension
    // that is doing the work. The pooling is stated in the tooltip, not implied.
    if (n.live) {
      const g2 = el("g", { class: "dg-spark", id: "dg-spark" });
      const x0 = b.x + 7, wpx = b.w - 14;
      const bw = wpx / SPARK;
      for (let i = 0; i < SPARK; i++) {
        g2.appendChild(el("rect", { x: (x0 + i * bw).toFixed(2), y: b.y + 30,
                                    width: Math.max(0.7, bw - 0.35).toFixed(2), height: 0.6,
                                    class: "dg-spark-bar" }));
      }
      const ttl = el("title");
      ttl.textContent = "the residual entering this iteration: 192 signed dimensions, " +
                        `max-pooled to ${SPARK} bars, live at the current token`;
      g2.appendChild(ttl);
      g.appendChild(g2);
      sparkBars = [...g2.querySelectorAll("rect")];
      sparkMid = b.y + 34;
    }

    const fire = () => onPick && onPick(n.stop);
    g.addEventListener("click", fire);
    g.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fire(); }
    });
    svg.appendChild(g);
    gs.set(n.id, g);
  }

  let t0 = 0;

  return {
    svg,
    /**
     * Paint the residual currently entering the block. Signed: amber above the midline, blue
     * below, which is the sign encoding every other figure on this page uses.
     *
     * This is the answer to "everything moves except v*". The box used to be a static label on a
     * diagram whose other boxes at least lit up per stop; the tensor it names was in the pack the
     * whole time and simply was not drawn anywhere.
     */
    setResidual(v) {
      if (!sparkBars.length || !v || !v.length) return;
      const per = v.length / SPARK;
      let peak = 0;
      for (let i = 0; i < v.length; i++) peak = Math.max(peak, Math.abs(v[i]));
      peak = peak || 1;
      for (let i = 0; i < SPARK; i++) {
        let best = 0;
        for (let k = Math.floor(i * per); k < Math.min(v.length, Math.floor((i + 1) * per)); k++) {
          if (Math.abs(v[k]) > Math.abs(best)) best = v[k];
        }
        const u = Math.min(1, Math.abs(best) / peak) ** 0.6;
        const hgt = Math.max(0.6, u * 8);
        const r = sparkBars[i];
        r.setAttribute("height", hgt.toFixed(2));
        r.setAttribute("y", (best >= 0 ? sparkMid - hgt : sparkMid).toFixed(2));
        r.setAttribute("class", `dg-spark-bar ${best < 0 ? "neg" : "pos"}`);
      }
    },
    /** Light the nodes belonging to `stop`; stop 0 lights everything at equal weight. */
    highlight(stop) {
      for (const n of NODES) {
        const on = stop === 0 || n.stop === stop;
        gs.get(n.id).classList.toggle("on", on);
        gs.get(n.id).classList.toggle("off", !(stop === 0) && n.stop !== stop);
      }
      for (const p of edges) {
        const to = boxes.get(p.dataset.to);
        p.classList.toggle("on", stop === 0 || to.stop === stop);
      }
      loop.classList.toggle("on", stop === 0 || stop === 6);
      loopLabel.classList.toggle("on", stop === 0 || stop === 6);
      for (const f of flows) f.classList.toggle("running", stop === 0);
    },
    /** Advance the flow dots. Only visible on stop 0; cheap enough to call every frame. */
    tick(now) {
      if (!flows[0] || !flows[0].classList.contains("running")) return;
      t0 = now;
      edges.forEach((p, i) => {
        const len = p.getTotalLength ? p.getTotalLength() : 0;
        if (!len) return;                       // jsdom has no geometry; skip silently
        const u = (((now / 1400) + i * 0.13) % 1);
        const pt = p.getPointAtLength(u * len);
        flows[i].setAttribute("cx", pt.x);
        flows[i].setAttribute("cy", pt.y);
        flows[i].setAttribute("opacity", Math.sin(u * Math.PI).toFixed(3));
      });
    },
    nodeIds: NODES.map((n) => n.id),
  };
}

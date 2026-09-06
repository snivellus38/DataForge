// THE FLOW DIAGRAM — one forward pass drawn as columns of nodes and weighted edges.
//
// Canvas for the edges and dots (hundreds of curves, redrawn every token), absolutely-positioned
// DOM for the column headers, the value labels and the inspector card — the same split the field
// uses, and for the same reasons: canvas is fast enough to animate and DOM is what a screen
// reader and a text-selection can reach.
//
// All of the arithmetic is in flow-data.js. This file decides where things sit and what colour
// they are, and nothing else. If a number appears here it came in through `frame`.
//
// COLOUR
// ------
// Amber positive, blue negative — the sign encoding arcs.js already uses on this site, so a blue
// line means the same thing in both places. Deliberately NOT the green/red of the usual network
// diagram: red/green is the worst pair for colour vision deficiency, and this page already has a
// signed encoding of its own that a viewer will have learned two stops earlier.

import { COLUMNS, TICKS, glyph } from "./flow-data.js";

const POS = "#e0872a", NEG = "#3987e5", DEAD = "#4a4640";
const INK = "#f3f1ec", INK2 = "#a8a49c", INK3 = "#6f6b64";

/** Signed magnitude -> stroke colour. `u` is 0..1 after the curve. */
function edgeColour(wv, u, alpha) {
  const c = wv < 0 ? NEG : POS;
  const r = parseInt(c.slice(1, 3), 16), g = parseInt(c.slice(3, 5), 16), b = parseInt(c.slice(5, 7), 16);
  // fade toward the surface at low magnitude rather than going transparent, so a weak edge still
  // reads as a line and the picture does not turn into disconnected dots
  const k = 0.35 + 0.65 * u;
  return `rgba(${Math.round(r * k)},${Math.round(g * k)},${Math.round(b * k)},${alpha})`;
}

export function createFlow(host, { onPick = null } = {}) {
  host.innerHTML = "";
  host.classList.add("flow-host");
  const canvas = document.createElement("canvas");
  canvas.className = "flow-canvas";
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label",
    "One token's path through one iteration: the input byte, the residual, the 12,288 neurons " +
    "before and after the ReLU, the state read-out, the gate, the residual update and the " +
    "distribution over the next byte. Edges are weight times source activation.");
  const heads = document.createElement("div");
  heads.className = "flow-heads";
  const labels = document.createElement("div");
  labels.className = "flow-labels";
  const card = document.createElement("div");
  card.className = "flow-card";
  card.hidden = true;
  host.append(canvas, heads, labels, card);

  const ctx = canvas.getContext("2d");
  let frame = null, layout = null, hits = [], hover = null, pinned = null, pinnedId = null;
  let adj = null, trace = null;
  let W = 0, Hh = 0, dpr = 1;

  const nkey = (col, i) => `${col}:${i}`;

  /** Adjacency over the frame's edges, built once per frame rather than per pointer move. */
  function buildAdj(f) {
    const out = new Map(), inn = new Map();
    const add = (m, k, e) => { const a = m.get(k); if (a) a.push(e); else m.set(k, [e]); };
    for (const e of f.edges) { add(out, nkey(e.from, e.si), e); add(inn, nkey(e.to, e.ti), e); }
    return { out, inn };
  }

  /**
   * Everything up- and downstream of one node, with its distance in hops.
   *
   * Selecting used to light only the edges TOUCHING the node, which on a diagram whose whole
   * subject is a path answers the wrong question -- you could see that a neuron had three inputs
   * and not where any of it went. This walks both directions to the ends of the graph, so
   * clicking a neuron shows the route from the input byte, through the state, through the gate,
   * into the residual and out to the byte distribution. Hop distance drives the brightness, so
   * the immediate connections still read as the immediate ones.
   */
  function traceFrom(col, i) {
    if (!adj) return null;
    const edges = new Map(), nodes = new Map([[nkey(col, i), 0]]);
    for (const [m, other] of [[adj.out, (e) => nkey(e.to, e.ti)],
                              [adj.inn, (e) => nkey(e.from, e.si)]]) {
      let front = [nkey(col, i)], hop = 1;
      while (front.length && hop <= COLUMNS.length) {
        const next = [];
        for (const k of front) {
          for (const e of m.get(k) || []) {
            if (!edges.has(e) || edges.get(e) > hop) edges.set(e, hop);
            const t = other(e);
            if (!nodes.has(t)) { nodes.set(t, hop); next.push(t); }
          }
        }
        front = next;
        hop++;
      }
    }
    return { key: nkey(col, i), edges, nodes };
  }

  function retrace() {
    const sel = pinned || hover;
    if (!sel) { trace = null; return; }
    const k = nkey(sel.col, sel.i);
    if (!trace || trace.key !== k) trace = traceFrom(sel.col, sel.i);
  }

  // ── layout ──────────────────────────────────────────────────────────────────────────────────
  //
  // The side padding is deliberately TIGHT. Column headers are clamped into the frame
  // independently (see paintHeaders), so they need no padding at all; the only real constraints
  // are the input byte's disc on the left and lm_head's value labels on the right -- "·213 100%"
  // at 9.5px mono, about 51px, printed beside their dots. Anything past that is plot width
  // thrown away, which is exactly what a first pass at fixing the clipping did: it stopped the
  // labels being cut by shrinking the diagram, when the width was there to be taken from the
  // narration rail instead (loop.css: 372px -> 300px).
  // PAD_T clears the three-line column header (about 50px from the frame); PAD_B was 34px of
  // nothing. Both are as small as the content allows -- the canvas is the visualisation, and
  // every pixel of inset is one the plot does not get.
  const PAD_T = 58, PAD_B = 14;
  const HEAD_W = 124;                 // must match .flow-head's width in loop.css
  const padL = () => 22;
  const padR = () => Math.max(66, Math.min(78, W * 0.058));

  function measure() {
    W = Math.max(320, host.clientWidth || 960);
    Hh = Math.max(260, host.clientHeight || 520);
    dpr = Math.min(2, (typeof devicePixelRatio === "number" ? devicePixelRatio : 1) || 1);
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(Hh * dpr);
    canvas.style.width = `${W}px`;
    canvas.style.height = `${Hh}px`;
    if (ctx.setTransform) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  /** Where every node sits. Columns are evenly spaced; rows depend on the column's unit. */
  function place(f) {
    const n = COLUMNS.length;
    const L = padL(), R = padR();
    const step = (W - L - R) / (n - 1);
    const top = PAD_T, bot = Hh - PAD_B;
    const cols = [];
    hits = [];

    for (let ci = 0; ci < n; ci++) {
      const col = COLUMNS[ci];
      const x = L + ci * step;
      const items = f.nodes[col.id];
      const pts = [];
      if (col.unit === "dim") {
        // all 192 dimensions, one tick each -- no sampling, so no caveat needed
        const h = bot - top, dy = h / TICKS;
        for (let i = 0; i < items.length; i++) pts.push({ x, y: top + (i + 0.5) * dy, r: dy * 0.42 });
      } else if (col.unit === "neuron") {
        // The dot grows into whatever row height it is given rather than sitting at a fixed 5.5px
        // in a taller frame, so a bigger window shows a bigger diagram and not the same diagram
        // with more gaps in it.
        const h = bot - top, dy = h / Math.max(items.length, 1);
        const r = Math.max(4, Math.min(9, dy * 0.34));
        for (let i = 0; i < items.length; i++) pts.push({ x, y: top + (i + 0.5) * dy, r });
      } else if (col.id === "byte") {
        pts.push({ x, y: (top + bot) / 2, r: Math.max(13, Math.min(19, (bot - top) * 0.030)) });
      } else {
        const h = bot - top, dy = Math.min(52, h / Math.max(items.length + 1, 1));
        const y0 = (top + bot) / 2 - (items.length - 1) * dy / 2;
        const r = Math.max(6, Math.min(11, dy * 0.30));
        for (let i = 0; i < items.length; i++) pts.push({ x, y: y0 + i * dy, r });
      }
      cols.push({ col, x, pts, items });
      // Every column is selectable, the 192-tick ones included -- "click any node" has to mean
      // any node. Their ticks are ~2.5px apart, so the hit radius is widened horizontally by
      // giving them a generous constant rather than pts[i].r, and the nearest-wins rule in
      // hitAt() picks the right one.
      const pad = col.unit === "dim" ? 7 : 5;
      for (let i = 0; i < pts.length; i++) {
        hits.push({ col: col.id, ci, i, x: pts[i].x, y: pts[i].y,
                    r: Math.max(col.unit === "dim" ? 2.2 : pts[i].r, pts[i].r) + pad });
      }
    }
    return { cols, byId: Object.fromEntries(cols.map((c, i) => [c.col.id, i])) };
  }

  // ── painting ────────────────────────────────────────────────────────────────────────────────
  function peakOf(f) {
    let mx = 0;
    for (const e of f.edges) mx = Math.max(mx, Math.abs(e.w));
    return mx || 1;
  }

  function draw() {
    if (!frame || !layout) return;
    const f = frame;
    ctx.clearRect(0, 0, W, Hh);

    // column guides, so the eye can find a column even where its nodes are sparse
    ctx.strokeStyle = "rgba(255,255,255,.045)";
    ctx.lineWidth = 1;
    for (const c of layout.cols) {
      ctx.beginPath();
      ctx.moveTo(c.x, PAD_T - 12);
      ctx.lineTo(c.x, Hh - PAD_B + 8);
      ctx.stroke();
    }

    const peak = peakOf(f);
    const sel = pinned || hover;
    retrace();

    const stroke = (e, alpha, width) => {
      const A = layout.cols[layout.byId[e.from]], B = layout.cols[layout.byId[e.to]];
      const p0 = A.pts[e.si], p1 = B.pts[e.ti];
      if (!p0 || !p1) return;
      const u = Math.min(1, Math.abs(e.w) / peak) ** 0.45;
      ctx.strokeStyle = trace && alpha < 0.1
        ? `rgba(150,146,140,${alpha})`            // off the path: no hue, so sign reads as signal
        : edgeColour(e.w, u, alpha);
      ctx.lineWidth = width(u);
      const mx = (p0.x + p1.x) / 2;
      ctx.beginPath();
      ctx.moveTo(p0.x, p0.y);
      ctx.bezierCurveTo(mx, p0.y, mx, p1.y, p1.x, p1.y);
      ctx.stroke();
    };

    if (!trace) {
      // nothing selected: weak first so strong edges land on top
      for (const e of [...f.edges].sort((p, q) => Math.abs(p.w) - Math.abs(q.w))) {
        const u = Math.min(1, Math.abs(e.w) / peak) ** 0.45;
        stroke(e, 0.16 + 0.62 * u, (uu) => 0.6 + 2.4 * uu);
      }
    } else {
      // off the path first and very faint, then the path itself, furthest hop first so the
      // immediate connections finish on top
      const on = [], off = [];
      for (const e of f.edges) (trace.edges.has(e) ? on : off).push(e);
      for (const e of off.sort((p, q) => Math.abs(p.w) - Math.abs(q.w))) {
        stroke(e, 0.035, () => 0.5);
      }
      on.sort((p, q) => (trace.edges.get(q) - trace.edges.get(p)) || (Math.abs(p.w) - Math.abs(q.w)));
      for (const e of on) {
        const hop = trace.edges.get(e);
        // a wide, faint underlay under the immediate connections so they read as lit, not merely
        // as un-dimmed -- the previous version left the selection at its ordinary brightness and
        // only dimmed everything else, which is why a click looked like nothing had happened
        if (hop === 1) stroke(e, 0.30, (u) => 5 + 7 * u);
        const alpha = hop === 1 ? 0.98 : hop === 2 ? 0.62 : 0.34;
        const w0 = hop === 1 ? 1.7 : hop === 2 ? 1.0 : 0.7;
        stroke(e, alpha, (u) => w0 + (hop === 1 ? 3.4 : 1.8) * u);
      }
    }

    // nodes
    for (const c of layout.cols) {
      for (let i = 0; i < c.pts.length; i++) {
        const pt = c.pts[i], nd = c.items[i];
        const isSel = sel && sel.col === c.col.id && sel.i === i;
        const hop = trace ? trace.nodes.get(nkey(c.col.id, i)) : undefined;
        paintNode(c.col, nd, pt, isSel, f, trace ? hop : null);
      }
    }
    paintLabels(f);
  }

  function paintNode(col, nd, pt, isSel, f, hop) {
    // hop === null   nothing selected
    // hop === undefined  this node is not on the selected path
    const faded = hop === undefined && hop !== null;
    let colour = DEAD, mag = 0;
    if (col.unit === "dim") {
      const scale = col.id === "v" ? f.scale.v : col.id === "a" ? f.scale.a : f.scale.dv;
      mag = Math.min(1, Math.abs(nd.value) / scale) ** 0.5;
      colour = nd.value < 0 ? NEG : POS;
      ctx.globalAlpha = (0.25 + 0.75 * mag) * (faded ? 0.22 : 1);
      ctx.fillStyle = colour;
      const th = Math.max(1.4, Math.min(3, pt.r * 1.7));
      ctx.fillRect(pt.x - 5 - 5 * mag, pt.y - th / 2, 10 + 10 * mag, th);
      ctx.globalAlpha = 1;
      return;
    }
    if (col.id === "xpre") {
      mag = Math.min(1, Math.abs(nd.value) / f.scale.pre) ** 0.5;
      colour = Number.isNaN(nd.value) ? DEAD : nd.value < 0 ? NEG : POS;
    } else if (col.id === "byteout") {
      mag = nd.p;
      colour = nd.truth ? POS : INK3;
    } else if (col.id === "byte") {
      mag = 1;
      colour = POS;
    } else {
      const scale = col.id === "x" ? f.scale.x : f.scale.y;
      mag = nd.value > 0 ? Math.min(1, nd.value / scale) ** 0.5 : 0;
      colour = nd.value > 0 ? POS : DEAD;
    }

    const r = pt.r * (0.55 + 0.45 * (mag || 0.15));
    ctx.globalAlpha = faded ? 0.24 : 1;
    if (mag > 0.02) {
      const g = ctx.createRadialGradient(pt.x, pt.y, 0, pt.x, pt.y, r * 3.2);
      g.addColorStop(0, hexA(colour, 0.35 * mag));
      g.addColorStop(1, hexA(colour, 0));
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, r * 3.2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = mag > 0.02 ? colour : DEAD;
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, r, 0, Math.PI * 2);
    ctx.fill();

    // a hollow ring marks a dot that is only here because it leads THIS token, so the reader can
    // tell the fixed cast from the visitors without reading the caption again
    if (nd.stable === false) {
      ctx.strokeStyle = hexA(INK, 0.55);
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, r + 3, 0, Math.PI * 2);
      ctx.stroke();
    }
    // a node the path passes through gets a ring, so the route is readable at the nodes as well
    // as along the edges
    if (hop > 0 && hop <= 2) {
      ctx.strokeStyle = hexA(INK, hop === 1 ? 0.7 : 0.34);
      ctx.lineWidth = 1.1;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, r + 3.5, 0, Math.PI * 2);
      ctx.stroke();
    }
    if (isSel) {
      ctx.strokeStyle = INK;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, r + 5.5, 0, Math.PI * 2);
      ctx.stroke();
      ctx.strokeStyle = hexA(POS, 0.55);
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, r + 9, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  const hexA = (hex, a) => {
    const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${a})`;
  };

  // ── DOM: headers, value labels, inspector ───────────────────────────────────────────────────
  function paintHeaders(f) {
    // Centred on the column where there is room, nudged inward at the ends. A header is 124px
    // wide and the end columns sit near the frame, so centring them unconditionally clips the
    // first and last -- which is exactly what shipped.
    const half = HEAD_W / 2 + 4;
    heads.innerHTML = COLUMNS.map((c, i) => {
      const live = f.live[c.id] || "";
      const x = Math.max(half, Math.min(W - half, layout.cols[i].x));
      return `<div class="flow-head" style="left:${x.toFixed(1)}px">
        <b>${c.name}</b><i>${c.sub}</i><em>${live}</em></div>`;
    }).join("");
  }

  function paintLabels(f) {
    // the strongest few dots in each sampled column get their value printed, like a chart's
    // data labels -- the reference this is modelled on does the same, and it is the difference
    // between "some dots are brighter" and "this neuron is at 0.74"
    const out = [];
    for (const c of layout.cols) {
      if (c.col.unit === "dim") continue;
      const rank = c.items.map((nd, i) => [i, c.col.id === "byteout" ? nd.p : nd.value])
        .filter(([, v]) => v > 0)
        .sort((a, b) => b[1] - a[1])
        .slice(0, c.col.id === "byteout" ? 9 : 6);
      for (const [i, v] of rank) {
        const pt = c.pts[i];
        const txt = c.col.id === "byteout"
          ? `${c.items[i].label} ${(v * 100).toFixed(0)}%`
          : v.toFixed(2);
        out.push(`<span class="flow-val${c.items[i].truth ? " is-truth" : ""}" ` +
                 `style="left:${pt.x + (c.col.id === "byteout" ? 13 : -13)}px;top:${pt.y}px;` +
                 `transform:translate(${c.col.id === "byteout" ? "0" : "-100%"},-50%)">${txt}</span>`);
      }
    }
    const b = layout.cols[0];
    out.push(`<span class="flow-byte" style="left:${b.x}px;top:${b.pts[0].y}px">` +
             `${f.nodes.byte[0].label}</span>`);
    labels.innerHTML = out.join("");
  }

  function showCard(h) {
    if (!h) { card.hidden = true; return; }
    const c = layout.cols[h.ci], nd = c.items[h.i];
    card.hidden = false;
    // flip to the left of the node when there is no room on the right, rather than clamping it
    // on top of the column it is describing
    const CW = 234;
    const right = h.x + 16;
    card.style.left = `${right + CW > W - 8 ? Math.max(6, h.x - 16 - CW) : Math.max(6, right)}px`;
    card.style.top = `${Math.min(Hh - 150, Math.max(6, h.y - 20))}px`;
    card.innerHTML = describe(c.col, nd, frame) + tracedLine();
  }

  /** How much of the diagram the current selection actually reaches. */
  function tracedLine() {
    if (!trace) return "";
    const hops = [...trace.edges.values()];
    const direct = hops.filter((h) => h === 1).length;
    const cols = new Set([...trace.nodes.keys()].map((k) => k.split(":")[0])).size;
    return `<p class="flow-card-path"><b>${direct}</b> direct &middot; ` +
           `<b>${hops.length}</b> on its path, across ${cols} of ${COLUMNS.length} columns</p>`;
  }

  function describe(col, nd, f) {
    const row = (k, v) => `<div><dt>${k}</dt><dd>${v}</dd></div>`;
    if (col.unit === "neuron") {
      const head = Math.floor(nd.i / f.stats.headN);
      const fireFrac = nd.fire !== undefined
        ? `${(nd.fire / f.stats.corpusSteps * 100).toFixed(1)}% of corpus tokens` : "—";
      return `<h4>neuron ${nd.i}</h4><dl>` +
        row("value here", nd.value ? nd.value.toFixed(4) : "0 — silent") +
        row("head · local", `${head} · ${nd.i - head * f.stats.headN}`) +
        row("in the cast", nd.stable ? "always drawn"
              : nd.why === "gate" ? "leads this token, past the gate"
              : "leads this token, before the gate") +
        (col.id === "y" ? row("gate input x", nd.gate ? nd.gate.toFixed(4) : "0 — switched off") : "") +
        (col.id === "x" && nd.inHead ? row("after RoPE", nd.roped.toFixed(4)) : "") +
        (nd.fire !== undefined ? row("fires on", fireFrac) : "") +
        `</dl>`;
    }
    if (col.id === "xpre") {
      return `<h4>neuron ${nd.i}, pre-activation</h4><dl>` +
        row("x_pre", Number.isNaN(nd.value) ? "not in the traced head" : nd.value.toFixed(4)) +
        row("survives ReLU", nd.value > 0 ? "yes" : "no — set to exactly 0") +
        `</dl>`;
    }
    if (col.id === "byteout") {
      return `<h4>byte ${nd.i} “${nd.label}”</h4><dl>` +
        row("logit", nd.value.toFixed(3)) +
        row("probability", `${(nd.p * 100).toFixed(2)}%`) +
        row("", nd.truth ? "this is what came next" : nd.chosen ? "the model's choice" : "") +
        `</dl>`;
    }
    return `<h4>${col.name}</h4><dl>${row("value", String(nd.value))}</dl>`;
  }

  // ── interaction ─────────────────────────────────────────────────────────────────────────────
  function hitAt(cx, cy) {
    let best = null, bd = Infinity;
    for (const h of hits) {
      const d = (h.x - cx) ** 2 + (h.y - cy) ** 2;
      if (d < h.r * h.r && d < bd) { bd = d; best = h; }
    }
    return best;
  }

  canvas.addEventListener("pointermove", (e) => {
    const r = canvas.getBoundingClientRect();
    const h = hitAt(e.clientX - r.left, e.clientY - r.top);
    const changed = (h && hover ? h.col !== hover.col || h.i !== hover.i : h !== hover);
    hover = h;
    if (changed) { draw(); showCard(pinned || h); }   // draw() computes the trace the card cites
  });
  canvas.addEventListener("pointerleave", () => {
    hover = null; draw(); showCard(pinned);
  });
  canvas.addEventListener("click", (e) => {
    const r = canvas.getBoundingClientRect();
    const h = hitAt(e.clientX - r.left, e.clientY - r.top);
    pinned = (h && pinned && h.col === pinned.col && h.i === pinned.i) ? null : h;
    // Remember WHICH node, not which slot. The drawn cast changes as the token advances -- the
    // visitors are this token's leaders -- so a pin held by row index would silently jump to a
    // different neuron on the next token while still claiming to be the one that was clicked.
    pinnedId = pinned ? layout.cols[pinned.ci].items[pinned.i].i : null;
    draw();
    showCard(pinned || hover);
    if (onPick) onPick(pinned ? layout.cols[pinned.ci].items[pinned.i] : null);
  });

  return {
    /** Hand it a frame from buildFrame(); scales are derived here so the model stays pure. */
    setFrame(f) {
      f.scale = scalesFor(f);
      f.live = liveFor(f);
      frame = f;
      adj = buildAdj(f);
      trace = null;                       // the edge objects are new, so any cached trace is stale
      measure();
      layout = place(f);
      paintHeaders(f);
      // re-seat the pin on the same node in the new frame, or drop it if it is no longer drawn
      if (pinned) {
        const col = layout.cols[pinned.ci];
        const at = col ? col.items.findIndex((n) => n.i === pinnedId) : -1;
        if (at < 0) { pinned = null; pinnedId = null; }
        else pinned = hits.find((h) => h.ci === pinned.ci && h.i === at) || null;
      }
      draw();
      showCard(pinned || hover);
    },
    resize() { if (frame) { measure(); layout = place(frame); paintHeaders(frame); draw(); } },
    pinned: () => pinned,
    clearPin() { pinned = null; pinnedId = null; trace = null; showCard(null); draw(); },
    /** For the gates: what is pinned, and how far its path reaches. */
    selection: () => (pinned
      ? { col: pinned.col, id: pinnedId,
          edges: trace ? trace.edges.size : 0,
          direct: trace ? [...trace.edges.values()].filter((h) => h === 1).length : 0 }
      : null),
    destroy() { host.innerHTML = ""; },
  };
}

/**
 * Per-column normalisation. p99-style: the strongest value in the column, so one huge dimension
 * cannot flatten the rest -- the same lesson as the sigma heatmaps (CLAUDE.md item 34), applied
 * to a node-link picture. Absolute, because these columns are signed.
 */
function scalesFor(f) {
  const mx = (arr, key = "value") => {
    let m = 0;
    for (const nd of arr) { const v = Math.abs(nd[key]); if (v > m && Number.isFinite(v)) m = v; }
    return m || 1;
  };
  return {
    v: mx(f.nodes.v), a: mx(f.nodes.a), dv: mx(f.nodes.dv),
    pre: mx(f.nodes.xpre), x: mx(f.nodes.x), y: mx(f.nodes.y),
  };
}

/** The live count under each column header. Every one of these is measured, not decorative. */
function liveFor(f) {
  const s = f.stats;
  const pc = (v, n = 1) => `${(v * 100).toFixed(n)}%`;
  return {
    byte: `“${glyph(s.byte)}” · ${s.t + 1} of ${s.T}`,
    v: "all 192 drawn",
    xpre: `${pc(s.preNegFrac)} below zero`,
    x: `${s.xFiring.toLocaleString("en-US")} firing · ${pc(s.xFrac, 2)}`,
    a: "all 192 drawn",
    y: `${s.yFiring.toLocaleString("en-US")} firing · ${pc(s.yFrac, 2)}`,
    dv: s.ySurvivors
      ? `${s.ySurvivors} drawn of ${s.yFiring.toLocaleString("en-US")} that got through`
      : `none of the drawn ${s.drawn} survived here`,
    byteout: s.top ? `${glyph(s.top.i)} at ${pc(s.top.p)}` : "",
  };
}

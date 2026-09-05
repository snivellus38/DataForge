// Minimal line chart for the capacity curves.
//
// Follows the dataviz procedure: form first (magnitude over an ordered x -> line), colour by job
// (two categorical slots, validated all-pairs in both modes), 2px lines, >=8px markers, recessive
// grid, legend always present for >=2 series plus direct labels, hover crosshair, and a table view
// so identity is never colour-alone.
import { SERIES, INK, mode } from "./palette.js";

const NS = "http://www.w3.org/2000/svg";
const el = (n, a = {}) => {
  const e = document.createElementNS(NS, n);
  for (const [k, v] of Object.entries(a)) e.setAttribute(k, v);
  return e;
};

/**
 * @param {HTMLElement} host
 * @param {Array<{name:string, points:Array<{k:number,acc:number}>}>} series
 * @param {{marker?:number, onHover?:function}} opts
 */
export function drawCapacity(host, series, opts = {}) {
  const m = mode(), ink = INK[m], pal = SERIES[m];
  const W = host.clientWidth || 560, H = 260;
  const pad = { l: 46, r: 108, t: 14, b: 34 };
  host.textContent = "";

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H,
                          role: "img", "aria-label": "Retrieval accuracy versus number of bindings" });
  const ks = series[0].points.map((p) => p.k);
  const xmin = Math.log2(ks[0]), xmax = Math.log2(ks[ks.length - 1]);
  const X = (k) => pad.l + ((Math.log2(k) - xmin) / (xmax - xmin)) * (W - pad.l - pad.r);
  const Y = (a) => pad.t + (1 - a) * (H - pad.t - pad.b);

  // recessive grid + axes
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    svg.append(el("line", { x1: pad.l, x2: W - pad.r, y1: Y(v), y2: Y(v),
                            stroke: ink.grid, "stroke-width": 1 }));
    const t = el("text", { x: pad.l - 8, y: Y(v) + 4, "text-anchor": "end",
                           fill: ink.muted, "font-size": 11 });
    t.textContent = `${v * 100}%`;
    svg.append(t);
  }
  for (const k of ks) {
    const t = el("text", { x: X(k), y: H - pad.b + 16, "text-anchor": "middle",
                           fill: ink.muted, "font-size": 11 });
    t.textContent = k;
    svg.append(t);
  }
  const xl = el("text", { x: (pad.l + W - pad.r) / 2, y: H - 2, "text-anchor": "middle",
                          fill: ink.muted, "font-size": 11 });
  xl.textContent = "simultaneous bindings k (log scale)";
  svg.append(xl);

  series.forEach((s, i) => {
    const c = pal[i % pal.length];
    const d = s.points.map((p, j) => `${j ? "L" : "M"}${X(p.k)},${Y(p.acc)}`).join(" ");
    svg.append(el("path", { d, fill: "none", stroke: c, "stroke-width": 2,
                            "stroke-linejoin": "round", "stroke-linecap": "round" }));
    for (const p of s.points)
      svg.append(el("circle", { cx: X(p.k), cy: Y(p.acc), r: 4, fill: c,
                                stroke: ink.surface, "stroke-width": 2 }));
    // direct label -- identity never depends on colour alone
    const last = s.points[s.points.length - 1];
    const lab = el("text", { x: X(last.k) + 10, y: Y(last.acc) + 4, fill: ink.secondary,
                             "font-size": 11.5, "font-weight": 600 });
    lab.textContent = s.name;
    svg.append(lab);
  });

  if (opts.marker) {
    const x = X(opts.marker);
    svg.append(el("line", { x1: x, x2: x, y1: pad.t, y2: H - pad.b,
                            stroke: ink.muted, "stroke-width": 1, "stroke-dasharray": "3 3" }));
  }
  host.append(svg);
  return svg;
}

/** Table view: required so the chart is readable without colour, and for screen readers. */
export function capacityTable(host, series) {
  host.textContent = "";
  const t = document.createElement("table");
  t.className = "datatable";
  const ks = series[0].points.map((p) => p.k);
  const head = document.createElement("tr");
  head.append(document.createElement("th"));
  for (const k of ks) { const th = document.createElement("th"); th.textContent = k; head.append(th); }
  t.append(head);
  for (const s of series) {
    const tr = document.createElement("tr");
    const th = document.createElement("th"); th.textContent = s.name; tr.append(th);
    for (const p of s.points) {
      const td = document.createElement("td");
      td.textContent = `${Math.round(p.acc * 100)}%`;
      tr.append(td);
    }
    t.append(tr);
  }
  host.append(t);
}

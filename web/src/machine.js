// The machine room: step through the computation token by token, with everything linked.
//
// Hovering or selecting a token drives every panel at once — its neuron activations, the
// attention scores it produces, the rank-one write it makes into sigma, and the state that
// results. Nothing here is precomputed for display; every array comes from trace(), which is
// verified to reproduce the forward pass exactly.
import { divergingRGB, INK, SERIES, mode } from "./palette.js";

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

/** p99 of |v|, so a handful of outliers don't wash a map out. */
function scaleOf(a) {
  const s = [];
  for (let i = 0; i < a.length; i += Math.max(1, (a.length / 4096) | 0)) s.push(Math.abs(a[i]));
  s.sort((x, y) => x - y);
  return s[Math.floor(s.length * 0.99)] || 1;
}

function paint(canvas, data, rows, cols, scale, cw, ch) {
  canvas.width = cols * cw; canvas.height = rows * ch;
  const ctx = canvas.getContext("2d", { alpha: false });
  const img = ctx.createImageData(cols * cw, rows * ch);
  const m = mode();
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      const [R, G, B] = divergingRGB(data[r * cols + c] / scale, m);
      for (let y = 0; y < ch; y++) {
        let o = (((r * ch + y) * cols * cw) + c * cw) * 4;
        for (let x = 0; x < cw; x++, o += 4) {
          img.data[o] = R; img.data[o + 1] = G; img.data[o + 2] = B; img.data[o + 3] = 255;
        }
      }
    }
  ctx.putImageData(img, 0, 0);
}

export class MachineRoom {
  /**
   * @param {HTMLElement} root container
   * @param {(t:number)=>void} onSelect notified when the focused token changes
   */
  constructor(root, onSelect = () => {}) {
    this.root = root; this.onSelect = onSelect;
    this.t = 0; this.playing = false; this.raf = null; this.hover = null;
    root.innerHTML = `
      <div class="mr-strip" id="mr-tokens"></div>
      <div class="mr-transport">
        <button id="mr-play" class="mr-btn" aria-label="Play">▶</button>
        <button id="mr-prev" class="mr-btn" aria-label="Previous token">◀</button>
        <button id="mr-next" class="mr-btn" aria-label="Next token">▶|</button>
        <input type="range" id="mr-scrub" min="0" max="0" value="0" aria-label="Token position">
        <span class="mr-pos">token <b id="mr-t">0</b> / <span id="mr-tmax">0</span></span>
      </div>
      <div class="mr-grid">
        <div class="mr-cell">
          <h4>neurons firing <span class="mr-hint" id="mr-sparse"></span></h4>
          <canvas id="mr-neurons"></canvas>
          <p class="mr-cap">ReLU'd activations for this token. Non-negative by construction —
            that is what the whole page is about.</p>
        </div>
        <div class="mr-cell">
          <h4>attention <span class="mr-hint">this token &rarr; earlier tokens</span></h4>
          <canvas id="mr-scores"></canvas>
          <p class="mr-cap" id="mr-scores-cap">Row <b>t</b>, columns <b>s &lt; t</b>. No softmax,
            so scores are unnormalised — and RoPE lets them go negative.</p>
        </div>
        <div class="mr-cell">
          <h4>this token's write <span class="mr-hint">rope(K)<sub>t</sub> &otimes; V<sub>t</sub></span></h4>
          <canvas id="mr-write"></canvas>
          <p class="mr-cap">Rank one: one row pattern times one column pattern.</p>
        </div>
        <div class="mr-cell">
          <h4>&sigma; after this token <span class="mr-hint" id="mr-dens"></span></h4>
          <canvas id="mr-sigma"></canvas>
          <p class="mr-cap">Everything written so far, summed. Fixed shape, whatever the length.</p>
        </div>
      </div>
      <div class="mr-tip" id="mr-tip" hidden></div>`;

    this.el = {
      strip: root.querySelector("#mr-tokens"), play: root.querySelector("#mr-play"),
      scrub: root.querySelector("#mr-scrub"), t: root.querySelector("#mr-t"),
      tmax: root.querySelector("#mr-tmax"), sparse: root.querySelector("#mr-sparse"),
      dens: root.querySelector("#mr-dens"), tip: root.querySelector("#mr-tip"),
      neurons: root.querySelector("#mr-neurons"), scores: root.querySelector("#mr-scores"),
      write: root.querySelector("#mr-write"), sigma: root.querySelector("#mr-sigma"),
      scap: root.querySelector("#mr-scores-cap"),
    };

    this.el.play.onclick = () => this.toggle();
    root.querySelector("#mr-prev").onclick = () => this.seek(this.t - 1);
    root.querySelector("#mr-next").onclick = () => this.seek(this.t + 1);
    this.el.scrub.oninput = (e) => { this.stop(); this.seek(+e.target.value); };
    this.el.scores.onmousemove = (e) => this.scoreHover(e);
    this.el.scores.onmouseleave = () => { this.el.tip.hidden = true; };
  }

  load(tr, labels) {
    this.tr = tr; this.labels = labels;
    this.el.scrub.max = tr.T - 1;
    this.el.tmax.textContent = tr.T - 1;
    this.sScale = scaleOf(tr.scores);
    this.wScale = scaleOf(tr.writes[tr.T - 1]);
    this.gScale = scaleOf(tr.snaps[tr.T - 1]);
    this.nScale = scaleOf(tr.xSparse);
    this.buildStrip();
    this.seek(Math.min(this.t, tr.T - 1));
  }

  buildStrip() {
    const s = this.el.strip;
    s.textContent = "";
    this.labels.forEach((l, i) => {
      const b = document.createElement("button");
      b.className = `mr-tok mr-tok-${l.kind}`;
      b.textContent = l.label;
      b.title = `token ${i} · byte ${l.byte}`;
      if (l.kind === "tgt") b.style.setProperty("--chip-hue", l.hue);
      b.onmouseenter = () => this.peek(i);
      b.onmouseleave = () => this.peek(null);
      b.onclick = () => { this.stop(); this.seek(i); };
      s.append(b);
    });
  }

  peek(i) { this.hover = i; this.paintStrip(); }

  paintStrip() {
    [...this.el.strip.children].forEach((b, i) => {
      b.classList.toggle("on", i === this.t);
      b.classList.toggle("dim", this.t !== null && i > this.t);
      b.classList.toggle("peek", this.hover === i);
    });
  }

  toggle() { this.playing ? this.stop() : this.play(); }

  play() {
    if (!this.tr) return;
    this.playing = true; this.el.play.textContent = "❚❚";
    if (this.t >= this.tr.T - 1) this.t = 0;
    let last = 0;
    const step = (now) => {
      if (!this.playing) return;
      if (now - last > 420) {
        last = now;
        if (this.t >= this.tr.T - 1) { this.stop(); return; }
        this.seek(this.t + 1);
      }
      this.raf = requestAnimationFrame(step);
    };
    this.raf = requestAnimationFrame(step);
  }

  stop() {
    this.playing = false; this.el.play.textContent = "▶";
    if (this.raf) cancelAnimationFrame(this.raf);
  }

  seek(t) {
    if (!this.tr) return;
    this.t = clamp(t, 0, this.tr.T - 1);
    this.el.scrub.value = this.t;
    this.el.t.textContent = this.t;
    this.paintStrip();
    this.draw();
    this.onSelect(this.t);
  }

  draw() {
    const { tr, t } = this, { N, D, T } = tr;

    // neuron activations for this token, as a tall strip of N cells
    const act = tr.xSparse.subarray(t * N, (t + 1) * N);
    const cols = 32, rows = Math.ceil(N / cols);
    const grid = new Float32Array(rows * cols);
    for (let i = 0; i < N; i++) grid[i] = act[i];
    paint(this.el.neurons, grid, rows, cols, this.nScale, 9, 9);
    this.el.sparse.textContent = `${(tr.sparsity[t] * 100).toFixed(0)}% of ${N}`;

    // the causal score matrix, with the current row marked by the caption
    paint(this.el.scores, tr.scores, T, T, this.sScale, Math.max(6, (240 / T) | 0), Math.max(6, (240 / T) | 0));
    this.el.scap.innerHTML = `Row <b>${t}</b>, columns <b>s &lt; ${t}</b>. No softmax, so scores
      are unnormalised — and RoPE lets them go negative.`;

    paint(this.el.write, tr.writes[t], N, D, this.wScale, 3, 2);
    paint(this.el.sigma, tr.snaps[t], N, D, this.gScale, 3, 2);
    let nz = 0; const g = tr.snaps[t];
    for (let i = 0; i < g.length; i++) if (g[i] !== 0) nz++;
    this.el.dens.textContent = `${(100 * nz / g.length).toFixed(0)}% non-zero`;
  }

  scoreHover(e) {
    const { tr } = this; if (!tr) return;
    const r = this.el.scores.getBoundingClientRect();
    const s = Math.floor(((e.clientX - r.left) / r.width) * tr.T);
    const t = Math.floor(((e.clientY - r.top) / r.height) * tr.T);
    if (s < 0 || t < 0 || s >= tr.T || t >= tr.T || s >= t) { this.el.tip.hidden = true; return; }
    const v = tr.scores[t * tr.T + s];
    const tip = this.el.tip;
    tip.hidden = false;
    tip.innerHTML = `<b>${this.labels[t].label}</b> attends to <b>${this.labels[s].label}</b>
      <span class="mr-num">${v >= 0 ? "+" : ""}${v.toFixed(2)}</span>`;
    tip.style.left = `${clamp(e.clientX - r.left + 12, 0, r.width - 170)}px`;
    tip.style.top = `${e.clientY - r.top + 14}px`;
  }
}

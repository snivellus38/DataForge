// PHASE 1 GATE — the walk pack, and the two things it deliberately does not ship.
//
// export_walk.py leaves out q_roped and sigma on the grounds that the browser can derive both.
// That is a claim about JS code, so it is checked HERE, against the model's own numbers, from
// the shipped bytes. If the RoPE lane order is ever flipped or the read/write order in the
// recurrence is swapped, this fails loudly instead of drawing a plausible wrong picture.
//
// It also re-derives every headline number from the payload rather than trusting the JSON:
// 5.13% x-sparsity, 0.94% y-sparsity, the negative-score share, and the G*/silence correlation.
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import {
  loadPack, sparseAt, rope, sigmaRun, deltaSigma, rowEnergy, attnMatrix,
  negShare, softmaxRow, gateAt, topBytes, silenceMCC, degreeBySilence, f16,
} from "../src/walkdata.js";
import { preflight, xPre, buildFrame, sigmaRows, castFor, COLUMNS } from "../src/flow-data.js";

const root = new URL("../..", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const read = (p) => readFileSync(root + p);
const json = (p) => JSON.parse(read(p, "utf8"));

const man = json("web/public/walk.json");
const gz = read("web/public/walk.bin.gz");
const buf = gunzipSync(gz);           // the page inflates with DecompressionStream; same bytes
const pack = loadPack(man, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const fix = json("web/test/walk_fixture.json");

const checks = [];
const ok = (name, cond, detail = "") => checks.push([name, !!cond, detail]);
const { N, D, H, L } = pack;
const W = man.walk[0], w = 0, LD = man.default_layer, HD = man.default_head;

// ── 1. quantisation round-trip against PRE-quantisation PyTorch values ────────────────────────
let worst = 0;
for (const s of fix.samples) {
  const sp = sparseAt(pack, `w${s.w}.L${s.L}.x`, s.t);
  if (!sp.n && s.x) { worst = Infinity; break; }
  const i = sp.ids.indexOf(s.neuron);
  if (i < 0) { worst = Infinity; break; }
  worst = Math.max(worst, Math.abs(sp.val[i] - s.x) / (s.scale / 255));
}
ok("x round-trips within 0.5 quantisation steps", worst <= 0.5,
   `worst ${worst.toFixed(3)} steps`);

const nSpot = fix.samples.filter((s) => s.w === 0).length;
ok("fixture actually addressed real tensors", nSpot > 0 && worst < Infinity);

// ── 2. THE RoPE CLAIM: derived JS-side must reproduce the shipped scores ──────────────────────
// scores_{t,s} = <rope(x)_t, rope(x)_s>. If our rotation is right, re-multiplying our own roped
// vectors reproduces the attention matrix the model actually ran. This is what licenses not
// shipping q_roped at all.
{
  const T = W.T;
  const M = attnMatrix(pack, w, LD, HD, T);
  const K = [];
  for (let t = 0; t < T; t++) K.push(rope(pack, sparseAt(pack, `w${w}.L${LD}.x`, t), t, HD));
  let maxAbs = 0, ref = 0;
  for (let r = 1; r < T; r++) {
    for (let c = 0; c < r; c++) {
      let s = 0;
      for (let n = 0; n < N; n++) s += K[r][n] * K[c][n];
      maxAbs = Math.max(maxAbs, Math.abs(s - M[r * T + c]));
      ref = Math.max(ref, Math.abs(M[r * T + c]));
    }
  }
  // ~1e-3 relative, and that floor is x's uint8 quantisation (1/255 per value, over a 3072-term
  // dot product), not the rotation. Python checked the rotation itself at 0.0 against the
  // model's q_roped. A loose tolerance only means something if it still discriminates, so:
  ok("JS RoPE reproduces the shipped attention scores", maxAbs / ref < 5e-3,
     `max err ${maxAbs.toExponential(2)} / peak ${ref.toFixed(2)} = ${(maxAbs / ref).toExponential(2)} rel`);

  // NEGATIVE CONTROL. Swapping the two rotation lanes is the exact bug the comment in
  // walkdata.js warns about, and it produces a picture that still looks like attention.
  // If this control ever passes, the test above has stopped testing anything.
  const flip = (v, t) => {
    const o = new Float32Array(N), TWO_PI = Math.PI * 2;
    const freqs = pack.t.get("rope_freqs").view;
    for (let n = 0; n < N; n += 2) {
      const ph = ((t * freqs[n]) % 1) * TWO_PI, c = Math.cos(ph), sn = Math.sin(ph);
      o[n] = v[n] * c + v[n + 1] * sn;          // lanes swapped on purpose
      o[n + 1] = v[n + 1] * c - v[n] * sn;
    }
    return o;
  };
  const raw = [];
  for (let t = 0; t < T; t++) {
    const sp = sparseAt(pack, `w${w}.L${LD}.x`, t);
    const d = new Float32Array(N);
    for (let i = 0; i < sp.ids.length; i++) {
      const g = sp.ids[i];
      if (g >= HD * N && g < (HD + 1) * N) d[g - HD * N] = sp.val[i];
    }
    raw.push(flip(d, t));
  }
  let badMax = 0;
  for (let r = 1; r < T; r++) {
    for (let c = 0; c < r; c++) {
      let s = 0;
      for (let n = 0; n < N; n++) s += raw[r][n] * raw[c][n];
      badMax = Math.max(badMax, Math.abs(s - M[r * T + c]));
    }
  }
  ok("...and the wrong rotation would have been caught", badMax / ref > 5e-2,
     `flipped lanes: ${(badMax / ref).toExponential(2)} rel, ${(badMax / maxAbs).toFixed(0)}x worse`);
}

// ── 3. THE SIGMA CLAIM: the recurrence reproduces the parallel read-out ───────────────────────
// CLAUDE.md item 17: this is a float32-scale number (~1e-6 relative). It is NOT comparable to
// the toy's 2.8e-14, which is float64.
{
  const { out } = sigmaRun(pack, w, LD, HD);
  const ref = f16(pack, `w${w}.L${LD}.H${HD}.a`);
  let maxErr = 0, sq = 0;
  for (let i = 0; i < ref.length; i++) {
    maxErr = Math.max(maxErr, Math.abs(out[i] - ref[i]));
    sq += ref[i] * ref[i];
  }
  const rms = Math.sqrt(sq / ref.length);
  const rel = maxErr / rms;
  // Again the floor is the payload, not the recurrence: x is uint8 and a_ast is f16, where
  // Python computing on the raw tensors got 3.35e-6. So pair it with a control.
  ok("sigma accumulated in JS reproduces a_ast", rel < 5e-2,
     `max err / rms = ${rel.toExponential(2)} (f16 payload; python f64 said ${W.sigma_residual_rel.toExponential(2)})`);

  // Correlation is scale-free and does not care about quantisation, so it is the sharper test
  // of whether we reproduced the right VECTOR rather than something of the right size.
  let sx = 0, sy = 0, sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < ref.length; i++) {
    sx += out[i]; sy += ref[i]; sxy += out[i] * ref[i];
    sxx += out[i] * out[i]; syy += ref[i] * ref[i];
  }
  const n = ref.length;
  const r = (n * sxy - sx * sy) / Math.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy));
  ok("...and it is the same vector, not merely the same magnitude", r > 0.999,
     `pearson r = ${r.toFixed(6)}`);

  // NEGATIVE CONTROL: writing before reading lets a token attend to itself. The model masks
  // with tril(-1), so this must be visibly wrong -- it is the classic off-by-one in a recurrence.
  {
    const v = f16(pack, `w${w}.L${LD}.v`);
    const bad = new Float32Array(ref.length), sig = new Float32Array(N * D);
    for (let t = 0; t < W.T; t++) {
      const k = rope(pack, sparseAt(pack, `w${w}.L${LD}.x`, t), t, HD);
      for (let q = 0; q < N; q++) {
        if (k[q] === 0) continue;
        for (let d = 0; d < D; d++) sig[q * D + d] += k[q] * v[t * D + d];   // write FIRST
      }
      for (let q = 0; q < N; q++) {
        if (k[q] === 0) continue;
        for (let d = 0; d < D; d++) bad[t * D + d] += k[q] * sig[q * D + d];
      }
    }
    let bm = 0;
    for (let i = 0; i < ref.length; i++) bm = Math.max(bm, Math.abs(bad[i] - ref[i]));
    ok("...and writing before reading would have been caught", bm / rms > 1,
       `self-attending: ${(bm / rms).toExponential(2)} vs ${rel.toExponential(2)}`);
  }
}

// ── 4. read-before-write: the mask is tril(-1), a token cannot attend to itself ───────────────
{
  const { out } = sigmaRun(pack, w, LD, HD, 1);
  let s = 0;
  for (let d = 0; d < D; d++) s += Math.abs(out[d]);
  ok("token 0 reads an empty state (strict-causal mask)", s === 0, `|a_0| = ${s}`);
}

// ── 5. headline numbers re-derived FROM THE BYTES, not read from the JSON ─────────────────────
{
  let xn = 0, yn = 0, cells = 0;
  for (let Lx = 0; Lx < L; Lx++) {
    const T = W.T;
    for (let t = 0; t < T; t++) {
      xn += sparseAt(pack, `w${w}.L${Lx}.x`, t).n;
      yn += sparseAt(pack, `w${w}.L${Lx}.y`, t).n;
      cells += N * H;
    }
  }
  const xf = xn / cells, yf = yn / cells;
  ok("x sparsity re-derives to ~5%", xf > 0.035 && xf < 0.075, `${(xf * 100).toFixed(2)}%`);
  ok("y sparsity re-derives to ~1%", yf > 0.004 && yf < 0.02, `${(yf * 100).toFixed(2)}%`);
  ok("the gate is several times sparser than x", xf / yf > 3,
     `${(xf / yf).toFixed(1)}x sparser`);
}

// ── 6. negative scores: real, and NOT one number ──────────────────────────────────────────────
{
  const a = negShare(pack, 0, LD, HD).frac;
  const b = negShare(pack, 1, LD, HD).frac;
  ok("causal scores really do go negative", a > 0 && b > 0,
     `w0 ${(a * 100).toFixed(1)}%  w1 ${(b * 100).toFixed(1)}%`);
  ok("negative share matches the exporter per sentence",
     Math.abs(a - man.walk[0].neg_share_default) < 1e-4 &&
     Math.abs(b - man.walk[1].neg_share_default) < 1e-4);
  // The short sentence is unrepresentative; the page must never quote one over a picture of the
  // other. This asserts the two really do differ, so the page is forced to say which is which.
  ok("the two sentences disagree enough to need labelling", Math.abs(a - b) > 0.1,
     `${(a * 100).toFixed(1)}% vs ${(b * 100).toFixed(1)}%`);
  const L0 = negShare(pack, 0, 0, HD).frac, L5 = negShare(pack, 0, 5, HD).frac;
  ok("interference is strongest in the first iteration", L0 > L5,
     `L0 ${(L0 * 100).toFixed(1)}% -> L5 ${(L5 * 100).toFixed(1)}%`);
}

// ── 7. softmax is the one that breaks it ──────────────────────────────────────────────────────
{
  const T = W.T, M = attnMatrix(pack, w, LD, HD, T), r = T - 1;
  const sm = softmaxRow(M, T, r);
  let s = 0, anyNeg = false;
  for (let c = 0; c < r; c++) { s += sm[c]; if (M[r * T + c] < 0) anyNeg = true; }
  ok("softmax row normalises to 1 (which raw scores never do)", Math.abs(s - 1) < 1e-5);
  let raw = 0;
  for (let c = 0; c < r; c++) raw += M[r * T + c];
  ok("raw scores are not normalised -- that is what buys constant memory",
     Math.abs(raw - 1) > 0.1, `row sum ${raw.toFixed(3)}`);
  ok("and the raw row contains negatives softmax could never produce", anyNeg);
}

// ── 8. structure predicts silence, computed from the shipped bytes ────────────────────────────
{
  const fman = json("web/public/big/field.json");
  const fbuf = read("web/public/big/field.bin");
  const fab = fbuf.buffer.slice(fbuf.byteOffset, fbuf.byteOffset + fbuf.byteLength);
  const tOf = (n) => {
    const e = fman.tensors.find((x) => x.name === n);
    return new Uint16Array(fab, e.offset, e.bytes / 2);
  };
  const outDeg = new Uint16Array(N * H), inDeg = new Uint16Array(N * H);
  for (let h = 0; h < H; h++) {
    outDeg.set(tOf(`h${h}.out_deg`), h * N);
    inDeg.set(tOf(`h${h}.in_deg`), h * N);
  }
  const fire = pack.t.get("fire_count").view;
  const m = silenceMCC(fire, outDeg, inDeg);
  const dg = degreeBySilence(fire, outDeg, inDeg);
  ok("G* predicts silence at high MCC", m.mcc > 0.85,
     `MCC ${m.mcc.toFixed(3)}  P(silent|isolated) ${(m.pSilentGivenIsolated * 100).toFixed(1)}% ` +
     `vs base rate ${(m.silentFrac * 100).toFixed(1)}%`);
  ok("silent neurons are the poorly connected ones", dg.firing > dg.silent * 10,
     `mean degree ${dg.silent.toFixed(1)} silent vs ${dg.firing.toFixed(1)} firing`);
  ok("about a third of the model never fires on this corpus",
     m.silentFrac > 0.25 && m.silentFrac < 0.5, `${(m.silentFrac * 100).toFixed(1)}%`);
}

// ── 9. the remaining decoders actually decode ─────────────────────────────────────────────────
{
  // x_pre is NOT shipped: it is v @ dx_head. That is a claim about JS arithmetic against a
  // matrix, so check it the way q_roped and sigma are checked -- against the model's own answer.
  // The ReLU's survivors ARE the positive entries of x_pre by definition, and `x` ships exactly
  // that support, so the support is the reference and nothing here is self-confirming.
  preflight(pack);
  const xp = xPre(pack, f16(pack, `w${w}.L${LD}.v`), 3 * D);
  const sup = new Set();
  for (const g of sparseAt(pack, `w${w}.L${LD}.x`, 3).ids) if (g < N) sup.add(g);
  let neg = 0, agree = 0;
  for (let i = 0; i < N; i++) {
    if (xp[i] < 0) neg++;
    if ((xp[i] > 0) === sup.has(i)) agree++;
  }
  ok("x_pre is mostly negative -- that is what the ReLU removes", neg / N > 0.8,
     `${((neg / N) * 100).toFixed(1)}% below zero`);
  ok("derived x_pre agrees with the ReLU support the model actually produced",
     agree / N > 0.999, `${agree} of ${N} neurons agree on sign`);
  // and a NEGATIVE control: the same derivation with the matrix transposed must not agree, or
  // the check above is passing on something other than the arithmetic it claims to test
  {
    const bad = new Float32Array(N);
    const v = f16(pack, `w${w}.L${LD}.v`);
    for (let d = 0; d < D; d++) for (let n = 0; n < N; n++) bad[n] += v[3 * D + d] * pack.flowDx[n * D + d];
    let bagree = 0;
    for (let i = 0; i < N; i++) if ((bad[i] > 0) === sup.has(i)) bagree++;
    ok("...and a mis-indexed decoder_x does NOT agree, so that check discriminates",
       bagree / N < 0.9, `${((bagree / N) * 100).toFixed(1)}% with the wrong stride`);
  }

  const g = gateAt(pack, w, LD, 10);
  ok("the gate is an AND: survivors are a subset of what fired",
     g.survived <= g.fired && g.survived > 0,
     `${g.fired} fired -> ${g.survived} survived (${(g.survival * 100).toFixed(0)}%)`);

  const top = topBytes(pack, w, 5, 8);
  const sum = top.reduce((s, x) => s + x.p, 0);
  ok("next-byte distribution decodes and is a distribution",
     top.length === 8 && sum > 0 && sum <= 1.0001,
     `top byte ${top[0].byte} at p=${top[0].p.toFixed(3)}`);

  const { sigma } = sigmaRun(pack, w, LD, HD, 12);
  const e = rowEnergy(sigma, N, D);
  let live = 0;
  for (let i = 0; i < N; i++) if (e[i] > 0) live++;
  ok("sigma row energy is populated but not everywhere", live > 0 && live < N,
     `${live} of ${N} rows carry energy after 12 tokens`);

  const ds = deltaSigma(pack, w, LD, HD, 7);
  let nz = 0;
  for (let i = 0; i < N; i++) if (ds.k[i] !== 0) nz++;
  ok("a single write is rank-one and sparse in the neuron axis", nz > 0 && nz < N / 2,
     `${nz} of ${N} rows touched by one token's write`);
}

// ── 10. the flow diagram's model: every edge is a real contribution ──────────────────────────
//
// The picture on stop 0 is only worth drawing if the lines mean something, so the arithmetic
// behind them is checked here rather than assumed by the smoke gate, which can only see that
// curves were painted.
{
  ok("the pack carries the flow pool and its weight slices",
     !!pack.t.get("flow.ids") && !!pack.t.get("flow.dx") && !!pack.t.get("flow.dy") &&
     !!pack.t.get("flow.enc") && !!pack.t.get("lm_head") && !!pack.t.get("dx_head"));

  const sig = sigmaRows(pack, w, LD, HD);
  for (let t = 0; t < 12; t++) sig.advance(t);
  const f = buildFrame(pack, { w, layer: LD, head: HD, t: 12, sig });

  ok("every column has nodes", COLUMNS.every((c) => f.nodes[c.id].length > 0),
     COLUMNS.map((c) => `${c.id}:${f.nodes[c.id].length}`).join(" "));
  ok("the 192-wide columns are drawn in FULL, not sampled",
     f.nodes.v.length === D && f.nodes.a.length === D && f.nodes.dv.length === D);
  ok("edges exist between every adjacent pair of columns",
     new Set(f.edges.map((e) => `${e.from}->${e.to}`)).size === COLUMNS.length - 1,
     [...new Set(f.edges.map((e) => `${e.from}->${e.to}`))].join(" "));
  ok("edges are SIGNED -- both directions of contribution are present",
     f.edges.some((e) => e.w > 0) && f.edges.some((e) => e.w < 0),
     `${f.edges.filter((e) => e.w < 0).length} negative of ${f.edges.length}`);

  // v* -> x_pre edges must literally be v[d] * decoder_x[d, n]
  {
    const v = f16(pack, `w${w}.L${LD}.v`);
    let worst = 0;
    for (const e of f.edges.filter((x) => x.from === "v" && x.to === "xpre")) {
      const nd = f.nodes.xpre[e.ti];
      const local = nd.i - HD * N;
      if (local < 0 || local >= N) continue;
      const want = v[12 * D + e.si] * pack.flowDx[e.si * N + local];
      worst = Math.max(worst, Math.abs(want - e.w));
    }
    ok("v* -> x_pre edge weights ARE weight x activation", worst < 1e-5,
       `max deviation ${worst.toExponential(1)}`);
  }

  // y -> dv edges must be y[g] * encoder[g, d], and their sum must approach the shipped dv
  {
    const dv = f16(pack, `w${w}.L${LD}.dv`);
    let worst = 0;
    for (const e of f.edges.filter((x) => x.from === "y" && x.to === "dv")) {
      const nd = f.nodes.y[e.si];
      const want = nd.value * pack.flowEnc[nd.pool * D + e.ti];
      worst = Math.max(worst, Math.abs(want - e.w));
    }
    ok("y -> residual edge weights ARE weight x activation", worst < 1e-4,
       `max deviation ${worst.toExponential(1)}`);
    let mx = 0;
    for (let d = 0; d < D; d++) mx = Math.max(mx, Math.abs(dv[12 * D + d]));
    ok("the residual update column carries the model's own dv", mx > 0,
       `peak |dv| ${mx.toFixed(3)}`);
  }

  ok("the drawn cast is a strict, disclosed sample of 12,288",
     f.stats.drawn > 0 && f.stats.drawn < 64 && f.stats.drawn < f.stats.xFiring,
     `${f.stats.drawn} drawn, ${f.stats.xFiring} firing`);
  ok("...and the update it builds is a fraction of the real one, not all of it",
     f.stats.dvRatio > 0 && f.stats.dvRatio < 1, `${(f.stats.dvRatio * 100).toFixed(1)}%`);
  // The gate kills ~98% of what fires, so a cast chosen by x alone leaves the last third of the
  // diagram empty on most tokens. Reserved slots for gate survivors are what fixes that, and the
  // rule only works if the pool actually contains survivors to reserve them for.
  ok("the cast reserves slots for neurons that got PAST the gate",
     f.cast.some((c) => c.why === "gate"), f.cast.filter((c) => c.why === "gate").length + " of them");
  ok("...so the gate column is not empty", f.stats.ySurvivors > 0,
     `${f.stats.ySurvivors} survivors drawn of ${f.stats.yFiring}`);
  {
    let empty = 0, steps = 0;
    for (const wk of man.walk) {
      for (const Lx of [0, LD, L - 1]) {
        const sg = sigmaRows(pack, wk.i, Lx, HD);
        for (let t = 0; t < wk.T; t++) {
          if (buildFrame(pack, { w: wk.i, layer: Lx, head: HD, t, sig: sg }).stats.ySurvivors === 0) empty++;
          steps++;
          sg.advance(t);
        }
      }
    }
    ok("the gate column has something to draw on almost every step", empty / steps < 0.05,
       `${empty} empty of ${steps} steps (${((empty / steps) * 100).toFixed(1)}%)`);
  }
  ok("the pool's coverage of true per-token leaders is measured, not asserted",
     man.flow.leader_coverage > 0.5 && man.flow.leader_coverage < 1,
     `${(man.flow.leader_coverage * 100).toFixed(1)}% of top-${man.flow.leader_k}`);

  // the stable cast must NOT change with the token; the visitors must
  const c0 = castFor(pack, new Map());
  ok("the stable cast is token-independent -- a dot keeps its row",
     c0.filter((c) => c.stable).map((c) => c.id).join(",") ===
     f.cast.filter((c) => c.stable).map((c) => c.id).join(","));
  const f2 = buildFrame(pack, { w, layer: LD, head: HD, t: 30, sig });
  ok("...while the token's leaders do change",
     f2.cast.filter((c) => !c.stable).map((c) => c.id).join(",") !==
     f.cast.filter((c) => !c.stable).map((c) => c.id).join(","));

  // every walk sentence, every iteration -- the coverage this whole change is about
  let built = 0;
  for (const wk of man.walk) {
    for (let Lx = 0; Lx < L; Lx++) {
      const sg = sigmaRows(pack, wk.i, Lx, HD);
      sg.advance(0);
      const ff = buildFrame(pack, { w: wk.i, layer: Lx, head: HD, t: 1, sig: sg });
      if (ff.edges.length > 20 && ff.nodes.x.length > 0) built++;
    }
  }
  // The reason a click used to light almost nothing: the cast came from all four heads, but
  // x_pre is derived from decoder_x[head] and sigma is accumulated for one head, so a neuron from
  // any other head had no pre-activation and no state row -- a dot with nothing entering or
  // leaving it. The pool is head-restricted now, and this is what says so.
  {
    const head = man.flow.head;
    ok("the flow pool is confined to the traced head",
       [...pack.t.get("flow.ids").view].every((g) => Math.floor(g / N) === head),
       `head ${head}`);
    ok("...so every drawn neuron has a real pre-activation",
       f.nodes.xpre.every((n) => n.inHead && Number.isFinite(n.value)),
       `${f.nodes.xpre.filter((n) => !n.inHead).length} outside the head`);

    // and a path can actually be walked from a neuron to both ends of the diagram
    const key = (c, i) => `${c}:${i}`;
    const out = new Map(), inn = new Map();
    const add = (m, k, e) => { const a = m.get(k); if (a) a.push(e); else m.set(k, [e]); };
    for (const e of f.edges) { add(out, key(e.from, e.si), e); add(inn, key(e.to, e.ti), e); }
    const reach = (c, i, m, step) => {
      const seen = new Set(), E = new Set();
      let front = [key(c, i)];
      while (front.length) {
        const nx = [];
        for (const k of front) for (const e of m.get(k) || []) {
          E.add(e);
          const t = step(e);
          if (!seen.has(t)) { seen.add(t); nx.push(t); }
        }
        front = nx;
      }
      return E;
    };
    let traced = 0, best = 0;
    for (let i = 0; i < f.nodes.x.length; i++) {
      const fwd = reach("x", i, out, (e) => key(e.to, e.ti));
      const back = reach("x", i, inn, (e) => key(e.from, e.si));
      const total = new Set([...fwd, ...back]).size;
      best = Math.max(best, total);
      if (total > 4) traced++;
    }
    ok("a click on a drawn neuron traces a real path, not one or two stubs",
       traced >= f.nodes.x.length * 0.5 && best > 20,
       `${traced} of ${f.nodes.x.length} neurons reach past their immediate edges; best ${best} edges`);
  }

  ok("the flow diagram builds for EVERY sentence at EVERY iteration",
     built === man.walk.length * L, `${built} of ${man.walk.length * L}`);
}

// ── report ────────────────────────────────────────────────────────────────────────────────────
let bad = 0;
for (const [n, good, d] of checks) {
  if (!good) bad++;
  console.log(`  ${good ? "ok  " : "FAIL"}  ${n}${d ? `  —  ${d}` : ""}`);
}
console.log(`\npayload ${(gz.byteLength / 1e6).toFixed(2)} MB shipped from ${(buf.byteLength / 1e6).toFixed(2)} MB raw (gzip ${(gz.byteLength / buf.byteLength).toFixed(2)}), ${man.tensors.length} tensors`);
if (bad) { console.log(`\nWALK GATE FAILED (${bad})`); process.exit(1); }
console.log("WALK GATE PASSED");

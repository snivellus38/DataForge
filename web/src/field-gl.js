// THE FIELD: 12,288 neurons of the 8M BDH, drawn as a living graph.
//
// WebGL2, hand-rolled, no dependency. 12,288 gl.POINTS plus a few thousand additive lines is
// trivial work for a GPU, so the cost of a library would buy nothing; the cost of a CDN import
// would be a hard external dependency in an artifact that must run from a static folder.
//
// THIS FILE IS THE THIN PART ON PURPOSE. Everything that can actually be wrong -- decoding the
// pack, indexing a token's active set, the decay recurrence, hit-testing, colour ramps -- lives
// in field-data.js and bigdata.js as pure functions, and is covered by web/test/field.mjs under
// Node where there is no GPU at all. What is left here is buffer plumbing and two shaders.
// (CLAUDE.md item 15: a suite that never executes the real path is not testing the product.)
//
// createField() returns { ok:false } instead of throwing when WebGL2 is unavailable -- jsdom,
// forced software-rendering, ancient browsers. Callers render a static fallback in that case;
// the page must never be blank because a GPU context was refused.

import { rampStops } from "./field-data.js";

const VERT = `#version 300 es
in vec2 a_pos;
in float a_act;
in vec3 a_tint;
uniform vec2  u_scale, u_off;
uniform float u_size, u_dpr;
out float v_act;
out vec3 v_tint;
void main() {
  v_tint = a_tint;
  gl_Position  = vec4(a_pos * u_scale + u_off, 0.0, 1.0);
  // Silent neurons stay visible but small, so the shape of the model is always on screen and
  // firing reads as a change in brightness rather than as points appearing from nothing.
  gl_PointSize = u_dpr * u_size * (0.55 + 1.85 * a_act);
  v_act = a_act;
}`;

const FRAG = `#version 300 es
precision highp float;
in float v_act;
in vec3 v_tint;
uniform vec3 u_c0, u_c1, u_c2, u_c3;
uniform float u_gain, u_idle, u_useTint;
out vec4 frag;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  float r2 = dot(d, d) * 4.0;
  if (r2 > 1.0) discard;              // round points, not squares
  float fall = 1.0 - r2; fall *= fall;
  float a = clamp(v_act, 0.0, 1.0);
  vec3 col = a < 0.34 ? mix(u_c0, u_c1, a / 0.34)
           : a < 0.67 ? mix(u_c1, u_c2, (a - 0.34) / 0.33)
                      : mix(u_c2, u_c3, (a - 0.67) / 0.33);
  // Categorical modes (graph community) carry an explicit per-node hue; sequential modes leave
  // u_useTint at 0 and read the ramp. One shader, both colour jobs, no second draw path.
  col = mix(col, v_tint * (0.30 + 0.70 * a), u_useTint);
  float alpha = fall * (u_idle + (1.0 - u_idle) * a) * u_gain;
  frag = vec4(col * alpha, alpha);    // premultiplied; blend is additive
}`;

// Additive blending is HDR: where many neurons fire close together the buffer sums well past 1.0.
// Clipping that turns the densest, most interesting cluster into a flat white blob. So points are
// drawn into a float target and tone-mapped with 1-exp(-x*e), which compresses highlights and
// keeps the cluster's shape. Needs EXT_color_buffer_float; without it we fall back to drawing
// straight to the canvas at reduced gain, which clips but still renders.
const TVERT = `#version 300 es
out vec2 v_uv;
void main() {
  // fullscreen triangle from gl_VertexID -- no vertex buffer needed
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  v_uv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const TFRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_tex;
uniform vec3 u_surface;
uniform float u_exposure;
out vec4 frag;
void main() {
  vec3 hdr = texture(u_tex, v_uv).rgb;
  vec3 c = u_surface + (1.0 - exp(-hdr * u_exposure));
  frag = vec4(pow(clamp(c, 0.0, 1.0), vec3(1.0 / 1.05)), 1.0);
}`;

const EVERT = `#version 300 es
in vec2 a_pos;
in float a_w;
uniform vec2 u_scale, u_off;
out float v_w;
void main() { gl_Position = vec4(a_pos * u_scale + u_off, 0.0, 1.0); v_w = a_w; }`;

const EFRAG = `#version 300 es
precision highp float;
in float v_w;
uniform vec3 u_pos, u_neg;
uniform float u_alpha;
out vec4 frag;
void main() {
  // Sign is the encoding, not decoration: G* entries are signed and the negative ones are the
  // destructive-interference half of the mechanism.
  vec3 c = v_w >= 0.0 ? u_pos : u_neg;
  float a = clamp(abs(v_w), 0.0, 1.0) * u_alpha;
  frag = vec4(c * a, a);
}`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    throw new Error("field-gl shader: " + gl.getShaderInfoLog(s));
  }
  return s;
}

function program(gl, vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error("field-gl link: " + gl.getProgramInfoLog(p));
  }
  return p;
}

/**
 * @param canvas   HTMLCanvasElement
 * @param data     { xy: Float32Array (2n), edges?: {i,j,w} }
 * @param opts     { surface, ramp, pointSize }
 */
export function createField(canvas, data, opts = {}) {
  const gl = canvas.getContext("webgl2", {
    alpha: false, antialias: false, premultipliedAlpha: true, powerPreference: "high-performance",
  });
  if (!gl) return { ok: false, reason: "no webgl2" };

  const n = data.xy.length / 2;
  const act = new Float32Array(n);
  const prog = program(gl, VERT, FRAG);
  const eprog = data.edges ? program(gl, EVERT, EFRAG) : null;

  // ── static geometry ────────────────────────────────────────────────────────────────────────
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const posBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
  gl.bufferData(gl.ARRAY_BUFFER, data.xy, gl.STATIC_DRAW);
  const aPos = gl.getAttribLocation(prog, "a_pos");
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

  const tint = new Float32Array(n * 3).fill(1);
  const tintBuf = gl.createBuffer();

  const actBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, actBuf);
  gl.bufferData(gl.ARRAY_BUFFER, act, gl.DYNAMIC_DRAW);
  const aAct = gl.getAttribLocation(prog, "a_act");
  gl.enableVertexAttribArray(aAct);
  gl.vertexAttribPointer(aAct, 1, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, tintBuf);
  gl.bufferData(gl.ARRAY_BUFFER, tint, gl.DYNAMIC_DRAW);
  const aTint = gl.getAttribLocation(prog, "a_tint");
  gl.enableVertexAttribArray(aTint);
  gl.vertexAttribPointer(aTint, 3, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);

  // ── edges: rebuilt on selection, so DYNAMIC and sized for the worst case ────────────────────
  let evao = null, eposBuf = null, ewBuf = null, eCount = 0;
  if (eprog) {
    evao = gl.createVertexArray();
    gl.bindVertexArray(evao);
    eposBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, eposBuf);
    const eAP = gl.getAttribLocation(eprog, "a_pos");
    gl.enableVertexAttribArray(eAP);
    gl.vertexAttribPointer(eAP, 2, gl.FLOAT, false, 0, 0);
    ewBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, ewBuf);
    const eAW = gl.getAttribLocation(eprog, "a_w");
    gl.enableVertexAttribArray(eAW);
    gl.vertexAttribPointer(eAW, 1, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
  }

  const u = (p, nm) => gl.getUniformLocation(p, nm);
  const U = {
    scale: u(prog, "u_scale"), off: u(prog, "u_off"), size: u(prog, "u_size"),
    dpr: u(prog, "u_dpr"), gain: u(prog, "u_gain"), idle: u(prog, "u_idle"),
    useTint: u(prog, "u_useTint"),
    c: [u(prog, "u_c0"), u(prog, "u_c1"), u(prog, "u_c2"), u(prog, "u_c3")],
  };
  const EU = eprog ? {
    scale: u(eprog, "u_scale"), off: u(eprog, "u_off"),
    pos: u(eprog, "u_pos"), neg: u(eprog, "u_neg"), alpha: u(eprog, "u_alpha"),
  } : null;

  const hdrOK = !!gl.getExtension("EXT_color_buffer_float");
  const tprog = hdrOK ? program(gl, TVERT, TFRAG) : null;
  const TU = tprog ? {
    tex: u(tprog, "u_tex"), surface: u(tprog, "u_surface"), exposure: u(tprog, "u_exposure"),
  } : null;
  const tvao = tprog ? gl.createVertexArray() : null;   // fullscreen triangle needs no attributes
  let fbo = null, tex = null, fbW = 0, fbH = 0;

  function ensureTarget(w, h) {
    if (!tprog || (fbo && fbW === w && fbH === h)) return;
    if (tex) { gl.deleteTexture(tex); gl.deleteFramebuffer(fbo); }
    tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    fbW = w; fbH = h;
  }

  const surface = hexRGB(opts.surface || "#100f0e");
  let stops = rampStops(opts.ramp);
  let view = { zoom: 1, cx: 0, cy: 0 };
  let pointSize = opts.pointSize || 3.2;
  let gain = 1;
  let useTint = 0;
  let idle = opts.idle ?? 0.26;
  let exposure = opts.exposure ?? 1.35;
  let edgeColors = { pos: hexRGB("#e0872a"), neg: hexRGB("#3987e5") };

  // model-space bounds, so the view maths is independent of the export's coordinate range
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = data.xy[i * 2], y = data.xy[i * 2 + 1];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    if (x < x0) x0 = x; if (x > x1) x1 = x;
    if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
  const span = Math.max(x1 - x0, y1 - y0) || 1;

  function transform() {
    const w = canvas.width || 1, h = canvas.height || 1;
    const asp = w / h;
    // fit the model box into clip space with a margin, preserving aspect
    const s = (1.8 / span) * view.zoom;
    return {
      sx: s / Math.max(1, asp), sy: s * Math.min(1, asp),
      ox: -(mx + view.cx) * (s / Math.max(1, asp)),
      oy: -(my + view.cy) * (s * Math.min(1, asp)),
    };
  }

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    return dpr;
  }

  function drawPoints(t, dpr, size, data) {
    gl.useProgram(prog);
    gl.uniform2f(U.scale, t.sx, t.sy);
    gl.uniform2f(U.off, t.ox, t.oy);
    gl.uniform1f(U.size, size);
    gl.uniform1f(U.dpr, dpr);
    gl.uniform1f(U.gain, fbo ? gain : gain * 0.72);   // clipping path needs headroom
    gl.uniform1f(U.idle, idle);
    gl.uniform1f(U.useTint, useTint);
    for (let i = 0; i < 4; i++) gl.uniform3fv(U.c[i], stops[i]);
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, actBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);
    gl.drawArrays(gl.POINTS, 0, n);
    gl.bindVertexArray(null);
  }

  function tonemap() {
    if (!fbo) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.disable(gl.SCISSOR_TEST);
    gl.disable(gl.BLEND);
    gl.useProgram(tprog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(TU.tex, 0);
    gl.uniform3fv(TU.surface, surface);
    gl.uniform1f(TU.exposure, exposure);
    gl.bindVertexArray(tvao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
  }

  function render() {
    const dpr = resize();
    const t = transform();
    ensureTarget(canvas.width, canvas.height);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);          // null when HDR is unavailable
    gl.viewport(0, 0, canvas.width, canvas.height);
    // The surface colour is added by the tone-map pass, so the HDR target starts at zero.
    if (fbo) gl.clearColor(0, 0, 0, 1);
    else gl.clearColor(surface[0], surface[1], surface[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);            // additive: overlapping glow accumulates

    if (eprog && eCount) {
      gl.useProgram(eprog);
      gl.uniform2f(EU.scale, t.sx, t.sy);
      gl.uniform2f(EU.off, t.ox, t.oy);
      gl.uniform3fv(EU.pos, edgeColors.pos);
      gl.uniform3fv(EU.neg, edgeColors.neg);
      gl.uniform1f(EU.alpha, 0.85);
      gl.bindVertexArray(evao);
      gl.drawArrays(gl.LINES, 0, eCount * 2);
    }

    drawPoints(t, dpr, pointSize, act);

    tonemap();
  }

  /**
   * Small multiples: the SAME 12,288 neurons drawn several times over, each tile with its own
   * activation. This is the picture weight tying earns -- n_layer is an iteration count of one
   * shared operator, not depth (CLAUDE.md item 6), so every tile is literally the same neurons
   * re-firing rather than a different layer's parameters.
   *
   * Scissor matters: gl.POINTS are clipped in NDC, but a point's SIZE can still spill past the
   * viewport edge, so without a scissor rect neurons from one tile bleed into its neighbour and
   * the small multiples quietly stop being comparable.
   *
   * @param tiles [{ x, y, w, h, act }] in CSS pixels, origin top-left
   */
  function renderTiles(tiles) {
    const dpr = resize();
    ensureTarget(canvas.width, canvas.height);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.disable(gl.SCISSOR_TEST);
    if (fbo) gl.clearColor(0, 0, 0, 1);
    else gl.clearColor(surface[0], surface[1], surface[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.enable(gl.SCISSOR_TEST);

    for (const tl of tiles) {
      const x = Math.round(tl.x * dpr), w = Math.round(tl.w * dpr);
      const h = Math.round(tl.h * dpr);
      const y = Math.round(canvas.height - (tl.y + tl.h) * dpr);   // GL origin is bottom-left
      gl.viewport(x, y, w, h);
      gl.scissor(x, y, w, h);
      const asp = w / h;
      const s = (1.8 / span) * view.zoom;
      const sx = s / Math.max(1, asp), sy = s * Math.min(1, asp);
      drawPoints({ sx, sy, ox: -(mx + view.cx) * sx, oy: -(my + view.cy) * sy },
                 dpr, pointSize * Math.min(1, Math.sqrt(w / canvas.width) * 1.6), tl.act);
    }
    gl.disable(gl.SCISSOR_TEST);
    tonemap();
  }

  return {
    ok: true,
    n,
    activation: act,                       // callers write into this directly, then render()
    render,
    renderTiles,
    setRamp(r) { stops = rampStops(r); useTint = 0; },
    /** Per-node rgb in 0..1, length 3n. Switches the field to categorical colouring. */
    setTint(rgb) {
      tint.set(rgb);
      useTint = 1;
      gl.bindBuffer(gl.ARRAY_BUFFER, tintBuf);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, tint);
    },
    setGain(g) { gain = g; },
    setIdle(v) { idle = v; },
    setExposure(e) { exposure = e; },
    hdr: hdrOK,
    setPointSize(s) { pointSize = s; },
    setView(v) { view = { ...view, ...v }; },
    getView: () => ({ ...view }),
    bounds: () => ({ x0, x1, y0, y1, mx, my, span }),
    /** clip-space -> model-space, for hit testing against a pointer position */
    toModel(px, py) {
      const r = canvas.getBoundingClientRect();
      const t = transform();
      const cx = ((px - r.left) / r.width) * 2 - 1;
      const cy = 1 - ((py - r.top) / r.height) * 2;
      return [(cx - t.ox) / t.sx, (cy - t.oy) / t.sy];
    },
    /** @param e {i:Uint16Array,j:Uint16Array,w:Float32Array} already filtered by the caller */
    setEdges(e) {
      if (!eprog) return;
      eCount = e ? e.i.length : 0;
      if (!eCount) return;
      const pts = new Float32Array(eCount * 4), ws = new Float32Array(eCount * 2);
      for (let k = 0; k < eCount; k++) {
        pts[k * 4] = data.xy[e.i[k] * 2];     pts[k * 4 + 1] = data.xy[e.i[k] * 2 + 1];
        pts[k * 4 + 2] = data.xy[e.j[k] * 2]; pts[k * 4 + 3] = data.xy[e.j[k] * 2 + 1];
        ws[k * 2] = ws[k * 2 + 1] = e.w[k];
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, eposBuf);
      gl.bufferData(gl.ARRAY_BUFFER, pts, gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, ewBuf);
      gl.bufferData(gl.ARRAY_BUFFER, ws, gl.DYNAMIC_DRAW);
    },
    setEdgeColors(pos, neg) { edgeColors = { pos: hexRGB(pos), neg: hexRGB(neg) }; },
    destroy() {
      gl.deleteBuffer(posBuf); gl.deleteBuffer(actBuf); gl.deleteBuffer(tintBuf);
      gl.deleteVertexArray(vao);
      if (eprog) { gl.deleteBuffer(eposBuf); gl.deleteBuffer(ewBuf); gl.deleteVertexArray(evao); }
      if (tex) { gl.deleteTexture(tex); gl.deleteFramebuffer(fbo); }
    },
  };
}

export function hexRGB(h) {
  return new Float32Array([1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255));
}

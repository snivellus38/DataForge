// SMOKE TEST for field.html: boot the instrument page for real, on BOTH render paths.
//
// Same lesson as web/test/smoke.mjs (CLAUDE.md item 15): field.mjs tests pure functions and would
// stay green while the page threw on load and rendered nothing. So this executes inspector.js
// against the real HTML, the real manifests and the real binaries.
//
// Two passes, because both happen to real viewers:
//   1. NO WEBGL   - getContext returns null. Must fall back to numbers, never a blank screen.
//   2. FAKE WEBGL - a recording stub. Cannot validate shader semantics (no GPU compiles them),
//                   but it does execute every buffer/uniform/draw path in field-gl.js and the
//                   whole playback loop, which is where structural bugs actually live.
import { readFileSync } from "node:fs";
import { JSDOM, VirtualConsole } from "jsdom";

const root = new URL("../..", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const read = (p) => readFileSync(root + p);

let fail = 0;
const check = (ok, msg) => { if (!ok) { fail++; console.log(`  FAIL ${msg}`); } };

/**
 * The [hidden] cascade check, done on the CSS text rather than through jsdom.
 *
 * This bug shipped: `.fallback { display: grid }` is an author rule and beats the UA stylesheet's
 * [hidden] { display: none }, so the "No WebGL2" overlay covered the live field permanently. The
 * `hidden` ATTRIBUTE was correct the whole time, so an attribute assertion passed happily.
 *
 * jsdom cannot catch it either -- its getComputedStyle honours `hidden` regardless of author CSS,
 * so it reports display:none even with the bug present (verified by reintroducing it). So this
 * reads the stylesheet directly: if any element in the HTML carries `hidden`, and any CSS rule
 * matching one of that element's classes sets `display`, then the [hidden] reset MUST exist.
 */
// Classes whose `hidden` attribute is set by JS at runtime, so the source scan below cannot see
// them. .hud-body is how the side panels collapse: a `display` rule matching it would beat the
// [hidden] reset and leave a "collapsed" panel fully visible while its handle claimed otherwise.
// jsdom's getComputedStyle honours `hidden` regardless of author CSS, so only this static read of
// the stylesheet can catch that -- the runtime assertion cannot.
const RUNTIME_HIDDEN = ["hud-body"];

function checkHiddenCascade(html, css) {
  css = css.replace(/\/\*[\s\S]*?\*\//g, "");   // comments are not rules
  const hiddenClasses = new Set(RUNTIME_HIDDEN);
  for (const tag of html.match(/<[a-z]+[^>]*\shidden[\s>]/gi) || []) {
    for (const c of (tag.match(/class="([^"]+)"/i)?.[1] || "").split(/\s+/)) if (c) hiddenClasses.add(c);
  }
  const risky = [], forced = [];
  for (const rule of css.split("}")) {
    const [sel, body = ""] = rule.split("{");
    const decl = body.match(/(?:^|[;\s])display\s*:[^;]*/);
    if (!decl) continue;
    for (const c of hiddenClasses) {
      if (new RegExp(`\\.${c}\\b`).test(sel) && !/:not\(\[hidden\]\)/.test(sel)) {
        risky.push(`${sel.trim()} { display: ... }`);
        // The reset saves you from a plain rule. It does NOT save you from an !important one on a
        // more specific selector -- `.hud.is-collapsed .hud-body { display: block !important }`
        // beats `[hidden] { display: none !important }` outright, and no cascade order fixes
        // that. So it is a hard failure, not a "did you remember the reset" warning.
        if (/!important/.test(decl[0])) forced.push(`${sel.trim()} {${decl[0].trim()} }`);
      }
    }
  }
  const hasReset = /\[hidden\]\s*\{[^}]*display\s*:\s*none\s*!important/.test(css);
  return { hiddenClasses: [...hiddenClasses], risky, forced, hasReset };
}

/** A WebGL2 context that records draw calls. Unknown methods no-op; unknown constants are ints. */
function fakeGL(calls) {
  const obj = {
    getShaderParameter: () => true, getProgramParameter: () => true,
    getShaderInfoLog: () => "", getProgramInfoLog: () => "",
    createShader: () => ({}), createProgram: () => ({}), createBuffer: () => ({}),
    createVertexArray: () => ({}), createTexture: () => ({}), createFramebuffer: () => ({}),
    getAttribLocation: () => 0, getUniformLocation: () => ({}),
    getExtension: (n) => (n === "EXT_color_buffer_float" ? {} : null),
    drawArrays: (mode, first, count) => calls.push({ mode, count }),
  };
  return new Proxy(obj, {
    get(t, k) {
      if (k in t) return t[k];
      if (typeof k === "string" && /^[A-Z0-9_]+$/.test(k)) return 1;   // GL constant
      return () => {};                                                 // GL method
    },
  });
}

async function boot({ gl }) {
  const errors = [];
  const vc = new VirtualConsole();
  vc.on("jsdomError", (e) => errors.push(e.message + "\n" + (e.detail?.stack || "")));
  vc.on("error", (m) => errors.push(String(m)));

  const dom = new JSDOM(read("web/field.html"), {
    url: "http://localhost/", runScripts: "dangerously", pretendToBeVisual: true, virtualConsole: vc,
  });
  const { window } = dom;
  const calls = [];

  // Inject the real stylesheet. jsdom does not fetch <link>, and without the CSS in the document
  // getComputedStyle cannot see the cascade -- which is precisely how the overlay bug below got
  // through: `.fallback{display:grid}` is an author rule and beats the UA's [hidden]{display:none},
  // so the fallback covered the live field permanently while the `hidden` ATTRIBUTE looked right.
  const style = window.document.createElement("style");
  style.textContent = read("web/src/field.css").toString("utf8");
  window.document.head.append(style);

  window.HTMLCanvasElement.prototype.getContext = gl ? () => fakeGL(calls) : () => null;
  for (const [k, v] of [["clientWidth", 1200], ["clientHeight", 800]]) {
    Object.defineProperty(window.HTMLCanvasElement.prototype, k, { value: v, configurable: true });
  }
  window.HTMLCanvasElement.prototype.getBoundingClientRect =
    () => ({ left: 0, top: 0, width: 1200, height: 800, right: 1200, bottom: 800 });
  window.devicePixelRatio = 1;
  window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
  window.fetch = async (u) => {
    const buf = read("web/" + String(u).replace(/^https?:\/\/[^/]+\//, ""));
    return {
      ok: true,
      json: async () => JSON.parse(buf.toString("utf8")),
      arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
    };
  };

  // Run inspector.js in THIS process against the jsdom globals -- that is what exercises the
  // real module graph. rAF is pumped a fixed number of times so the loop runs but terminates.
  let frames = 0;
  Object.assign(globalThis, {
    window, document: window.document, fetch: window.fetch,
    addEventListener: window.addEventListener.bind(window),
    requestAnimationFrame: (fn) => { if (frames++ < 6000) setTimeout(() => fn(frames * 100), 0); return frames; },
  });

  // The harness needs the sentence list to drive the selector; read it from the same manifest
  // the module fetches rather than duplicating the corpus here.
  window.__tracesMeta = JSON.parse(read("web/public/big/traces.json").toString("utf8")).sentences;

  const bust = `../src/inspector.js?p=${gl ? "gl" : "nogl"}`;
  await import(bust);

  // Poll for readiness instead of sleeping a guessed number of milliseconds. Boot has to fetch
  // and decode ~4 MB and lay out 12,288 nodes; it lands around 450ms here, and a fixed 400ms
  // sleep passed only by luck until one more module pushed it over. Wait for the thing that
  // actually signals "the app is running" -- a populated HUD, or the fallback having taken over.
  const ready = () => {
    const d = window.document;
    return !d.getElementById("nogl").hidden ||
           d.getElementById("c-active").textContent.trim() !== "—";
  };
  const deadline = Date.now() + 8000;
  while (!ready() && Date.now() < deadline) await new Promise((r) => setTimeout(r, 25));
  if (!ready()) errors.push(`page never became ready within 8s`);
  const shown = (el) => window.getComputedStyle(el).display !== "none";
  return { window, errors, calls, shown, $: (id) => window.document.getElementById(id) };
}

// ── pass 0: the CSS cascade around [hidden] ─────────────────────────────────────────────────
{
  const html = read("web/field.html").toString("utf8");
  const css = read("web/src/field.css").toString("utf8");
  const { hiddenClasses, risky, forced, hasReset } = checkHiddenCascade(html, css);
  check(forced.length === 0,
        `these rules FORCE display on an element that gets hidden, and no [hidden] reset can ` +
        `outrank them:\n` + forced.map((r) => `      ${r}`).join("\n"));
  check(risky.length === 0 || hasReset,
        `these rules override [hidden] and would paint a hidden element over the page:\n` +
        risky.map((r) => `      ${r}`).join("\n") +
        `\n      add: [hidden] { display: none !important; }`);
  console.log(`cascade:   [hidden] used on .${hiddenClasses.join(", .")} · ` +
              `${risky.length} display rule(s) could override it (${forced.length} !important) · ` +
              `reset ${hasReset ? "present" : "MISSING"}`);
}

// ── pass 1: no WebGL ────────────────────────────────────────────────────────────────────────
{
  const { errors, $, shown } = await boot({ gl: false });
  check(errors.length === 0, `no-WebGL boot threw:\n${errors.join("\n")}`);
  check(!$("nogl").hidden, "fallback stayed hidden despite getContext returning null");
  check(shown($("nogl")), "fallback is not actually rendered (computed display is none)");
  const stats = $("nogl-stats");
  check(stats.children.length >= 8, `fallback stats not populated (${stats.children.length} nodes)`);
  for (const want of ["12,288", "modularity", "active per token"]) {
    check(stats.textContent.includes(want), `fallback stats missing "${want}"`);
  }
  console.log(`no-webgl:  fallback shown, ${stats.children.length / 2} stats — ` +
              `${stats.textContent.replace(/\s+/g, " ").trim().slice(0, 68)}`);
}

// ── pass 2: with a GL context ───────────────────────────────────────────────────────────────
{
  const { errors, calls, $, shown, window: document0w } = await boot({ gl: true });
  const document0 = document0w.document;
  check(errors.length === 0, `WebGL boot threw:\n${errors.join("\n")}`);
  check($("nogl").hidden, "fallback attribute set even though a GL context was available");
  // COMPUTED, not the attribute. The attribute was correct while the overlay covered the canvas:
  // `.fallback{display:grid}` beat the UA's [hidden]{display:none} and the page looked dead.
  check(!shown($("nogl")),
        "fallback overlay is RENDERED over the live field — a CSS display rule is beating [hidden]");
  check(!shown($("card")), "neuron card is rendered before anything is selected");

  // the field must actually have drawn all 12,288 points, and the tone-map pass must have run
  const points = calls.filter((c) => c.count === 12288);
  check(points.length > 0, `no draw call for 12,288 points (saw ${calls.length} draws)`);
  const tonemap = calls.filter((c) => c.count === 3);
  check(tonemap.length > 0, "tone-map fullscreen pass never drew (HDR path broken)");

  // the HUD must be populated by the loop, not left at its placeholder em-dashes
  for (const id of ["c-active", "c-pct", "c-tok", "c-byte", "c-layer"]) {
    check($(id).textContent.trim() !== "—" && $(id).textContent.trim() !== "",
          `HUD #${id} never populated (still "${$(id).textContent}")`);
  }
  check(/^\d/.test($("c-active").textContent), `active count is not a number: "${$("c-active").textContent}"`);
  check($("strip").children.length > 20, `byte strip not built (${$("strip").children.length})`);
  check($("legend").textContent.length > 0, "legend empty");
  check($("mode-note").textContent.length > 20, "mode note empty");

  // Every colour mode, driven through the real change handler. The community mode shipped
  // broken once -- the legend advertised three hues while the shader only had the amber ramp --
  // and nothing caught it because only the default mode was ever executed.
  const sel = $("sel-mode");
  const seen = new Set();
  for (const m of ["sigma", "degree", "community", "act"]) {
    sel.value = m;
    sel.onchange({ target: sel });
    const legend = $("legend").innerHTML;
    check($("mode-note").textContent.length > 20, `mode "${m}" has no explanatory note`);
    check(legend.length > 0, `mode "${m}" rendered no legend`);
    check(!seen.has(legend), `mode "${m}" reuses another mode's legend — encoding is wrong`);
    seen.add(legend);
    if (m === "community") {
      check(/#3987e5/i.test(legend), "community legend must show the categorical swatches");
      check(!/linear-gradient/.test(legend), "community is categorical, not a ramp");
    } else {
      check(/linear-gradient/.test(legend), `mode "${m}" is a magnitude and needs a ramp legend`);
    }
  }
  console.log(`modes:     ${seen.size} distinct legends across 4 modes, all with notes`);

  // Every view, driven through the real handler and then actually rendered. Each view owns a
  // different canvas and a different draw path, so a view that is merely selectable but never
  // paints is exactly the kind of thing that reaches a viewer.
  const vsel = $("sel-view");
  const before = calls.length;
  for (const v of ["tiles", "arcs", "orthant", "field"]) {
    vsel.value = v;
    vsel.onchange({ target: vsel });
    const n0 = calls.length;
    await new Promise((r) => setTimeout(r, 120));   // let a frame or two run
    if (v === "tiles") {
      check($("field").hidden === false, "tiles view hid the WebGL canvas");
      check(!$("tile-labels").hidden, "tiles view did not show its labels");
      check($("tile-labels").children.length === 6,
            `expected 6 iteration labels, got ${$("tile-labels").children.length}`);
      check(/iteration/.test($("tile-labels").textContent), "tile labels are empty");
    }
    if (v === "arcs") {
      check(!$("arcs").hidden, "arcs view did not reveal its canvas");
      check($("field").hidden, "arcs view left the WebGL canvas visible underneath");
      check(!$("arc-stats").hidden, "arcs view did not show the negative-share readout");
      check($("sel-mode").disabled, "colour-by should be disabled in the arcs view");
    }
    if (v === "orthant") {
      check(!$("orthant").hidden, "orthant view did not reveal its canvas");
      check($("field").hidden && $("arcs").hidden, "orthant left another view's canvas visible");
      check(!$("orth-panel").hidden, "orthant control panel stayed hidden");
      check(document0.querySelector(".transport").hidden,
            "transport is still shown in the orthant view, implying a timeline it does not have");
      // the two headline numbers are real measurements, not placeholders
      const hi = $("o-hi").textContent, hs = $("o-hisign").textContent;
      check(/^\d+\.\d°$/.test(hi), `ReLU'd 3072-D angle not rendered: "${hi}"`);
      check(/^\d+\.\d°$/.test(hs), `signed 3072-D angle not rendered: "${hs}"`);
      const a = parseFloat(hi), b = parseFloat(hs);
      check(Math.abs(a - 71.4) < 2.5, `ReLU'd angle ${a}° is not near arccos(1/pi) = 71.4°`);
      check(Math.abs(b - 90) < 2.5, `signed angle ${b}° is not near 90°`);
      check(b - a > 12, `the whole point is the gap; got only ${(b - a).toFixed(1)}°`);
    }
    if (v === "field") {
      check(!$("field").hidden && $("arcs").hidden, "returning to the field view did not restore it");
      check(!$("sel-mode").disabled, "colour-by stayed disabled after leaving the arcs view");
    }
  }
  console.log(`views:     tiles → 6 labelled iterations · arcs → own canvas + neg readout · ` +
              `orthant → ${$("o-hi").textContent} vs ${$("o-hisign").textContent} · ` +
              `field restored (${calls.length - before} extra draws)`);
  // ── EVERY SENTENCE, not just the starred one ──────────────────────────────────────────────
  //
  // The regression this exists for: the traces pack gave the hero sentence all six iterations,
  // attention and sigma row energy and gave the other six one layer and nothing else, so the
  // iteration selector was disabled, the small-multiples and arc views refused to draw, and the
  // sigma colour mode painted an empty field -- on 6 of the 7 sentences. Every gate passed,
  // because every gate only ever looked at the hero. So drive all of them, through the real
  // change handlers, and require numbers that DIFFER between sentences: a readout that is
  // identical everywhere is a readout that is not reading this sentence.
  const ssel = $("sel-sent");
  const scrub = $("scrub");
  const settle = () => new Promise((r) => setTimeout(r, 90));
  const rows = [];
  for (const meta of window.__tracesMeta) {
    ssel.value = String(meta.i);
    ssel.onchange({ target: ssel });
    await settle();

    check($("sel-layer").options.length === 6,
          `s${meta.i}: iteration selector has ${$("sel-layer").options.length} options, needs 6`);
    check(!$("sel-layer").disabled, `s${meta.i}: iteration selector is disabled`);
    check($("strip").children.length > 5, `s${meta.i}: byte strip is empty`);

    // park on a mid-sentence token so the readouts have something to say
    scrub.value = String(Math.floor(meta.T / 2));
    scrub.oninput({ target: scrub });

    // 1. the field view must still be counting live activations for THIS sentence
    vsel.value = "field"; vsel.onchange({ target: vsel });
    sel.value = "act"; sel.onchange({ target: sel });
    await settle();
    const active = parseInt($("c-active").textContent.replace(/,/g, ""), 10);
    check(Number.isFinite(active) && active > 0, `s${meta.i}: active count is "${$("c-active").textContent}"`);

    // 2. sigma energy: exported per sentence now, so the mode must not be an empty field
    sel.value = "sigma"; sel.onchange({ target: sel });
    await settle();
    check(!/not exported|re-run/i.test($("mode-note").textContent),
          `s${meta.i}: sigma mode reports missing data -- "${$("mode-note").textContent.slice(0, 60)}"`);
    sel.value = "act"; sel.onchange({ target: sel });

    // 3. all six iterations, with their overlap-against-iteration-1 labels
    vsel.value = "tiles"; vsel.onchange({ target: vsel });
    await settle();
    check(!$("tile-labels").hidden, `s${meta.i}: tiles view did not show its labels`);
    check($("tile-labels").children.length === 6,
          `s${meta.i}: ${$("tile-labels").children.length} iteration labels, expected 6`);
    const shared = ($("tile-labels").textContent.match(/(\d+)% shared with 1/g) || []).length;
    check(shared === 5, `s${meta.i}: ${shared} overlap readouts, expected 5 (iterations 2..6)`);
    const lit = [...$("tile-labels").children].map((b) => b.textContent.match(/([\d,]+) lit/)?.[1]);
    check(lit.every(Boolean), `s${meta.i}: an iteration label has no active count`);

    // 4. the arc diagram, with a per-token negative share
    vsel.value = "arcs"; vsel.onchange({ target: vsel });
    await settle();
    check(!$("arcs").hidden, `s${meta.i}: arcs view did not reveal its canvas`);
    check(!$("arc-stats").hidden, `s${meta.i}: arcs view did not show the negative-share readout`);
    check($("sel-layer").disabled,
          `s${meta.i}: iteration selector is live in the arcs view but does not drive it`);
    const neg = $("a-neg").textContent.trim();
    check(/^\d+%$/.test(neg), `s${meta.i}: negative share reads "${neg}"`);
    rows.push({ i: meta.i, active, neg, tiles: $("tile-labels").textContent.replace(/\s+/g, " ") });
    vsel.value = "field"; vsel.onchange({ target: vsel });
  }
  check(rows.length === 7, `swept ${rows.length} sentences, expected 7`);
  // identical readouts across seven different sentences would mean the views are showing one
  // sentence's data no matter which is selected -- the failure this whole change is about
  check(new Set(rows.map((r) => r.active)).size > 1, "active count is identical on every sentence");
  check(new Set(rows.map((r) => r.tiles)).size === rows.length,
        "the six-iteration labels are identical across sentences");
  console.log(`sentences: all ${rows.length} drive every view — active ` +
              `${rows.map((r) => r.active).join("/")} · arcs neg ${rows.map((r) => r.neg).join("/")}`);

  // ── the side panels get out of the way ──────────────────────────────────────────────────
  //
  // Both HUDs float over a full-bleed canvas and cover a real slice of it. They collapse to a
  // handle; the contents are hidden rather than translated away, so this checks the COMPUTED
  // display (a CSS `display` rule on .hud-body would beat [hidden] and leave a "collapsed" panel
  // fully visible while its button claimed otherwise) and that nothing focusable survives.
  {
    const focusables = (id) => $(id).querySelectorAll("select, input, button, a").length;
    check(!!$("tgl-l") && !!$("tgl-r"), "the side panels have no collapse handles");
    check(focusables("hud-l-body") > 0 && focusables("panel-body") > 0,
          "the panel bodies are empty — the wrappers are in the wrong place");

    for (const [btn, body, host] of [["tgl-l", "hud-l-body", "hud-l"], ["tgl-r", "panel-body", "panel"]]) {
      $(btn).dispatchEvent(new document0w.MouseEvent("click", { bubbles: true }));
      check($(body).hidden, `${btn} did not hide ${body}`);
      check(!shown($(body)), `${body} is still RENDERED after collapsing — a display rule beats [hidden]`);
      check($(host).classList.contains("is-collapsed"), `${host} did not take the collapsed class`);
      check($(btn).getAttribute("aria-expanded") === "false", `${btn} still reports expanded`);
      check(shown($(btn)), `${btn} vanished with the panel — there is no way back`);
      $(btn).dispatchEvent(new document0w.MouseEvent("click", { bubbles: true }));
      check(!$(body).hidden && shown($(body)), `${btn} did not restore ${body}`);
      check($(btn).getAttribute("aria-expanded") === "true", `${btn} still reports collapsed`);
    }

    // H hides both at once, and brings both back
    const press = (t) => document0w.dispatchEvent(
      new document0w.KeyboardEvent("keydown", { key: "h", bubbles: true, cancelable: true }));
    press();
    check($("hud-l-body").hidden && $("panel-body").hidden, "H did not hide both panels");
    press();
    check(!$("hud-l-body").hidden && !$("panel-body").hidden, "H did not bring both panels back");
    // and it must not fire while a select has focus, or picking a sentence starting with h breaks
    const sel = $("sel-mode");
    sel.dispatchEvent(new document0w.KeyboardEvent("keydown", { key: "h", bubbles: true, cancelable: true }));
    check(!$("panel-body").hidden, "H fired from inside a form control");
    console.log(`panels:    both collapse to a handle and come back · H toggles both · ` +
                `${focusables("panel-body")} controls hidden with the right-hand panel`);
  }

  console.log(`overlay:   fallback computed display = ` +
              `${$("nogl").ownerDocument.defaultView.getComputedStyle($("nogl")).display} (must be none)`);

  console.log(`webgl:     ${calls.length} draw calls — ${points.length} point passes, ` +
              `${tonemap.length} tone-map passes`);
  console.log(`hud:       firing ${$("c-active").textContent} (${$("c-pct").textContent}), ` +
              `token ${$("c-tok").textContent}, iteration ${$("c-layer").textContent}`);
  console.log(`strip:     ${$("strip").children.length} bytes rendered`);
}

console.log(fail === 0 ? "\nFIELD SMOKE PASSED" : `\nFIELD SMOKE FAILED (${fail} check(s))`);
process.exit(fail === 0 ? 0 : 1);

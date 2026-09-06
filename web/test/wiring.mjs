// Structural checks that would otherwise only surface by opening a browser:
//  - every #id the page controllers reach for exists in their HTML
//  - every id in the HTML is actually used (dead markup)
//  - the layers override really changes computation
import { readFileSync } from "node:fs";
import { BDHModel } from "../src/bdh.js";

const html = readFileSync("web/price.html", "utf8") + readFileSync("web/loop.html", "utf8");
const js = readFileSync("web/src/price.js", "utf8") + readFileSync("web/src/loop.js", "utf8")
         + readFileSync("web/src/diagram.js", "utf8");

const inHtml = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]));
const wanted = new Set([...js.matchAll(/\$\("([A-Za-z0-9_-]+)"\)/g)].map((m) => m[1]));

const missing = [...wanted].filter((i) => !inHtml.has(i));
const unused = [...inHtml].filter((i) => !wanted.has(i) && !js.includes(`"${i}"`));

console.log(`ids in html: ${inHtml.size}   referenced by the controllers: ${wanted.size}`);
if (missing.length) console.log("  MISSING in html: " + missing.join(", "));
if (unused.length) console.log("  note, not referenced by id: " + unused.join(", "));

// selectors used besides ids
const classSel = [...js.matchAll(/\$\("\.([A-Za-z0-9_-]+)"\)/g)].map((m) => m[1]);
const badClass = classSel.filter((c) => !html.includes(`class="${c}`) && !html.includes(` ${c}"`) && !html.includes(`${c} `));
if (badClass.length) console.log("  MISSING class targets: " + badClass.join(", "));

// Classes generated in JS rather than written in the HTML are invisible to the check above and
// to jsdom, which has no layout engine. tokens.js builds `chip chip-<kind>` names by string
// concatenation, so a stylesheet that defines `.chip.big` instead of `.chip-big` renders
// unstyled boxes and every gate stays green. Assert the generated names exist in the CSS.
{
  const css = readFileSync("web/src/price.css", "utf8") + readFileSync("web/src/theme.css", "utf8");
  const need = ["chip", "chip-src", "chip-tgt", "chip-sep", "chip-qry", "chip-big"];
  const miss = need.filter((c) => !css.includes(`.${c}`));
  console.log(`  ${miss.length ? "FAIL" : "ok  "}  every class tokens.js generates is styled` +
              (miss.length ? `  —  missing ${miss.join(", ")}` : ""));
  if (miss.length) {
    console.log("");
    console.log("WIRING FAILED — generated class names are not styled");
    process.exit(1);
  }
}

// layers override must actually change the computation
// ── the three-page nav must be the SAME control on all three pages ──────────────────────────
//
// It is the one thing that appears on every page, so it is the one thing that must not move or
// restyle between them. field.html carries it as `.pagenav` inside its HUD and is FROZEN;
// loop/price carry it as `.sitenav`, a fixed bar. The two rulesets are deliberate duplicates
// (field.css cannot be edited), which is exactly the arrangement that drifts silently -- so the
// order of the links, which one is marked current, and the pill declarations are asserted here.
const NAV_PAGES = [
  ["web/field.html", "pagenav", "field.html", "web/src/field.css"],
  ["web/loop.html", "sitenav", "loop.html", "web/src/theme.css"],
  ["web/price.html", "sitenav", "price.html", "web/src/theme.css"],
];
const navOrder = [];
const navCurrent = [];
for (const [page, cls, self] of NAV_PAGES) {
  const src = readFileSync(page, "utf8");
  const nav = src.slice(src.indexOf(`class="${cls}"`), src.indexOf("</nav>", src.indexOf(`class="${cls}"`)));
  const links = [...nav.matchAll(/<a href="([^"]+)"([^>]*)>([^<]+)<\/a>/g)];
  navOrder.push(links.map((m) => `${m[1]}:${m[3].trim()}`).join(" | "));
  navCurrent.push(links.filter((m) => m[2].includes('aria-current="page"')).map((m) => m[1]).join(","));
  // the links must come before any wordmark/spacer, i.e. the toggle sits on the LEFT edge
  const firstLink = nav.indexOf("<a ");
  const spacer = nav.indexOf('class="spacer"');
  navOrder.push(spacer === -1 || spacer > firstLink ? "left" : "RIGHT");
  navCurrent.push(self);
}
const orders = navOrder.filter((_, i) => i % 2 === 0);
const sides = navOrder.filter((_, i) => i % 2 === 1);
const currents = navCurrent.filter((_, i) => i % 2 === 0);
const selves = navCurrent.filter((_, i) => i % 2 === 1);

// the pill declarations themselves, read out of both stylesheets
const decl = (css, sel, prop) => {
  const i = css.indexOf(sel + " {");
  if (i < 0) return null;
  const body = css.slice(i, css.indexOf("}", i));
  return body.match(new RegExp(prop + "\\s*:\\s*([^;]+)"))?.[1].trim() || null;
};
const fieldCss = readFileSync("web/src/field.css", "utf8");
const themeCss = readFileSync("web/src/theme.css", "utf8");
const pillProps = ["border-radius", "padding", "background"];
const pillDrift = pillProps.filter(
  (prop) => decl(fieldCss, ".pagenav a", prop) !== decl(themeCss, ".sitenav a", prop));
const currentProps = ["color", "background", "border-color"];
const currentDrift = currentProps.filter(
  (prop) => decl(fieldCss, '.pagenav a[aria-current="page"]', prop) !==
            decl(themeCss, '.sitenav a[aria-current="page"]', prop));

console.log(`\nnav:  ${orders[0]}`);
console.log(`      current = ${currents.join(" / ")}   position = ${sides.join(" / ")}`);

const navChecks = [
  ["all three pages list the same links in the same order", new Set(orders).size === 1],
  ["each page marks itself as the current one", currents.every((c, i) => c === selves[i])],
  ["the toggle sits on the left on every page", sides.every((x) => x === "left")],
  ["the pill geometry matches field.css", pillDrift.length === 0],
  ["the current-page pill matches field.css", currentDrift.length === 0],
];

const man = JSON.parse(readFileSync("web/public/model.json", "utf8"));
const bin = readFileSync("web/public/model.bin");
const m = new BDHModel(man, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
const toks = JSON.parse(readFileSync("web/public/presets.json", "utf8")).hook.bytes;
const base = m.forward(toks, { keepSigma: false });
const l2 = m.forward(toks, { keepSigma: false, layers: 2 });
const l4 = m.forward(toks, { keepSigma: false, layers: 4 });
let dSame = 0, dDiff = 0;
for (let i = 0; i < base.logits.length; i++) {
  dSame = Math.max(dSame, Math.abs(base.logits[i] - l4.logits[i]));
  dDiff = Math.max(dDiff, Math.abs(base.logits[i] - l2.logits[i]));
}

const checks = [
  ...navChecks,
  ["no missing element ids", missing.length === 0],
  ["no missing class targets", badClass.length === 0],
  ["layers:4 == default (model trained at 4)", dSame === 0],
  ["layers:2 changes the output", dDiff > 1e-3],
];
let bad = 0;
console.log();
for (const [n, ok] of checks) { if (!ok) bad++; console.log(`  ${ok ? "ok  " : "FAIL"}  ${n}`); }
console.log(`\n  L=4 delta ${dSame.toExponential(1)}   L=2 delta ${dDiff.toExponential(1)}`);
console.log(bad ? `\nWIRING FAILED (${bad})` : "\nWIRING OK");
process.exit(bad ? 1 : 0);

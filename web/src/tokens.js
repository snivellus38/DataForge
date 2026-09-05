// Display mapping for model tokens.
//
// The model reads raw bytes (SRC 33..80, TGT 128..175); TGT renders as \x81 etc, so the UI must
// map them. IDENTITY IS NEVER COLOUR-ALONE: the validated categorical palette clears only three
// slots on the all-pairs gate, and the sandbox can put up to 12 tokens on screen compared
// pairwise. So the LETTER carries identity and colour only reinforces it.
import { SERIES, mode } from "./palette.js";

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

export function makeTokenMap(manifest) {
  const src = manifest.tokens.SRC, tgt = manifest.tokens.TGT;
  const byByte = new Map();
  src.forEach((b, i) => byByte.set(b, { kind: "src", i, label: LETTERS[i % 26], byte: b }));
  tgt.forEach((b, i) => byByte.set(b, { kind: "tgt", i, label: String(i + 1), byte: b }));
  byByte.set(manifest.tokens.SEP, { kind: "sep", label: "·", byte: manifest.tokens.SEP });
  byByte.set(manifest.tokens.QRY, { kind: "qry", label: "?", byte: manifest.tokens.QRY });
  return byByte;
}

/** Hue for a target token. Decorative reinforcement only -- the label is the identity. */
export function tgtHue(i, m = mode()) {
  const s = SERIES[m];
  return i < s.length ? s[i] : `hsl(${(i * 47) % 360} 55% ${m === "dark" ? 58 : 45}%)`;
}

export function chip(tok, { big = false } = {}) {
  const el = document.createElement("span");
  el.className = `chip chip-${tok.kind}${big ? " chip-big" : ""}`;
  el.textContent = tok.label;
  if (tok.kind === "tgt") el.style.setProperty("--chip-hue", tgtHue(tok.i));
  el.title = `byte ${tok.byte}`;
  return el;
}

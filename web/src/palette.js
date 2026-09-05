// Validated with the dataviz palette validator (scripts/validate_palette.js).
// 2-series categorical, --pairs all, BOTH modes: all checks PASS
//   light #2a78d6/#eb6834  CVD dE 24.7, normal dE 33.6
//   dark  #3987e5/#d95926  CVD dE 26.8, normal dE 31.8
export const SERIES = {
  light: ["#2a78d6", "#eb6834", "#1baf7a"],
  dark:  ["#3987e5", "#d95926", "#199e70"],
};

// sigma is SIGNED (measured range [-32.8, +23.9]), so magnitude is DIVERGING, never sequential:
// two poles with a neutral gray midpoint that reads as "nothing here".
export const DIVERGING = {
  light: { neg: "#2a78d6", mid: "#f0efec", pos: "#e34948" },
  dark:  { neg: "#3987e5", mid: "#383835", pos: "#e66767" },
};

export const INK = {
  light: { primary: "#0b0b0b", secondary: "#52514e", muted: "#8a8880", surface: "#fcfcfb", grid: "#e6e5e1" },
  dark:  { primary: "#ffffff", secondary: "#c3c2b7", muted: "#8a8880", surface: "#1a1a19", grid: "#2e2e2b" },
};

export const mode = () =>
  document.documentElement.dataset.theme ||
  (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");

const hex2rgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

/** Diverging ramp for a signed value in [-1,1]. Gray at 0 means "nothing written here". */
export function divergingColor(t, m = mode()) {
  const p = DIVERGING[m];
  const a = hex2rgb(p.mid), b = hex2rgb(t < 0 ? p.neg : p.pos);
  const k = Math.min(1, Math.abs(t));
  return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * k)).join(",")})`;
}

/** Same ramp as packed RGB, for direct ImageData writes (no per-cell string alloc). */
export function divergingRGB(t, m = mode()) {
  const p = DIVERGING[m];
  const a = hex2rgb(p.mid), b = hex2rgb(t < 0 ? p.neg : p.pos);
  const k = Math.min(1, Math.abs(t));
  return a.map((v, i) => (v + (b[i] - v) * k) | 0);
}

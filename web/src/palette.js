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

// ── THE FIELD ────────────────────────────────────────────────────────────────────────────────
// The neuron field is ALWAYS dark, in both page themes. It is an instrument read-out, not a
// figure on paper: additive glow needs a dark ground, and 12,288 luminous points on white is
// not a legible picture. The page chrome around it still follows the theme.
export const FIELD_SURFACE = "#100f0e";

// ACTIVATION = magnitude -> sequential, ONE hue (amber), monotonically lighter with value.
// Four stops rather than two so the low end stays near the surface (a silent neuron should
// barely exist) while the top end goes near-white. Monotonicity is asserted in web/test/field.mjs.
export const IGNITION = ["#342b25", "#8a4a1e", "#e0872a", "#ffe4b0"];

// SIGMA ROW ENERGY = a second, simultaneous magnitude -> the next sequential hue (blue), per the
// reference palette's rule for two sequential contexts. Accumulated state reads cool; the
// transient firing reads hot, so the two never get confused on screen.
export const SIGMA_RAMP = ["#191d24", "#1c5cab", "#3987e5", "#cde2fb"];

// COMMUNITIES = identity -> categorical. Only THREE slots, plus gray.
// Validated all-pairs on #141413: worst CVD dE 9.4 (deutan), worst normal-vision dE 20.9, all
// >= 3:1 vs surface. A neuron field is a scatter, so every pair is adjacent and --pairs all
// applies; the 4th slot (yellow) fails against orange at dE 4.8 and is not available.
// This cap is a good thing here: G* modularity is only ~0.08 and the Louvain communities have no
// spatial separation (silhouette ~ -0.12), so painting ten of them in ten hues would draw
// structure the measurement says is not there. Top 3 get a hue; everything else is "other".
export const COMMUNITY = ["#3987e5", "#d95926", "#199e70"];
export const COMMUNITY_OTHER = "#4a4945";

// Heads are NOT colour-coded: four categorical hues cannot pass all-pairs, and the measured
// head-separation silhouette is -0.012, so hue would imply a grouping that does not exist.
// Show heads as small multiples instead.

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

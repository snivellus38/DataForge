"""
Offline render of THE FIELD, reproducing web/src/field-gl.js's compositing.

WHY THIS EXISTS
---------------
Two jobs. First, tuning: point size, glow falloff, decay and gain are aesthetic parameters, and
iterating on them through a browser is slow. This renders the same maths to a PNG in a second.
Second, stills: the README and the one-page summary need a hero image, and it must be the real
field from the real weights rather than an illustration.

IT IS AN APPROXIMATION AND MUST BE LABELLED AS ONE. It mirrors the shader (radial falloff
(1-r^2)^2, premultiplied additive accumulation, the four-stop ignition ramp) but it is a separate
implementation, so it can drift. It is a tuning aid, never a source of numbers.
"""
import argparse, gzip, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--field", default="web/public/big/field")
ap.add_argument("--traces", default="web/public/big/traces")
ap.add_argument("--layout", default="global.xy")
ap.add_argument("--sentence", type=int, default=None, help="default: the hero sentence")
ap.add_argument("--layer", type=int, default=None)
ap.add_argument("--token", type=int, default=None, help="default: 70% through the sentence")
ap.add_argument("--decay", type=float, default=0.86)
ap.add_argument("--trail", type=int, default=14, help="tokens of afterglow to accumulate")
ap.add_argument("--size", type=int, default=1500)
ap.add_argument("--point", type=float, default=3.2, help="base point radius in px at size 1500")
ap.add_argument("--gain", type=float, default=1.0)
ap.add_argument("--curve", type=float, default=1.0, help="activation gamma; <1 lifts the mid-range")
ap.add_argument("--idle", type=float, default=0.13, help="alpha floor for a silent neuron")
ap.add_argument("--idle-hex", default="#241f1b")
ap.add_argument("--exposure", type=float, default=0.0,
                help="if >0, tone-map 1-exp(-x*e) instead of hard-clipping the additive buffer")
ap.add_argument("--out", default="hero.png")
a = ap.parse_args()

IGNITION = ["#241f1b", "#8a4a1e", "#e0872a", "#ffe4b0"]     # must match web/src/palette.js
SURFACE = "#100f0e"
hexf = lambda h: np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def load(stem):
    man = json.load(open(stem + ".json"))
    # traces ships gzipped (every sentence carries six iterations, attention and sigma); field
    # does not. The manifest says which, so this does not have to know the filenames.
    if man.get("compression") == "gzip":
        raw = gzip.decompress(open(stem + ".bin.gz", "rb").read())
    else:
        raw = open(stem + ".bin", "rb").read()
    buf = np.frombuffer(raw, dtype=np.uint8)
    t = {}
    for e in man["tensors"]:
        t[e["name"]] = np.frombuffer(buf, dtype=np.dtype(e["dtype"]),
                                     count=int(np.prod(e["shape"])),
                                     offset=e["offset"]).reshape(e["shape"])
    return man, t


fman, ft = load(a.field)
tman, tt = load(a.traces)
si = a.sentence if a.sentence is not None else tman["hero"]
Lx = a.layer if a.layer is not None else tman["default_layer"]
sent = next(s for s in tman["sentences"] if s["i"] == si)
T = sent["T"]
tok = a.token if a.token is not None else int(T * 0.7)

# positions, un-quantised exactly as web/src/bigdata.js positions() does
ent = next(e for e in fman["tensors"] if e["name"] == a.layout)
q = ft[a.layout].astype(np.float64)
lo, hi = np.array(ent["box"][0]), np.array(ent["box"][1])
xy = lo + q * (hi - lo) / 65535.0
ok = np.isfinite(xy).all(1)

# activations with the same decay recurrence as field-data.js decayInto()
off, idx, val, scale = (tt[f"s{si}.L{Lx}.off"], tt[f"s{si}.L{Lx}.idx"],
                        tt[f"s{si}.L{Lx}.val"], tt[f"s{si}.L{Lx}.scale"])
ref = np.percentile(np.concatenate([val[off[t]:off[t + 1]] * scale[t] / 255
                                    for t in range(0, T, max(1, T // 24))]), 99)
act = np.zeros(len(xy))
for t in range(max(0, tok - a.trail), tok + 1):
    act *= a.decay
    v = np.minimum(1.0, (val[off[t]:off[t + 1]] * scale[t] / 255) / ref) ** a.curve
    np.maximum.at(act, idx[off[t]:off[t + 1]].astype(np.int64), v)

# ── splat, additive, premultiplied -- the shader's blend ─────────────────────────────────────
S = a.size
m = ok
p = xy[m]
lo2, hi2 = p.min(0), p.max(0)
span = (hi2 - lo2).max()
cen = (lo2 + hi2) / 2
px = ((p - cen) / span * 1.72 * 0.5 + 0.5) * S
A = act[m]

R = int(np.ceil(a.point * (S / 1500) * 2.4)) + 2
yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
canvas = np.zeros((S + 2 * R, S + 2 * R, 3))

rad = a.point * (S / 1500) * (0.55 + 1.85 * A)
stops = np.array([hexf(h) for h in ([a.idle_hex] + IGNITION[1:])])


def ramp(v):
    v = np.clip(v, 0, 1)
    out = np.empty((len(v), 3))
    s1, s2 = v < 0.34, (v >= 0.34) & (v < 0.67)
    s3 = ~(s1 | s2)
    out[s1] = stops[0] + (stops[1] - stops[0]) * (v[s1] / 0.34)[:, None]
    out[s2] = stops[1] + (stops[2] - stops[1]) * ((v[s2] - 0.34) / 0.33)[:, None]
    out[s3] = stops[2] + (stops[3] - stops[2]) * ((v[s3] - 0.67) / 0.33)[:, None]
    return out


col = ramp(A)
ix = np.round(px[:, 0]).astype(int) + R
iy = np.round(S - px[:, 1]).astype(int) + R
order = np.argsort(A)                                   # dim first so bright lands on top
for k in order:
    r = rad[k]
    if r < 0.4:
        continue
    rr = int(np.ceil(r * 2.4))
    sy, sx = np.mgrid[-rr:rr + 1, -rr:rr + 1]
    d2 = (sx * sx + sy * sy) / (r * r * 4.0 + 1e-9)
    fall = np.clip(1.0 - d2, 0, 1) ** 2
    alpha = fall * (a.idle + (1.0 - a.idle) * A[k]) * a.gain
    y0, x0 = iy[k] - rr, ix[k] - rr
    if y0 < 0 or x0 < 0 or y0 + 2 * rr + 1 > canvas.shape[0] or x0 + 2 * rr + 1 > canvas.shape[1]:
        continue
    canvas[y0:y0 + 2 * rr + 1, x0:x0 + 2 * rr + 1] += alpha[..., None] * col[k]

img = canvas[R:-R, R:-R]
# Additive blending is HDR: a dense cluster of firing neurons sums past 1.0 and hard-clipping
# turns it into a flat white blob, destroying exactly the structure worth looking at. Tone-map
# instead -- highlights compress smoothly and the cluster keeps its shape.
if a.exposure > 0:
    img = 1.0 - np.exp(-img * a.exposure)
img = hexf(SURFACE) + img
img = np.clip(img, 0, 1)
img = img ** (1 / 1.05)

try:
    from PIL import Image
    Image.fromarray((img * 255).astype(np.uint8)).save(a.out)
except ImportError:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imsave(a.out, img)

lit = int((A > 0.02).sum())
print(f"{a.out}  {S}x{S}")
print(f"  sentence {si} layer {Lx} token {tok}/{T}  trail {a.trail}  decay {a.decay}")
print(f"  {lit:,} of {len(A):,} neurons visibly lit ({lit/len(A):.1%}) at this instant")
print(f"  text: {sent['text'][:78]}")

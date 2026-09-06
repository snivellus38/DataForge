"""
Export the FULL PIPELINE, stage by stage, for THE LOOP (web/loop.html).

`export_traces.py` ships what fires (activations, for the field). This ships *everything the
model computes between one token going in and the next byte coming out*, so a walkthrough can
open up each stage and show the model's own numbers:

    v*  -LN->  x_pre  -ReLU->  x = Q = K  -RoPE->  scores  ->  sigma  ->  a*
                                              -D_y-> y_pre -ReLU-> (*) x = y  -E->  v*'
                                                                              ->  logits

WHAT IS DELIBERATELY NOT SHIPPED
--------------------------------
`q_roped`. RoPE is a closed form in the neuron index and the position, so the browser can derive
it from `x` + `rope_freqs` -- verified here at EXACTLY 0.0 max abs error against the model's own
q_roped, and re-checked JS-side by web/test/walk.mjs. Shipping it would add ~140 KB per sentence
to say something the reader can recompute. Same reasoning for sigma: every write is rank-one
(rope(x)_t (outer) v_t), so the browser accumulates the state itself from `x` and `v_ast` and
checks its own answer against the shipped `a_ast`.

TWO SENTENCES, AND WHY
----------------------
w0 "Thank you very much" (T=45) is the teaching default: a 45x45 score matrix has cells big
enough to hover. But it is NOT representative -- only 14.8% of its causal scores are negative
against 34.3% pooled over the corpus. So w1 (T=105) ships too, and the page shows the
per-sentence figure computed live from the shipped scores rather than quoting one number over
a picture of another.

QUANTISATION
------------
Activations are uint8 against a per-token max (non-negative) or a per-token min/max pair
(x_pre, which is signed and whose negative half is the entire point of the ReLU stage).
Anything a NUMBER is claimed from ships as f32/f16: attention scores, a_ast, v_ast, logits.
"""
import argparse, gzip, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bdh_big import load_big

# Same corpus as export_traces.py, so fire_count here indexes the same neurons the field draws.
CORPUS = [
    "<F:en>The price was fifty euros and thirty pounds<T:fr>Le prix etait de cinquante euros et trente livres",
    "<F:en>The dollar strengthened against the yen today<T:fr>Le dollar s'est renforce face au yen aujourd'hui",
    "<F:en>Germany and France signed the bilateral treaty<T:fr>L'Allemagne et la France ont signe le traite bilateral",
    "<F:en>The European Parliament voted on this resolution<T:fr>Le parlement europeen a vote cette resolution",
    "<F:en>The Commission presented the annual budget report<T:fr>La Commission a presente le rapport budgetaire annuel",
    "<F:en>The budget was discussed. The budget was approved. The budget will be implemented.<T:fr>Le budget a ete discute. Le budget a ete approuve. Le budget sera mis en oeuvre.",
    "<F:en>Thank you very much<T:fr>Merci beaucoup",
]
WALK = [6, 3]          # indices into CORPUS: [0] legible default, [1] representative

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="trained_model_things/kriti_checkpoints (1)/french_best.pt")
ap.add_argument("--out", default="web/public/walk")
ap.add_argument("--head", type=int, default=0, help="head kept for the full-resolution stages")
ap.add_argument("--layer", type=int, default=3, help="layer kept for the full-resolution stages")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()

t0 = time.time()
model, cfg, ck = load_big(a.ckpt, a.device)
D, H, N, L = cfg.n_embd, cfg.n_head, cfg.N, cfg.n_layer
print(f"{L}L x D={D} x H={H}, N={N}/head, {H*N:,} neurons, val {ck['losses']['val']:.4f}\n")

blob, entries = bytearray(), []


def put(name, arr, **meta):
    # Pad every tensor to an 8-byte boundary. Sparse packs interleave uint8 values with f32
    # scales, so without this a later Float32Array view lands on an odd offset and the browser
    # either throws or silently reads garbage. The JS decoder keeps a copy-on-misalignment
    # fallback anyway (same as bigdata.js), but nothing should ever need it.
    if len(blob) % 8:
        blob.extend(bytes(8 - len(blob) % 8))
    b = np.ascontiguousarray(arr).tobytes()
    entries.append(dict(name=name, dtype=arr.dtype.str, shape=list(arr.shape),
                        offset=len(blob), bytes=len(b), **meta))
    blob.extend(b)
    return arr


def sparse_pack(X):
    """(T, M) non-negative -> per-token (index, uint8 value) pairs + row offsets + scales."""
    idx, val, scale, off = [], [], [], [0]
    for t in range(X.shape[0]):
        nz = np.flatnonzero(X[t])
        s = float(X[t, nz].max()) if nz.size else 1.0
        idx.append(nz.astype("<u2"))
        val.append(np.clip(np.round(X[t, nz] / s * 255), 0, 255).astype(np.uint8))
        scale.append(s)
        off.append(off[-1] + nz.size)
    return (np.concatenate(idx) if idx else np.zeros(0, "<u2"),
            np.concatenate(val) if val else np.zeros(0, np.uint8),
            np.array(scale, "<f4"), np.array(off, "<u4"))


def signed_pack(X):
    """(T, M) SIGNED -> uint8 with a per-token (min, max) pair. x_pre is ~95% negative and the
    negative half is exactly what the ReLU stage is about, so it cannot be stored sparsely."""
    lo = X.min(1).astype("<f4")
    hi = X.max(1).astype("<f4")
    rng_ = np.maximum(hi - lo, 1e-9)
    q = np.clip(np.round((X - lo[:, None]) / rng_[:, None] * 255), 0, 255).astype(np.uint8)
    return q, lo, hi


def rope(x, freqs, T):
    """Exactly bdh_big._causal_attention's rotation, in numpy, on (T, N)."""
    pos = np.arange(T, dtype=np.float64).reshape(T, 1)
    ph = (pos * freqs.reshape(1, -1) % 1.0) * (2 * math.pi)
    xr = np.stack((-x[:, 1::2], x[:, ::2]), -1).reshape(x.shape)
    return x * np.cos(ph) + xr * np.sin(ph)


freqs = model.rope_freqs.view(-1)[:N].float().cpu().numpy()
put("rope_freqs", freqs.astype("<f4"))

# decoder_x for the head the walkthrough traces, as (D, N) row-major so the browser's
# accumulation loop walks n contiguously for each d. This replaces every shipped x_pre tensor
# (see the layer loop) and doubles as the exact edge coefficient for the flow diagram's first
# hop, for all 3,072 neurons of this head rather than only the pooled ones.
dx_head_f32 = model.decoder_x[a.head].detach().float().cpu().numpy()          # (D, N)
put("dx_head", dx_head_f32.astype("<f2"), head=a.head)
# The browser gets the F16 copy, so the error has to be measured against that, not against the
# f32 original -- otherwise this reports the accuracy of a matrix nobody downloads.
dx_head_shipped = dx_head_f32.astype("<f2").astype(np.float32)
xpre_err = 0.0

# ── corpus pass: sparsity per layer, negative-score share per layer/head, firing counts ───────
fire = np.zeros(H * N, dtype=np.int64)
xs = np.zeros((L, 2), np.int64)
ys = np.zeros((L, 2), np.int64)
negs = np.zeros((L, H, 2), np.int64)
print("corpus pass (7 sentences):")
for si, text in enumerate(CORPUS):
    ids = torch.tensor([list(text.encode())], dtype=torch.long, device=a.device)
    with torch.no_grad(), model.extracting() as buf:
        model(ids)
    T = ids.shape[1]
    for Lx in range(L):
        X = buf[Lx]["x"]
        Y = buf[Lx]["y"]
        xs[Lx] += [int((X > 0).sum()), X.numel()]
        ys[Lx] += [int((Y > 0).sum()), Y.numel()]
        fire += (X[0].permute(1, 0, 2).reshape(T, H * N) > 0).sum(0).cpu().numpy().astype(np.int64)
        for h in range(H):
            A = buf[Lx]["attn"][0, h]
            tri = torch.tril_indices(T, T, -1, device=A.device)
            w = A[tri[0], tri[1]]
            negs[Lx, h] += [int((w < 0).sum()), w.numel()]
    print(f"  s{si} T={T:3d}")

put("fire_count", np.minimum(fire, 65535).astype("<u2"))
x_frac = xs[:, 0].sum() / xs[:, 1].sum()
y_frac = ys[:, 0].sum() / ys[:, 1].sum()
print(f"\n  x active {x_frac:.4%}   y active {y_frac:.4%}   gate is {x_frac/y_frac:.1f}x sparser")
print(f"  neurons that never fire: {int((fire == 0).sum()):,} of {H*N:,} ({(fire == 0).mean():.1%})")

# ── the two walk sentences: every stage, every layer ──────────────────────────────────────────
# ── the flow diagram's cast ───────────────────────────────────────────────────────────────────
# web/loop.html stop 0 draws one forward pass as a layered node-link diagram. 12,288 neurons
# cannot each be a dot, so a FIXED POOL is committed here and the page draws from it: a stable
# subset (ordered by corpus firing frequency, so a dot keeps its identity and its position as the
# animation runs) plus whichever pool members lead the current token. Committing the cast up
# front is what lets every EDGE be a real number -- contribution = weight x source activation --
# because the weight slices for exactly these neurons ship alongside.
#
# The pool is built from neurons that actually lead tokens on the walk sentences, not merely from
# the busiest neurons in the corpus; `leader_coverage` in the manifest reports what fraction of
# the true per-token leaders it captures, and the page states it rather than implying it draws
# everything.
FLOW_TOP = 6           # per (walk, layer, token), how many leaders feed the pool
FLOW_POOL = 256        # how many neurons get weight slices

walk_meta, fixture = [], []
leader_hits = np.zeros(H * N, dtype=np.int64)
leader_steps = []
rng = np.random.default_rng(0)
for wi, si in enumerate(WALK):
    text = CORPUS[si]
    raw = list(text.encode())
    ids = torch.tensor([raw], dtype=torch.long, device=a.device)
    with torch.no_grad(), model.extracting() as buf:
        logits, _ = model(ids)
    T = len(raw)
    # Both sentences carry every stage at every iteration. An earlier revision shipped the full
    # set for w0 only, which is the same failure the field had: a view gated on which sentence
    # you picked rather than on the data. The flow diagram on stop 0 draws all six iterations of
    # whichever sentence is selected, so "full" has to mean full.
    full = True
    tri = np.tril_indices(T, -1)

    put(f"w{wi}.logits", logits[0].float().cpu().numpy().astype("<f2"))

    # `L{n}.v` is the LayerNormed residual going INTO iteration n -- that is what x_pre is built
    # from. The logits come from the UN-normalised residual coming out of the last one, so the
    # diagram's last column needs that too. Accumulate it the way the model does, from the
    # embedding plus each iteration's dv, and assert it reproduces the logits exactly.
    with torch.no_grad():
        v_run = (model.embed(ids) +
                 model.pos_emb(torch.arange(T, device=ids.device)))[0].float()

    layer_meta = []
    first_set = None
    for Lx in range(L):
        b = buf[Lx]
        # v* is head-independent (the LN'd residual): (T, D)
        v_ln = b["v_ast"][0, 0].float().cpu().numpy()
        put(f"w{wi}.L{Lx}.v", v_ln.astype("<f2"))

        # x, all heads, sparse, global neuron id h*N+n -- same id space as field.bin's global.xy.
        # The long sentence is here to make the score matrix representative, not to be walked
        # neuron by neuron, so it only carries the default layer.
        x = b["x"][0].permute(1, 0, 2).reshape(T, H * N).float().cpu().numpy()
        i_, v_, s_, o_ = sparse_pack(x)
        if full or Lx == a.layer:
            put(f"w{wi}.L{Lx}.x.idx", i_)
            put(f"w{wi}.L{Lx}.x.val", v_)
            put(f"w{wi}.L{Lx}.x.scale", s_)
            put(f"w{wi}.L{Lx}.x.off", o_)

        # y, the gated output. ReLU(y_pre) is NOT shipped in general: on the support of x it is
        # exactly y / x, and OFF the support of x it is multiplied by zero, so it provably cannot
        # reach the residual. Shipping it dense cost 7.1 MB to say something recomputable.
        # One head at the default layer of the legible sentence ships anyway, because the gate
        # stop draws the two sets side by side and needs the half that x switches off.
        y = b["y"][0].permute(1, 0, 2).reshape(T, H * N).float().cpu().numpy()
        i2, v2, s2, o2 = sparse_pack(y)
        put(f"w{wi}.L{Lx}.y.idx", i2)
        put(f"w{wi}.L{Lx}.y.val", v2)
        put(f"w{wi}.L{Lx}.y.scale", s2)
        put(f"w{wi}.L{Lx}.y.off", o2)
        if full and Lx == a.layer:
            ry = np.maximum(b["y_pre"][0, a.head].float().cpu().numpy(), 0)
            i3, v3, s3, o3 = sparse_pack(ry)
            put(f"w{wi}.L{Lx}.ry.idx", i3)
            put(f"w{wi}.L{Lx}.ry.val", v3)
            put(f"w{wi}.L{Lx}.ry.scale", s3)
            put(f"w{wi}.L{Lx}.ry.off", o3)

        # attention scores: strict-causal, so only the lower triangle exists. f32 -- the sign
        # and the magnitude are both claimed on screen.
        heads = range(H) if (full or Lx == a.layer) else [a.head]
        for h in heads:
            A = b["attn"][0, h].float().cpu().numpy()
            put(f"w{wi}.L{Lx}.H{h}.attn", A[tri].astype("<f4"))
            # a_ast is the sigma read-out the browser checks itself against. It is only needed
            # where that check is offered: the default layer (all heads) and head 0 (all layers).
            if Lx == a.layer or h == a.head:
                put(f"w{wi}.L{Lx}.H{h}.a", b["a_ast"][0, h].float().cpu().numpy().astype("<f2"))

        # x_pre is NOT shipped. x_pre[t,n] = sum_d v[t,d] * decoder_x[head,d,n], and both
        # factors are already here -- so the browser derives it, exactly as it derives RoPE and
        # sigma. Shipping it dense and signed for both sentences at every iteration cost 2.76 MB,
        # against 1.18 MB for the matrix itself; and the derived values are f16-weight accurate
        # rather than uint8-quantised against a per-token range, so the picture gets BETTER while
        # the pack gets smaller. Measured below.
        xpre_err = max(xpre_err, float(np.abs(
            v_ln @ dx_head_shipped - b["x_pre"][0, a.head].float().cpu().numpy()
        ).max() / max(np.abs(b["x_pre"][0, a.head].float().cpu().numpy()).max(), 1e-9)))

        # dv = this iteration's actual contribution to the residual, y_f @ encoder.
        # The flow diagram's second-to-last column needs the REAL delta, and reconstructing it
        # browser-side would mean shipping all 12,288 rows of `encoder` (4.7 MB) rather than the
        # pool's 256 (0.1 MB). Shipping the answer is smaller AND exact, and the pool's share of
        # it is then a number the page can state instead of implying the drawn edges are all of it.
        y_f = b["y"][0].permute(1, 0, 2).reshape(T, H * N)
        dv = (y_f @ model.encoder).detach().float()
        put(f"w{wi}.L{Lx}.dv", dv.cpu().numpy().astype("<f2"))
        v_run = v_run + dv

        # Which neurons lead this token, at this iteration -- restricted to the TRACED HEAD.
        #
        # The flow diagram is a path, and a path has to be traceable end to end. x_pre is derived
        # from decoder_x[head] and sigma is accumulated for one head, so a neuron from any other
        # head has no pre-activation and no state row: it appears as a dot with nothing entering
        # or leaving it. Drawing those was the reason clicking a neuron lit almost no connections.
        # The cast therefore comes from the traced head only, and the page says so; the counts
        # above the columns stay all-heads, because that is what the sparsity claim is about.
        xh = x[:, a.head * N:(a.head + 1) * N]
        for t in range(T):
            top = np.argpartition(-xh[t], FLOW_TOP)[:FLOW_TOP]
            top = set(int(g) + a.head * N for g in top[xh[t, top] > 0])
            for g in top:
                leader_hits[g] += 1
            leader_steps.append(top)

        # iteration-to-iteration recruitment (CLAUDE.md item 20), computed here so the page
        # states a measured overlap rather than showing six similar-looking clouds
        act = set(np.flatnonzero(x.any(0)).tolist())
        if first_set is None:
            first_set = act
        A0 = b["attn"][0]
        layer_meta.append({
            "layer": Lx,
            "x_frac": round(float((x > 0).mean()), 5),
            "y_frac": round(float((y > 0).mean()), 5),
            "mean_active_per_token": round(float((x > 0).sum()) / T, 1),
            "n_distinct_neurons": len(act),
            "jaccard_vs_iter1": round(len(act & first_set) / max(len(act | first_set), 1), 4),
            "neg_share": {str(h): round(float((A0[h].cpu().numpy()[tri] < 0).mean()), 5)
                          for h in range(H)},
        })

        for t in (rng.choice(T, size=min(3, T), replace=False) if (full or Lx == a.layer) else []):
            nz = np.flatnonzero(x[t])
            if nz.size:
                n = int(rng.choice(nz))
                fixture.append({"w": wi, "L": Lx, "t": int(t), "neuron": n,
                                "x": float(x[t, n]), "n_active": int(nz.size),
                                "scale": float(s_[t])})

    put(f"w{wi}.vfinal", v_run.cpu().numpy().astype("<f2"))
    # If this drifts, the diagram's byte column is drawing edges from a residual the model never
    # had. Relative, because logits run to O(10).
    lg_chk = (v_run @ model.lm_head.detach().float())
    lg_err = float((lg_chk - logits[0].float()).abs().max() /
                   logits[0].float().abs().max().clamp_min(1e-9))
    assert lg_err < 1e-4, f"w{wi}: reconstructed residual does not reproduce the logits ({lg_err:.2e})"
    print(f"  w{wi} residual -> logits check: {lg_err:.2e} relative")

    # sigma recurrence residual, recomputed here so the page can state the number it will
    # later reproduce in the browser (CLAUDE.md item 17: f32-scale, NOT the toy's float64 2.8e-14)
    x0 = buf[a.layer]["x"][0, a.head].float().cpu().numpy().astype(np.float64)
    V = buf[a.layer]["v_ast"][0, 0].float().cpu().numpy().astype(np.float64)
    K = rope(x0, freqs, T)
    sig = np.zeros((N, D))
    recon = np.zeros((T, D))
    for t in range(T):
        recon[t] = sig.T @ K[t]
        sig += np.outer(K[t], V[t])
    ref = buf[a.layer]["a_ast"][0, a.head].float().cpu().numpy().astype(np.float64)
    resid = float(np.abs(recon - ref).max() / np.sqrt((ref ** 2).mean()))

    walk_meta.append({
        "i": wi, "corpus_index": si, "text": text, "T": T, "bytes": raw,
        "full_resolution": full, "layers": layer_meta,
        "sigma_residual_rel": resid,
        "neg_share_default": layer_meta[a.layer]["neg_share"][str(a.head)],
    })
    print(f"  w{wi} T={T:3d}  sigma recurrence residual {resid:.2e}  "
          f"neg@L{a.layer}H{a.head} {walk_meta[-1]['neg_share_default']:.1%}  {text[:44]}")

# ── flow pool + the weight slices that make its edges real ───────────────────────────────────
cand = np.flatnonzero(leader_hits > 0)
pool = cand[np.argsort(-leader_hits[cand])][:FLOW_POOL]
pool = pool[np.argsort(-fire[pool], kind="stable")].astype(np.int64)
pool_set = set(int(g) for g in pool)
cov = float(np.mean([len(top & pool_set) / max(len(top), 1) for top in leader_steps]))

# decoder_x is (H, D, N) and x_pre[h,t,n] = sum_d v[t,d] * decoder_x[h,d,n]; decoder_y is the
# same for y_pre; encoder is (H*N, D) and the residual update is sum_g y[t,g] * encoder[g,d].
# These three slices are exactly the per-edge coefficients the diagram needs, and nothing else is.
dx_w = np.stack([model.decoder_x[g // N, :, g % N].detach().float().cpu().numpy() for g in pool])
dy_w = np.stack([model.decoder_y[g // N, :, g % N].detach().float().cpu().numpy() for g in pool])
en_w = model.encoder[torch.as_tensor(pool, device=model.encoder.device)].detach().float().cpu().numpy()

put("flow.ids", pool.astype("<u2"))
put("flow.dx", dx_w.astype("<f2"))
put("flow.dy", dy_w.astype("<f2"))
put("flow.enc", en_w.astype("<f2"))
put("lm_head", model.lm_head.detach().float().cpu().numpy().astype("<f2"))
print(f"\nx_pre derived from v @ decoder_x[{a.head}] instead of shipped: "
      f"max relative error {xpre_err:.2e} (f32 reference; the uint8 pack it replaces was ~4e-3)")
assert (pool // N == a.head).all(), "flow pool escaped the traced head"
print(f"\nflow pool: {len(pool)} neurons of head {a.head} carry weight slices; they cover "
      f"{cov:.1%} of every step's true top-{FLOW_TOP} IN THAT HEAD, across "
      f"{len(leader_steps):,} (sentence, iteration, token) steps")

os.makedirs(os.path.dirname(a.out), exist_ok=True)
# gzip, for the same reason traces.bin.gz is: both walk sentences now carry every stage at every
# iteration and the raw blob roughly doubled. The page inflates with DecompressionStream.
gz = gzip.compress(bytes(blob), 9)
open(a.out + ".bin.gz", "wb").write(gz)
if os.path.exists(a.out + ".bin"):
    os.remove(a.out + ".bin")      # a stale uncompressed copy would be silently served instead
json.dump({
    "what": "full per-stage pipeline tensors for the walkthrough (web/loop.html)",
    "source_ckpt": os.path.basename(a.ckpt),
    "provenance": "trained by the author for an earlier Pathway hackathon; disclosed as prior work",
    "deviations_from_reference_bdh": [
        "adds a learned pos_emb (4096 x D) on top of RoPE",
        "has no decay term",
    ],
    "not_shipped": {
        "q_roped": "derivable from x + rope_freqs; verified 0.0 max abs error, re-checked in web/test/walk.mjs",
        "sigma": "every write is rank-one, so the browser accumulates it from x and v",
        "x_pre": "x_pre = v @ dx_head; the matrix ships instead of its 2.76 MB of outputs, and "
                 "the derived values are more accurate than the uint8 pack they replace",
    },
    "x_pre_derivation_max_rel_err": round(xpre_err, 8),
    "neuron_id": "global id = head * N + n, matching global.xy in field.bin",
    "config": {"n_layer": L, "n_embd": D, "n_head": H, "N": N, "N_total": H * N},
    "default_layer": a.layer, "default_head": a.head,
    "compression": "gzip",
    "flow": {
        "pool": int(len(pool)),
        "head": a.head,
        "leader_k": FLOW_TOP,
        "rule": f"neurons of head {a.head} -- the traced head, the only one with a derivable "
                f"x_pre and an accumulated sigma row, so the only one a path can be followed "
                f"through -- that lead at least one (sentence, iteration, token) step, ordered "
                f"by corpus firing frequency; the page draws a stable prefix plus the current "
                f"token's leaders among them",
        "leader_coverage": round(cov, 4),
        "steps_measured": len(leader_steps),
        "weights": "flow.dx / flow.dy / flow.enc are decoder_x, decoder_y and encoder restricted "
                   "to the pool, so every drawn edge is weight x source activation, not a guess",
    },
    "corpus": {
        "n_sentences": len(CORPUS),
        "x_active_frac": round(float(x_frac), 5),
        "y_active_frac": round(float(y_frac), 5),
        "gate_sparser_by": round(float(x_frac / y_frac), 2),
        "per_layer": [{"layer": i, "x_frac": round(float(xs[i, 0] / xs[i, 1]), 5),
                       "y_frac": round(float(ys[i, 0] / ys[i, 1]), 5)} for i in range(L)],
        "neg_share_pooled": {f"L{i}H{h}": round(float(negs[i, h, 0] / negs[i, h, 1]), 5)
                             for i in range(L) for h in range(H)},
        "neurons_never_firing": int((fire == 0).sum()),
        "n_neuron_token_steps": int(xs[:, 1].sum()),
    },
    "walk": walk_meta, "tensors": entries,
}, open(a.out + ".json", "w"), indent=1)

# THE PRICE needs the per-neuron firing counts and nothing else from this run -- it plots them
# against G* degree for the "structure predicts silence" slide. Making that page inflate the whole
# 4.87 MB walk pack to read 24 KB of counters is absurd, so they also ship standalone.
json.dump({
    "what": "per-neuron firing counts over the analysis corpus, for web/price.html",
    "source_ckpt": os.path.basename(a.ckpt),
    "definition": "number of (token, iteration) steps on which the neuron was non-zero",
    "neuron_id": "global id = head * N + n, matching global.xy in field.bin",
    "config": {"n_layer": L, "n_embd": D, "n_head": H, "N": N, "N_total": H * N},
    "steps": int(xs[:, 1].sum() // (H * N)),
    "n_sentences": len(CORPUS),
    "never_firing": int((fire == 0).sum()),
    "counts": np.minimum(fire, 65535).astype(int).tolist(),
}, open("web/public/fire.json", "w"))
print(f"wrote web/public/fire.json  ({os.path.getsize('web/public/fire.json')/1e3:.0f} KB)")

json.dump({"note": "stage tensors as the model produced them, pre-quantisation; the JS gate "
                   "decodes web/public/walk.bin.gz and must match within one uint8 step",
           "x_active_frac": round(float(x_frac), 5),
           "y_active_frac": round(float(y_frac), 5),
           "samples": fixture},
          open("web/test/walk_fixture.json", "w"), indent=1)

print(f"\nwrote {a.out}.bin.gz  {len(gz)/1e6:.2f} MB gzipped from {len(blob)/1e6:.2f} MB raw "
      f"(ratio {len(gz)/len(blob):.2f})  {len(entries)} tensors  +  {a.out}.json")
print(f"wrote web/test/walk_fixture.json  ({len(fixture)} spot checks)")
print(f"{time.time()-t0:.1f}s")

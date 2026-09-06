"""
Export per-token TRACES: what actually fires, token by token, when the 8M model reads French.

This is what animates THE FIELD. `export_field.py` fixes WHERE each of the 12,288 neurons sits;
this fixes WHICH of them light up at each token, and how brightly.

WHY THE PAYLOAD IS SPARSE, AND WHY THAT IS THE POINT
-----------------------------------------------------
The model runs at ~5% active (measured: x_active 5.15% over the analysis corpus), so storing
dense activations would be 95% zeros. Storing (index, value) pairs for the survivors is both
20x smaller AND a direct encoding of the claim the page is making: non-negativity plus a ReLU
means almost every neuron is silent almost always. The compression ratio IS the sparsity number.

Neuron ids are GLOBAL: id = h * N + n, matching the ordering of `global.xy` in field.bin, so a
trace index is directly a node index in the field.

WHAT delta-sigma IS HERE
------------------------
Each token adds exactly one rank-one write, delta-sigma_t = rope(K)_t (outer) V_t. Since V_t is
shared across all rows of that outer product, WHICH neurons the write lands on -- and how hard --
is given entirely by rope(K)_t, whose support is the active set. So the active-neuron payload is
the write pattern; no separate edge list is needed, and claiming one would be inventing
structure. (CLAUDE.md item 13: individual writes are rank-one, the accumulated state is dense.)

WHAT EVERY SENTENCE GETS
------------------------
All of it. Earlier revisions exported the six iterations, the attention scores and the sigma row
energy for the hero sentence ONLY, and one layer for the rest -- so on any other sentence the
iteration selector, the "all six iterations" view, the arc diagram and the sigma colour mode were
either dead or silently showed nothing. Every sentence now carries all six iterations plus
attention and sigma row energy at (--layer, --attn-head), and the manifest records the per-sentence
negative-score share and recurrence residual instead of one hero number.

That roughly trebles the raw blob, so it is gzipped on the way out; see the write block at the
bottom for why that is nearly free here.

QUANTISATION
------------
Values are uint8 against a per-(token, layer) max, which is stored as fp32. Activations are used
here only to drive brightness and node size, where 1/255 of the frame maximum is far below what
an eye resolves. Nothing numerical is claimed from these files -- the equivalence residual and
the capacity curves are computed live from the toy model, not read from here.
"""
import argparse, gzip, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bdh_big import load_big

# The analysis corpus from the training notebook: grouped so that concept structure is probeable
# (currency / country / institution), plus a deliberate repetition sentence for the Hebbian trace
# and two short ones. Kept verbatim so the numbers line up with the prior run's telemetry.
CORPUS = [
    "<F:en>The price was fifty euros and thirty pounds<T:fr>Le prix etait de cinquante euros et trente livres",
    "<F:en>The dollar strengthened against the yen today<T:fr>Le dollar s'est renforce face au yen aujourd'hui",
    "<F:en>Germany and France signed the bilateral treaty<T:fr>L'Allemagne et la France ont signe le traite bilateral",
    "<F:en>The European Parliament voted on this resolution<T:fr>Le parlement europeen a vote cette resolution",
    "<F:en>The Commission presented the annual budget report<T:fr>La Commission a presente le rapport budgetaire annuel",
    "<F:en>The budget was discussed. The budget was approved. The budget will be implemented.<T:fr>Le budget a ete discute. Le budget a ete approuve. Le budget sera mis en oeuvre.",
    "<F:en>Thank you very much<T:fr>Merci beaucoup",
]

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="models/checkpoints/french_best.pt")
ap.add_argument("--out", default="web/public/big/traces")
ap.add_argument("--hero", type=int, default=5,
                help="index into CORPUS marked as the default/starred sentence in the UI")
ap.add_argument("--layer", type=int, default=3,
                help="layer whose attention scores and sigma row energy are exported")
ap.add_argument("--attn-head", type=int, default=0)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()

t0 = time.time()
model, cfg, ck = load_big(a.ckpt, a.device)
D, H, N, L = cfg.n_embd, cfg.n_head, cfg.N, cfg.n_layer
print(f"{cfg.n_layer}L x D={D} x H={H}, N={N}/head, {H*N:,} neurons, val {ck['losses']['val']:.4f}\n")

blob, entries = bytearray(), []


def put(name, arr, **meta):
    b = np.ascontiguousarray(arr).tobytes()
    entries.append(dict(name=name, dtype=arr.dtype.str, shape=list(arr.shape),
                        offset=len(blob), bytes=len(b), **meta))
    blob.extend(b)


def sparse_pack(X):
    """(T, H*N) non-negative -> per-token (index, uint8 value) pairs + row offsets + scales."""
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


# Spot-check fixture for the JS parity gate. These are the activations as the MODEL produced
# them, before quantisation -- so the gate tests the whole chain (model -> pack -> bin -> JS
# decode), not just that numpy can read its own output back.
rng = np.random.default_rng(0)
fixture = []

sent_meta, total_active, total_cells = [], 0, 0
neg_hits, neg_total = 0, 0
for si, text in enumerate(CORPUS):
    raw = list(text.encode("utf-8"))
    ids = torch.tensor([raw], dtype=torch.long, device=a.device)
    with torch.no_grad(), model.extracting() as buf:
        model(ids)
    T = len(raw)

    per_layer = []
    for Lx in range(L):
        # x is (B, H, T, N) -> (T, H*N) with global id h*N+n, matching field.bin's global.xy
        x = buf[Lx]["x"][0].permute(1, 0, 2).reshape(T, H * N).float().cpu().numpy()
        i_, v_, s_, o_ = sparse_pack(x)
        put(f"s{si}.L{Lx}.idx", i_); put(f"s{si}.L{Lx}.val", v_)
        put(f"s{si}.L{Lx}.scale", s_); put(f"s{si}.L{Lx}.off", o_)
        act = i_.size / (T * H * N)
        total_active += i_.size; total_cells += T * H * N
        per_layer.append({"layer": Lx, "n_active": int(i_.size), "active_frac": round(act, 5)})

        for t in rng.choice(T, size=min(4, T), replace=False):
            nz = np.flatnonzero(x[t])
            if not nz.size:
                continue
            n = int(rng.choice(nz))
            fixture.append({"s": si, "L": Lx, "t": int(t), "neuron": n,
                            "value": float(x[t, n]), "n_active": int(nz.size),
                            "scale": float(s_[t])})

    # attention scores for the arc diagram: one head, one layer, strict-causal so the upper
    # triangle and the diagonal are structurally zero and are not stored.
    at = buf[a.layer]["attn"][0, a.attn_head].float().cpu().numpy()          # (T, T)
    tri = np.tril_indices(T, -1)
    w = at[tri].astype("<f4")
    put(f"s{si}.attn_i", tri[0].astype("<u2")); put(f"s{si}.attn_j", tri[1].astype("<u2"))
    put(f"s{si}.attn_w", w)
    neg = float((w < 0).mean())
    neg_hits += int((w < 0).sum()); neg_total += w.size

    # sigma row energy: sigma_t = sum_{s<t} rope(K)_s (outer) V_s, so ||sigma_t[n,:]|| is the
    # accumulated write weight on neuron n. Computed recurrently, which is also a live check
    # that the recurrent form reproduces the parallel one this model actually ran.
    K = buf[a.layer]["q_roped"][0, a.attn_head].float()                      # (T, N) rope'd keys
    V = buf[a.layer]["v_ast"][0, 0].float()                                  # (T, D)
    sig = torch.zeros(N, D, device=K.device)
    energy = np.zeros((T, N), dtype=np.float32)
    recon = torch.zeros(T, D, device=K.device)
    for t in range(T):
        energy[t] = sig.norm(dim=1).cpu().numpy()
        recon[t] = sig.T @ K[t]                                              # read BEFORE writing
        sig += torch.outer(K[t], V[t])
    # Relative to the RMS of what is being reconstructed. An absolute residual is meaningless
    # here: a_ast entries run to O(1e2), so 1e-4 absolute is 1e-6 relative -- ordinary float32
    # accumulation over T steps, not a broken recurrence. (The toy's 2.8e-14 in CLAUDE.md
    # item 3 is a float64 number and is not the comparison to make against this one.)
    ref = buf[a.layer]["a_ast"][0, a.attn_head].float()
    rms = ref.pow(2).mean().sqrt().clamp_min(1e-12)
    resid = ((recon - ref).abs().max() / rms).item()
    emax = energy.max(1, keepdims=True).clip(1e-9)
    put(f"s{si}.sigma_energy", np.clip(np.round(energy / emax * 255), 0, 255).astype(np.uint8))
    put(f"s{si}.sigma_energy_scale", emax[:, 0].astype("<f4"))

    sent_meta.append({"i": si, "text": text, "T": T, "layers": per_layer,
                      "bytes": raw, "hero": si == a.hero,
                      "n_scores": int(w.size), "neg_share": round(neg, 5),
                      "sigma_residual": float(resid)})
    print(f"s{si}: T={T:3d}  6 iterations  active "
          f"{np.mean([p['active_frac'] for p in per_layer]):.2%}  "
          f"{w.size:>6,} scores, {neg:5.1%} negative  sigma resid {resid:.1e}  {text[:34]}...")

os.makedirs(os.path.dirname(a.out), exist_ok=True)
# gzip, because every sentence now carries all six iterations plus attention and sigma, and
# the raw blob roughly trebled. The streams compress very unevenly -- sorted uint16 indices
# to ~0.39 and the smooth sigma-energy ramp to ~0.19, while the uint8 activation values are
# noise and barely move -- so the win is real and free. The browser inflates with
# DecompressionStream and the Node gates with zlib; nothing about the layout changes.
gz = gzip.compress(bytes(blob), 9)
open(a.out + ".bin.gz", "wb").write(gz)
if os.path.exists(a.out + ".bin"):
    os.remove(a.out + ".bin")        # stale uncompressed copy would be silently served
json.dump({
    "what": "per-token neuron activations (sparse), attention scores and sigma row energy",
    "source_ckpt": os.path.basename(a.ckpt),
    "neuron_id": "global id = head * N + n, matching global.xy in field.bin",
    "config": {"n_layer": L, "n_embd": D, "n_head": H, "N": N, "N_total": H * N},
    "hero": a.hero, "default_layer": a.layer, "attn_head": a.attn_head,
    "compression": "gzip",
    "every_sentence": {"layers": L, "attention": True, "sigma_energy": True},
    "attention_scope": f"layer {a.layer}, head {a.attn_head} (one layer/head for every sentence)",
    "measured_active_frac": round(total_active / total_cells, 5),
    "corpus_neg_share": round(neg_hits / neg_total, 5),
    "sentences": sent_meta, "tensors": entries,
}, open(a.out + ".json", "w"), indent=1)

json.dump({"note": "activations as the model produced them, pre-quantisation; the JS gate "
                   "decodes web/public/big/traces.bin.gz and must match within one uint8 step",
           "measured_active_frac": round(total_active / total_cells, 5),
           "samples": fixture},
          open("web/test/big_fixture.json", "w"), indent=1)
print(f"wrote web/test/big_fixture.json  ({len(fixture)} spot checks)")

print(f"\nmeasured mean active fraction: {total_active/total_cells:.2%} "
      f"({total_active:,} of {total_cells:,} neuron-token slots)")
print(f"corpus negative causal scores: {neg_hits/neg_total:.2%} "
      f"({neg_hits:,} of {neg_total:,}) at layer {a.layer} head {a.attn_head}")
print(f"{time.time()-t0:.1f}s")
print(f"wrote {a.out}.bin.gz  {len(gz)/1e6:.2f} MB gzipped from {len(blob)/1e6:.2f} MB raw "
      f"(ratio {len(gz)/len(blob):.2f})  {len(entries)} tensors  +  {a.out}.json")

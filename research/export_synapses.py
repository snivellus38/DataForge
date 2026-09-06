"""
Watch individual synapses fire.

BDH's interpretability claim is about SYNAPSES, not just neurons: a standing neuron-to-neuron
graph, with a dynamic state written on top of it. This exports both halves for a handful of
specific pairs so the page can show one synapse over the course of a sentence.

    G*[i,j] = (decoder_x[h]^T @ encoder[h]^T)[i,j]     the STANDING weight, from the weights alone
    co_ij(t) = x_i(t) * x_j(t)                          the HEBBIAN coactivation at token t

G* needs no data at all -- it is two weight matrices multiplied together. co_ij(t) is what the
sentence does to it. A synapse is "active" when a strongly wired pair is firing together, and
that is the pair worth drawing.

WHAT THIS IS NOT
----------------
This is not sigma. sigma is N x D (CLAUDE.md item 4) and is not a neuron-by-neuron matrix, so it
cannot be read as a synapse between two named neurons. G* can. Do not conflate them on screen:
G* is standing structure, sigma is the per-sequence state.

USAGE
    python research/export_synapses.py
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bdh_big import load_big

CORPUS = [
    "<F:en>The price was fifty euros and thirty pounds<T:fr>Le prix etait de cinquante euros et trente livres",
    "<F:en>The European Parliament voted on this resolution<T:fr>Le parlement europeen a vote cette resolution",
    "<F:en>The budget was discussed. The budget was approved. The budget will be implemented.<T:fr>Le budget a ete discute. Le budget a ete approuve. Le budget sera mis en oeuvre.",
    "<F:en>Thank you very much<T:fr>Merci beaucoup",
]

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="models/checkpoints/french_best.pt")
ap.add_argument("--out", default="web/public/synapses.json")
ap.add_argument("--head", type=int, default=0)
ap.add_argument("--layer", type=int, default=3)
ap.add_argument("--track", type=int, default=24, help="how many synapses to follow")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()

t0 = time.time()
model, cfg, ck = load_big(a.ckpt, a.device)
D, H, N = cfg.n_embd, cfg.n_head, cfg.N
h = a.head

# ── the standing graph, from the weights alone ───────────────────────────────────────────────
with torch.no_grad():
    G = (model.decoder_x[h].T @ model.encoder[h * N:(h + 1) * N].T).float()   # (N, N)
print(f"G* for head {h}: {G.shape[0]}x{G.shape[1]}, "
      f"range [{G.min():.3f}, {G.max():.3f}], computed from weights only")

# ── activations across the corpus, so "which pairs actually co-fire" is measured, not guessed ─
acts, meta = [], []
for si, text in enumerate(CORPUS):
    ids = torch.tensor([list(text.encode())], dtype=torch.long, device=a.device)
    with torch.no_grad(), model.extracting() as buf:
        model(ids)
    acts.append(buf[a.layer]["x"][0, h].float())          # (T, N)
    meta.append({"i": si, "text": text, "T": ids.shape[1], "bytes": list(text.encode())})

# Candidate pairs: strongly wired AND actually co-active somewhere. Ranking on G* alone would
# mostly surface pairs that never fire together, which teaches nothing about the sentence.
allx = torch.cat(acts, 0)                                  # (sum T, N)
live = (allx > 0).float().mean(0)                          # firing rate per neuron
cand = torch.where(live > 0.02)[0]
print(f"{len(cand)} neurons fire on >2% of tokens; ranking pairs among those")

sub = G[cand][:, cand].abs()
sub.fill_diagonal_(0)
co = (allx[:, cand] > 0).float()
cofire = (co.T @ co) / co.shape[0]                          # fraction of tokens both are on
score = sub * cofire                                        # wired AND co-active
k = min(a.track, score.numel())
flat = torch.topk(score.flatten(), k).indices
pairs = []
for f in flat.tolist():
    i, j = divmod(f, len(cand))
    gi, gj = int(cand[i]), int(cand[j])
    if gi == gj:
        continue
    pairs.append({"i": gi, "j": gj,
                  "w": round(float(G[gi, gj]), 5),
                  "cofire": round(float(cofire[i, j]), 4)})
pairs = pairs[:a.track]
print(f"tracking {len(pairs)} synapses")

# ── the timeline: co-activation of each tracked pair, token by token ─────────────────────────
sent_out = []
for si, x in enumerate(acts):
    T = x.shape[0]
    vals = []
    for p in pairs:
        v = (x[:, p["i"]] * x[:, p["j"]]).cpu().numpy()
        vals.append(v)
    V = np.stack(vals)                                      # (n_pairs, T)
    mx = float(V.max()) or 1.0
    sent_out.append({
        **meta[si],
        "peak": mx,
        "vals": [[round(float(v), 4) for v in row] for row in (V / mx)],
    })
    print(f"  s{si}  T={T:3d}  peak coactivation {mx:.3f}")

out = {
    "what": "standing synapse weights G* plus per-token Hebbian coactivation for tracked pairs",
    "source_ckpt": os.path.basename(a.ckpt),
    "definitions": {
        "G*": "decoder_x[h]^T @ encoder[h]^T -- the standing neuron-to-neuron weight, computed "
              "from the model's parameters with no data whatsoever",
        "coactivation": "x_i(t) * x_j(t) at layer %d, head %d -- what this sentence does to that "
                        "standing wire" % (a.layer, h),
    },
    "not_sigma": "sigma is N x D, not neuron-by-neuron, so it cannot be read as a weight between "
                 "two named neurons. G* can. These are different objects and the page must not "
                 "conflate them.",
    "caveats": [
        "Pairs are ranked by |G*| x co-firing rate over this 4-sentence corpus, so they are the "
        "pairs that matter HERE, not the globally strongest synapses.",
        "Head %d, layer %d only." % (h, a.layer),
        "Coactivation is normalised per sentence to its own peak, for display.",
    ],
    "config": {"n_layer": cfg.n_layer, "n_head": H, "N": N, "head": h, "layer": a.layer},
    "pairs": pairs,
    "sentences": sent_out,
}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
sz = os.path.getsize(a.out) / 1e3
print(f"\nwrote {a.out}  ({sz:.0f} KB)  {time.time() - t0:.0f}s")

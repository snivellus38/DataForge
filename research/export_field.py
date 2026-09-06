"""
Export THE FIELD: the neuron-and-synapse graph of a trained BDH, laid out for the browser.

WHAT G* IS, AND WHY IT IS THE RIGHT THING TO DRAW
-------------------------------------------------
BDH's claim to being "a network of neurons and synapses" is not a metaphor about the residual
stream -- it is a concrete matrix. Every layer READS the residual through decoder_x[h] (D x N)
and WRITES it back through encoder[h] (N x D). Composing the two gives a neuron -> neuron
operator that is INDEPENDENT of any input:

    G*[h] = decoder_x[h]^T @ encoder[h]^T          (N x N)

G*[i,j] is how strongly neuron j's write lands on neuron i's read, i.e. the standing synapse
between them. This is the object a Transformer has no analogue for: its per-layer projections
live in different bases, so there is no single neuron set to draw edges between. BDH is
weight-tied (CLAUDE.md item 6), so ONE G* governs every one of the n_layer iterations.

WHY THIS EXPORTER SHIPS A LAYOUT RATHER THAN EDGES ALONE
--------------------------------------------------------
Force-directed layout of ~12K nodes in the browser costs seconds and is non-deterministic across
machines. We solve it once here, ship (x, y) as quantised uint16, and the page becomes a pure
renderer. Determinism also means the figure in the README is the figure the reader sees.

THREE LAYOUTS ARE SHIPPED, CHOSEN BY LOOKING AT THEM
-----------------------------------------------------
  global    t-SNE over ALL H*N neurons at once on the [read ; write] signature -- neuron i's
            read vector decoder_x[h][:, i] concatenated with its write vector encoder[h][i, :],
            each half L2-normalised so neither dominates by scale. This is the hero: every
            neuron in the model in one cloud, and neurons that DO similar things sit together,
            which is what makes concept lighting legible.
  semantic  the same embedding computed per head.
  graph     Fruchterman-Reingold on the thresholded G*, over the CONNECTED CORE ONLY.

The first attempt laid every node out under FR and had to be thrown away. At the p99 threshold
39% of neurons have no edge, so they feel only repulsion and gravity, equilibrate at a single
radius, and draw a crisp ring whose radius is set by the gravity constant rather than by
anything in the model -- while squeezing the connected core into an unreadable dot. Rendering
that would have been a picture of my own hyperparameter. FR now runs on the connected subgraph
and `h*.xy_graph_placed` marks who has a position; the page states how many it omits.

Run with --preview DIR to write PNGs. Layout quality is a judgement about legibility, and the
silhouette/edge-length numbers this prints did not settle it -- looking at the images did.

HONESTY NOTE
------------
Everything here is computed from OUR OWN trained weights. Nothing is replayed from a paper.
Two numbers this prints are unflattering and must survive onto the page rather than being tuned
away by moving the threshold: modularity is only ~0.08, and the Louvain communities have no
spatial separation in either layout (silhouette ~ -0.12). The graph is heavy-tailed, not modular.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import networkx as nx

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="models/checkpoints/french_best.pt")
ap.add_argument("--out", default="web/public/big/field")
ap.add_argument("--percentile", type=float, default=99.0,
                help="edge threshold on |G*|; 99.0 reproduces the prior run's 94,373 edges")
ap.add_argument("--edges-per-head", type=int, default=6000, help="strongest edges shipped per head")
ap.add_argument("--tsne-iter", type=int, default=750)
ap.add_argument("--fr-iter", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--heads", default="all", help="'all' or a comma list, e.g. '0' while iterating")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--preview", default="", help="directory for PNG layout previews")
ap.add_argument("--report-only", action="store_true", help="print stats, write nothing")
a = ap.parse_args()

torch.manual_seed(a.seed)
np.random.seed(a.seed)
t0 = time.time()

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = ck["config"]
D, H = cfg["n_embd"], cfg["n_head"]
N = D * cfg["mlp_dim_mult"] // H
sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model_state_dict"].items()}
encoder = sd["encoder"].view(H, N, D).float()        # (H, N, D)  neuron -> residual  (WRITE)
decoder_x = sd["decoder_x"].float()                  # (H, D, N)  residual -> neuron  (READ)

val = ck.get("losses", {}).get("val", float("nan"))
print(f"ckpt {a.ckpt}")
print(f"  {cfg['n_layer']}L x D={D} x H={H}, N={N}/head, {H*N:,} neurons total, "
      f"iter {ck.get('iteration','?')}, val loss {val:.4f}")
print(f"  weight-tied: ONE G* per head governs all {cfg['n_layer']} iterations\n")

from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


def fr_layout(n, ei, ej, ew, iters, seed, device, gravity=0.02, chunk=512):
    """Fruchterman-Reingold with gravity, vectorised.

    networkx's spring_layout loops over nodes in Python once per iteration -- 3072 x 200 = 614K
    Python-level steps, which ran for over ten minutes without finishing a single head. The same
    algorithm as batched tensor ops takes seconds. Repulsion is all-pairs (O(n^2) = 9.4M pairs)
    but chunked over rows so peak memory stays near 12 MB regardless of n.

    GRAVITY IS NOT COSMETIC HERE. At the p99 threshold 39% of neurons have no edge at all, so
    under plain FR they feel repulsion and nothing else and stream off to the boundary, crushing
    the connected core into a dot. A weak pull toward the centroid holds the unconnected majority
    in a halo at a readable radius. It changes where isolated nodes sit, never which nodes are
    connected -- the edges shipped alongside are untouched by it.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    pos = (torch.rand((n, 2), generator=g, device=device, dtype=torch.float32) - 0.5) * 2.0
    ei = torch.as_tensor(np.asarray(ei), dtype=torch.long, device=device)
    ej = torch.as_tensor(np.asarray(ej), dtype=torch.long, device=device)
    w = torch.as_tensor(np.abs(np.asarray(ew)), dtype=torch.float32, device=device)
    w = (w / w.max().clamp_min(1e-12)).unsqueeze(-1)
    k = (1.0 / n) ** 0.5
    t = 0.1                                   # temperature: max step as a fraction of the domain
    dt = t / (iters + 1)
    for _ in range(iters):
        disp = torch.zeros_like(pos)
        for s in range(0, n, chunk):          # repulsion, chunked over rows
            e = min(s + chunk, n)
            d = pos[s:e, None, :] - pos[None, :, :]           # (chunk, n, 2)
            d2 = (d * d).sum(-1).clamp_min(1e-6)              # squared distance
            disp[s:e] = (d * (k * k / d2).unsqueeze(-1)).sum(1)
        de = pos[ei] - pos[ej]                                # attraction along synapses
        dl = de.norm(dim=-1, keepdim=True).clamp_min(1e-3)
        fa = de * (dl / k) * w                                # |f_a| = d^2/k, direction de/dl
        disp.index_add_(0, ei, -fa)
        disp.index_add_(0, ej, fa)
        c = pos - pos.mean(0, keepdim=True)                   # gravity toward the centroid
        disp = disp - c * (gravity / k)
        dlen = disp.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        pos = pos + disp / dlen * dlen.clamp(max=t)           # step capped by temperature
        t -= dt
    return pos.double().cpu().numpy()


want = list(range(H)) if a.heads == "all" else [int(x) for x in a.heads.split(",")]
heads, report = [], []
for h in want:
    G = (decoder_x[h].T @ encoder[h].T).numpy()      # (N, N) = (N,D) @ (D,N)
    absG = np.abs(G)
    thr = float(np.percentile(absG, a.percentile))
    ii, jj = np.where(absG >= thr)
    keep = ii != jj                                   # drop self-synapses; they are not edges
    ii, jj = ii[keep], jj[keep]
    w = G[ii, jj]

    out_deg = np.bincount(ii, minlength=N).astype(np.int32)
    in_deg = np.bincount(jj, minlength=N).astype(np.int32)

    # --- communities: networkx >= 3.0 ships Louvain, so python-louvain is not needed ----------
    UG = nx.Graph()
    UG.add_nodes_from(range(N))
    UG.add_weighted_edges_from(zip(ii.tolist(), jj.tolist(), np.abs(w).tolist()))
    comms = nx.community.louvain_communities(UG, weight="weight", seed=a.seed)
    comms = sorted(comms, key=len, reverse=True)
    mod = nx.community.modularity(UG, comms, weight="weight")
    # 39% of neurons have no edge at the p99 threshold, so Louvain returns each of them as its own
    # singleton community. The prior run reported "12 clusters" only because it built its graph
    # from edges alone and never added the isolated nodes. Both counts are reported here; the
    # singleton count IS the heavy tail and belongs on the page, not swept under a threshold.
    real = [c for c in comms if len(c) > 1]
    n_singleton = len(comms) - len(real)
    cluster = np.full(N, 255, dtype=np.uint8)     # 255 = unclustered / isolated
    for c, members in enumerate(real[:255]):
        cluster[list(members)] = c
    deg_all = np.bincount(np.concatenate([ii, jj]), minlength=N)
    n_isolated = int((deg_all == 0).sum())
    half = int((np.sort(deg_all)[::-1].cumsum() < deg_all.sum() * 0.5).sum())

    # --- layout A: force-directed, CONNECTED CORE ONLY -----------------------------------------
    # Laying out all N nodes together produced a lie: isolated neurons feel only repulsion and
    # gravity, so they equilibrate at one radius and draw a perfect ring whose radius is set by
    # my gravity constant, not by the model -- while crushing the connected core to a dot. The
    # honest picture lays out only the nodes that HAVE synapses; the page states the count of
    # those it left out rather than drawing them somewhere arbitrary.
    t1 = time.time()
    core = np.flatnonzero(deg_all > 0)
    remap = np.full(N, -1, dtype=np.int64)
    remap[core] = np.arange(len(core))
    xy_core = fr_layout(len(core), remap[ii], remap[jj], w, a.fr_iter, a.seed, a.device)
    xy_graph = np.full((N, 2), np.nan)
    xy_graph[core] = xy_core
    t_fr = time.time() - t1

    # --- layout B: t-SNE on each neuron's [read ; write] signature -----------------------------
    # Both halves are L2-normalised separately so neither dominates by scale: read vectors and
    # write vectors are trained under different gradients and their norms are not comparable.
    read = decoder_x[h].T.numpy()                     # (N, D)
    write = encoder[h].numpy()                        # (N, D)
    nrm = lambda M: M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    sig = np.hstack([nrm(read), nrm(write)]).astype(np.float32)
    t1 = time.time()
    xy_sem = TSNE(n_components=2, init="pca", random_state=a.seed, perplexity=40,
                  max_iter=a.tsne_iter, verbose=0).fit_transform(sig)
    t_sne = time.time() - t1

    def score(xy, name):
        ok = np.isfinite(xy).all(1)                       # graph layout omits isolated neurons
        xy = (xy - np.nanmean(xy, 0)) / (np.nanstd(xy, 0).mean() + 1e-9)
        elen = np.linalg.norm(xy[ii] - xy[jj], axis=1)
        # only over non-singleton communities: with 39% of neurons isolated, every singleton
        # scores a perfect 0 and drags the mean toward noise.
        m = (cluster != 255) & ok
        sil = float(silhouette_score(xy[m], cluster[m])) if len(set(cluster[m])) > 1 else float("nan")
        return {"layout": name, "n_placed": int(ok.sum()),
                "median_edge_len": float(np.median(elen)),
                "p90_edge_len": float(np.percentile(elen, 90)), "louvain_silhouette": sil}

    sg, ss = score(xy_graph, "graph"), score(xy_sem, "semantic")
    report += [dict(head=h, **sg), dict(head=h, **ss)]
    print(f"head {h}: {len(ii):>7,} edges @ |G*|>={thr:.4f}  modularity {mod:.4f}")
    print(f"         {len(real)} communities of size>1 (+{n_singleton} singletons), "
          f"sizes {[len(c) for c in real[:8]]}")
    print(f"         heavy tail: {n_isolated} of {N} neurons ({n_isolated/N:.0%}) have NO edge; "
          f"{half} neurons ({half/N:.0%}) carry half the edge endpoints; "
          f"max out-degree {out_deg.max()} (neuron {int(out_deg.argmax())})")
    print(f"         layout graph    median edge {sg['median_edge_len']:.3f}  "
          f"silhouette {sg['louvain_silhouette']:+.3f}  ({t_fr:.1f}s)")
    print(f"         layout semantic median edge {ss['median_edge_len']:.3f}  "
          f"silhouette {ss['louvain_silhouette']:+.3f}  ({t_sne:.1f}s)")

    # --- ship only the strongest edges ---------------------------------------------------------
    order = np.argsort(-np.abs(w))[: a.edges_per_head]
    heads.append(dict(h=h, thr=thr, n_edges_total=int(len(ii)), modularity=float(mod),
                      n_comms=len(comms), comm_sizes=[len(c) for c in comms[:32]],
                      ei=ii[order].astype("<u2"), ej=jj[order].astype("<u2"),
                      ew=w[order].astype("<f4"),
                      n_comms_real=len(real), n_singleton=n_singleton,
                      n_isolated=n_isolated, n_carry_half=half,
                      out_deg=out_deg, in_deg=in_deg, cluster=cluster,
                      xy_graph=xy_graph, xy_sem=xy_sem))

    if a.preview:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(a.preview, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(16, 16), facecolor="#141413")
        deg_c = np.log1p(deg_all)
        for ax, (xy, nm) in zip(axes.ravel()[:2], [(xy_graph, "graph"), (xy_sem, "semantic")]):
            f = np.isfinite(xy).all(1)
            ax.scatter(xy[f, 0], xy[f, 1], c=deg_c[f], s=3, cmap="magma", linewidths=0)
            ax.set_title(f"h{h} {nm} - colour = log degree", color="w", fontsize=11)
        for ax, (xy, nm) in zip(axes.ravel()[2:], [(xy_graph, "graph"), (xy_sem, "semantic")]):
            f = np.isfinite(xy).all(1)
            m = (cluster != 255) & f
            ax.scatter(xy[~m & f, 0], xy[~m & f, 1], c="#333", s=2, linewidths=0)
            ax.scatter(xy[m, 0], xy[m, 1], c=cluster[m] % 20, s=3, cmap="tab20", linewidths=0)
            ax.set_title(f"h{h} {nm} - colour = Louvain community", color="w", fontsize=11)
        for ax in axes.ravel():
            ax.set_facecolor("#141413"); ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(); fig.savefig(f"{a.preview}/field_h{h}.png", dpi=110,
                                        facecolor="#141413"); plt.close(fig)
        print(f"         preview -> {a.preview}/field_h{h}.png")

# ── the global layout: all H*N neurons in ONE cloud ──────────────────────────────────────────
# The per-head layouts answer "how is this head organised". The hero shot needs the whole model
# at once -- "every dot is a neuron, you are looking at all 12,288 of them" -- so this embeds
# every neuron of every head in a single t-SNE over the same [read ; write] signature. It also
# settles a question we cannot answer by inspection: do the four heads occupy DIFFERENT regions
# of read/write space, or are they interleaved? The printed head-silhouette is that answer.
xy_global = head_of = None
if len(want) == H:
    t1 = time.time()
    nrm = lambda M: M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    sig_all = np.vstack([np.hstack([nrm(decoder_x[h].T.numpy()), nrm(encoder[h].numpy())])
                         for h in range(H)]).astype(np.float32)
    head_of = np.repeat(np.arange(H), N).astype(np.uint8)
    xy_global = TSNE(n_components=2, init="pca", random_state=a.seed, perplexity=50,
                     max_iter=a.tsne_iter, verbose=0).fit_transform(sig_all)
    hs = float(silhouette_score(xy_global, head_of))
    print(f"global: {H*N:,} neurons in one t-SNE  ({time.time()-t1:.1f}s)")
    print(f"        head separation silhouette {hs:+.3f}  "
          f"({'heads occupy distinct regions' if hs > 0.25 else 'heads are interleaved, not spatially distinct'})")
    report.append({"layout": "global", "head_silhouette": hs, "n_placed": int(H * N)})

    if a.preview:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 11), facecolor="#141413")
        ax.scatter(xy_global[:, 0], xy_global[:, 1], c=head_of, s=2, cmap="tab10", linewidths=0)
        ax.set_title(f"all {H*N:,} neurons, colour = head  (silhouette {hs:+.3f})",
                     color="w", fontsize=12)
        ax.set_facecolor("#141413"); ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(); fig.savefig(f"{a.preview}/field_global.png", dpi=110,
                                        facecolor="#141413"); plt.close(fig)
        print(f"        preview -> {a.preview}/field_global.png")

print(f"\n{time.time()-t0:.1f}s")
if a.report_only:
    sys.exit(0)

# -- pack -------------------------------------------------------------------------------------
# Positions are quantised to uint16 over each head's own bounding box; the manifest carries the
# box so the page can un-quantise. 1/65535 of a screen width is far below one device pixel, so
# the quantisation is invisible and it halves the payload versus float32.
def quant(xy):
    ok = np.isfinite(xy).all(1)
    lo, hi = np.nanmin(xy, 0), np.nanmax(xy, 0)
    z = np.where(ok[:, None], xy, lo)                     # placeholder; consumers gate on `placed`
    q = np.clip(np.round((z - lo) / (hi - lo + 1e-12) * 65535), 0, 65535).astype("<u2")
    return q, lo.astype(float).tolist(), hi.astype(float).tolist(), ok.astype(np.uint8)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
blob, entries = bytearray(), []

def put(name, arr, **meta):
    b = np.ascontiguousarray(arr).tobytes()
    entries.append(dict(name=name, dtype=arr.dtype.str, shape=list(arr.shape),
                        offset=len(blob), bytes=len(b), **meta))
    blob.extend(b)

head_meta = []
for hd in heads:
    h = hd["h"]
    qg, glo, ghi, okg = quant(hd["xy_graph"])
    qs, slo, shi, _ = quant(hd["xy_sem"])
    put(f"h{h}.xy_graph", qg, box=[glo, ghi])
    put(f"h{h}.xy_graph_placed", okg)                     # 0 = isolated, has no graph position
    put(f"h{h}.xy_semantic", qs, box=[slo, shi])
    put(f"h{h}.cluster", hd["cluster"])
    put(f"h{h}.out_deg", hd["out_deg"].astype("<u2"))
    put(f"h{h}.in_deg", hd["in_deg"].astype("<u2"))
    put(f"h{h}.edge_i", hd["ei"])
    put(f"h{h}.edge_j", hd["ej"])
    put(f"h{h}.edge_w", hd["ew"])
    head_meta.append({k: hd[k] for k in ("h", "thr", "n_edges_total", "modularity",
                                         "n_comms", "comm_sizes")})

if xy_global is not None:
    qg, glo, ghi, _ = quant(xy_global)
    put("global.xy", qg, box=[glo, ghi])
    put("global.head", head_of)

open(a.out + ".bin", "wb").write(bytes(blob))
json.dump({
    "what": "G* = decoder_x[h]^T @ encoder[h]^T, the standing neuron->neuron synapse matrix",
    "source_ckpt": os.path.basename(a.ckpt),
    "provenance": "trained by the author for an earlier Pathway hackathon; disclosed as prior work",
    "config": cfg, "N_per_head": N, "n_neurons_total": H * N,
    "iteration": ck.get("iteration"), "val_loss": ck.get("losses", {}).get("val"),
    "percentile": a.percentile, "edges_per_head": a.edges_per_head, "seed": a.seed,
    "heads": head_meta, "layout_report": report, "tensors": entries,
}, open(a.out + ".json", "w"), indent=1)

sz = len(blob) / 1e6
print(f"wrote {a.out}.bin  {sz:.2f} MB  ({len(entries)} tensors)  +  {a.out}.json")
print(f"budget: {sz:.2f} / 5.00 MB used by the field")

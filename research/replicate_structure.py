#!/usr/bin/env python
"""Do the structural findings replicate on a second, independently trained model?

WHY THIS SCRIPT EXISTS
----------------------
Every structural number this project reports -- 5.13% of neurons active, the heavy-tailed synapse
graph G*, and the big one, "G* predicts which neurons stay silent from the weights alone" at
MCC +0.944 -- was measured on ONE checkpoint, `french_best.pt`. A property of one checkpoint and a
property of the architecture are different claims, and only the second is worth teaching.

The control is sitting in the same directory. `portuguese_best.pt` is the same architecture and the
same recipe trained on a different language pair, from a different random initialisation, for a
different number of iterations (40,000 against 50,000). It shares no weights with the French model
-- that is exactly what `research/merge_replicate.py` established the hard way when merging the two
collapsed. So if a number reproduces across them, it is a fact about the architecture and not a
fact about one run.

WHAT IS COMPARED, AND ON WHAT
-----------------------------
Two kinds of measurement, and the distinction matters when reading the output.

  * WEIGHTS ONLY -- G*'s edge count, isolated fraction, degree concentration. No data enters these
    at all: G*[h] = decoder_x[h].T @ encoder[h].T. A replication here is as clean as it gets,
    because there is no corpus to confound it.

  * WEIGHTS AND DATA -- activation sparsity, negative attention share, and the silence prediction.
    These need text, and the two models do not speak the same language. Feeding French to the
    Portuguese model would measure how badly it handles out-of-distribution input, not how its
    neurons behave. So each model is measured on ITS OWN output: the same seven English source
    sentences, each completed greedily by the model in its own target language. Same inputs, same
    decoding, each model in the regime it was trained for.

    That is the fair comparison and it is not a perfectly controlled one -- the two corpora differ
    in content and slightly in length. Say so when quoting these rows. The weights-only rows carry
    no such caveat.

THE THRESHOLD, AND WHY IT IS A PERCENTILE
-----------------------------------------
"Isolated" is defined by a threshold on |G*|, and the count moves if the threshold moves. We use
the p99 of |G*| per head, the same convention as `research/export_field.py`, so the edge budget is
identical for both models by construction and only the STRUCTURE of the graph can differ. A fixed
absolute threshold would confound structure with weight scale, which is not what we are asking.

USAGE
    python research/replicate_structure.py
    python research/replicate_structure.py --device cpu --out research/runs/replication.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from bdh_big import load_big  # noqa: E402

# The seven English sources from the analysis corpus. Each model completes these itself, in its own
# target language, so neither is measured on text the other produced.
SOURCES = [
    "The price was fifty euros and thirty pounds",
    "The dollar strengthened against the yen today",
    "Converting francs to marks was expensive",
    "Germany and France signed the bilateral treaty",
    "Sweden and Finland proposed a new allocation",
    "The European Parliament voted against this resolution",
    "The budget was discussed at length yesterday",
]

MODELS = [
    ("french", "models/checkpoints/french_best.pt", "fr"),
    ("portuguese", "models/checkpoints/portuguese_best.pt", "pt"),
]


@torch.no_grad()
def complete(model, prompt, n, device):
    """Greedy byte-level completion, stopping if the model starts a fresh example."""
    ids = list(prompt.encode("utf-8"))
    start = len(ids)
    for _ in range(n):
        logits, _ = model(torch.tensor([ids], dtype=torch.long, device=device))
        ids.append(int(logits[0, -1].argmax()))
        if b"<F:en>" in bytes(ids[start:]):
            ids = ids[:-len(b"<F:en>")]
            break
    return ids


@torch.no_grad()
def activation_stats(model, cfg, corpora, device):
    """x and y sparsity per iteration, the negative causal-score share, and which neurons fire."""
    H, N, L = cfg.n_head, cfg.N, cfg.n_layer
    x_nz = np.zeros(L, dtype=np.int64)
    y_nz = np.zeros(L, dtype=np.int64)
    slots = np.zeros(L, dtype=np.int64)
    neg = tot = 0
    neg_l = np.zeros(L, dtype=np.int64)
    tot_l = np.zeros(L, dtype=np.int64)
    neg_l3h0 = tot_l3h0 = 0
    per_sentence_neg = []
    ever = np.zeros(H * N, dtype=bool)   # fired at least once anywhere in the corpus

    for ids in corpora:
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        with model.extracting() as buf:
            model(idx)
        s_neg = s_tot = 0
        for li, d in buf.items():
            x = d["x"][0]                     # (H, T, N)
            y = d["y"][0]
            x_nz[li] += int((x > 0).sum())
            y_nz[li] += int((y > 0).sum())
            slots[li] += x.numel()
            ever |= (x > 0).any(dim=1).reshape(-1).cpu().numpy()
            sc = d["attn"][0]                 # (H, T, T), zero on and above the diagonal
            T = sc.shape[-1]
            causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=sc.device), -1)
            vals = sc[:, causal]
            s_neg += int((vals < 0).sum())
            s_tot += int(vals.numel())
            neg_l[li] += int((vals < 0).sum())
            tot_l[li] += int(vals.numel())
            if li == 3:
                h0 = sc[0][causal]
                neg_l3h0 += int((h0 < 0).sum())
                tot_l3h0 += int(h0.numel())
        neg += s_neg
        tot += s_tot
        per_sentence_neg.append(s_neg / max(s_tot, 1))

    return {
        "x_active": float(x_nz.sum() / slots.sum()),
        "y_active": float(y_nz.sum() / slots.sum()),
        "x_active_per_iteration": [round(float(a / b), 5) for a, b in zip(x_nz, slots)],
        "y_active_per_iteration": [round(float(a / b), 5) for a, b in zip(y_nz, slots)],
        "neuron_token_slots": int(slots.sum()),
        "neg_score_share_pooled": float(neg / max(tot, 1)),
        "neg_score_share_per_iteration": [round(float(a / max(b, 1)), 4) for a, b in zip(neg_l, tot_l)],
        "neg_score_share_L3H0": float(neg_l3h0 / max(tot_l3h0, 1)),
        "neg_score_share_per_sentence": [round(v, 4) for v in per_sentence_neg],
        "causal_scores": int(tot),
        "silent_fraction": float(1.0 - ever.mean()),
    }, ever


def graph_stats(sd, cfg, pct=99.0):
    """G*[h] = decoder_x[h].T @ encoder[h].T, thresholded at a percentile of |G*|.

    Weights only. No data enters this function."""
    H, N, D = cfg.n_head, cfg.N, cfg.n_embd
    encoder = sd["encoder"].view(H, N, D).float()
    decoder_x = sd["decoder_x"].float()
    per_head = []
    degree = np.zeros(H * N, dtype=np.int64)
    for h in range(H):
        G = (decoder_x[h].T @ encoder[h].T).abs().numpy()
        thr = float(np.percentile(G, pct))
        ei, ej = np.nonzero(G >= thr)
        out_deg = np.bincount(ei, minlength=N)
        endpoints = out_deg + np.bincount(ej, minlength=N)
        degree[h * N:(h + 1) * N] = endpoints
        order = np.sort(endpoints)[::-1]
        cum = np.cumsum(order) / max(order.sum(), 1)
        half = int(np.searchsorted(cum, 0.5)) + 1
        per_head.append({
            "head": h,
            "edges": int(ei.size),
            "threshold": thr,
            "isolated_share": float((endpoints == 0).mean()),
            "max_out_degree": int(out_deg.max()),
            "neurons_carrying_half_the_endpoints": round(float(half / N), 4),
        })
    return {
        "per_head": per_head,
        "edges_total": int(sum(p["edges"] for p in per_head)),
        "isolated_share": float((degree == 0).mean()),
    }, degree


def mcc(a, b):
    """Matthews correlation between two boolean vectors, plus the confusion counts."""
    tp = int((a & b).sum())
    tn = int((~a & ~b).sum())
    fp = int((a & ~b).sum())
    fn = int((~a & b).sum())
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return ((tp * tn - fp * fn) / den if den > 0 else 0.0), (tp, fp, fn, tn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gen", type=int, default=90, help="bytes generated per source sentence")
    ap.add_argument("--pct", type=float, default=99.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "research/runs/replication.json"))
    args = ap.parse_args()

    results = {}
    for name, rel, tag in MODELS:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            print("missing %s -- download it with:\n  gh release download v1.0-models -p '%s' "
                  "-D models/checkpoints/" % (rel, os.path.basename(rel)))
            return 1
        model, cfg, ck = load_big(path, device=args.device)
        sd = {k.replace("_orig_mod.", ""): v.cpu() for k, v in ck["model_state_dict"].items()}

        corpora = [complete(model, "<F:en>%s<T:%s>" % (s, tag), args.gen, args.device)
                   for s in SOURCES]
        texts = [bytes(c).decode("utf-8", errors="replace") for c in corpora]

        act, ever = activation_stats(model, cfg, corpora, args.device)
        gph, degree = graph_stats(sd, cfg, args.pct)

        silent = ~ever
        isolated = degree == 0
        m, (tp, fp, fn, tn) = mcc(isolated, silent)

        results[name] = {
            "checkpoint": rel,
            "iterations": int(ck.get("iteration", -1)),
            "val_loss": float(ck.get("losses", {}).get("val", float("nan"))),
            "params": int(sum(p.numel() for p in model.parameters())),
            "config": {"n_layer": cfg.n_layer, "n_embd": cfg.n_embd, "n_head": cfg.n_head,
                       "N": cfg.N, "N_total": cfg.N_total},
            "corpus_bytes": int(sum(len(c) for c in corpora)),
            "activations": act,
            "graph": gph,
            "silence_prediction": {
                "mcc": round(float(m), 4),
                "p_silent_given_isolated": round(float(tp / max(tp + fp, 1)), 4),
                "base_rate_silent": round(float(silent.mean()), 4),
                "mean_degree_silent": round(float(degree[silent].mean()), 2),
                "mean_degree_firing": round(float(degree[~silent].mean()), 2),
                "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            },
            "samples": [t.split("<T:%s>" % tag, 1)[-1] for t in texts],
        }
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"percentile": args.pct, "sources": SOURCES, "models": results}, f, indent=2)

    fr, pt = results["french"], results["portuguese"]
    w = 36
    print("\n" + "=" * 78)
    print("  DOES IT REPLICATE?   french_best (50k iters)   vs   portuguese_best (40k iters)")
    print("=" * 78)
    rows = [
        ("__WEIGHTS ONLY -- no data involved", None, None),
        ("G* edges at p%g (fixed by the threshold)" % args.pct,
         "{:,}".format(fr["graph"]["edges_total"]), "{:,}".format(pt["graph"]["edges_total"])),
        ("neurons with no synapse",
         "%.1f%%" % (100 * fr["graph"]["isolated_share"]),
         "%.1f%%" % (100 * pt["graph"]["isolated_share"])),
        ("max out-degree (head 0)",
         fr["graph"]["per_head"][0]["max_out_degree"], pt["graph"]["per_head"][0]["max_out_degree"]),
        ("share carrying half the endpoints",
         "%.1f%%" % (100 * fr["graph"]["per_head"][0]["neurons_carrying_half_the_endpoints"]),
         "%.1f%%" % (100 * pt["graph"]["per_head"][0]["neurons_carrying_half_the_endpoints"])),
        ("__WEIGHTS AND DATA -- each on its own output", None, None),
        ("x active",
         "%.2f%%" % (100 * fr["activations"]["x_active"]), "%.2f%%" % (100 * pt["activations"]["x_active"])),
        ("y active (the gate)",
         "%.2f%%" % (100 * fr["activations"]["y_active"]), "%.2f%%" % (100 * pt["activations"]["y_active"])),
        ("negative causal scores, all layers",
         "%.1f%%" % (100 * fr["activations"]["neg_score_share_pooled"]),
         "%.1f%%" % (100 * pt["activations"]["neg_score_share_pooled"])),
        ("  ...at L3/H0 (the figure the repo quotes)",
         "%.1f%%" % (100 * fr["activations"]["neg_score_share_L3H0"]),
         "%.1f%%" % (100 * pt["activations"]["neg_score_share_L3H0"])),
        ("  ...iteration 1 -> 6",
         "%.0f%% -> %.0f%%" % (100 * fr["activations"]["neg_score_share_per_iteration"][0],
                               100 * fr["activations"]["neg_score_share_per_iteration"][-1]),
         "%.0f%% -> %.0f%%" % (100 * pt["activations"]["neg_score_share_per_iteration"][0],
                               100 * pt["activations"]["neg_score_share_per_iteration"][-1])),
        ("neurons never firing",
         "%.1f%%" % (100 * fr["activations"]["silent_fraction"]),
         "%.1f%%" % (100 * pt["activations"]["silent_fraction"])),
        ("MCC(isolated, silent)",
         "%+.3f" % fr["silence_prediction"]["mcc"], "%+.3f" % pt["silence_prediction"]["mcc"]),
        ("P(silent | isolated)",
         "%.1f%%" % (100 * fr["silence_prediction"]["p_silent_given_isolated"]),
         "%.1f%%" % (100 * pt["silence_prediction"]["p_silent_given_isolated"])),
        ("mean degree, silent vs firing",
         "%.1f / %.1f" % (fr["silence_prediction"]["mean_degree_silent"],
                          fr["silence_prediction"]["mean_degree_firing"]),
         "%.1f / %.1f" % (pt["silence_prediction"]["mean_degree_silent"],
                          pt["silence_prediction"]["mean_degree_firing"])),
    ]
    for label, a, b in rows:
        if a is None:
            print("\n  " + label[2:])
        else:
            print("  %-*s %14s %14s" % (w, label, a, b))
    print("\n  fr:", fr["samples"][5][:72])
    print("  pt:", pt["samples"][5][:72])
    print("\n  wrote", os.path.relpath(args.out, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

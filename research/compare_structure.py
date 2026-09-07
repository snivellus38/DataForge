#!/usr/bin/env python
"""Sparsity and weight-only structure: our BDH beside a Transformer, and beside its own null.

WHY THIS SCRIPT EXISTS
----------------------
Two of this project's results are stated without anything to compare them to:

    "5.13% of neurons are active"          -- sparse compared to what?
    "G* predicts which neurons stay silent
     from the weights alone, MCC +0.944"   -- is that special, or would any network do it?

The first question needs a second architecture. The second needs a NULL, and the null is the more
important of the two, because it is the one that makes our own claim falsifiable regardless of
what any other model does.

WHAT IS MEASURED

1. ACTIVATION DENSITY, and the definitional trap in it.
   BDH's x = ReLU(...) produces EXACT zeros, so "active" is unambiguous. A Transformer's MLP uses
   GELU, which is never exactly zero -- so "> 0" is roughly half of everything by symmetry and
   means nothing. Reporting that as "the Transformer is 50% dense" would be a cheap win off a bad
   definition. Three predicates are therefore reported for both models: exactly zero, greater than
   zero, and above 1% of that token's largest activation. The last is the one to compare, and it
   is the one where a GELU network can in principle look sparse.

2. WHETHER STRUCTURE PREDICTS FUNCTION, in each architecture and against a null.
   For BDH, G*[h] = decoder_x[h].T @ encoder[h].T is a neuron -> neuron map over ONE shared
   population, because the model is weight-tied: the same neurons, the same matrices, every
   iteration. Isolation in that graph predicts silence on real text.

   The closest thing a Transformer has is W_out[l] @ W_in[l+1]: the MLP neurons of layer l mapped
   onto the MLP neurons of layer l+1. It is a real object and it is computed here, but it is not
   the same KIND of object, and that difference is the actual finding rather than whichever MCC
   comes out larger. A Transformer has no single neuron population -- layer l and layer l+1 hold
   different neurons -- and its token mixing runs through a softmax whose weights are computed
   from the data and are nowhere in the weights. So the question "which neurons will never fire,
   derived from the weights alone" does not have a well-posed answer there in the way it does for
   a weight-tied model.

3. THE NULL, which is what licenses claim 2 for our own model.
   Degrees are shuffled across neurons and the prediction is recomputed. If MCC survives that, the
   result was never about G* -- it was about the base rates. It does not survive: that is the
   point of running it.

WHAT IS NOT CONTROLLED, STATED PLAINLY
--------------------------------------
GPT-2 is a different model trained by other people on other data at another scale. Nothing here
isolates an architecture's contribution. Each model is observed in its own regime -- GPT-2 on
English, ours on the French it generates -- because measuring GPT-2 on tagged French would
describe how it handles out-of-distribution input rather than how its neurons behave. To show that
choice is not doing the work, GPT-2 is ALSO measured on the French text, and the two densities are
printed side by side: if they barely move, the choice of text is not what produces the contrast.

USAGE
    python research/compare_structure.py
    python research/compare_structure.py --skip-hf      # ours and the null only, no downloads
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
from replicate_structure import SOURCES, complete, mcc  # noqa: E402

ENGLISH = (
    "The Commission has presented a report on this question and the Council has taken note of it. "
    "In accordance with the rules of procedure the President opened the debate on the proposal. "
    "Several members asked for the vote to be postponed until the next part-session. "
    "It is impossible to say with any precision how long the negotiations will take, but the "
    "committee has agreed to review the matter again before the end of the current year. "
)


def density(acts, thresh=0.01):
    """Three predicates on a (tokens, units) activation matrix. See the docstring's trap note."""
    a = acts.abs()
    peak = a.max(dim=-1, keepdim=True).values.clamp_min(1e-9)
    return {
        "exactly_zero": float((acts == 0).float().mean()),
        "greater_than_zero": float((acts > 0).float().mean()),
        "above_1pct_of_peak": float((a > thresh * peak).float().mean()),
    }


@torch.no_grad()
def bdh_side(model, cfg, corpora, device, pct):
    """BDH: activation density on x, plus whether G* isolation predicts silence -- and the null."""
    H, N = cfg.n_head, cfg.N
    ever = np.zeros(H * N, dtype=bool)
    dens = []
    for ids in corpora:
        with model.extracting() as buf:
            model(torch.tensor([ids], dtype=torch.long, device=device))
        for d in buf.values():
            x = d["x"][0]                                     # (H, T, N)
            flat = x.permute(1, 0, 2).reshape(x.shape[1], -1)  # (T, H*N)
            dens.append(density(flat))
            ever |= (x > 0).any(dim=1).reshape(-1).cpu().numpy()
    return ever, {k: float(np.mean([d[k] for d in dens])) for k in dens[0]}


def bdh_graph_degree(sd, cfg, pct):
    H, N, D = cfg.n_head, cfg.N, cfg.n_embd
    encoder = sd["encoder"].view(H, N, D).float()
    decoder_x = sd["decoder_x"].float()
    deg = np.zeros(H * N, dtype=np.int64)
    for h in range(H):
        G = (decoder_x[h].T @ encoder[h].T).abs().numpy()
        ei, ej = np.nonzero(G >= np.percentile(G, pct))
        deg[h * N:(h + 1) * N] = np.bincount(ei, minlength=N) + np.bincount(ej, minlength=N)
    return deg


@torch.no_grad()
def gpt2_side(name, texts, pct, device):
    """The same two questions asked of a Transformer, with the closest analogue of G*."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    lm = AutoModelForCausalLM.from_pretrained(name).to(device).eval()

    blocks = lm.transformer.h
    acts = {i: [] for i in range(len(blocks))}
    hooks = [b.mlp.act.register_forward_hook(
        lambda m, i, o, i_=i: acts[i_].append(o.detach()[0].float().cpu())) for i, b in enumerate(blocks)]

    out = {}
    for label, text in texts.items():
        for v in acts.values():
            v.clear()
        ids = tok(text, return_tensors="pt").to(device)
        lm(**ids)
        dens = [density(torch.cat(acts[i], 0)) for i in range(len(blocks))]
        out[label] = {k: float(np.mean([d[k] for d in dens])) for k in dens[0]}
        if label == "english":
            fired = {i: (torch.cat(acts[i], 0) > 0).any(0).numpy() for i in range(len(blocks))}
    for h in hooks:
        h.remove()

    # W_out[l] @ W_in[l+1]: layer l's MLP neurons onto layer l+1's. Conv1D stores weights
    # transposed relative to nn.Linear, hence the shapes below.
    mccs, isolated_shares = [], []
    for l in range(len(blocks) - 1):
        w_out = blocks[l].mlp.c_proj.weight.detach().float().cpu()        # (4D, D)
        w_in = blocks[l + 1].mlp.c_fc.weight.detach().float().cpu()       # (D, 4D)
        G = (w_out @ w_in).abs().numpy()                                  # (4D, 4D)
        ei, ej = np.nonzero(G >= np.percentile(G, pct))
        deg = np.bincount(ei, minlength=G.shape[0]) + np.bincount(ej, minlength=G.shape[0])
        iso = deg == 0
        sil = ~fired[l]
        isolated_shares.append(float(iso.mean()))
        if iso.any() and sil.any():
            m, _ = mcc(iso, sil)
            mccs.append(float(m))
    params = sum(p.numel() for p in lm.parameters())
    del lm
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "params": params,
        "mlp_neurons": int(blocks[0].mlp.c_fc.weight.shape[1]) * len(blocks),
        "density": out,
        "silence_prediction": {
            "mcc_per_layer_pair": [round(m, 4) for m in mccs],
            "mcc_mean": round(float(np.mean(mccs)), 4) if mccs else None,
            "isolated_share_mean": round(float(np.mean(isolated_shares)), 4),
            "note": ("W_out[l] @ W_in[l+1] maps one layer's MLP neurons onto the NEXT layer's. It "
                     "is the closest available analogue, not the same object: a Transformer has no "
                     "single neuron population across layers, and its token mixing is a "
                     "data-dependent softmax that is not in the weights at all."),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models/checkpoints/french_best.pt"))
    ap.add_argument("--hf", default="gpt2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pct", type=float, default=99.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-hf", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "research/runs/structure.json"))
    args = ap.parse_args()

    model, cfg, ck = load_big(args.ckpt, device=args.device)
    sd = {k.replace("_orig_mod.", ""): v.cpu() for k, v in ck["model_state_dict"].items()}
    corpora = [complete(model, "<F:en>%s<T:fr>" % s, 90, args.device) for s in SOURCES]
    french = "".join(bytes(c).decode("utf-8", errors="replace") for c in corpora)

    ever, dens = bdh_side(model, cfg, corpora, args.device, args.pct)
    deg = bdh_graph_degree(sd, cfg, args.pct)
    silent, isolated = ~ever, deg == 0
    m, _ = mcc(isolated, silent)

    rng = np.random.default_rng(args.seed)
    nulls = []
    for _ in range(200):
        shuffled = isolated.copy()
        rng.shuffle(shuffled)
        nulls.append(mcc(shuffled, silent)[0])

    res = {
        "percentile": args.pct,
        "bdh": {
            "params": int(sum(p.numel() for p in model.parameters())),
            "neurons": int(cfg.N_total),
            "density": {k: round(v, 5) for k, v in dens.items()},
            "silence_prediction": {
                "mcc": round(float(m), 4),
                "isolated_share": round(float(isolated.mean()), 4),
                "silent_share": round(float(silent.mean()), 4),
                "null_mcc_mean": round(float(np.mean(nulls)), 4),
                "null_mcc_p95": round(float(np.percentile(nulls, 95)), 4),
                "null_draws": len(nulls),
            },
        },
    }
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    print("  BDH  density  exactly zero %.1f%%   > 0 %.2f%%   > 1%% of peak %.2f%%"
          % (100 * dens["exactly_zero"], 100 * dens["greater_than_zero"],
             100 * dens["above_1pct_of_peak"]))
    print("  BDH  G* isolation predicts silence   MCC %+.3f    shuffled null %+.3f (p95 %+.3f)"
          % (m, np.mean(nulls), np.percentile(nulls, 95)))

    if not args.skip_hf:
        try:
            res["transformer"] = gpt2_side(args.hf, {"english": ENGLISH, "french": french},
                                           args.pct, args.device)
            t = res["transformer"]
            for lbl in ("english", "french"):
                d = t["density"][lbl]
                print("  %-5s density (%s)  exactly zero %.1f%%   > 0 %.2f%%   > 1%% of peak %.2f%%"
                      % (args.hf, lbl, 100 * d["exactly_zero"], 100 * d["greater_than_zero"],
                         100 * d["above_1pct_of_peak"]))
            print("  %-5s W_out@W_in isolation predicts silence   MCC %s  (isolated %.1f%%)"
                  % (args.hf, t["silence_prediction"]["mcc_mean"],
                     100 * t["silence_prediction"]["isolated_share_mean"]))
        except Exception as e:
            print("  transformer side failed: %s" % e)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2)
    print("  wrote", os.path.relpath(args.out, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

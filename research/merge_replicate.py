"""
Replicate the BDH paper's model-merging experiment, and find out why ours failed.

THE PAPER'S CLAIM (arXiv:2509.26507 §7.1, "Model merging: concatenating two models"), verbatim:

    "(a) concatenate all parameter tensors that have an 'n' dimension (e.g. D_y, D_x, E, RoPE
     frequency buffers) along their n dimension, (b) average all other parameters (e.g. token
     embeddings and token prediction weights)."

and, as steps 1-3, a PRECONDITION that is easy to miss:

    1. Train a base model on a chosen language pair  (En-Es, n=24576)
    2. "Clone the base model and continue training on two datasets: English-French (En-Fr) and
       English-Portuguese (En-Pt)."
    3. Merge the En-Fr and En-Pt models.

Their Table 2 reports the merged model at 0.39-0.43 translating INTO English and 0.77-1.45 going
out -- degraded but coherent. They call it "human-like degradation".

WHAT WE HAVE, AND HOW IT DIFFERS
--------------------------------
The author's prior hackathon run trained French (50k iters) and Portuguese (40k iters)
INDEPENDENTLY, from scratch -- see notebook Cells 6 and 7, two separate train() calls with no
cloning. Its merge (Cell 8) then reported a loss of 3276 and generated "iiiiiiii".

A cross-entropy over 256 classes cannot exceed ln(256) = 5.55 unless something has overflowed,
so 3276 is not "degraded", it is broken. Before blaming the missing common base we have to rule
out a second difference: Cell 8 does

    merged['rope_freqs'] = sd_a['rope_freqs']            # COPIED, not concatenated

but _build_rope_freqs makes the schedule theta^(idx/N), which is N-DEPENDENT. At the merged
N=6144 every neuron would be rotated at a rate it was never trained with. The paper lists RoPE
buffers among the tensors to CONCATENATE precisely so that each half keeps its own schedule.

That is a confound in exactly the shape of CLAUDE.md item 7, so this script runs the variants
side by side and lets the numbers separate them.

USAGE
    python research/merge_replicate.py
"""
import json, os, sys, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bdh_big import BigBDH, BigBDHConfig, load_big, generate

CKPT = "models/checkpoints"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Held-out-ish probe sentences. NOT a proper validation corpus -- the training .bin files are not
# in this repo -- so every number below is labelled as a small-sample probe, never as a val loss.
PROBE = {
    "fr": [
        "<F:en>The European Parliament voted on this resolution<T:fr>Le parlement europeen a vote cette resolution",
        "<F:en>Thank you very much<T:fr>Merci beaucoup",
        "<F:en>The Commission presented the annual budget report<T:fr>La Commission a presente le rapport budgetaire annuel",
        "<F:en>Germany and France signed the bilateral treaty<T:fr>L'Allemagne et la France ont signe le traite bilateral",
    ],
    "pt": [
        "<F:en>The European Parliament voted on this resolution<T:pt>O parlamento europeu votou esta resolucao",
        "<F:en>Thank you very much<T:pt>Muito obrigado",
        "<F:en>The price in euros was fifty pounds<T:pt>O preco em euros foi de cinquenta libras",
        "<F:en>Germany and France signed the treaty<T:pt>A Alemanha e a Franca assinaram o tratado",
    ],
}


def clean(sd):
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def merge(sd_a, sd_b, cfg, rope_mode):
    """The paper's recipe. `rope_mode` is the variable under test."""
    m = {}
    # (b) average everything without an n dimension
    for k in ("embed.weight", "lm_head", "pos_emb.weight"):
        m[k] = (sd_a[k] + sd_b[k]) / 2.0
    # (a) concatenate everything with one
    m["encoder"] = torch.cat([sd_a["encoder"], sd_b["encoder"]], dim=0)        # (2n, D)
    m["decoder_x"] = torch.cat([sd_a["decoder_x"], sd_b["decoder_x"]], dim=2)  # (H, D, 2n)
    m["decoder_y"] = torch.cat([sd_a["decoder_y"], sd_b["decoder_y"]], dim=2)

    if rope_mode == "concat":            # what the paper says to do
        m["rope_freqs"] = torch.cat([sd_a["rope_freqs"], sd_b["rope_freqs"]], dim=-1)
    elif rope_mode == "copy":            # what the prior run did
        m["rope_freqs"] = sd_a["rope_freqs"]
    elif rope_mode == "rebuild":         # let the model derive a fresh schedule at the new N
        pass
    else:
        raise ValueError(rope_mode)

    mcfg = dict(cfg)
    mcfg["mlp_dim_mult"] = cfg["mlp_dim_mult"] * 2
    return m, mcfg


def build(state, cfg_d):
    cfg = BigBDHConfig(**cfg_d)
    model = BigBDH(cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)
    hard = [k for k in missing if k != "rope_freqs"]
    if hard or unexpected:
        raise RuntimeError(f"state mismatch: missing={hard} unexpected={list(unexpected)}")
    return model.to(DEV).eval(), cfg


@torch.no_grad()
def probe_loss(model, sentences):
    """Mean next-byte cross-entropy over the probe sentences. ln(256)=5.545 is chance."""
    tot, n = 0.0, 0
    for s in sentences:
        ids = torch.tensor([list(s.encode())], dtype=torch.long, device=DEV)
        logits, _ = model(ids[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, 256).float(), ids[:, 1:].reshape(-1))
        v = float(loss)
        if not math.isfinite(v):
            return float("inf")
        tot += v
        n += 1
    return tot / n


@torch.no_grad()
def max_logit(model):
    """Logit magnitude. The specialists already run large, so this matters only as a RATIO --
    it is how the E-halving control is shown to have done exactly what it claims."""
    ids = torch.tensor([list(PROBE["fr"][1].encode())], dtype=torch.long, device=DEV)
    return float(model(ids)[0].abs().max())



def main():
    fr_ck = torch.load(f"{CKPT}/french_best.pt", map_location="cpu", weights_only=False)
    pt_ck = torch.load(f"{CKPT}/portuguese_best.pt", map_location="cpu", weights_only=False)
    sd_a, sd_b = clean(fr_ck["model_state_dict"]), clean(pt_ck["model_state_dict"])
    cfg = fr_ck["config"]

    # rope_freqs is a buffer, not a parameter, so it may not be in the saved state dict at all.
    for sd, ck in ((sd_a, fr_ck), (sd_b, pt_ck)):
        if "rope_freqs" not in sd:
            tmp = BigBDH(BigBDHConfig(**cfg))
            sd["rope_freqs"] = tmp.rope_freqs.clone()

    print(f"French     val loss {fr_ck['losses']['val']:.4f}   ({fr_ck['iteration']:,} iters)")
    print(f"Portuguese val loss {pt_ck['losses']['val']:.4f}   ({pt_ck['iteration']:,} iters)")
    print(f"\nboth trained FROM SCRATCH, independently -- no common base, unlike the paper's §7.1\n")
    print(f"{'model':<42}{'fr probe':>11}{'pt probe':>11}   (chance = {math.log(256):.3f})")
    print("-" * 88)

    results = {}
    fr_model, _, _ = load_big(f"{CKPT}/french_best.pt", DEV)
    pt_model, _, _ = load_big(f"{CKPT}/portuguese_best.pt", DEV)
    for name, mdl in (("French specialist", fr_model), ("Portuguese specialist", pt_model)):
        r = {"fr": probe_loss(mdl, PROBE["fr"]), "pt": probe_loss(mdl, PROBE["pt"]),
             "max_abs_logit": max_logit(mdl)}
        results[name] = r
        print(f"{name:<42}{r['fr']:>11.3f}{r['pt']:>11.3f}   "
              f"max|logit| {r['max_abs_logit']:>9.1f}")
    del fr_model, pt_model
    torch.cuda.empty_cache() if DEV == "cuda" else None

    # The three merged variants, plus one control. `halve_E` exists because the merged residual
    # receives the SUM of both models' contributions, so the first thing to suspect is that the
    # residual simply doubled. Halving E removes that explanation -- and, as it turns out, does
    # not fix the model, which is exactly why the control is worth running.
    samples = {}
    VARIANTS = (
        ("concat", False, "merged - paper recipe (RoPE concatenated)"),
        ("copy", False, "merged - RoPE copied from A (prior run)"),
        ("rebuild", False, "merged - RoPE rebuilt at the new n"),
        ("concat", True, "control - paper recipe, E scaled by 1/2"),
    )
    for mode, halve, label in VARIANTS:
        state, mcfg = merge(sd_a, sd_b, cfg, mode)
        if halve:
            state["encoder"] = state["encoder"] * 0.5
        try:
            mdl, c = build(state, mcfg)
        except Exception as e:
            print(f"{label:<42}{'load failed':>22}   {type(e).__name__}: {str(e)[:44]}")
            results[label] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            continue
        r = {"fr": probe_loss(mdl, PROBE["fr"]), "pt": probe_loss(mdl, PROBE["pt"]),
             "n_neurons": c.N_total, "n_params": sum(p.numel() for p in mdl.parameters()),
             "max_abs_logit": max_logit(mdl)}
        results[label] = r
        print(f"{label:<42}{r['fr']:>11.3f}{r['pt']:>11.3f}   "
              f"max|logit| {r['max_abs_logit']:>9.1f}")
        samples[label] = [
            generate(mdl, "<F:en>The European Parliament<T:fr>", n=48, device=DEV).replace(chr(10), " "),
            generate(mdl, "<F:en>Thank you very much<T:pt>", n=48, device=DEV).replace(chr(10), " "),
        ]
        del mdl
        if DEV == "cuda":
            torch.cuda.empty_cache()

    print("\nsamples from each merged variant:")
    for k, v in samples.items():
        print(f"\n  {k}")
        for s in v:
            print(f"    {s[:110]}")

    out = {
        "what": "replication of arXiv:2509.26507 section 7.1 model merging, on our own checkpoints",
        "paper_recipe": "concatenate every tensor with an n dimension (D_x, D_y, E, RoPE frequency "
                        "buffers); average the rest (embeddings, lm_head)",
        "paper_precondition": "the paper's two models are CLONES of a common En-Es base "
                              "(section 7.1 steps 1-3). Ours were trained independently from "
                              "scratch, so they share no neuron basis.",
        "paper_result_table2": {"Es->En": 0.43, "Fr->En": 0.40, "Pt->En": 0.39,
                                "En->Es": 1.45, "En->Fr": 0.77, "En->Pt": 0.86},
        "our_eval_caveat": "mean next-byte cross-entropy over 4 short probe sentences per "
                           "language, NOT a held-out validation corpus. Chance is ln(256)=5.545. "
                           "Not comparable to the paper's Table 2 numbers in absolute terms.",
        "chance": math.log(256),
        "results": results,
        "samples": samples,
    }
    os.makedirs("web/public", exist_ok=True)
    json.dump(out, open("web/public/merge.json", "w"), indent=1)
    print("\nwrote web/public/merge.json")


if __name__ == "__main__":
    main()

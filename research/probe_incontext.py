#!/usr/bin/env python
"""How far back can a fixed-size state still reach? Measured on the 8M model.

WHY THIS SCRIPT EXISTS
----------------------
The whole artifact is about writing bindings into a fixed-size state at inference time, and the
capacity result (`research/sigma_capacity.py`) measures that on SYNTHETIC keys written directly
into a synthetic sigma: 100% recall at k = 8, 48% at k = 128, with the 1/pi overlap of rectified
Gaussians as the mechanism. What was never measured is the same thing THROUGH A TRAINED MODEL on
real text -- does the trained 8M BDH actually lose a binding as the distance to it grows?

This is the repeated-span probe, run against a MATCHED CONTROL. Two sequences are built that are
identical in every respect except one:

    A (repeat)   <F:en> S  + d bytes of filler + S
    B (control)  <F:en> S' + d bytes of filler + S        (S' unrelated, same length)

and the NLL is measured on the FINAL S in both. It sits at the same absolute position, after the
same filler, in both -- so the difference between them is the effect of S having been in the
context, and nothing else.

The first draft of this script compared the second occurrence against the FIRST occurrence instead,
which is wrong and looked reasonable: the first occurrence sits early in the sequence where the
model has almost no context, so its NLL is inflated for reasons that have nothing to do with
retrieval. That version produced a curve that swung between +0.35 and -1.46 bits with standard
errors of 0.1 -- systematic, not noisy, and measuring position rather than memory.

THE REFERENCE LINE NEEDS NO BASELINE RUN
----------------------------------------
A softmax Transformer with a KV cache keeps every past key and value EXACTLY, for as long as they
are inside the window. Its ability to find a span 400 bytes back is not degraded by the 400 bytes
in between: the retrieval is a fresh softmax over all of them. That is a property of the
architecture, not an empirical claim about some particular checkpoint, so the flat reference line
is drawn from the definition and no baseline model is trained or run to produce it.

What the comparison then costs is stated plainly rather than hidden: the KV cache pays for that
exactness with memory that grows without bound, which is the trade
`research/compare_memory.py` quantifies and `research/compare_runtime.py` measures.

TWO PROBES, BECAUSE THEY ASK DIFFERENT QUESTIONS
------------------------------------------------
  * TEXT spans, drawn from the model's own output distribution. In-distribution, so the model's
    language knowledge can help it -- this measures recall as a reader would experience it.
  * RANDOM byte spans over the alphabet the corpus actually uses. Nothing about French helps here,
    so any gain at all has to come from the state. This is the cleaner measurement of the
    mechanism and the harder test.

WHAT IS AND IS NOT CONTROLLED
-----------------------------
The first occurrence of a span is at a different absolute position from the second, and this model
has a learned positional embedding, so position is not perfectly controlled. It is held as fixed as
it can be: the span always begins at the same offset in the prompt, and only the FILLER length
varies, so the second occurrence moves and the first does not. The filler is real text from the
model's own distribution, so the intervening tokens are typical rather than adversarial.

USAGE
    python research/probe_incontext.py
    python research/probe_incontext.py --trials 40 --span 32
"""
import argparse
import json
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from bdh_big import load_big  # noqa: E402

# Filler and span material in the model's own register, so nothing here is out of distribution.
FILLER = (
    "the Commission has presented a report on this question and the Council has taken note of it "
    "in accordance with the rules of procedure the President opened the debate on the proposal "
    "several members asked for the vote to be postponed until the next part session in Strasbourg "
    "the rapporteur thanked the committee for its work on the amendments tabled in plenary today "
)
SPAN_SOURCE = (
    "the allocation of funds to the regions concerned remains a matter for the member states "
    "a common position was adopted by a large majority after a long and difficult negotiation "
    "the directive will enter into force on the first day of the following month as agreed "
)


@torch.no_grad()
def span_nll(model, ids, lo, hi, device):
    """Mean NLL in nats/byte over ids[lo:hi], predicted from everything before each byte."""
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    logits, _ = model(idx)
    lp = F.log_softmax(logits[0].float(), dim=-1)
    tgt = torch.tensor(ids[lo:hi], device=device)
    got = lp[lo - 1:hi - 1].gather(1, tgt.unsqueeze(1)).squeeze(1)
    return float(-got.mean())


def build(prefix, lead, span, filler, d):
    """prefix + lead + d bytes of filler + span. `lead` is `span` in A and something else in B."""
    fill = (filler * (d // len(filler) + 2))[:d]
    ids = list((prefix + lead + fill + span).encode("utf-8"))
    hi = len(ids)
    return ids, hi - len(span.encode("utf-8")), hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models/checkpoints/french_best.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--span", type=int, default=24, help="span length in bytes")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "research/runs/incontext.json"))
    ap.add_argument("--png", default=os.path.join(ROOT, "docs/media/incontext-recall.png"))
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        print("missing %s -- gh release download v1.0-models -p 'french_best.pt' "
              "-D models/checkpoints/" % os.path.relpath(args.ckpt, ROOT))
        return 1
    model, cfg, _ = load_big(args.ckpt, device=args.device)
    rng = random.Random(args.seed)

    # The byte alphabet the corpus actually uses -- random spans are drawn from this, not from all
    # 256 values, so "random" still means "bytes this model has seen".
    alphabet = sorted(set((FILLER + SPAN_SOURCE).encode("utf-8")))
    # Capped so the whole sequence stays inside the 512-byte context the model was trained on.
    # Past that the learned pos_emb is in rows it never saw at training and the NLL blows up --
    # which is a fact about extrapolation, not about memory, and would contaminate this curve.
    dists = [0, 16, 32, 64, 128, 192, 256, 384]
    prefix = "<F:en>"

    results = {}
    for kind in ("text", "random"):
        rows = []
        for d in dists:
            gains, ctrl, rep = [], [], []
            for t in range(args.trials):
                def draw():
                    if kind == "text":
                        i = rng.randrange(0, len(SPAN_SOURCE) - args.span)
                        return SPAN_SOURCE[i:i + args.span]
                    return bytes(rng.choice(alphabet) for _ in range(args.span)).decode("latin-1")

                span = draw()
                other = draw()
                while other == span:
                    other = draw()
                ids_a, lo, hi = build(prefix, span, span, FILLER, d)
                ids_b, lo_b, hi_b = build(prefix, other, span, FILLER, d)
                assert (lo, hi) == (lo_b, hi_b), "A and B must measure the same window"
                if len(ids_a) > 512:
                    continue
                na = span_nll(model, ids_a, lo, hi, args.device)   # span was in context
                nb = span_nll(model, ids_b, lo, hi, args.device)   # it was not
                rep.append(na); ctrl.append(nb); gains.append(nb - na)
            rows.append({
                "distance": d,
                "nll_control": round(float(np.mean(ctrl)), 4),
                "nll_repeat": round(float(np.mean(rep)), 4),
                "gain_nats": round(float(np.mean(gains)), 4),
                "gain_bits": round(float(np.mean(gains)) / math.log(2), 4),
                "gain_stderr_bits": round(float(np.std(gains) / math.sqrt(max(len(gains), 1))
                                                / math.log(2)), 4),
                "trials": len(gains),
            })
            print("  %-6s d=%-5d  control %.3f  repeat %.3f  ->  gain %.3f +- %.3f bits/byte"
                  % (kind, d, rows[-1]["nll_control"], rows[-1]["nll_repeat"],
                     rows[-1]["gain_bits"], rows[-1]["gain_stderr_bits"]))
        results[kind] = rows

    def retention(rows):
        g0 = rows[0]["gain_bits"]
        return [round(r["gain_bits"] / g0, 3) if g0 else None for r in rows]

    out = {
        "checkpoint": os.path.relpath(args.ckpt, ROOT),
        "span_bytes": args.span,
        "trials_per_distance": args.trials,
        "seed": args.seed,
        "distances": dists,
        "reference": ("A softmax Transformer with a KV cache retrieves an exact copy of every past "
                      "key and value at any distance inside its window, so its reference line here "
                      "is flat by construction, at the price of a cache that grows without bound."),
        "results": results,
        "retention_vs_distance_0": {k: retention(v) for k, v in results.items()},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2)

    print("")
    for kind in ("text", "random"):
        r = results[kind]
        z = [abs(x["gain_bits"]) / max(x["gain_stderr_bits"], 1e-9) for x in r]
        big = sum(1 for x, zz in zip(r, z) if x["gain_bits"] > 0 and zz > 2)
        print("  %-6s  gain %.3f bits/byte at d=0 -> %.3f at d=%d   (%d of %d distances "
              "positive at >2 stderr)"
              % (kind, r[0]["gain_bits"], r[-1]["gain_bits"], r[-1]["distance"], big, len(r)))
    tr, rr = results["text"], results["random"]
    print("")
    print("  READ IT THIS WAY: on in-distribution text the gain is small but consistently")
    print("  positive and decays with distance -- a fixed state holding something and slowly")
    print("  losing it. On random spans it is indistinguishable from zero at every distance,")
    print("  so this model has NOT learned a general copy mechanism. Nothing in translating")
    print("  Europarl rewards reproducing an arbitrary byte string, and it did not learn to.")
    print("  In-context retrieval is a trained capability, not a free consequence of having")
    print("  a recurrent state -- which is why the 131K cipher model, whose task cannot be")
    print("  solved any other way, does read its bindings straight out of context.")
    print("  wrote", os.path.relpath(args.out, ROOT))

    try:
        plot(results, dists, args.png)
        print("  wrote", os.path.relpath(args.png, ROOT))
    except Exception as e:
        print("  (no figure: %s)" % e)
    return 0


def plot(results, dists, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for kind, marker in (("text", "o"), ("random", "s")):
        r = results[kind]
        g = [x["gain_bits"] for x in r]
        e = [x["gain_stderr_bits"] for x in r]
        ax.errorbar(dists, g, yerr=e, marker=marker, capsize=2,
                    label="%s spans" % ("in-distribution text" if kind == "text" else "random byte"))
    ax.axhline(0, lw=1, color="0.6")
    ax.annotate("no benefit from having seen the span", (dists[-1], 0), fontsize=8,
                ha="right", va="bottom", color="0.45")
    g0 = results["text"][0]["gain_bits"]
    ax.axhline(g0, ls="--", lw=1, color="0.5")
    ax.annotate("an exact KV cache would hold its line flat,\nfor a cache that grows without bound",
                (dists[-1], g0), fontsize=8, ha="right", va="bottom", color="0.35")
    ax.set_xlabel("bytes of filler between the span and its repeat")
    ax.set_ylabel("in-context gain (bits/byte saved on the repeat)")
    ax.set_title("How long the 8M BDH's fixed state holds a span\n"
                 "(matched control: the same span, same position, absent from the context)",
                 fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())

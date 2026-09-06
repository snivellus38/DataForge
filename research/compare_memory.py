#!/usr/bin/env python
"""What BDH's fixed-size state actually costs, against an equal-dimension KV cache.

WHY THIS SCRIPT EXISTS
----------------------
"Constant memory" is the headline every linear-attention architecture gets, and on its own it is
misleading. Constant is not the same as small. This script computes, from the two configs this
repo actually ships, how large BDH's state is, how large the equivalent KV cache is, and the
context length at which the constant one starts winning. Every number the README states about
memory comes from here; nothing is typed by hand.

It needs no weights and no torch -- only the two committed config files -- so it runs on a clean
clone.

WHAT IS BEING COMPARED, PRECISELY
---------------------------------
A decoder-only Transformer with the SAME residual width D, the SAME number of layers L and the
SAME head count H, decoding one token at a time, must retain K and V for every past token in
every layer:

    kv_bytes(T)  =  2 * L * D * T * b          (2 = K and V; b = bytes per scalar)

BDH decoding one token at a time must retain, per layer and per head, the associative state
sigma, which is N x D because V = LN(v) has dimension D and not neuron dimension (see CLAUDE.md
item 4 -- the paper's conceptual neuron x neuron state is not what the code materialises):

    sigma_bytes  =  L * H * N * D * b          (independent of T)

Dividing one by the other, D, L and b all cancel:

    T*  =  (L*H*N*D*b) / (2*L*D*b)  =  H*N/2  =  N_total / 2

THE RESULT WORTH REMEMBERING
    BDH's state is cheaper than the equivalent KV cache exactly when the context is longer than
    HALF THE MODEL'S NEURON COUNT. That is a property of the architecture, not of a dtype or a
    hyperparameter, and it holds for both models in this repo.

The same cancellation on arithmetic per generated token -- 2*T*d_head for attention against
2*N*D for a rank-one write plus a read, with d_head = D/H -- puts the compute crossover at
T = N*H = N_total, twice the memory one.

This is not "BDH loses". A KV cache grows without bound and BDH's state does not; below the
crossover you are paying a fixed premium for that guarantee, and above it you are collecting.
Saying which side of the line a model is on is the honest version of the claim.

USAGE
    python research/compare_memory.py [--bytes 2] [--png docs/media/memory-crossover.png]
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_configs():
    """The two models this repo ships, read from their own committed metadata."""
    big = json.load(open(os.path.join(ROOT, "models/telemetry/meta.json")))["config"]
    toy = json.load(open(os.path.join(ROOT, "web/public/model.json")))
    tc = toy["config"]
    return [
        dict(name="8M translation BDH", L=big["n_layer"], D=big["n_embd"], H=big["n_head"],
             N=big["N"], params=7962624, context=512, context_note="trained block size"),
        dict(name="131K cipher BDH", L=tc["n_layer"], D=tc["n_embd"], H=tc["n_head"],
             N=toy["N"], params=toy["params"], context=147,
             context_note="longest shipped prompt"),
    ]


def measure(cfg, b):
    L, D, H, N = cfg["L"], cfg["D"], cfg["H"], cfg["N"]
    kv_per_token = 2 * L * D * b
    sigma = L * H * N * D * b
    return dict(kv_per_token=kv_per_token,
                sigma=sigma,
                crossover=sigma / kv_per_token,
                closed_form=H * N / 2,
                compute_crossover=N * H,
                kv_at_context=kv_per_token * cfg["context"],
                ratio_at_context=sigma / (kv_per_token * cfg["context"]))


def mb(x):
    return x / 1e6


def report(rows, b):
    dt = {2: "fp16/bf16", 4: "fp32"}.get(b, str(b) + "-byte")
    print("\nBDH state vs KV cache   (scalar size %d bytes, %s)\n" % (b, dt))
    head = ("%-22s%3s%5s%3s%7s%12s%11s%11s%10s"
            % ("model", "L", "D", "H", "N", "KV B/token", "sigma", "crossover", "at ctx"))
    print(head)
    print("-" * len(head))
    for cfg, m in rows:
        print("%-22s%3d%5d%3d%7d%12s%10.1fM%10s t%8.1fx"
              % (cfg["name"], cfg["L"], cfg["D"], cfg["H"], cfg["N"],
                 format(m["kv_per_token"], ","), mb(m["sigma"]),
                 format(int(m["crossover"]), ","), m["ratio_at_context"]))
    print()
    for cfg, m in rows:
        assert abs(m["crossover"] - m["closed_form"]) < 1e-9, "closed form T* = H*N/2 disagrees"
        print("%s:" % cfg["name"])
        print("  sigma is a flat %.1f MB. An equal-dimension KV cache passes it at %s tokens,"
              % (mb(m["sigma"]), format(int(m["crossover"]), ",")))
        print("  which is H*N/2 = N_total/2 = %s -- what the algebra gives once D, L and the"
              % format(int(m["closed_form"]), ","))
        print("  dtype cancel.")
        print("  At its %d-token %s the cache holds only %.2f MB, so BDH is carrying %.1fx MORE"
              % (cfg["context"], cfg["context_note"], mb(m["kv_at_context"]),
                 m["ratio_at_context"]))
        print("  state than a Transformer would have to at that length.")
        print("  Arithmetic per generated token crosses over later, at N*H = %s tokens.\n"
              % format(m["compute_crossover"], ","))


def plot(rows, b, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, dim, grid = "#e8e4dc", "#8a8378", "#2a2825"
    amber, blue = "#e8a33d", "#5aa9e6"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), facecolor="#141311")
    for ax, (cfg, m) in zip(axes, rows):
        T = [2 ** i for i in range(1, 18)]
        ax.set_facecolor("#141311")
        ax.plot(T, [mb(m["kv_per_token"] * t) for t in T], color=blue, lw=2.0,
                label="KV cache, same D / L / H")
        ax.plot(T, [mb(m["sigma"])] * len(T), color=amber, lw=2.0,
                label="BDH state (constant)")
        ax.axvspan(m["crossover"], 2 ** 17, color=amber, alpha=0.07, lw=0)
        ax.axvline(m["crossover"], color=dim, lw=1.0, ls=(0, (4, 3)))
        ax.axvline(cfg["context"], color=ink, lw=1.0, ls=(0, (1, 3)))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlim(2, 2 ** 17)
        ax.set_title("%s  (%s params)" % (cfg["name"], format(cfg["params"], ",")),
                     color=ink, fontsize=10, loc="left", pad=10)
        ax.set_xlabel("context length (tokens)", color=dim, fontsize=8.5)
        ax.set_ylabel("state (MB)", color=dim, fontsize=8.5)
        ax.tick_params(colors=dim, labelsize=7.5)
        for s in ax.spines.values():
            s.set_color(grid)
        ax.grid(True, color=grid, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        floor = mb(m["kv_per_token"] * 2)
        ax.annotate("crossover %s = N_total/2" % format(int(m["crossover"]), ","),
                    xy=(m["crossover"], floor), color=dim, fontsize=7.5, rotation=90,
                    va="bottom", ha="right", xytext=(-4, 2), textcoords="offset points")
        ax.annotate("its own context\n%d" % cfg["context"], xy=(cfg["context"], floor),
                    color=ink, fontsize=7.5, va="bottom", ha="right",
                    xytext=(-4, 2), textcoords="offset points")
        ax.annotate("BDH's state is the\ncheaper one from here",
                    xy=(2 ** 16.6, mb(m["sigma"]) * 0.055), color=amber, fontsize=7.5,
                    va="center", ha="right")
        leg = ax.legend(loc="upper left", fontsize=7.5, frameon=False)
        for t in leg.get_texts():
            t.set_color(ink)
    fig.suptitle("Constant is not the same as small", color=ink, fontsize=12, x=0.008,
                 ha="left", y=0.99)
    fig.text(0.008, 0.015,
             "Both axes log. Scalars at %d bytes. Produced by research/compare_memory.py from "
             "the shipped configs -- no weights required." % b,
             color=dim, fontsize=7.5, ha="left")
    fig.tight_layout(rect=(0, 0.045, 1, 0.955))
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    print("wrote %s" % path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=2,
                    help="bytes per stored scalar; 2 = fp16/bf16 (default), 4 = fp32")
    ap.add_argument("--png", default=os.path.join(ROOT, "docs/media/memory-crossover.png"))
    a = ap.parse_args()
    rows = [(c, measure(c, a.bytes)) for c in load_configs()]
    report(rows, a.bytes)
    plot(rows, a.bytes, a.png)

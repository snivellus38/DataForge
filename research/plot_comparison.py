#!/usr/bin/env python
"""Two figures for the README's comparison section, drawn from the committed JSON.

WHY THIS FILE IS SEPARATE FROM THE SCRIPTS THAT MEASURED THE NUMBERS
--------------------------------------------------------------------
`research/compare_structure.py` needs torch and a 96 MB checkpoint;
`research/replicate_structure.py` needs two of them. Neither is available on a clean clone, and
neither is needed to DRAW something that has already been measured. This file reads
`research/runs/structure.json` and `research/runs/replication.json` and nothing else, so the
figures in the README can be rebuilt by anyone with matplotlib, the same property
`research/compare_memory.py` has.

WHAT IT DRAWS

1. sparsity-grid.png
   Two 100x100 grids of cells, one per model, with the share of cells lit set to the share of units
   each model has above 1% of that token's peak activation. BDH lights 493 of 10,000; GPT-2 lights
   8,629. The predicate is on the figure because it has to be: BDH's ReLU produces exact zeros and
   GPT-2's GELU never does, so "active" is only comparable once it means a magnitude.

   A grid rather than a bar chart because the quantity being compared is a density over a
   population of neurons, and a density is what a grid shows. It is also the same way THE FIELD
   draws the model, so a reader arriving from the artifact recognises it.

2. replication-dumbbell.png
   Every structural measurement on the Portuguese model divided by the same measurement on the
   French one, against a line at parity. Two models, same architecture and recipe, different
   language pair, different initialisation, different iteration count, no shared weights.

   THERE IS DELIBERATELY NO SHADED "ACCEPTABLE" BAND. Picking a tolerance after seeing the data is
   how a replication figure becomes decoration: widen the band until everything is inside it and
   the picture says nothing. The line at 1.0 is the only reference, every row carries its two raw
   values, and the one row that lands far from parity (the negative-score share, at 1.36) is drawn
   exactly where it falls.

USAGE
    python research/plot_comparison.py
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ACCENT = "#d98032"      # activation amber, as THE FIELD uses it
DIM = "#e9e7e4"
INK = "#2f2d2a"
BLUE = "#2b6ca8"


def sparsity_grid(structure, path, side=100, seed=0):
    """Two grids of `side` x `side` cells, lit in proportion to each model's active share."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bdh = structure["bdh"]["density"]["above_1pct_of_peak"]
    gpt = structure["transformer"]["density"]["english"]["above_1pct_of_peak"]
    panels = [
        ("8M BDH", bdh, "%.2f%% of units" % (100 * bdh)),
        ("GPT-2 (124M)", gpt, "%.1f%% of units" % (100 * gpt)),
    ]

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 5.0))
    n = side * side
    for ax, (name, share, sub) in zip(axes, panels):
        lit = int(round(share * n))
        cells = np.zeros(n)
        cells[rng.choice(n, size=lit, replace=False)] = 1.0
        ax.imshow(cells.reshape(side, side), cmap=matplotlib.colors.ListedColormap([DIM, ACCENT]),
                  interpolation="nearest", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#cfcdc9")
        ax.set_title("%s\n%s,  %s of 10,000 cells lit" % (name, sub, format(lit, ",")),
                     fontsize=12, color=INK, pad=10, linespacing=1.7)

    fig.suptitle("How much of each model is switched on, one cell per 0.01%%   (%.1fx apart)"
                 % (gpt / bdh), fontsize=12.5, color=INK, y=1.0)
    fig.text(0.5, 0.03,
             "Active means above 1% of that token's peak activation, the only predicate "
             "comparable across the two: BDH's ReLU produces\nexact zeros and GPT-2's GELU never "
             "does. Each model measured on the text it was built for.",
             ha="center", fontsize=8, color="#6b6862")
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return lit


def replication_dumbbell(rep, path):
    """Portuguese over French for every structural measurement, against a line at parity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fr, pt = rep["models"]["french"], rep["models"]["portuguese"]
    fg, pg = fr["graph"], pt["graph"]
    fa, pa = fr["activations"], pt["activations"]
    fs, ps = fr["silence_prediction"], pt["silence_prediction"]

    rows = [
        ("gate y active", fa["y_active"] * 100, pa["y_active"] * 100, "%.2f%%"),
        ("neurons carrying half the edges", fg["per_head"][0]["neurons_carrying_half_the_endpoints"] * 100,
         pg["per_head"][0]["neurons_carrying_half_the_endpoints"] * 100, "%.1f%%"),
        ("max out-degree, head 0", fg["per_head"][0]["max_out_degree"],
         pg["per_head"][0]["max_out_degree"], "%.0f"),
        ("MCC(isolated, silent)", fs["mcc"], ps["mcc"], "%.3f"),
        ("P(silent | isolated)", fs["p_silent_given_isolated"] * 100,
         ps["p_silent_given_isolated"] * 100, "%.1f%%"),
        ("x active", fa["x_active"] * 100, pa["x_active"] * 100, "%.2f%%"),
        ("neurons with no synapse", fg["isolated_share"] * 100, pg["isolated_share"] * 100, "%.1f%%"),
        ("neurons never firing", fa["silent_fraction"] * 100, pa["silent_fraction"] * 100, "%.1f%%"),
        ("negative scores, L3/H0", fa["neg_score_share_L3H0"] * 100,
         pa["neg_score_share_L3H0"] * 100, "%.1f%%"),
    ]
    rows.sort(key=lambda r: abs(r[2] / r[1] - 1.0))

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ys = np.arange(len(rows))
    for y, (label, f, p, fmt) in zip(ys, rows):
        ratio = p / f
        ax.plot([1.0, ratio], [y, y], "-", lw=1.6, color="#d5d3cf", zorder=1)
        ax.plot([ratio], [y], "o", ms=10, color=ACCENT if abs(ratio - 1) < 0.15 else BLUE,
                zorder=3)
        ax.annotate((fmt + "  to  " + fmt) % (f, p), (ratio, y), fontsize=8.5, color="#5a5854",
                    xytext=(14 if ratio >= 1 else -14, 0), textcoords="offset points",
                    va="center", ha="left" if ratio >= 1 else "right", zorder=4)

    ax.axvline(1.0, lw=1.4, color=INK, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10, color=INK)
    ax.set_xlim(0.64, 1.52)   # room for the leftmost row to carry its label without touching the spine
    ax.set_xlabel("Portuguese model as a multiple of the French one   (1.0 = identical)",
                  fontsize=10, color=INK, labelpad=8)
    ax.set_title("The same measurements on an independently trained sibling", fontsize=12.5,
                 color=INK, pad=12)
    ax.grid(axis="x", alpha=0.2, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.text(0.5, -0.02,
             "Same architecture and recipe, different language pair, different initialisation, "
             "50k iterations against 40k, no shared weights.\nNo tolerance band is drawn: choosing "
             "one after seeing the data is how a replication figure stops meaning anything.",
             ha="center", fontsize=8, color="#6b6862")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", default=os.path.join(ROOT, "research/runs/structure.json"))
    ap.add_argument("--replication", default=os.path.join(ROOT, "research/runs/replication.json"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "docs/media"))
    args = ap.parse_args()

    for p in (args.structure, args.replication):
        if not os.path.exists(p):
            print("missing %s. Run npm run compare first." % os.path.relpath(p, ROOT))
            return 1

    structure = json.load(open(args.structure, encoding="utf-8"))
    rep = json.load(open(args.replication, encoding="utf-8"))

    a = os.path.join(args.outdir, "sparsity-grid.png")
    lit = sparsity_grid(structure, a)
    print("  wrote %s   (BDH %d cells lit of 10,000)"
          % (os.path.relpath(a, ROOT), round(structure["bdh"]["density"]["above_1pct_of_peak"] * 10000)))

    b = os.path.join(args.outdir, "replication-dumbbell.png")
    rows = replication_dumbbell(rep, b)
    print("  wrote %s" % os.path.relpath(b, ROOT))
    worst = max(rows, key=lambda r: abs(r[2] / r[1] - 1.0))
    print("  closest to parity: %s (%.3f)   furthest: %s (%.3f)"
          % (rows[0][0], rows[0][2] / rows[0][1], worst[0], worst[2] / worst[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

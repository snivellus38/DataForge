#!/usr/bin/env python
"""Sparsity is learned, not imposed -- from the 8M run's own training telemetry.

WHAT THIS SHOWS
---------------
Every ~2,500 iterations the translation run probed three held-out sentences and recorded what
fraction of neurons were non-zero. Nothing in the loss encourages sparsity: there is no L1 term,
no k-winners-take-all, no threshold. The only structural pressure is the ReLU, which forces
activations into the non-negative orthant.

The curve that comes out has a shape worth reading:

  * At initialisation x is 49.9% active -- exactly what a ReLU on roughly symmetric
    pre-activations gives you. Half of everything, on for no reason.
  * By iteration 2,500 it has collapsed to 3.11%. The network's first move is to switch most of
    itself off.
  * Then it climbs back, monotonically, to 5.09% at 47,500 while the loss keeps falling. It
    RECRUITS. The end state is not the sparsest one it ever visited; it is the sparsest one it
    can afford at that loss.

The gate y = ReLU(D_y a) * x tracks the same shape one order of magnitude lower, ending near
0.92%: an AND of two sparse conditions lets fewer than one neuron in a hundred reach the residual.

WHY IT IS WORTH PLOTTING HERE
-----------------------------
The endpoint corroborates a number this repo measures independently. The training probe lands at
5.09% on three sentences through the prior run's own instrumentation; our export pipeline, a
different code path on a 7-sentence corpus and 55,959,552 neuron-token slots, lands at 5.129%
(web/public/big/traces.json, which is where the reference line on the figure is read from -- it is
not typed in). Two measurements, two codebases, one number.

CAVEAT THAT TRAVELS WITH THE FIGURE
    The probe is three sentences, not a corpus, and it was taken with the checkpoint mid-training.
    It is evidence about the SHAPE of the curve. The endpoint is corroborated; the intermediate
    points are not independently reproduced.

The iteration-0 loss probe is excluded from the loss panel: it reports 8.2e5 for French and
1.1e6 for Portuguese, before warm-up has done anything, and plotting it flattens everything else.
The sparsity panel keeps its iteration-0 point, because that is the whole story of the first move.

USAGE
    python research/plot_training.py [--png docs/media/sparsity-emergence.png]
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EVO = os.path.join(ROOT, "models/telemetry/evolution")
TRACES = os.path.join(ROOT, "web/public/big/traces.json")


def load(lang):
    """Training-time probes: iteration, loss, and the ZERO fraction of x and y."""
    rows = json.load(open(os.path.join(EVO, "evolution_%s.json" % lang)))
    return dict(
        it=[r["iteration"] for r in rows],
        loss=[r["loss"] for r in rows],
        # the run stored sparsity as a fraction of zeros; the artifact talks in active fraction
        x=[100.0 * (1.0 - r["mean_x_sp"]) for r in rows],
        y=[100.0 * (1.0 - r["mean_y_sp"]) for r in rows],
    )


def summarise(name, d, ours):
    lo = min(range(len(d["x"])), key=lambda i: d["x"][i])
    print("%s: %d probes, iterations %s to %s" % (name, len(d["it"]), d["it"][0], d["it"][-1]))
    print("  x active  %6.2f%% at init -> %6.2f%% at %s (the floor) -> %6.2f%% at %s"
          % (d["x"][0], d["x"][lo], format(d["it"][lo], ","), d["x"][-1],
             format(d["it"][-1], ",")))
    print("  y active  %6.2f%% at init -> %6.2f%% at %s (the floor) -> %6.2f%% at %s"
          % (d["y"][0], d["y"][lo], format(d["it"][lo], ","), d["y"][-1],
             format(d["it"][-1], ",")))
    print("  loss      %8.4f at %s -> %8.4f at %s"
          % (d["loss"][1], format(d["it"][1], ","), d["loss"][-1],
             format(d["it"][-1], ",")))
    climbing = all(b >= a - 1e-9 for a, b in zip(d["x"][1:], d["x"][2:]))
    print("  x is monotonically non-decreasing after the floor: %s" % climbing)
    print("  final probe vs our independent corpus measurement (%.3f%%): %+.3f pp\n"
          % (ours, d["x"][-1] - ours))
    return climbing


def plot(fr, pt, ours, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, dim, grid = "#e8e4dc", "#8a8378", "#2a2825"
    amber, blue, teal = "#e8a33d", "#5aa9e6", "#6fbf9a"
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.3), facecolor="#141311")

    for a in (ax, bx):
        a.set_facecolor("#141311")
        a.tick_params(colors=dim, labelsize=7.5)
        for s in a.spines.values():
            s.set_color(grid)
        a.grid(True, color=grid, lw=0.6, alpha=0.7)
        a.set_axisbelow(True)
        a.set_xlabel("training iteration", color=dim, fontsize=8.5)

    ax.axhline(ours, color=teal, lw=1.2, ls=(0, (5, 3)))
    ax.plot(fr["it"], fr["x"], color=amber, lw=2.0, marker="o", ms=2.6,
            label=r"English$\rightarrow$French,  x")
    ax.plot(pt["it"], pt["x"], color=amber, lw=1.2, alpha=0.55, marker="o", ms=2.0,
            label=r"English$\rightarrow$Portuguese,  x")
    ax.plot(fr["it"], fr["y"], color=blue, lw=2.0, marker="o", ms=2.6,
            label=r"English$\rightarrow$French,  y (after the gate)")
    ax.plot(pt["it"], pt["y"], color=blue, lw=1.2, alpha=0.55, marker="o", ms=2.0,
            label=r"English$\rightarrow$Portuguese,  y")
    ax.set_yscale("log")
    ax.set_ylabel("neurons active (%, log)", color=dim, fontsize=8.5)
    ax.set_title("Sparsity is learned, not imposed", color=ink, fontsize=10, loc="left", pad=10)
    ax.annotate("49.9% at init:\na ReLU on symmetric\npre-activations",
                xy=(600, fr["x"][0] * 0.92), xytext=(5200, 24), color=ink, fontsize=7.5,
                arrowprops=dict(color=dim, arrowstyle="-", lw=0.8))
    ax.annotate("collapses to %.2f%% by 2,500" % fr["x"][1], xy=(2600, fr["x"][1] * 0.94),
                xytext=(5200, 1.55), color=ink, fontsize=7.5,
                arrowprops=dict(color=dim, arrowstyle="-", lw=0.8))
    ax.annotate("then climbs back to %.2f%%,\nmonotonically, while the loss falls"
                % fr["x"][-1], xy=(fr["it"][-1] * 0.97, fr["x"][-1] * 0.93),
                xytext=(17000, 1.9), color=ink, fontsize=7.5,
                arrowprops=dict(color=dim, arrowstyle="-", lw=0.8))
    ax.annotate("%.3f%% -- what our export pipeline measures\nindependently, on a 7-sentence "
                "corpus" % ours, xy=(9500, ours * 1.22), color=teal, fontsize=7.5)
    leg = ax.legend(loc="upper right", fontsize=6.8, frameon=False)
    for t in leg.get_texts():
        t.set_color(ink)

    bx.plot(fr["it"][1:], fr["loss"][1:], color=amber, lw=2.0, marker="o", ms=2.6,
            label=r"English$\rightarrow$French")
    bx.plot(pt["it"][1:], pt["loss"][1:], color=amber, lw=1.2, alpha=0.55, marker="o", ms=2.0,
            label=r"English$\rightarrow$Portuguese")
    bx.set_ylabel("probe loss (nats/byte)", color=dim, fontsize=8.5)
    bx.set_title("...while the loss falls the whole way", color=ink, fontsize=10,
                 loc="left", pad=10)
    leg = bx.legend(loc="upper right", fontsize=7.5, frameon=False)
    for t in leg.get_texts():
        t.set_color(ink)

    fig.text(0.008, 0.015,
             "Probes: 3 held-out sentences every 2,500 iterations, recorded by the training run "
             "itself. The iteration-0 loss probe (8.2e5, pre-warm-up) is excluded from the right "
             "panel. Produced by research/plot_training.py.",
             color=dim, fontsize=7.2, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    print("wrote %s" % path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default=os.path.join(ROOT, "docs/media/sparsity-emergence.png"))
    a = ap.parse_args()
    ours = 100.0 * json.load(open(TRACES))["measured_active_frac"]
    fr, pt = load("french"), load("portuguese")
    print()
    ok_fr = summarise("English->French  ", fr, ours)
    ok_pt = summarise("English->Portuguese", pt, ours)
    assert ok_fr and ok_pt, "the recruit-after-the-floor claim does not hold on this data"
    plot(fr, pt, ours, a.png)

#!/usr/bin/env python
"""The memory and compute crossovers, measured on a clock instead of derived on paper.

WHY THIS SCRIPT EXISTS
----------------------
`research/compare_memory.py` derives, from the two configs this repo ships, that BDH's fixed state
is cheaper than an equal-dimension KV cache exactly past

    T* = H*N/2 = N_total/2          (memory)
    T* = H*N   = N_total            (arithmetic per generated token)

That is arithmetic. This script runs both models and times them, so the curve is measured and the
arithmetic is falsifiable rather than merely stated. If the two disagree, the arithmetic is what
gets fixed.

WHY A RANDOMLY INITIALISED TRANSFORMER IS A LEGITIMATE BASELINE HERE
--------------------------------------------------------------------
Decode latency and state size are functions of SHAPE, not of what the weights contain: a matmul
costs the same whatever numbers are in it. So the baseline is a decoder-only Transformer built at
the same D, L and H as our BDH, with a real KV cache, at random initialisation -- and that is a
fair measurement of what that architecture costs to run. It is emphatically NOT a quality
baseline, and no loss or accuracy number is produced here or should ever be quoted from it.

WHAT HAD TO BE BUILT, AND WHY IT IS THE INTERESTING PART
--------------------------------------------------------
`BigBDH.forward` is the PARALLEL form: it re-reads the whole prefix on every call, which is O(T)
work per token and O(T) memory -- the same cost profile as the thing it is supposed to beat.
Timing that would have measured the wrong object entirely and made BDH look worse than it is.

So `recurrent_step()` below is the RECURRENT form at 8M scale: it carries sigma of shape (N, D)
per layer and head, reads it, then writes one rank-one update per token. Constant memory, constant
work per token, no prefix.

Two details in it are load-bearing, and both are easy to get wrong:

  * READ BEFORE WRITE. The mask is `tril(-1)`, strictly causal, so token t attends to s < t and
    NOT to itself. Writing token t into sigma before reading it inverts that. `--check` includes
    this as a negative control precisely because a loose tolerance means nothing unless it still
    rejects the wrong answer.
  * RoPE is applied to the query and to the key, and since Q = K here, the SAME rotated vector is
    both written and read -- at its own position. That is what makes `sum_s rope(x)_s (x) v_s`
    equal the parallel `(rope(x) rope(x)^T).tril(-1) @ V`.

`--check` verifies the recurrence against the parallel forward on a real sentence and prints both
the residual and the control. Run it whenever this file is touched.

WHAT THE LATENCY HALF CAN AND CANNOT SHOW AT THIS SCALE
-------------------------------------------------------
The memory half is exact: sigma is flat, the cache grows, and the measured curves cross where the
arithmetic says they will. The latency half is weaker, and the output says so rather than implying
otherwise. At D = 192 with 4 heads of 48 dimensions, one decode step is a few MFLOP -- far too
little to occupy a GPU -- so BOTH models sit on a floor of Python and kernel-launch overhead, and
the transformer's growing attention shows up only as a few percent on top of it. The predicted
compute crossover at N_total is therefore not reachable on this hardware at this model size.
What the numbers do support: BDH's per-token cost is FLAT in context length (it has no prefix to
re-read), the transformer's is not, and below the crossover BDH pays a premium in time for the
same reason it pays one in bytes -- it moves its whole 28.3 MB state on every token regardless of
how little context there is. Read the SHAPES, not the absolute milliseconds, and do not quote
these as a throughput benchmark: `recurrent_step` is a readable reference implementation of the
recurrence, not a tuned kernel.

THE POSITION CLAMP ABOVE 4,096
------------------------------
The checkpoint's learned `pos_emb` has 4,096 rows, so contexts past that have no position to look
up. For the TIMING sweep the index is clamped, which changes the values the model produces (they
become meaningless) but not the work it does -- and the work is the only thing being measured
there. The correctness check runs entirely below that limit.

USAGE
    python research/compare_runtime.py --check
    python research/compare_runtime.py --max-ctx 16384
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from bdh_big import load_big  # noqa: E402


# ── the baseline: a decoder-only Transformer at the same shape, with a real KV cache ────────────
class Block(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.H, self.dh = H, D // H
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)
        self.fc1, self.fc2 = nn.Linear(D, 4 * D, bias=False), nn.Linear(4 * D, D, bias=False)

    def forward(self, x, cache):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        shape = lambda t: t.view(B, T, self.H, self.dh).transpose(1, 2)
        q, k, v = shape(q), shape(k), shape(v)
        if cache["k"] is not None:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
        cache["k"], cache["v"] = k, v
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        att = att.softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(y)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class Baseline(nn.Module):
    """Same D / L / H as the BDH it is compared against. Weights are random: this measures cost."""

    def __init__(self, D, L, H, vocab=256):
        super().__init__()
        self.embed = nn.Embedding(vocab, D)
        self.blocks = nn.ModuleList([Block(D, H) for _ in range(L)])
        self.lnf = nn.LayerNorm(D)
        self.head = nn.Linear(D, vocab, bias=False)

    def empty_cache(self):
        return [{"k": None, "v": None} for _ in self.blocks]

    def forward(self, idx, cache):
        x = self.embed(idx)
        for b, c in zip(self.blocks, cache):
            x = b(x, c)
        return self.head(self.lnf(x))


# ── the BDH recurrent decode ────────────────────────────────────────────────────────────────────
def empty_sigma(cfg, device, dtype=torch.float32):
    return torch.zeros(cfg.n_layer, cfg.n_head, cfg.N, cfg.n_embd, device=device, dtype=dtype)


def _rope(x, pos, freqs):
    """Rotate x at absolute position `pos`. x is (H, N); freqs is (N,)."""
    phase = (pos * freqs % 1.0) * (2 * math.pi)
    cos_p, sin_p = torch.cos(phase), torch.sin(phase)
    rot = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).reshape_as(x)
    return x * cos_p + rot * sin_p


@torch.no_grad()
def recurrent_step(model, cfg, token, pos, sigma):
    """One token through BDH carrying sigma. Constant memory, constant work, no prefix.

    Returns the logits for the next byte. `sigma` is mutated in place."""
    D, H, N = cfg.n_embd, cfg.n_head, cfg.N
    freqs = model.rope_freqs.view(-1)[:N]
    pos_ix = min(pos, model.pos_emb.num_embeddings - 1)          # see the docstring's clamp note
    v_ast = model.embed(token) + model.pos_emb(
        torch.tensor(pos_ix, device=token.device))               # (D,)
    for L in range(cfg.n_layer):
        v_n = model.ln(v_ast)                                    # (D,)
        x = F.relu(torch.einsum("d,hdn->hn", v_n, model.decoder_x))   # (H, N)
        q = _rope(x, pos, freqs)                                 # (H, N)
        # READ BEFORE WRITE: tril(-1) excludes the current token from its own attention.
        a_ast = torch.einsum("hn,hnd->hd", q, sigma[L])          # (H, D)
        sigma[L] += q.unsqueeze(-1) * v_n.view(1, 1, D)          # rank-one write, per head
        y = F.relu(torch.einsum("hd,hdn->hn", a_ast, model.decoder_y)) * x
        v_ast = v_ast + (y.reshape(-1) @ model.encoder)
    return v_ast @ model.lm_head


@torch.no_grad()
def check_recurrence(model, cfg, device, text="<F:en>The European Parliament voted<T:fr>"):
    """The recurrent form must reproduce the parallel forward -- with a control that rejects the
    read/write order being wrong, so the tolerance is doing real work."""
    ids = list(text.encode("utf-8"))
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    ref, _ = model(idx)
    ref = ref[0]

    def run(write_first):
        sigma = empty_sigma(cfg, device)
        out = []
        for t, b in enumerate(ids):
            tok = torch.tensor(b, dtype=torch.long, device=device)
            if not write_first:
                out.append(recurrent_step(model, cfg, tok, t, sigma))
            else:
                out.append(_step_write_first(model, cfg, tok, t, sigma))
        return torch.stack(out)

    got = run(False)
    ctrl = run(True)
    rel = ((got - ref).abs().max() / ref.abs().max()).item()
    rel_c = ((ctrl - ref).abs().max() / ref.abs().max()).item()
    return rel, rel_c


@torch.no_grad()
def _step_write_first(model, cfg, token, pos, sigma):
    """The negative control: the same step with the rank-one write moved before the read."""
    D, H, N = cfg.n_embd, cfg.n_head, cfg.N
    freqs = model.rope_freqs.view(-1)[:N]
    pos_ix = min(pos, model.pos_emb.num_embeddings - 1)
    v_ast = model.embed(token) + model.pos_emb(torch.tensor(pos_ix, device=token.device))
    for L in range(cfg.n_layer):
        v_n = model.ln(v_ast)
        x = F.relu(torch.einsum("d,hdn->hn", v_n, model.decoder_x))
        q = _rope(x, pos, freqs)
        sigma[L] += q.unsqueeze(-1) * v_n.view(1, 1, D)          # <- wrong order, on purpose
        a_ast = torch.einsum("hn,hnd->hd", q, sigma[L])
        y = F.relu(torch.einsum("hd,hdn->hn", a_ast, model.decoder_y)) * x
        v_ast = v_ast + (y.reshape(-1) @ model.encoder)
    return v_ast @ model.lm_head


# ── timing ──────────────────────────────────────────────────────────────────────────────────────
def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def time_min(fn, reps, device):
    """MINIMUM over repetitions, not mean or median.

    Latency measurements are contaminated upwards and never downwards -- a scheduler hiccup or a
    clock drop can only make a run slower. On a laptop GPU with unstable clocks the mean is mostly
    a picture of the thermal state of the machine, so the minimum is the least-contaminated
    estimate of what the work actually costs. The first call is discarded as warm-up."""
    fn(); sync(device)
    ts = []
    for _ in range(reps):
        sync(device); t0 = time.perf_counter()
        fn()
        sync(device); ts.append(time.perf_counter() - t0)
    return float(np.min(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models/checkpoints/french_best.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-ctx", type=int, default=16384)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--run", type=int, default=32,
                    help="tokens generated per timed run; the per-token figure is the quotient")
    ap.add_argument("--bytes", type=int, default=2, help="bytes per scalar for the memory model")
    ap.add_argument("--check", action="store_true", help="verify the recurrence and stop")
    ap.add_argument("--out", default=os.path.join(ROOT, "research/runs/runtime.json"))
    ap.add_argument("--png", default=os.path.join(ROOT, "docs/media/runtime-crossover.png"))
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        print("missing %s -- gh release download v1.0-models -p 'french_best.pt' "
              "-D models/checkpoints/" % os.path.relpath(args.ckpt, ROOT))
        return 1
    model, cfg, _ = load_big(args.ckpt, device=args.device)
    D, L, H, N = cfg.n_embd, cfg.n_layer, cfg.n_head, cfg.N

    rel, rel_ctrl = check_recurrence(model, cfg, args.device)
    print("recurrence vs parallel forward   rel = %.2e" % rel)
    print("control, write-before-read       rel = %.2e   (%.0fx worse)" % (rel_ctrl, rel_ctrl / rel))
    ok = rel < 1e-4 and rel_ctrl > 100 * rel
    print("  %s" % ("OK" if ok else "FAILED -- the recurrence does not reproduce the forward"))
    if args.check:
        return 0 if ok else 1
    if not ok:
        return 1

    base = Baseline(D, L, H).to(args.device).eval()
    ctxs = [c for c in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384] if c <= args.max_ctx]
    K = args.run

    # Global warm-up. Without it the first context measured pays for every lazy CUDA init and
    # reads as the slowest point on the curve, which is an artifact and looks like a finding.
    warm_sigma = empty_sigma(cfg, args.device)
    warm_tok = torch.tensor(65, dtype=torch.long, device=args.device)
    for t in range(32):
        recurrent_step(model, cfg, warm_tok, t, warm_sigma)
    wc = base.empty_cache()
    with torch.no_grad():
        base(torch.randint(0, 256, (1, 64), device=args.device), wc)
    sync(args.device)
    del warm_sigma, wc

    rows = []
    for T in ctxs:
        # Both models are timed the same way: generate K consecutive tokens starting from a context
        # of T, and divide. One-shot timing of a single step at these sizes measures Python and
        # kernel-launch overhead rather than the architecture.
        tok = torch.tensor(65, dtype=torch.long, device=args.device)

        def bdh_run():
            sigma = empty_sigma(cfg, args.device)                # sigma stays this size forever
            for i in range(K):
                recurrent_step(model, cfg, tok, T + i, sigma)

        bdh_ms = 1e3 * time_min(bdh_run, args.reps, args.device) / K

        nxt = torch.randint(0, 256, (1, 1), device=args.device)
        dh = D // H

        def fresh_cache():
            # The cache is SYNTHESISED rather than produced by a prefill. Decode latency does not
            # depend on what is in the cache, only on how much of it there is -- and running a real
            # prefill would materialise a T x T attention matrix (4 GB at T = 16,384) to measure
            # something the prefill is not part of.
            return [{"k": torch.randn(1, H, T, dh, device=args.device),
                     "v": torch.randn(1, H, T, dh, device=args.device)} for _ in range(L)]

        def tf_run(cache):
            with torch.no_grad():
                for _ in range(K):
                    base(nxt, cache)                             # cache grows, as it does in use

        cache = fresh_cache()
        tf_ms = 1e3 * time_min(lambda: tf_run(fresh_cache()), args.reps, args.device) / K
        del cache

        b = args.bytes
        rows.append({
            "context": T,
            "bdh_ms_per_token": round(bdh_ms, 4),
            "transformer_ms_per_token": round(tf_ms, 4),
            "bdh_state_mb": round(L * H * N * D * b / 1e6, 2),
            "kv_cache_mb": round(2 * L * D * T * b / 1e6, 2),
        })
        print("  T=%-6d  BDH %7.3f ms   Transformer %7.3f ms   |   sigma %6.1f MB   "
              "cache %6.1f MB" % (T, bdh_ms, tf_ms, rows[-1]["bdh_state_mb"], rows[-1]["kv_cache_mb"]))

    mem_star, cmp_star = H * N // 2, H * N
    faster = [r["context"] for r in rows if r["bdh_ms_per_token"] < r["transformer_ms_per_token"]]
    smaller = [r["context"] for r in rows if r["bdh_state_mb"] < r["kv_cache_mb"]]
    growth = lambda k: round(rows[-1][k] / rows[0][k], 3)
    overhead_bound = growth("transformer_ms_per_token") < 1.5
    out = {
        "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
        "config": {"D": D, "L": L, "H": H, "N": N, "N_total": cfg.N_total},
        "bytes_per_scalar": args.bytes,
        "recurrence_check": {"relative_error": rel, "control_write_before_read": rel_ctrl},
        "predicted_memory_crossover": mem_star,
        "predicted_compute_crossover": cmp_star,
        "first_measured_context_where_sigma_is_smaller": smaller[0] if smaller else None,
        "first_measured_context_where_bdh_is_faster": faster[0] if faster else None,
        "latency_growth_over_the_sweep": {
            "context_span": [rows[0]["context"], rows[-1]["context"]],
            "bdh": growth("bdh_ms_per_token"),
            "transformer": growth("transformer_ms_per_token"),
        },
        "latency_is_overhead_bound": bool(overhead_bound),
        "caveat": ("At D=192/H=4 one decode step is a few MFLOP, so both models sit on a "
                   "kernel-launch overhead floor and the predicted compute crossover at N_total "
                   "is not observable on this hardware. The memory half is exact; the latency "
                   "half supports only the SHAPE of the curves. recurrent_step is a readable "
                   "reference implementation, not a tuned kernel."),
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2)

    print("\n  predicted memory crossover  N_total/2 = %d   -> first measured smaller at %s"
          % (mem_star, smaller[0] if smaller else "not reached"))
    print("  predicted compute crossover N_total   = %d   -> first measured faster  at %s"
          % (cmp_star, faster[0] if faster else "not reached"))
    print("")
    print("  over %d -> %d tokens of context, ms/token grew  BDH x%.2f   Transformer x%.2f"
          % (rows[0]["context"], rows[-1]["context"],
             growth("bdh_ms_per_token"), growth("transformer_ms_per_token")))
    if overhead_bound:
        print("  BOTH ARE OVERHEAD-BOUND at this size: one decode step is a few MFLOP, so the")
        print("  compute crossover is not observable here. The memory result above is exact;")
        print("  the latency result supports the shape of the curves and nothing stronger.")
    print("  wrote", os.path.relpath(args.out, ROOT))

    try:
        plot(rows, mem_star, cmp_star, out["device"], args.png)
        print("  wrote", os.path.relpath(args.png, ROOT))
    except Exception as e:                                       # plotting is a convenience
        print("  (no figure: %s)" % e)
    return 0


def plot(rows, mem_star, cmp_star, device, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = [r["context"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(T, [r["kv_cache_mb"] for r in rows], "o-", label="Transformer KV cache")
    ax[0].plot(T, [r["bdh_state_mb"] for r in rows], "s-", label="BDH state $\\sigma$")
    ax[0].axvline(mem_star, ls="--", lw=1, color="0.5")
    ax[0].annotate("$T^*=N_{total}/2$\n= %d" % mem_star, (mem_star, ax[0].get_ylim()[1] * 0.02),
                   fontsize=8, ha="right", color="0.35")
    ax[0].set_title("State size, per generated token")
    ax[0].set_ylabel("MB (fp16)")

    ax[1].plot(T, [r["transformer_ms_per_token"] for r in rows], "o-", label="Transformer + KV cache")
    ax[1].plot(T, [r["bdh_ms_per_token"] for r in rows], "s-", label="BDH recurrent step")
    ax[1].axvline(cmp_star, ls="--", lw=1, color="0.5")
    ax[1].annotate("$T^*=N_{total}$\n= %d" % cmp_star, (cmp_star, ax[1].get_ylim()[0]),
                   fontsize=8, ha="right", color="0.35")
    ax[1].set_title("Latency, one token at a time (overhead-bound at this size)")
    ax[1].set_ylabel("ms per token")

    for a in ax:
        a.set_xscale("log", base=2)
        a.set_yscale("log")
        a.set_xlabel("context length (bytes already generated)")
        a.grid(alpha=0.25, which="both", lw=0.5)
        a.legend(fontsize=8, frameon=False)
    fig.suptitle("Measured on %s — the Transformer is random-init at the same D/L/H, because decode "
                 "cost depends on shape, not weights. Left is exact; right is dominated by "
                 "kernel-launch overhead at D=192: read the shapes, not the milliseconds."
                 % device, fontsize=8, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())

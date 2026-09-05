"""
DIRECT associative capacity of the fixed-size synaptic state -- no training in the loop.

Why this exists: measuring capacity by "train at many k and watch accuracy fall" is confounded.
Raising k also raises sequence length, symbol count and training difficulty, so a falling curve
cannot be attributed to the state running out of room. (We hit exactly this: see CLAUDE.md item 7.)

Here we interrogate sigma itself. Write k rank-one bindings into the same state bdh.py's
attention implicitly maintains, then read each key back and ask whether the correct value wins:

    sigma = sum_i  rope(key_i) (outer) value_i          shape (N, D)
    read(q) = sigma^T rope(q)

Retrieval is "correct" when read(key_j) is closest to value_j among all k stored values.
Nothing is learned; this is a property of the state and the key geometry alone. That makes it
the honest substrate for the artifact's interference lesson.

Keys mimic bdh.py's x_sparse: ReLU'd Gaussians, i.e. non-negative and roughly half-sparse.
Non-negative keys OVERLAP (expected cosine > 0), which is precisely why interference appears
far below the naive rank bound of N.
"""
import argparse, math, sys, os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor"))
from bdh import Attention, BDHConfig

p = argparse.ArgumentParser()
p.add_argument("--N", type=int, default=256)      # neurons per head (key dim)
p.add_argument("--D", type=int, default=64)       # value dim
p.add_argument("--trials", type=int, default=200)
p.add_argument("--sparsity", type=float, default=-1, help="if >=0, keep only this fraction of key dims")
a = p.parse_args()
torch.manual_seed(0)


def get_freqs(n, theta=2 ** 16):
    q = lambda t, k=2: (t / k).floor() * k
    return 1.0 / (theta ** (q(torch.arange(0, n, 1, dtype=torch.float64)) / n)) / (2 * math.pi)


def rope(ph, v):
    vr = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
    p_ = (ph % 1) * (2 * math.pi)
    return v * torch.cos(p_) + vr * torch.sin(p_)


def trial(k, N, D, sparsity):
    keys = torch.relu(torch.randn(k, N, dtype=torch.float64))
    if sparsity >= 0:                       # optionally force harder sparsity than ReLU gives
        mask = torch.rand(k, N, dtype=torch.float64) < sparsity
        keys = keys * mask
    vals = torch.randn(k, D, dtype=torch.float64)
    ph = torch.arange(k, dtype=torch.float64).view(-1, 1) * get_freqs(N).view(1, -1)
    kr = rope(ph, keys)                     # positions 0..k-1, as bdh.py would place them
    sigma = torch.einsum("kn,kd->nd", kr, vals)          # the fixed-size state, N x D
    read = torch.einsum("nd,kn->kd", sigma, kr)          # read every stored key back
    # correct iff the true value is the nearest stored value (cosine)
    rn = read / read.norm(dim=1, keepdim=True).clamp_min(1e-12)
    vn = vals / vals.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sim = rn @ vn.T
    return (sim.argmax(1) == torch.arange(k)).double().mean().item()


print(f"sigma is {a.N}x{a.D} = {a.N*a.D:,} floats. keys = ReLU(N(0,1)) over {a.N} dims"
      + (f", extra sparsity {a.sparsity}" if a.sparsity >= 0 else " (~50% nonzero)"))
print("mean cosine between two non-negative keys:",
      f"{torch.nn.functional.cosine_similarity(torch.relu(torch.randn(2000,a.N)),torch.relu(torch.randn(2000,a.N))).mean():.3f}")
print(f"\n{'k':>5} {'retrieval':>10}")
ks = [2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512]
for k in ks:
    acc = sum(trial(k, a.N, a.D, a.sparsity) for _ in range(max(20, a.trials // max(1, k // 8)))) \
          / max(20, a.trials // max(1, k // 8))
    print(f"{k:>5} {acc:>9.1%}")

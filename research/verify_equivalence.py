"""
Verifies the load-bearing claim of the "Demonstrations Are Weights" plan:

  bdh.py's parallel attention  scores = (rope(Q) @ rope(K).mT).tril(-1); out = scores @ V
  is the EXACT unrolling of a Hebbian fast-weight recurrence:

      sigma_t   = sum_{s<t} rope(K)_s  (outer)  V_s          # shape (N, D)
      output_t  = sigma_t^T @ rope(Q)_t
      sigma_{t+1} = sigma_t + rope(K)_t (outer) V_t

Results measured 2026-09-04 (torch 2.6.0+cu124):
    A. bdh.py as written (Q=K, ReLU, no softmax) : 2.8e-14  (float64)  -> EXACT
    B. Q != K                                    : 1.5e-05  (float32)  -> still exact
    C. no ReLU                                   : 2.7e-05  (float32)  -> still exact
    D. softmax ON                                : 1.1e+02             -> BREAKS

    min causal score, ReLU'd Q=K, no RoPE : +0.0000   (non-negative)
    min causal score, ReLU'd Q=K, w/ RoPE : -2.06     (0.9% of scores negative)

Conclusions that shaped the plan:
  * Only softmax destroys the fixed-size state. Q!=K and dropping ReLU do NOT --
    they cost you the *interpretation* (sigma as a Hebbian synapse matrix over one
    non-negative neuron basis), not the constant memory.
  * RoPE destroys attention-score non-negativity. BDH's "sparse and positive"
    property applies to NEURON ACTIVATIONS, not to attention scores in the public code.
    Those negative scores are the destructive-interference mechanism that stands in
    for the decay term bdh.py does not have.

Reference: github.com/pathwaycom/bdh (MIT), bdh.py -- Attention.forward / get_freqs / rope.
"""
import math
import torch
import torch.nn.functional as F


def get_freqs(n, theta, dtype):
    quantize = lambda t, q=2: (t / q).floor() * q
    return 1.0 / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n)) / (2 * math.pi)


def rope(phases, v):
    v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
    p = (phases % 1) * (2 * math.pi)
    return v * torch.cos(p) + v_rot * torch.sin(p)


def phases(T, N, dtype):
    return torch.arange(0, T, dtype=dtype).view(1, 1, -1, 1) * get_freqs(N, 2 ** 16, dtype).view(1, 1, 1, N)


def parallel(ph, Q, K, V, softmax=False):
    QR, KR = rope(ph, Q), rope(ph, K)
    if softmax:
        T = Q.shape[2]
        s = (QR @ KR.mT).masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 0), float("-inf"))
        s = torch.nan_to_num(torch.softmax(s, dim=-1), 0.0)
    else:
        s = (QR @ KR.mT).tril(diagonal=-1)
    return s @ V


def recurrent(ph, Q, K, V):
    """Explicit materialized sigma. This is what the artifact renders."""
    QR, KR = rope(ph, Q), rope(ph, K)
    B, nh, T, N = Q.shape
    D = V.shape[-1]
    out = torch.zeros(B, nh, T, D, dtype=Q.dtype)
    sigma = torch.zeros(B, nh, N, D, dtype=Q.dtype)          # <-- N x D, not N x N
    for t in range(T):
        out[:, :, t, :] = torch.einsum("bhnd,bhn->bhd", sigma, QR[:, :, t, :])
        sigma = sigma + torch.einsum("bhn,bhd->bhnd", KR[:, :, t, :], V[:, :, t, :])
    return out


def main():
    torch.manual_seed(0)
    B, nh, T, N, D = 1, 2, 24, 64, 8
    mx = lambda a, b: (a - b).abs().max().item()

    for dtype, name in ((torch.float64, "float64"), (torch.float32, "float32")):
        ph = phases(T, N, dtype)
        Q = F.relu(torch.randn(B, nh, T, N, dtype=dtype))
        V = torch.randn(B, nh, T, D, dtype=dtype)
        r = mx(parallel(ph, Q, Q, V), recurrent(ph, Q, Q, V))
        print(f"A. bdh.py as written ({name:7s})        : {r:.3e}")
        assert r < (1e-12 if dtype is torch.float64 else 1e-4), "equivalence broke"

    ph = phases(T, N, torch.float32)
    Q = F.relu(torch.randn(B, nh, T, N))
    K = F.relu(torch.randn(B, nh, T, N))
    Qn = torch.randn(B, nh, T, N)
    V = torch.randn(B, nh, T, D)

    b = mx(parallel(ph, Q, K, V), recurrent(ph, Q, K, V))
    c = mx(parallel(ph, Qn, Qn, V), recurrent(ph, Qn, Qn, V))
    d = mx(parallel(ph, Q, Q, V, softmax=True), recurrent(ph, Q, Q, V))
    print(f"B. Q != K                          : {b:.3e}  (survives)")
    print(f"C. no ReLU                         : {c:.3e}  (survives)")
    print(f"D. softmax ON                      : {d:.3e}  (BREAKS)")
    assert b < 1e-4 and c < 1e-4, "Q!=K / no-ReLU should NOT break the recurrence"
    assert d > 1.0, "softmax must break the recurrence"

    s_norope = (Q @ Q.mT).tril(-1)
    s_rope = (rope(ph, Q) @ rope(ph, Q).mT).tril(-1)
    print(f"\nmin causal score, no RoPE          : {s_norope.min().item():+.4f}")
    print(f"min causal score, with RoPE        : {s_rope.min().item():+.4f}")
    print(f"fraction of causal scores negative : {(s_rope[s_rope != 0] < 0).float().mean().item():.1%}")
    assert s_norope.min().item() >= 0.0, "ReLU'd Q=K should give non-negative scores without RoPE"
    assert s_rope.min().item() < 0.0, "RoPE should destroy score non-negativity"

    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()

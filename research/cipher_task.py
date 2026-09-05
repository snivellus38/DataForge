"""
In-context substitution cipher -- the artifact's live task.

Each example defines a FRESH random bijection over letters, visible only in the prompt:

    "ab cd ef|c" -> answer "d"
     ^^ ^^ ^^  ^
     |  |  |   query source symbol
     k demonstrations, each <src><tgt>

Nothing about the mapping is learnable from the weights: the bijection is resampled per
example, so the only way to answer is to read the demonstrations out of recurrent state.
That is exactly the property the artifact teaches, and it makes the oracle free.

Deliberately a miniature of BDH-CQ's own positive control (arXiv:2608.09888 sec 6.3: a fresh
color permutation defined entirely through demonstrations, applied to held-out queries,
solved 96/96 as simultaneous bindings scale 2->8). We are NOT reproducing BDH-CQ -- this is
an independent toy on the public BDH baseline.
"""
import torch

# Two disjoint byte ranges, 48 symbols each, so k can reach 48 simultaneous bindings.
# The first sweep used 13+13, which HARD-CAPPED k at 13 and made the capacity cliff
# unreachable by construction -- a well-trained model holds 13 bindings comfortably.
# Alphabet size is the load axis; it must exceed the state's capacity for interference
# to be observable at all.
SRC = "".join(chr(c) for c in range(33, 81))    # '!'..'P'   48 symbols
TGT = "".join(chr(c) for c in range(128, 176))  # byte 128..175, 48 symbols
SEP, QRY = ord(" "), ord("|")
assert len(set(SRC) & set(TGT)) == 0 and SEP not in map(ord, SRC + TGT) and QRY not in map(ord, SRC + TGT)


def seq_len(k):
    """Length of a k-demonstration episode, in bytes."""
    return 3 * k + 2 + 1   # k * "<src><tgt> " + "|" + query + answer


_SRC_B = torch.tensor([ord(c) for c in SRC])
_TGT_B = torch.tensor([ord(c) for c in TGT])


def make_batch(batch, k, device="cpu", generator=None):
    """Returns (idx, answer_pos, targets). idx is (B, T) of byte values.

    Fully vectorised -- built directly on `device`. The original per-example Python loop was
    the training bottleneck (256 randperm calls per step on CPU while the GPU idled), not the
    model. Per-row sampling without replacement uses the rand().argsort() trick.
    """
    T = seq_len(k)
    src_tbl, tgt_tbl = _SRC_B.to(device), _TGT_B.to(device)
    srcs = torch.rand(batch, len(SRC), device=device, generator=generator).argsort(dim=1)[:, :k]
    tgts = torch.rand(batch, len(TGT), device=device, generator=generator).argsort(dim=1)[:, :k]
    src_b, tgt_b = src_tbl[srcs], tgt_tbl[tgts]

    idx = torch.full((batch, T), SEP, dtype=torch.long, device=device)
    idx[:, 0:3 * k:3] = src_b          # demonstration sources
    idx[:, 1:3 * k:3] = tgt_b          # demonstration targets; 3i+2 stays SEP
    q = torch.randint(k, (batch,), device=device, generator=generator)
    rows = torch.arange(batch, device=device)
    idx[:, 3 * k] = QRY
    idx[:, 3 * k + 1] = src_b[rows, q]     # query symbol
    ans = tgt_b[rows, q]
    idx[:, 3 * k + 2] = ans                # answer, teacher-forced
    return idx, T - 2, ans                 # predicted FROM index T-2


def decode(row):
    return "".join(chr(int(c)) for c in row)

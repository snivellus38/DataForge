"""
The 8M-parameter BDH architecture, as trained for the English->French translation checkpoints.

PROVENANCE
----------
This is the architecture the author trained for an earlier Pathway hackathon; the class is
transcribed from that training notebook (`models/notebooks/bdh-translation-training.ipynb`, Cell 2)
so the checkpoints in `models/checkpoints/` load exactly. It is NOT the
vendored `pathwaycom/bdh` reference implementation -- that lives untouched in `vendor/bdh.py` and
must stay that way.

HOW IT DIFFERS FROM vendor/bdh.py, AND WHY THAT MATTERS FOR THE ARTIFACT
------------------------------------------------------------------------
Same in every load-bearing respect:
  * weight-tied across layers -- `encoder`, `decoder_x`, `decoder_y` are built once and reused
    inside `for L in range(n_layer)`, so n_layer is an ITERATION COUNT, not depth
    (CLAUDE.md item 6). There are no per-layer parameters.
  * Q = K = ReLU(decoder_x . LN(v)), so attention scores are built from non-negative activations
  * V = LN(v), i.e. dimension D, not neuron space -- so sigma is N x D, not N x N
    (CLAUDE.md item 4)
  * no softmax anywhere -- this is what buys the constant-size state
  * strict-causal mask, tril(diagonal=-1): a token cannot attend to itself
  * RoPE applied to Q and K only, and applied INSIDE attention so the x_sparse feeding the
    multiplicative gate is never rotated (the bug the Phase-2 parity gate caught)

Differs in two ways that must be disclosed wherever this model appears on the page:
  * it adds a LEARNED positional embedding (pos_emb, 4096 x D) on top of RoPE
  * it has no decay term

Both deviations are inherited from the checkpoint and cannot be changed without retraining.
"""
import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BigBDHConfig:
    n_layer: int = 6
    n_embd: int = 192
    n_head: int = 4
    dropout: float = 0.1
    vocab_size: int = 256
    mlp_dim_mult: int = 64

    @property
    def N(self):
        return self.n_embd * self.mlp_dim_mult // self.n_head

    @property
    def N_total(self):
        return self.N * self.n_head


class BigBDH(nn.Module):
    def __init__(self, config: BigBDHConfig):
        super().__init__()
        self.config = config
        D, H, N = config.n_embd, config.n_head, config.N
        self.embed = nn.Embedding(config.vocab_size, D)
        self.pos_emb = nn.Embedding(4096, D)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        self.drop = nn.Dropout(config.dropout)
        self.encoder = nn.Parameter(torch.empty(H * N, D))
        self.decoder_x = nn.Parameter(torch.empty(H, D, N))
        self.decoder_y = nn.Parameter(torch.empty(H, D, N))
        self.lm_head = nn.Parameter(torch.empty(D, config.vocab_size))
        self.register_buffer("rope_freqs", self._build_rope_freqs(N))
        self._extract = False
        self._buffer = {}

    def _build_rope_freqs(self, N, theta=2 ** 16):
        idx = torch.arange(N, dtype=torch.float32)
        idx_q = (idx / 2).floor() * 2
        return (1.0 / (theta ** (idx_q / N)) / (2 * math.pi)).view(1, 1, 1, N)

    def _causal_attention(self, Q, K, V):
        B, H, T, N = Q.size()
        positions = torch.arange(T, device=Q.device, dtype=Q.dtype).view(1, 1, T, 1)
        phases = (positions * self.rope_freqs[:, :, :, :N].to(Q.dtype) % 1.0) * (2 * math.pi)
        cos_p, sin_p = torch.cos(phases), torch.sin(phases)
        Q_rot = torch.stack((-Q[..., 1::2], Q[..., ::2]), dim=-1).reshape_as(Q)
        Q_roped = Q * cos_p + Q_rot * sin_p
        scores = torch.matmul(Q_roped, Q_roped.transpose(-2, -1))
        mask = torch.tril(torch.ones(T, T, device=Q.device, dtype=torch.bool), diagonal=-1)
        scores = scores.masked_fill(~mask, 0.0)
        return torch.matmul(scores, V), scores, Q_roped

    @contextmanager
    def extracting(self):
        self._extract, self._buffer = True, {}
        try:
            yield self._buffer
        finally:
            self._extract = False
            self._buffer = {}

    def forward(self, idx, targets=None):
        C = self.config
        B, T = idx.size()
        D, H, N = C.n_embd, C.n_head, C.N
        v_ast = (self.embed(idx) + self.pos_emb(torch.arange(T, device=idx.device))).unsqueeze(1)
        for L in range(C.n_layer):
            v_n = self.ln(v_ast.squeeze(1)).unsqueeze(1)
            x_pre = torch.einsum("bitd,hdn->bhtn", v_n, self.decoder_x)
            x = F.relu(x_pre)
            a_ast, attn, q_roped = self._causal_attention(x, x, v_n.expand(B, H, T, D))
            y_pre = torch.einsum("bhtd,hdn->bhtn", a_ast, self.decoder_y)
            y = F.relu(y_pre) * x
            if self._extract:
                self._buffer[L] = {"x_pre": x_pre.detach(), "x": x.detach(),
                                   "y_pre": y_pre.detach(), "y": y.detach(),
                                   "a_ast": a_ast.detach(), "v_ast": v_n.detach(),
                                   "attn": attn.detach(), "q_roped": q_roped.detach()}
            y_f = self.drop(y.permute(0, 2, 1, 3).reshape(B, T, H * N))
            v_ast = v_ast + torch.matmul(y_f, self.encoder).unsqueeze(1)
        logits = torch.matmul(v_ast.squeeze(1), self.lm_head)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, C.vocab_size), targets.view(-1))
        return logits, loss


def load_big(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = BigBDHConfig(**ck["config"])
    model = BigBDH(cfg)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model_state_dict"].items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # rope_freqs is a derived buffer and is allowed to be absent; anything else missing means the
    # transcription has drifted from the checkpoint and the extraction below would be garbage.
    hard = [k for k in missing if k != "rope_freqs"]
    if hard or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={hard} unexpected={list(unexpected)}")
    return model.to(device).eval(), cfg, ck


@torch.no_grad()
def generate(model, prompt: str, n=80, device="cpu", greedy=True, temp=0.8, seed=0):
    """Byte-level continuation. Used as the correctness check on the transcription."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = list(prompt.encode("utf-8"))
    for _ in range(n):
        x = torch.tensor([ids[-2048:]], dtype=torch.long, device=device)
        logits, _ = model(x)
        nxt = logits[0, -1]
        if greedy:
            ids.append(int(nxt.argmax()))
        else:
            p = torch.softmax(nxt.float().cpu() / temp, -1)
            ids.append(int(torch.multinomial(p, 1, generator=g)))
    return bytes(ids).decode("utf-8", errors="replace")

# Demonstrations Are Weights

**DataForge 2026 — Pathway track ("Explain the Frontier")**
Topic: **Test-Time Adaptation — Optimization versus Context**

> **Demonstrations are weights you write at inference time: a frozen model adapts by adding
> rank-one updates to a fixed-size associative state, so changing only the demonstrations changes
> the answer — but that state's capacity is set by how much its keys *overlap*, not by its size,
> so the non-negativity that makes BDH inspectable is also what makes it forget sooner.**

---

## Status — work in progress

This repository is **not a finished submission.** Being explicit about that, because the track
judges evidence discipline and a README that overstates its own completeness is the wrong start.

| | |
|---|---|
| Research + primary-source dossier | done |
| Trained model + browser substrate | done, five passing gates |
| Interactive artifact | narrative essay built; sandbox and polish outstanding |
| BDH / BDH-CQ module | built — equations, replayed Table 3, evidence ledger |
| Capacity (1/pi) section | built and live |
| One-page concept summary, blog, final README | not written |

See [`plan.md`](plan.md) for the full design and [`CLAUDE.md`](CLAUDE.md) for the working log,
including the methodology traps this project already fell into and climbed out of.

## What it is

A 131,072-parameter [BDH](https://github.com/pathwaycom/bdh) that we trained, running **live in
the browser** (257 KB, fp16) on an in-context substitution cipher. The mapping is resampled per
example and exists **only in the prompt**, so the only way to answer is to read it out of the
model's recurrent state. Remove a demonstration, keep the query bytes byte-identical, and the
answer changes — while `parameter updates` stays at 0.

It is **not** an official BDH model, and it is **not BDH-CQ**, whose weights are not public.

## Quickstart

```bash
npm test          # five gates: PyTorch parity, equivalence, app logic
npm run dev       # serve the artifact at http://localhost:8080
```

Retraining needs Python + PyTorch + a GPU (~65 min):

```bash
npm run train     # research/train.py -> research/runs/final.pt
npm run export    # weights + test fixture + verified presets
npm run verify:torch   # the PyTorch twin of the equivalence test
```

## Verified results

Everything below is reproducible from this repo.

**The parallel↔recurrent equivalence is exact.** `bdh.py`'s quadratic `tril` attention is the
unrolling of a Hebbian fast-weight recurrence
`σ_{t+1} = σ_t + rope(K)_t ⊗ V_t`, `out_t = σ_tᵀ rope(Q)_t`.

| Check | Result |
|---|---|
| float64 residual | `2.8e-14` — exact |
| Browser JS vs PyTorch logits | `5.2e-7` worst relative, argmax 100% |
| Q ≠ K | equivalence **survives** (`1.7e-5`) |
| no ReLU | equivalence **survives** (`1.6e-5`) |
| **softmax on** | equivalence **breaks** (`1.2e+02`) |

Only softmax costs you the fixed-size state — which is why a Transformer needs a KV cache that
grows. Q=K and the ReLU buy *readability*, not constant memory.

**Non-negativity costs associative capacity, at a rate of 1/π.** Writing `k` bindings into the
real N=256 × D=64 state and reading them back (`research/sigma_capacity.py`):

| k | 8 | 32 | 64 | 128 | 256 | 384 |
|---|---|---|---|---|---|---|
| non-negative keys (BDH) | 100% | 94% | 80% | 47% | 22% | 12% |
| signed keys (counterfactual) | 100% | 100% | 100% | 100% | 100% | 99% |

Mean pairwise cosine is **+0.319** for ReLU'd keys vs **−0.000** for signed — and +0.319 is
analytic: **1/π** for ReLU'd Gaussians. Capacity is bounded by key *overlap*, not by state size
(signed keys hold 384 bindings in a rank-256 state).

**Model accuracy** (trained on k ∈ [2,12], 48-symbol alphabet, chance = 2.1%):

| k | 2–12 | 16 | 20 | 24 | 32 | 48 |
|---|---|---|---|---|---|---|
| accuracy | **100%** | 64% | 40% | 20% | 9% | 3% |

Note: This curve is a **demonstration-coverage** limit, not a capacity limit — the same state
retrieves 99% at k=16 when measured directly. The two must not be conflated.

## What is live vs precomputed

- **Live:** everything currently on the page — the forward pass, σ, the equivalence residual, the
  break-it toggles. Computed in your browser from `web/public/model.bin`.
- **Precomputed:** the capacity curves above (seconds to regenerate; scripts included).
- **Replayed (not yet built):** all BDH-CQ numbers will be quoted from arXiv:2608.09888 with a
  section/table locator. No BDH-CQ result is ever computed here — its weights are not public.
- **Animated:** nothing.

## Honest limitations

- Our toy runs **~43% active neurons**; the BDH paper reports ~5% for trained BDH-GPU at scale on
  language. This model does not reproduce that, and does not claim to. *(Note: the 5% figure is still
  unverified against the primary source.)*
- σ is **dense**, not sparse ridges — 100% of its 16,384 cells are nonzero within a few tokens.
- The model is 131K parameters on a synthetic task. It is an honest miniature, not evidence about
  BDH at scale.

## Attribution and licenses

| Component | Source | License |
|---|---|---|
| `vendor/bdh.py` | [pathwaycom/bdh](https://github.com/pathwaycom/bdh) — **unmodified** | MIT (`vendor/LICENSE-bdh`) |
| Trained weights, task, browser port, artifact | this repo | MIT |
| Colour palette + validation method | Claude Code `dataviz` skill reference palette | — |

Primary sources:
- Kosowski, Uznański, Chorowski, Stamirowska, Bartoszkiewicz — *The Dragon Hatchling*, [arXiv:2509.26507](https://arxiv.org/abs/2509.26507)
- Engdahl et al. — *BDH-CQ: In-Context Learning with Recurrent Latent Reasoning*, [arXiv:2608.09888](https://arxiv.org/abs/2608.09888)
- Schlag, Irie, Schmidhuber — *Linear Transformers Are Secretly Fast Weight Programmers*, ICML 2021
- Katharopoulos et al. — *Transformers are RNNs*, ICML 2020

## AI assistance disclosure

Built with substantial assistance from **Claude (Opus 5)** via Claude Code: literature extraction,
experiment design, training and analysis code, the browser port, and drafting of this README and
`plan.md`. Every empirical claim here is backed by a script in this repo that the team runs and
must be able to defend. The architecture in `vendor/bdh.py` is Pathway's, unmodified.

A full disclosure covering every component ships with the final submission.

# Demonstrations are weights you write at inference time

**And the non-negativity that makes those weights readable is what makes them forget.**

*DataForge 2026, Pathway track — one-page concept summary. Topic: test-time adaptation,
optimization versus context.*

---

### The design pressure

A model meeting an unfamiliar task has two routes. It can **optimize** — turn the demonstrations
into training samples and take a backward pass first, as HRM
([arXiv:2506.21734](https://arxiv.org/abs/2506.21734)) and TRM
([arXiv:2510.04871](https://arxiv.org/abs/2510.04871)) do on ARC at ~$1.48–1.76 per task. Or it can
**adapt in state**: write them into a recurrent memory and answer with every parameter untouched.
BDH-CQ takes the second route three orders of magnitude cheaper. The question is not which is
cheaper, but what the second route *charges* instead.

### The mechanism, and the part everyone guesses wrong

A softmax-free attention with `Q = K` has an exact recurrent form. Where a Transformer keeps a
KV cache that grows with every token, the same computation can be carried in a fixed-size
associative state updated by rank-one writes:

> σₜ  =  σₜ₋₁ + rope(K)ₜ ⊗ Vₜ
> outₜ  =  σₜᵀ rope(Q)ₜ

Not an approximation: the official MIT `bdh.py` run both ways agrees to **2.8 × 10⁻¹⁴ in float64**,
and our 8M model's recurrent decode reproduces its parallel forward to **6 × 10⁻⁷**. The natural
assumption is that BDH's three unusual choices — no softmax, `Q = K`, a ReLU forcing non-negative
activations — all buy that constant memory. **They do not.** Softmax breaks the equivalence
(`1.2 × 10²`); `Q ≠ K` (`1.7 × 10⁻⁵`) and dropping the ReLU (`1.6 × 10⁻⁵`) leave it intact.
Softmax-freeness buys the fixed state. `Q = K` and the ReLU buy *readability*.

### What readability costs

Non-negative activations make BDH inspectable — sparse, positive, monosemantic structure is the
Dragon Hatchling paper's claim ([arXiv:2509.26507](https://arxiv.org/abs/2509.26507)). On an 8M BDH
we trained, 4.93% of units clear 1% of their peak against GPT-2's 86.3%, and `G* = Dₓᵀ Eᵀ` predicts
**from the weights alone, with no data**, which neurons stay silent on real text: MCC **+0.944**,
against **−0.001** for a shuffled null and **+0.908** on an independent Portuguese sibling.

That same non-negativity caps what the state can hold. Writing *k* rank-one bindings into a fixed
N×D state and reading them back gives 100% at k=8, 94% at k=32, 47% at k=128. The counterfactual
identifies the cause: with **signed** keys the identical state holds 384 bindings at 99%, past its
own rank. Mean pairwise cosine is **+0.319 for ReLU'd keys versus −0.000 for signed**, and +0.319
is analytic — **1/π** for rectified Gaussians. Non-negative vectors cannot be near-orthogonal, so
they interfere. Interpretability is paid for in memory, at a fixed rate. The ceiling belongs to the
linear-attention family, not to BDH: DeltaNet
([arXiv:2406.06484](https://arxiv.org/abs/2406.06484)) replaces "the additive update in linear
transformers" — BDH's write — with a delta rule, reporting it better at associative recall. Ours is
the mechanism and the constant.

| | KV-cache Transformer | BDH (public, MIT) | BDH-CQ | HRM / TRM |
|---|---|---|---|---|
| Memory per extra token | grows linearly | **constant**, cheaper past `N_total/2` | constant (dims undisclosed) | no context state |
| Adaptation lives in | the context window | recurrent state σ | recurrent state | **weights** — a backward pass |
| Inspectability | attention maps | **4.93%** active + a synapse graph | none published | none |
| Evidence here | cost only | **we train and run it** | replayed, cited | authors' report |

GPT-2 is not a control. The contrast that is structural rather than empirical: **38.7%** of BDH's
neurons are isolated in `G*` and they are the silent ones, against **0.1%** in GPT-2's nearest
analogue, which maps *two different* neuron populations. Weight-tying is what makes the question
well posed.

### The roles of BDH and BDH-CQ

**BDH is the public substrate** — MIT-licensed, runnable, and what every number above was measured
on. **BDH-CQ is evidence only**: no weights, no API, dimensions and update rules "remain
proprietary." The bridge is exact: BDH-CQ's update is Eq. (1) `Sₜ = U_θ(Sₜ₋₁, Dₜ)`, and §3.2 names
its special case `Sₜ = Sₜ₋₁ + U_θ(Dₜ)` as "linear attention being the conceptually simplest
standalone realization." **That special case is exactly what `bdh.py` computes.** One correction:
additive accumulation is the *named special case*, not BDH-CQ's rule — calling it BDH-CQ's
mechanism, as secondary summaries do, is an error.

### Reading the evidence honestly

BDH-CQ's headline is 29.5% pass@2 (118/400) on public ARC-AGI-1 at ~0.85 H200 GPU-s/task
(arXiv:2608.09888, Table 1). Four qualifications, all from the report:

- **$0.00070/task is computed, not billed** — 0.85 s × an assumed $3/H200-hour. §6.6 reports the
  *same* 118/400 at **$0.00265246**, 3.8× higher, unreconciled. There, "57× cheaper" is ~15×.
- **Not equal-accuracy**: GPT-5.6 Luna (Low) scores *higher* at 34.2%, and the Wilson interval on
  29.5%, [25.24, 34.15], overlaps it.
- The audit reproducing 29.5% was by **co-authors** — developer-reported, not external. BDH-CQ
  appears on no ARC Prize leaderboard.
- **ConceptARC is in the training mixture** (§4.2) and then evaluated on (59.38%); the paper says
  its control does not rule out training exposure.

**The sharpest limitation is structural, not numerical.** Latent reasoning returns grids, not
traces, so — the report's words — "a correct rule applied incompletely is observationally
indistinguishable from a narrower rule applied completely." Table 9: 89.7% of failures still have
correct output dimensions. Not verbalizing is what makes it cheap, and why you cannot audit it.

**Continue at** https://demonstrations-are-weights.vercel.app, then the Dragon Hatchling paper,
BDH-CQ §3.2 and Appendix A.3–A.4, and DeltaNet.

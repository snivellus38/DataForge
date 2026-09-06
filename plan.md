> **HISTORICAL — this is the concept plan as approved on 2026-09-05, kept for the record.**
> It describes a single-page artifact that no longer exists. The build diverged deliberately and
> the plan was not retro-fitted, because a plan rewritten to match what happened stops being
> evidence of what was decided in advance.
>
> For what was actually built and why, read [`README.md`](README.md); for the reasoning, the
> measurements and the dead ends, read [`CLAUDE.md`](CLAUDE.md). The two substantive changes:
> the artifact became **three pages** (THE FIELD / THE LOOP / THE PRICE) rather than one essay,
> and the claim's second clause was settled as the **1/π capacity trade-off**. Everything this
> plan says about the science — the equivalence, the BDH-CQ bridge, the evidence discipline —
> still holds and was carried through.

---

# DataForge 2026 — Pathway Track: "Demonstrations Are Weights"

## Context

DataForge's Pathway track ("Explain the Frontier") asks for **one self-contained interactive artifact** that makes a single frontier AI concept genuinely click, wired to Pathway's Dragon Hatchling (BDH) or BDH-CQ. It is modeled on the NeurIPS 2026 Education Track. There is no existing code — `Pathway_PS.md` is the only file in the repo, so this is a greenfield concept + build plan.

The track's own bar: **weak** = generic overview, animation passed off as computation, bolted-on BDH mention. **Strong** = one claim, real substrate, honest labeling, a BDH module that teaches, a disclosed limitation. **Exceptional** = a reusable substrate, a claim reproducible in under a minute, or a resource people keep using afterward.

**Team constraints (confirmed):** 3+ weeks · strong React/D3/canvas · own NVIDIA GPU · high-ceiling risk appetite. This is the best-case envelope, so the plan targets the "exceptional" bar: a real model we trained, running live in the browser, with its memory on screen.

### The decisive research finding

**BDH-CQ cannot be run. There are no public weights, no public API, no demo** — access is gated behind "reach out to us," and the report states "dimensions, exact update rules, and implementation details remain proprietary." Any team that plans a live BDH-CQ demo will fail. Every BDH-CQ number must be clearly-labeled replayed evidence.

But there is an exact, citable bridge from what *is* public to what isn't. BDH-CQ's memory update is Eq. (1), `S_t = U_θ(S_{t−1}, D_t)`, and the report names its special case explicitly:

> "…linear attention being the conceptually simplest standalone realization of linear correction rules on S, capturing the special case `S_t = S_{t−1} + U_θ(D_t)`."
> — arXiv:2608.09888 §3.2, p.3

**That special case is exactly what the official MIT-licensed `bdh.py` computes.** Because it uses no softmax and `K is Q`, its quadratic `tril` attention is the parallel unrolling of a Hebbian fast-weight recurrence:

> `σ_t = Σ_{s<t} QR_s ⊗ V_s` , `output_t = σ_tᵀ QR_t` , `σ_{t+1} = σ_t + QR_t ⊗ V_t`

**VERIFIED NUMERICALLY** (`torch 2.6`, local GPU box): float64 residual `max|parallel − recurrent| = 2.8e-14`. It is exact; the float32 figure of `2.3e-5` is pure accumulation error. So we can **materialize BDH's synapse matrix and put it on screen while it is being written**, and honestly say it instantiates the special case the BDH-CQ paper names — without ever claiming to run BDH-CQ.

⚠ Three subtleties, all verified, all worth defending live:
- The materializable state is **N×D**, not the paper's conceptual neuron×neuron N×N — `V = x` (dim D) and a LayerNorm between `scores @ V` and the projection back to neuron space blocks the exact reduction.
- **The public baseline has no decay term.** The paper's σ has damping; `bdh.py` has none. Forgetting comes from destructive RoPE phase interference instead.
- **RoPE destroys score non-negativity, and this was measured.** With `Q = K = relu(...)` and *no* RoPE, the minimum causal attention score is exactly `+0.0000`. With `bdh.py`'s RoPE applied, it is `−2.06`, and **0.9% of causal scores are negative**. So the "sparse and positive" property of BDH holds for *neuron activations*, not for attention scores in the public code. Those negative scores **are** the destructive-interference mechanism that stands in for the missing decay term. Conflating the two non-negativities is the subtlest available error here, and we teach the distinction rather than fall into it.

⚠ The DataForge brief's own hint ("state accumulates additively per demonstration") is a **half-truth** — additive accumulation is the named special case, not BDH-CQ's actual rule. Saying otherwise is a factual error. We correct it on screen; that correction is itself a teaching beat.

---

## The submission

**Declared topic:** Test-Time Adaptation — Optimization versus Context.
*(The brief annotates this topic with the exact HRM/TRM-vs-BDH-CQ contrast we build. Associative memory / fast weights is the mechanism, not a second topic — keeping it as mechanism preserves one central claim.)*

**One-sentence falsifiable claim:**

> **Demonstrations are weights you write at inference time: a frozen model adapts by adding rank-one updates to a fixed-size associative state, so changing only the demonstrations changes the answer — but that state's capacity is set by how much its keys overlap, not by its size, so the non-negativity that makes BDH inspectable is also what makes it forget sooner.**

**How a learner falsifies it.** Four independent attacks, all live: (1) hold the test input byte-identical, change only demonstrations — if the answer never changes, the first half is wrong (verified: correct at p=1.000 with the demonstration, wrong at p=0.972 without it); (2) check the weight hash before and after adaptation — if it changes, "frozen" is wrong; (3) flip the keys from non-negative to signed and re-measure retrieval — if capacity does **not** improve, the second half is wrong (measured: signed keys hold 100% at k=256 and 99% at k=384, where non-negative keys are at 22% and 12%); (4) push `k` past the state's rank N — if failure tracked *size* rather than overlap, signed keys would fail at N too, and they do not.

**Why this second clause and not the undemonstrated-value boundary.** Both are real limitations. This one is *our own measurement*, causally isolated by an explicit counterfactual, reproducible in seconds, and it ties sparsity, interpretability and memory into one mechanism with an analytic constant (1/π). BDH-CQ's Table 8 boundary stays in the artifact as clearly-labelled replayed evidence — it is simply not what the headline claim rests on.

**Audience:** ML practitioners and senior CS undergrads who know what a Transformer and a KV cache are, have used in-context learning, but have never seen the state that in-context learning writes to. **Prerequisites:** matrix multiply, outer product, softmax attention. Not required: RoPE, linear attention, BDH.

---

## Artifact architecture

Three layers, with the live/replayed boundary drawn hard and labeled in the UI.

### Layer A — live shrunk BDH (real, ours, in-browser)

Train `pathwaycom/bdh` at a deliberately shrunk config so σ is legible:

`n_layer=4, n_embd(D)=64, n_head(nh)=2, mlp_internal_dim_multiplier=8` → **N = 8·64/2 = 256** neurons/head, **131,072 params measured** (~0.26 MB fp16 — trivially shippable). σ per head = **256×64 = 16,384 floats**; 32,768 per layer at nh=2. Directly renderable as a heatmap.

**⚠ MEASURED, and it corrects an earlier estimate of ~426K: BDH is weight-tied across layers.**
`encoder`, `encoder_v` and `decoder` are constructed once in `__init__` and reused inside
`for level in range(n_layer)`. There are **no per-layer parameters**. So `n_layer` is not depth —
it is an **iteration count of one shared operator**, and the whole model is `3·nh·D·N + 2·vocab·D`.

This is a gift, not a nuisance. It maps directly onto BDH-CQ's latent reasoning loop
`H_{r+1} = F_θ(H_r, S_K)` (Eq. 3) — *the same θ applied R times*. BDH's layer loop has literally
that structure. So **"iterations" becomes a live control that is structurally the same knob as
BDH-CQ's reasoning effort**, with the honest caveat displayed on screen that the report never
states R and never says what physically changes between effort levels.

**Task: an in-context substitution cipher.** Byte sequences like `▲→q ●→m ■→k | ●→?` where the model must apply a bijection defined *only* in the prompt. Chosen deliberately: it is a faithful miniature of BDH-CQ's own positive control (§6.3 — a fresh color permutation defined entirely through demonstrations, solved 96/96 as simultaneous bindings scale 2→8). Ground truth is a deterministic oracle, so **truth-beside-estimate is free**, and `k` (number of bindings) is a real control that maps to a real paper experiment.

Inference path: hand-written JS/WebGPU forward pass. Matrices are 256×64; sub-millisecond per token. No ONNX dependency, and writing it ourselves is what lets us expose σ.

### Layer B — the σ inspector (the heart)

- **σ is dense** (measured: 100% of 16,384 cells nonzero within a few tokens, because keys are
  ~43% dense). So the renderer shows the things that are actually legible, not the raw state alone:
  **Δσ per token** (the write is genuinely rank-one and visibly structured), the **before/after
  difference map** when a demonstration is removed, and the **row-energy profile** ‖σₙ‖ showing
  which neurons carry the binding. A heatmap of raw σ by itself reads as noise.
- Reading a query highlights which σ rows it addresses.
- **The realness proof:** compute the answer twice — once by `bdh.py`'s parallel `tril` form, once by the explicit recurrence — and display `max|Δ|` live (`2.8e-14` in float64, `2.3e-5` in float32). **A scripted animation cannot do this.**
- **The break-it panel, corrected against measurement.** My first draft claimed Q≠K and removing the ReLU would kill the equivalence. They do not — measured `1.5e-5` and `2.7e-5`, i.e. still exact. **Only softmax breaks it** (`1.1e+02`). This correction makes the lesson *sharper*, not weaker, and it becomes the artifact's central insight:
  - **Softmax ON → the fixed-size state dies.** Normalization couples every score to every other, so you must keep all past keys. *This is precisely why a Transformer needs a KV cache that grows.* One toggle, and the learner watches a fixed-size state become an unbounded one.
  - **Q≠K, or ReLU removed → the recurrence survives, but the interpretation dies.** σ stops being a Hebbian synapse matrix over one non-negative neuron basis. These choices buy BDH its *readability*, not its constant memory. Learners — and most write-ups — assume otherwise.

### Layer C — the other route: optimization (real gradient steps)

Same unseen cipher, solved the other way: a small head fine-tuned on the demonstrations in-browser. Two counters run side by side throughout — **`parameter updates: 0` / `parameter updates: 47`** — plus `hash(θ)` before and after, bit-identical on the context route. This is HRM/TRM's regime versus BDH-CQ's, and the report's cost gap is cited beside it (ARC Prize: HRM $1.48/task, TRM $1.76/task, both transductive; BDH-CQ $0.00070–$0.00265/task, developer-reported).

### Layer D — BDH-CQ evidence panel (replayed, labeled)

Never live. Learner predicts before each reveal:
- **Table 3, the cleanest experiment in the paper** — test inputs byte-identical, only demonstrations change: ordering length-8 **0/24 → 13/24**, nesting depth-5 **19/24 → 24/24**.
- **Table 4** — rotation composes (72/72), reflection only partly (47/72), color swap never (0/72), *with* the paper's own caveat that shuffled families reach 1/24 alone.
- **Table 8** — the undemonstrated-value boundary.

---

## Controls (six, each one real variable)

| Control | Concept variable | Visible consequence |
|---|---|---|
| Add / remove / edit a demonstration | what is written into σ | a rank-one ridge appears or vanishes; answer flips |
| `k` = simultaneous bindings (2…12) | associative capacity | ridges overlap; reads corrupt |
| Query token | the read operation | highlighted σ rows; output vs oracle |
| Parallel ↔ recurrent residual | mathematical equivalence | `max|Δ|` ≈ 1e-6 |
| Softmax on/off | whether state can stay fixed-size | equivalence dies; state must grow (the KV-cache lesson) |
| Q≠K / ReLU off | what non-negativity actually buys | recurrence survives; σ stops being readable as synapses |
| Layer + head selector | where memory lives | different σ per layer/head |
| **Iterations** (`n_layer`, 1…6) | repeated application of one shared operator | accuracy vs compute; the structural analogue of BDH-CQ's Eq. (3) latent loop |

## The sixty-second moment

Page opens with the cipher **already running** — three demonstrations, three bright ridges in σ, correct answer beside a green oracle tick. The learner drags one demonstration out. A ridge disappears; the answer flips to wrong; the oracle turns red. **`parameter updates: 0`** never moves. The whole claim, in one gesture, before any reading.

## Limitations taught

0. **The toy does not reproduce BDH's reported sparsity, and says so.** Our model runs ~43% active
   neurons; the paper reports ~5% for trained BDH-GPU at scale on language. Stated on screen as a
   limitation of the toy, not smoothed over. (⚠ the 5% figure is still UNVERIFIED against the
   primary source — check arXiv:2509.26507 before quoting it.)
1. **Interference is the price of *non-negativity*, and we measured the exchange rate.**
   Writing `k` rank-one bindings into the real N=256×D=64 state and reading them back:
   100% to k=8, 94% at k=32, 47% at k=128, 22% at k=256. The counterfactual is the lesson —
   with **signed** keys retrieval holds **100% to k=256 and 99% at k=384**, past the rank bound.
   Mean pairwise cosine is **+0.319 (non-negative) vs −0.000 (signed)**, and +0.319 is analytic:
   **1/π** for ReLU'd Gaussians. So the ReLU that gives BDH sparse, positive, *interpretable*
   activations is exactly what caps its associative memory — non-negative vectors cannot be
   near-orthogonal. **Interpretability is paid for in capacity, at a rate of 1/π.**
   This is our own measurement (`research/sigma_capacity.py`), runs in seconds, and ships with
   its own reference curve — truth beside estimate, for free. Compare honestly against BDH-CQ's
   reported 24/24 at k=8 (§6.3), labeled as a bigger, trained, different system.
2. **The undemonstrated-value boundary.** Learner predicts interpolation is safe. Table 8 says it is not: 0/40 interpolated *equals* 0/80 extrapolated.
3. **Latent reasoning leaves no trace.** The report's sharpest self-criticism, quoted: "a correct rule applied incompletely is observationally indistinguishable from a narrower rule applied completely." That is the real cost of not verbalizing.

## Narrative arc

Hook (cipher already live) → open σ → prove it's real (equivalence + break-it) → the optimization route side by side → where it breaks (`k`, then the value boundary) → BDH module (Eq. 1 and its special case; Eqs. 2–4 with "R is never stated in the paper"; replayed Table 3) → evidence ledger → sandbox.

## The reusable substrate ("exceptional" bar)

Ship **`bdh-lens`**: a small MIT library (Python + JS) that materializes σ from any `bdh.py`-family model and renders it. The artifact is its first consumer. This is the thing people keep using after the hackathon.

---

## One-page concept summary (500–950 words)

**Thesis:** test-time adaptation splits into two mechanisms — changing parameters, or writing to state — and BDH-CQ is evidence the second can be competitive at a fraction of the cost, with a distinct and measurable failure surface.

**Comparison table rows:** BDH-CQ · HRM · TRM · a frontier CoT model (GPT-5.6 Luna Low). **Columns:** adaptation mechanism (state write vs backward pass) · parameters updated at inference · reported ARC-AGI-1 pass@2 · cost basis · evidence level.

**Evidence discipline is the differentiator** — every number labeled by tier:
- The report gives **two unreconciled costs for the same 118/400 result**: $0.00070 (§5) and $0.00265246 (§6.6). At the higher figure "57× cheaper" becomes ~15×.
- $0.00070 is **computed**, not billed — 0.85 H200-GPU-seconds × an assumed $3/H200-hour. Competitors' figures are API prices with vendor margin. Figure 2's own footnotes concede three accounting bases on one axis.
- **BDH-CQ is not on the ARC Prize leaderboard** (official or community, checked 2026-09-04). HRM (2.0%) and TRM (7.8%) are.
- The "independent" audit was conducted by **co-authors** (Kinas, Bielik; Zhong, NYU).
- **ConceptARC is in the training mixture** (§4.2) and then used as an evaluation set (§6.1, 59.38%); §6.5 concedes the control "does not… rule out exposure through training."
- The Łukasz Kaiser replication is **press-release only**. The "1B→600B scaling laws confirmed" claim (§9.2) is two sentences with no data.
- **No ARC-AGI-2 numbers exist anywhere.**
- 29.5%'s Wilson 95% CI is [25.24, 34.15] — it nearly touches Luna Low's 34.2%.

**Primary papers (≥3, 2022–2026):** Kosowski et al. 2025 (arXiv:2509.26507) · Engdahl et al. 2026 (arXiv:2608.09888) · Schlag et al. 2021, *Linear Transformers Are Secretly Fast Weight Programmers* · Katharopoulos et al. 2020 · Wang et al. 2025 (HRM) · Jolicoeur-Martineau 2025 (TRM).

**Never state:** what "CQ" stands for (undefined in every source); that effort levels = a specific R (never disclosed); that BDH is an SSM in the Mamba sense (the paper says "ReLU-low-rank transformations with linear attention," and never uses the term).

---

## Build plan — riskiest first

**Week 1 — kill the main risk.** Does a 426K-param shrunk BDH actually learn the in-context cipher? Train on GPU; sweep `k`; measure the capacity cliff. *If it does not learn, everything downstream changes* — fall back to a larger config or a single-binding task first. In parallel: verify the parallel↔recurrent equivalence numerically in PyTorch and pin the residual.

**Week 2 — substrate.** Port the forward pass to JS/WebGPU; validate against PyTorch to 1e-5; build the σ canvas renderer and the equivalence/break-it panel. Extract `bdh-lens`.

**Week 3 — narrative + deliverables.** Guided arc, optimization-route panel, replayed BDH-CQ panels, mobile pass, README, one-pager, provenance and AI-disclosure records.

**Cut line if time runs short — cut in this order:** the composition matrix (Table 4) → the layer/head selector → the optimization-route panel → mobile polish.
**Must not be cut:** the live σ inspector, the equivalence + break-it proof, the `parameter updates: 0` counter, the replayed Table 3, the evidence ledger, the one-pager.

## How this could still lose, and the mitigations

| Risk | Mitigation |
|---|---|
| ~~Shrunk BDH never learns the cipher~~ | **RETIRED 2026-09-05.** 131K-param BDH reaches **100% at every trained k (2–10)** and extrapolates: 100% at k=11–12, 99.4% at k=13. `research/week1_gate.py` |
| ~~Equivalence claim is wrong~~ | **Retired.** Verified numerically at float64: `2.8e-14` |
| No visible capacity cliff | **Diagnosed 2026-09-05, still open.** Two causes, both mine: (a) the 13-symbol alphabet hard-capped `k` at 13, so the cliff was unreachable by construction — alphabet is now 48+48; (b) shrinking `N` at a fixed 2500-step budget produced decay curves that were largely *undertraining* (N=64: 47%→70% at k=2 when trained 3.2× longer, still climbing). Correct method: train a capable model to convergence, then raise `k` until a fixed-size σ genuinely fails. Running now. **Do not present a small-`N` curve as interference unless k=2 is ~100%.** |
| A judge reads it as "BDH-CQ demo" | The word "BDH-CQ" never appears on a live panel; every replayed number carries a source locator badge |
| Overclaiming the additive form | On-screen correction of the brief's own half-truth, quoting §3.2's "special case" |
| σ heatmap is pretty but unreadable | Shrink until ridges are countable by eye; ridge count must equal demonstration count |
| Evidence ledger reads as paper-bashing | Frame as evidence *tiers*, not accusations; the report is unusually candid and we say so |

## Verification

- `pytest` asserting `max|parallel − recurrent| < 1e-12` in float64 and `< 1e-4` in float32 across sequence lengths; asserting it breaks **only** under softmax; and asserting the RoPE score-sign result (`min ≈ 0` without RoPE, `< 0` with). Already verified once by hand — port that script into the repo as the first test.
- JS forward pass diffed against PyTorch logits to 1e-5 in CI.
- Oracle-vs-model accuracy sweep over `k`, committed as the reproducible capacity curve.
- Every replayed number in the UI carries a `§/Table` locator; a link-check job runs in CI.
- Sixty-second test on five people who have not seen it: can they state the claim back and name one limitation?

## Open alternative (not recommended, recorded for the record)

Declaring **Cost-Accuracy Pareto Frontiers** instead, and building the evidence ledger into the main artifact — a "challenge the headline" tool with toggles for GPU price, the $0.00070-vs-$0.00265246 choice, and the Luna discount. It is a genuinely strong idea with an unusually sharp falsifiable claim, but the live substrate would be a spreadsheet rather than a model, which scores worse on "interactive substrate." Keeping it as the one-pager's spine gets most of the value.

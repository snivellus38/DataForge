# CLAUDE.md — session handoff

**Read this file fully before doing anything.** It carries context that exists nowhere else in
the repo. At the end of your session, append your own entry to the Session Log.

## What this project is

A submission for **DataForge 2026, Pathway track ("Explain the Frontier")** — one self-contained
**interactive** educational artifact that makes a single frontier AI concept click, connected to
Pathway's Dragon Hatchling (**BDH**) or **BDH-CQ**. Modeled on the NeurIPS 2026 Education Track.

- `Pathway_PS.md` — the problem statement. Authoritative. Read it.
- `plan.md` — the approved concept + build plan ("Demonstrations Are Weights").
- `research/bdh-cq-dossier.md` — **expensive primary-source extraction. Do not regenerate.**
  The full BDH-CQ technical report extracted verbatim, with a section/table locator and an
  evidence level on every claim. It cost most of a ~790K-token agent run, and it was *rescued
  from a temp directory that gets garbage-collected* — if it is ever lost, it is expensive to
  rebuild and the failure mode is subtle: an agent without it will produce plausible-looking
  BDH-CQ figures that are wrong. Treat it as canonical for every BDH-CQ number.
- `research/verify_equivalence.py` — runnable proof of the plan's core technical claim, and a
  **regression guard**. It asserts the positive result (parallel == recurrent) *and the negative
  ones* (Q≠K and no-ReLU must **not** break the recurrence; softmax must; RoPE must destroy score
  non-negativity). So if someone later "improves" the model and quietly breaks the σ story, this
  fails loudly. **Move it into the real test suite as soon as the repo has one.**

- `vendor/bdh.py` — unmodified official implementation, MIT (`vendor/LICENSE-bdh`). Do not edit.
- `research/cipher_task.py` — the in-context substitution-cipher task (fresh bijection per example).
- `research/week1_gate.py` — the feasibility gate. **Passed.** See Session Log.

**Status: no product code has been written yet.** Only the plan, the dossier, and the
verification script exist.

## Team constraints (confirmed with the user)

3+ weeks · strong React/D3/canvas · own NVIDIA GPU (torch 2.6.0+cu124, CUDA available) ·
high-ceiling risk appetite, aiming to win. Design for this envelope — do **not** retreat to
Streamlit/Gradio/a plain notebook.

## The five things you would get wrong without this file

**1. BDH-CQ cannot be run. Full stop.**
No public weights, no public API, no demo. Access is gated behind "reach out to us," and the
report states internals "remain proprietary." Every BDH-CQ number in the artifact must be
clearly-labeled *replayed* evidence from arXiv:2608.09888. The live substrate must be the
official MIT `pathwaycom/bdh` at tiny scale, or an explicitly-labeled toy. Never present either
as an official BDH/BDH-CQ model.

**2. There is an exact, citable bridge to BDH-CQ that does not require running it.**
BDH-CQ's memory update is Eq. (1) `S_t = U_θ(S_{t−1}, D_t)`. §3.2 names its special case:
`S_t = S_{t−1} + U_θ(D_t)` — "linear attention being the conceptually simplest standalone
realization." **That special case is exactly what the public `bdh.py` computes.** This is the
whole reason the plan works.

**3. The equivalence is VERIFIED, not assumed — don't re-derive it, and don't re-run agents on it.**
`bdh.py`'s parallel `scores=(rope(Q)@rope(K).mT).tril(-1); out=scores@V` is the exact unrolling of
`sigma_t = Σ_{s<t} rope(K)_s ⊗ V_s`, `out_t = sigma_t^T rope(Q)_t`. Measured **2.8e-14 in float64**.
Run `python research/verify_equivalence.py` to reproduce in seconds.

**4. Three corrections to the obvious intuitions — all measured, all load-bearing.**
- **Only softmax breaks the fixed-size state** (`1.1e+02`). Q≠K (`1.7e-5`) and dropping ReLU
  (`1.6e-5`) do **not** break the recurrence. They cost you the *interpretation* of σ as a Hebbian
  synapse matrix, not the constant memory. An earlier draft of the plan claimed otherwise and was
  wrong. Softmax-freeness buys constant memory; Q=K + ReLU buy readability.
- **RoPE destroys attention-score non-negativity.** ReLU'd `Q=K` without RoPE → min causal score
  `+0.0000`. With `bdh.py`'s RoPE → `−2.06`, with 0.9% of scores negative. BDH's "sparse and
  positive" property applies to **neuron activations**, not attention scores. Those negative scores
  *are* the destructive-interference mechanism standing in for the decay term `bdh.py` lacks.
- **The materializable state is `N×D`, not the paper's conceptual neuron×neuron `N×N`.** `V = x`
  (dim `D`), and a LayerNorm between `scores @ V` and the projection back to neuron space blocks
  the exact reduction.

**5. The PS itself contains a half-truth — do not repeat it.**
`Pathway_PS.md` says BDH-CQ's "state accumulates additively per demonstration." Additive
accumulation is the *named special case*, not BDH-CQ's actual rule (which is the general
`U_θ`, and is proprietary). Stating it as BDH-CQ's mechanism is a factual error. The plan
corrects this on screen as a teaching beat.

## Never state these (each is a factual error a judge will catch)

Full list with sources in `research/bdh-cq-dossier.md` under MISCONCEPTION TRAPS. The worst:

- What **"CQ"** stands for — undefined in every source. Do not guess.
- That effort levels = some number of latent iterations `R` — `R` is never disclosed.
- That BDH is a **state-space model in the Mamba sense** — the PS forbids it and the paper says
  "ReLU-low-rank transformations with linear attention," never "state-space model."
- That 29.5% is **ARC-Prize-verified** — BDH-CQ is on *neither* leaderboard (checked 2026-09-04).
- That the independent audit was third-party — **both auditors are co-authors**.
- That ConceptARC 59.38% is clean — **ConceptARC is in the training mixture** (§4.2).
- That any **ARC-AGI-2** result exists — none does, anywhere.
- Any number not in the dossier. If it isn't there, mark it as needing verification.

Also worth knowing: the report gives **two unreconciled costs for the same 118/400 result** —
$0.00070 (§5) and $0.00265246 (§6.6). At the higher figure "57× cheaper" becomes ~15×. This is
the sharpest honest hook available and the spine of the one-page summary.

**6. BDH is WEIGHT-TIED across layers. Measured, and it surprised me.**
`encoder`, `encoder_v` and `decoder` are built once in `__init__` and reused inside
`for level in range(n_layer)`. There are **no per-layer parameters**: total = `3·nh·D·N + 2·vocab·D`.
The plan's shrunk config is **131,072 params, not the ~426K I first estimated** (that estimate
assumed per-layer weights — wrong). `n_layer` is an *iteration count of one shared operator*, not
depth. This maps onto BDH-CQ Eq. (3) `H_{r+1} = F_θ(H_r, S_K)` — same θ applied R times — so
"iterations" is a live control with a real, defensible link to BDH-CQ's reasoning effort. Caveat
that must stay on screen: the report never states R, nor what changes between effort levels.

## The one unretired risk

~~Will the shrunk BDH learn the cipher?~~ **RETIRED 2026-09-05 — it does, at 100%.**

The open question is now the opposite one: **σ is too big to show interference.** Accuracy is
100% through k=13, so there is no visible capacity cliff, and "visible state" is a stated design
requirement. Shrink N until the cliff appears (`research/sweep_capacity.sh` sweeps
`mlp_internal_dim_multiplier` ∈ {1,2,4} → N ∈ {32,64,128}); results land in `research/runs/`.

**7. Two methodology traps in the capacity experiment. Both cost a day; do not re-enter them.**
- **Undertraining masquerades as interference.** A sweep over `N ∈ {32,64,128}` at 2500 steps
  produced beautiful monotone decay-to-chance curves at small `N`. They were mostly an artifact of
  the step budget: N=64 went **47% → 70% at k=2** when trained 2500 → 8000 steps, and was still
  climbing. N=96 @ 8000 also *beat* N=128 @ 2500. **Never compare configs at a fixed step budget
  and call the difference capacity.** The tell is low-`k` accuracy: if k=2 is not ~100%, the model
  has not learned the algorithm and nothing about its high-`k` behaviour is a capacity claim.
- **The task itself capped the load.** The original alphabet was 13 source symbols, so `k ≤ 13`
  by construction, and a trained model holds 13 bindings easily. The cliff was unreachable by
  design, not absent. The alphabet is now 48+48 (`research/cipher_task.py`), so load can exceed
  state capacity. **Alphabet size is the load axis** — if you shrink it, you re-cap `k`.

**8. The interference lesson has a rigorous, measured mechanism — and it is the strongest
result in the project so far.** Measuring capacity via training is confounded (item 7). Measure
sigma directly instead: `python research/sigma_capacity.py`. Writing k rank-one bindings into an
N=256, D=64 state and reading them back gives 100% to k=8, 94% at k=32, 47% at k=128, 22% at k=256.
The counterfactual identifies the cause: with **signed** keys, retrieval stays **100% to k=256 and
99% at k=384** — past the rank bound. Mean pairwise cosine is **+0.319 for ReLU'd (non-negative)
keys vs -0.000 for signed**, and +0.319 is analytic: **1/pi = 0.3183** for ReLU'd Gaussians.
So **BDH's non-negativity — the source of its sparse, positive, interpretable activations — is
precisely what caps its associative capacity**, because non-negative vectors cannot be
near-orthogonal. Interpretability is paid for in memory, at a rate of 1/pi. This is the artifact's
limitation beat, it is our own measurement, and it has an explicit counterfactual as its reference.

**9. MODEL degradation and STATE capacity are two different curves. Do not conflate them.**
Trained N=256 on k in [2,10] to loss 0.0000: 100% at k=2/5/8, then 86% at k=11, 62% at k=14,
31% at k=20. Looks like the state filling up. It is not. The SAME state, measured directly
(`sigma_capacity.py`), retrieves 100% at k=16 and 95% at k=32 -- far past where the model fails.
- **Model curve = extrapolation limit**, set by the k range seen in training. This is the
  artifact's *demonstration-coverage* lesson, and it maps onto BDH-CQ Table 3.
- **Sigma curve = associative capacity**, set by fixed state size and the 1/pi non-negative
  key overlap. This is the artifact's *interference* lesson.
They are separate beats teaching separate things. Labelling the model curve as "interference"
or "the state running out of room" is a factual error -- it is a training-coverage effect.
To probe capacity through the model rather than the state, training k must extend past the
eval range, otherwise coverage is the binding constraint, not capacity.

**10. Model tokens are raw bytes and are NOT human-readable. The UI must map them.**
`SRC` = bytes 33-80, `TGT` = bytes 128-175, so a demonstration prints as `'"' -> ''`.
Do not surface raw bytes to a learner. `web/public/*.json` carries a `display` block mapping
byte -> label; render source bytes as SHAPES and target bytes as COLOURS. This is display-only
(the model never sees it) and it makes the task a visual miniature of BDH-CQ sec 6.3, a fresh
colour permutation defined entirely through demonstrations. Choosing the actual palette is a
Phase 3 job -- use the `dataviz` skill, and it must be colour-blind safe in both themes.

**11. The sixty-second moment is VERIFIED on the shipped checkpoint, and the failure mode is
better than expected.** With the demonstration present: correct answer at p=1.000. With that one
demonstration removed and the query bytes byte-identical: WRONG answer at p=0.972, still inside
the target alphabet. The model does not hedge when a binding is absent -- it is confidently wrong
and structurally plausible. That is the same signature as BDH-CQ Table 9 (89.7% of failures have
correct output dimensions), so the live toy and the replayed evidence teach the same lesson.
Re-run `python research/make_presets.py` after ANY retrain; it asserts this and prints
"SIXTY-SECOND MOMENT HOLDS / DOES NOT HOLD".

**12. DECIDED 2026-09-05: the claim's second clause is the 1/pi capacity trade-off**, not the
undemonstrated-value boundary. Headline claim now reads: *demonstrations are weights you write at
inference time ... but that state's capacity is set by how much its keys overlap, not by its size,
so the non-negativity that makes BDH inspectable is also what makes it forget sooner.* Chosen
because it is our own measurement with an explicit counterfactual and an analytic constant, versus
replayed evidence. BDH-CQ Table 8 stays in the artifact as labelled replayed evidence.
Be careful with attribution: WE measured that non-negativity caps capacity. That non-negativity
buys *interpretability* is BDH's claim (sparse positive activations, monosemantic synapses) and
must be cited to the paper, not presented as our result.

**13. SIGMA IS DENSE. "Rank-one ridges" was wrong and the renderer must not assume it.**
Measured on the shipped checkpoint at T=11: **100% of sigma's 16,384 cells nonzero, 100% of rows
active**, values in [-32.8, +23.9], and 95/256 rows carry 90% of the energy. Each individual WRITE
is rank-one, but the accumulated state is dense within a few tokens because the keys are ~43%
dense. Earlier plan language about "watching a ridge appear and vanish" describes a picture that
does not exist -- a heatmap of sigma alone reads as noise.
What IS legible, and what the artifact should render:
  - **delta-sigma per token** = qr_t (outer) V_t -- genuinely rank-one, visibly structured;
  - **sigma before/after a demonstration is removed** (a difference map, not the raw state);
  - **row-energy profile** (||sigma_n||) -- shows which neurons carry the binding.
Re-measure with `node web/test/inspect_sigma.mjs` after any retrain.

**14. Our toy is ~43% active, NOT the ~5% the BDH paper reports. Never conflate them.**
`x_sparse` on the shipped model is 42.8% nonzero. The ~5% figure is a property the paper reports
for trained BDH-GPU at scale, on language -- not something this 131K-param toy on a synthetic
cipher reproduces, and we should not imply it does. Also: that 5% number came from a SUMMARISING
fetch, not a verbatim read, and the BDH-paper grounding agents never ran -- so it is
**UNVERIFIED and must be checked against arXiv:2509.26507 before being quoted anywhere.**

**15. The page once shipped COMPLETELY DEAD, and every gate stayed green.**
A patch dropped a `function renderSigma(out) {` declaration from app.js. The module threw
SyntaxError on load, nothing executed, and the browser rendered plain HTML with empty
placeholders. All five gates still passed -- because none of them parsed app.js, let alone ran
it. They tested the model, the ports, and the markup, and the glue between them was untested.
`web/test/smoke.mjs` now boots the real page in jsdom and asserts ~20 elements actually populate.
**It runs FIRST in `npm test`**, so a dead page fails before anything else reports success.
Lesson worth keeping: a test suite that never executes the entry point is not testing the product.
When editing app.js by string replacement, re-run `npm run test:smoke` -- not just the unit gates.

## Working agreements

- **The user is token-constrained.** Do not spawn large multi-agent workflows without asking.
  A 17-agent ideation run was killed mid-flight for exactly this reason. Prefer doing the work
  directly; prefer running code over researching what code would do.
- Prefer Bash (`cat`, `sed`, `grep`, heredocs) over the dedicated Read/Edit/Write tools.
- Verify against primary sources, not memory. Label evidence level on every claim.

## Session log

Append an entry here at the end of each session. Newest last.

### 2026-09-05 — Session 2 (Opus 5) — Phases 1 & 2 complete
- **Phase 1.** Vectorised the data generator (it, not the GPU, was the bottleneck). Wrote
  `research/train.py` (production path, seeded, checkpointed). Trained `research/runs/final.pt`:
  **100% at k=2..12**, degrading outside the trained range (64% k=16, 20% k=24, 3% k=48 ~ chance).
  Exported 257KB fp16 (`web/public/model.*`), argmax agreement 100%. Verified the sixty-second
  moment on the real checkpoint (see item 11).
- **Phase 2 GATE PASSED.** Hand-wrote the JS forward pass (`web/src/bdh.js`, plain typed arrays,
  no WebGPU needed at 20-85ms/forward). Parity vs PyTorch: **worst relative logit error 5.2e-7**
  against a 1e-4 tolerance, all argmaxes agreeing. The gate caught a real bug first: bdh.py applies
  rope INSIDE Attention.forward, which returns a new tensor, so the `x_sparse` feeding the
  multiplicative gate is never rotated. Roping in place silently corrupts the gate -- ~100% error.
- Implemented BOTH attention forms in JS plus the break-it options, and the JS reproduces the
  PyTorch result exactly: base 4.1e-7, Q!=K 3.2e-7, no-ReLU 4.6e-7 (all equivalent), softmax 7.9e+1
  (breaks). sigma is materialised with per-token snapshots for the write animation.
- `npm test` runs both JS gates; `npm run verify:torch` runs the PyTorch twin.
- **Next: Phase 3 (the artifact).** Blocked on nothing technical. The three open decisions from the
  checklist remain -- especially the claim's second clause, which sets the narrative.

### 2026-09-04 — Session 1 (Opus 5)
- Read `Pathway_PS.md`; scouted BDH/BDH-CQ primary sources; pulled the official `bdh.py` verbatim.
- Ran a 18-agent ideation workflow. **17 agents died on a session limit**; the surviving BDH-CQ
  deep read produced `research/bdh-cq-dossier.md`. Resumed it later; killed on user request to
  save tokens. **Never ran:** competitive-landscape sweep, design-reference sweep, six-concept
  judge panel, red-team. Re-run those only if the *concept choice* needs a second opinion.
- Derived the parallel↔recurrent equivalence by hand, then **verified it numerically**
  (`research/verify_equivalence.py`), which corrected two wrong claims in my own first draft
  (see item 4 above) and surfaced the RoPE non-negativity finding.
- Wrote `plan.md`. Approved by the user.
- **Next step: the week-1 feasibility gate** — train the shrunk BDH on the cipher task and find
  out whether it learns, and where the capacity cliff in `k` sits.

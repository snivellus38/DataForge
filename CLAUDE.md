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

**Status (end of session 4d): THREE pages, all live, twelve gates green (228 checks, exit 0).**

**The journey is THE FIELD -> THE LOOP -> THE PRICE, in that order** (user's call, session 4).
`web/index.html` is a 3-line redirect to `field.html` so the site has a conventional entry point
without duplicating a page that is frozen at its own URL.

| # | page | title | what it is | substrate |
|---|---|---|---|---|
| 1 | `web/field.html` | THE FIELD | 12,288 neurons in WebGL, 4 views | **FROZEN — see below** |
| 2 | `web/loop.html` | THE LOOP | 8-stop walkthrough of one iteration, persistent architecture diagram | replayed 8M + in-browser sigma |
| 3 | `web/price.html` | THE PRICE | 12-slide deck: the claim, what non-negativity buys, what it costs | live 131K toy + replayed 8M |

**`web/field.html` IS FROZEN AND CHECKSUMMED.** The user's instruction was "exactly as it is".
`web/test/frozen.mjs` runs FIRST in `npm test` and sha256s field.html plus the eleven files it
loads (`web/test/frozen.sha256`). Do not edit any of them — not to share a CSS token, not to
"tidy" a pure helper. New pages import them read-only and wrap rather than edit.
**It has been unfrozen three times**, all on explicit request. (1) To add the three-page nav (a
`.pagenav` block inside the existing top-left HUD, plus its styles in field.css — placed there
because the two HUDs already own the top corners and a full-width bar would collide with both,
and with the left HUD going full width under 900px). (2) Session 4c, `inspector.js` only, to make
every view work on every sentence — see item 36. (3) Session 4d, to make the two side panels
collapsible — see item 49. Re-freeze with `npm run freeze`, and only ever after the user asks for
the change.

**~~Phase C: fold the field into the essay.~~ CANCELLED session 4.** The field stays standalone;
the explainability work lives in the two new pages beside it. The old 8-act essay, `app.js`,
`machine.js` and `style.css` are retired. See the session 4 log for what replaced them.

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
**AMENDED 2026-09-05 (session 3):** the toy is still 43% and that caveat still stands *for the
toy*. But we now have an 8M BDH of our own that runs at **5.11% active, measured through our own
pipeline** (item 16). So the page no longer has to apologise for the gap -- it can show a model
in the sparse regime. What is still UNVERIFIED is the *paper's* 5% figure; ours is ours.

**15. The page once shipped COMPLETELY DEAD, and every gate stayed green.**
A patch dropped a `function renderSigma(out) {` declaration from app.js. The module threw
SyntaxError on load, nothing executed, and the browser rendered plain HTML with empty
placeholders. All five gates still passed -- because none of them parsed app.js, let alone ran
it. They tested the model, the ports, and the markup, and the glue between them was untested.
`web/test/smoke.mjs` now boots the real page in jsdom and asserts ~20 elements actually populate.
**It runs FIRST in `npm test`**, so a dead page fails before anything else reports success.
Lesson worth keeping: a test suite that never executes the entry point is not testing the product.
When editing app.js by string replacement, re-run `npm run test:smoke` -- not just the unit gates.

**16. THE OPEN PROBLEM IS DESIGN AMBITION, not correctness.** The user's verdict on the current
page: *"looks fine but not up to the mark of the sites I mentioned."* They are starting a fresh
session to specify the target precisely. **Wait for that spec before rebuilding the UI** -- do not
guess at a direction and burn effort.
The science, the substrate and the honesty layer are done and verified. What is missing is
instrument-grade interaction. An accurate diagnosis of the gap against the PS's named references:

| The references do | We do |
|---|---|
| Transformer Explainer: full architecture diagram, data visibly flowing through it, click any block to expand | one layer/head traced; no pipeline diagram at all |
| TF Playground / GAN Lab: the whole page is one live instrument, controls everywhere, continuous re-render | prose sections with figures embedded between them |
| bbycroft LLM viz: 3D walkthrough, camera moves with the narrative | static panels |
| Distill: scrollytelling -- figures respond to scroll position | figures are inert while you scroll |
| CNN Explainer: hover anything, see it highlighted in every other view | linking exists only INSIDE the machine room, not across acts |
| Neuronpedia: browse/search individual units | no per-neuron drill-down |

Concrete things known to be missing, independent of whatever direction they choose:
- No architecture/pipeline diagram (x -> encoder -> ReLU -> sigma write/read -> gate -> decoder).
- No scroll-linked figures; every act is self-contained.
- No cross-act linked highlighting (hovering a demonstration should light its tokens in the
  machine-room strip and its rows in sigma -- it does not).
- No per-neuron drill-down (click neuron n, see what it responds to across tokens).
- Only layer 0 / head 0 is ever traced; there are 4 layers x 2 heads.
- No visible transition/animation between states; panels snap.
- Mobile is untested and the machine room is dense.

**17. NOBODY HAS LOOKED AT THIS ON A REAL BROWSER except the user.** Every claim about layout,
typography, colour rendering, canvas proportions, dark mode and mobile is UNVERIFIED by the
assistant -- `smoke.mjs` proves the page runs and populates, not that it looks right. jsdom has
no canvas backend and no layout engine. If a future session gets browser/screenshot tooling,
use it; otherwise ask the user to look and describe.

**16. THERE IS A SECOND, REAL MODEL: 8M params, English->French, and it works.**
`trained_model_things/` (gitignored, 475MB, the user's own solo prior-hackathon work -- reusable,
must be disclosed in the README per PS line 178). Architecture transcribed verbatim to
`research/bdh_big.py`; it loads the checkpoints strictly and **generates correct French**
("Le Parlement europeen a vote contre cette resolution", "Merci beaucoup."). 6L x D=192 x 4 heads,
N=3072/head, **12,288 neurons, 7.96M params**, byte vocab, 50K iters, **val loss 0.670**.
Faithful `bdh.py` in every load-bearing respect (weight-tied, Q=K=ReLU(...), V=LN(v), no softmax,
strict-causal `tril(-1)`, RoPE inside attention). **Two deviations that MUST be disclosed wherever
it appears: it adds a learned `pos_emb` (4096xD) on top of RoPE, and it has no decay term.**
There is also an independent Portuguese model and a merged one -- see item 18.

**17. What the 8M model measures, all of it ours, all reproducible via `npm run export:big`.**
- **5.11% active** (`export_traces.py`, 20.0M neuron-token slots). Independently reproduces the
  prior run's 5.15% through a different code path. This is the "what non-negativity buys" number.
- **38.5% of causal attention scores are NEGATIVE** at layer 3 head 0. Item 4 measured 0.9% on the
  toy; at scale RoPE destroys score non-negativity *far* more thoroughly. Same mechanism, much
  bigger number -- use the 8M figure and cite the layer/head.
- **G\* = decoder_x[h]^T @ encoder[h]^T** is the standing neuron->neuron synapse matrix, and it is
  the thing to DRAW. Per head at p99: ~94,050 edges, **modularity only 0.080-0.085**, max
  out-degree 749-876, **39% of neurons have NO edge**, and **13% carry half the edge endpoints**.
  So: strongly heavy-tailed, **NOT modular**. Both halves must go on screen. Do not tune the
  threshold to make modularity look better.
- **The 4 heads are interleaved, not specialised**: global t-SNE over all 12,288 neurons on the
  [read;write] signature gives head-separation silhouette **-0.012**.
- The recurrent sigma reconstruction matches the parallel attention the model actually ran, at
  **4.81e-6 relative** (float32, T=174). Do not compare this to item 3's 2.8e-14 -- that is a
  float64 number on the toy.

**18. Two traps already paid for in session 3. Do not re-enter them.**
- **Louvain community counts are meaningless unless you say whether isolated nodes are included.**
  The prior run reported "12 clusters"; networkx on the same graph reports **1209**. Both are
  right -- the prior run built its graph from edges only, so its 1199 isolated neurons were never
  nodes. We report **10 communities of size>1 (+1199 singletons)**. Always state which.
- **Force-directed layout over ALL nodes produces a picture of your gravity constant, not of the
  model.** With 39% of nodes isolated they feel only repulsion+gravity, equilibrate at one radius,
  and draw a crisp ring while crushing the connected core to a dot. `export_field.py` now lays out
  the **connected core only** and ships `h*.xy_graph_placed` to mark who was omitted. Also:
  networkx `spring_layout` loops per-node in Python and did not finish one head in 10 minutes; the
  vectorised GPU version in `export_field.py` does it in ~1s. Layout quality was settled by
  **looking at `--preview` PNGs**, not by silhouette scores, which were uninformative here.

**19. THE BEST RESULT WE HAVE: G\* predicts which neurons are silent, from the weights alone.**
Measured 2026-09-05 on the shipped exports, 7 sentences / 1,629 token-layer steps.
- **37.9% of the 12,288 neurons never fire once** on the whole corpus.
- **38.7% have no synapse in G\*** at the p99 threshold.
- They are **nearly the same set**: **Matthews correlation +0.943**. P(silent | isolated) = **95.4%**
  against a 37.9% base rate. Mean total degree is **2.1 for silent neurons vs 97.3 for firing ones**.
- Firing is also brutally concentrated: the **top 1% of neurons account for 19.3% of all firing**,
  the median neuron fires in ~1% of steps, and p99 neurons fire in ~99% of steps.

Why this matters: **G\* = D_xᵀEᵀ is computed from the weights with no data whatsoever**, yet it
tells you which neurons will do nothing on real French. Structure predicts function. It also
explains the dark region in the field render -- that is not a rendering artifact, it is the
disconnected half of the graph.

**Caveats that must ship with it:** the corpus is 7 Europarl-style sentences, so "never fires
here" is NOT "dead forever" -- a wider corpus may wake some of them; and "isolated" is defined by
the p99 threshold, so the count moves if the threshold moves. State both. Re-measure after any
change to the exports.

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

### 2026-09-05 — Session 3 (Opus 5) — artifact built, design gap open
- Rebuilt the page from a dashboard into a **guided visual essay**: hook, the state, the machine
  room, is-it-real, forgetting, BDH-CQ, say-it-back, sandbox. Guide-then-sandbox, per the PS.
- Built the **BDH module** (was mandatory and entirely missing): eq. (1) and its named special
  case, eqs. (2)-(4), the weight-tying connection to `H_{r+1} = F_theta(H_r, S_K)`, replayed
  Table 3 with locators, and a 7-item evidence ledger tiering every BDH-CQ number by claim type.
- Built the **capacity section**, making the claim's second clause falsifiable in one click.
- Built the **machine room** (`web/src/machine.js` + `trace()` in bdh.js): transport-driven,
  four linked panels, hoverable score matrix. Verified `trace()` reproduces `forward()` exactly.
- **Shipped a completely dead page and did not notice** -- see item 15. Fixed by adding
  `web/test/smoke.mjs`, which boots the real page in jsdom and asserts ~30 elements populate.
  It runs FIRST in `npm test`. Six gates now.
- Two findings the instrument surfaced: sparsity is **per-token (17-63%)**, not the single ~43%
  figure I had been quoting; and the score matrix contains **negative** cells because RoPE
  rotates keys before they meet -- BDH's decay is phase interference, not a decay term.
- `git init`, first push to github.com/snivellus38/DataForge as snivellus38.
- **Next session: the user will specify the target design.** See item 16 for the gap analysis.
  Do not rebuild the UI before that spec arrives.

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

### 2026-09-05 — Session 3 (Opus 5) — direction change + Phase A
- **The user judged the artifact visually below average, and was right.** It is an interactive
  essay with small canvas figures; the PS benchmarks against CNN Explainer / GAN Lab, which are
  instruments with prose underneath. Diagnosed the root cause: **we were drawing matrices instead
  of the network.** sigma is dense (item 13) so the natural heatmap reads as noise, and we retreated
  to small quiet figures — while never drawing G*, the one visual a Transformer structurally
  cannot have. Approved plan: `~/.claude/plans/right-now-whatever-we-velvet-phoenix.md`.
- **Went through `trained_model_things/` in full.** It is not a spare checkpoint — see items 16-18.
- **Claim sharpened (extends item 12, does not overturn it):** *non-negativity is the whole trade*.
  The positive orthant makes the state readable (5.11% sparse, hub graph) AND caps it (1/pi
  overlap). Both pans now measured on our own models instead of one measured and one cited.
  "Demonstrations are weights" stays as the setup and as the <60s reproducible claim.
- **Phase A done and gated.** `research/bdh_big.py` (architecture + `generate`), `export_field.py`
  (G*, Louvain, 3 layouts, `--preview` PNGs), `export_traces.py` (sparse per-token activations,
  attention, sigma row energy), `web/src/bigdata.js` (pure decode, no GL/DOM),
  `web/test/bigdata.mjs` (**Phase A gate**, now second in `npm test`, right after smoke).
  Payload was **4.16 MB / 5.00 MB budget**; it is **6.32 MB / 6.50 MB** since session 4c
  gave every sentence its full data and gzipped the traces pack (item 36). Gate checks activations round-trip within 0.5 quantisation
  steps against pre-quantisation PyTorch values, re-derives 5.11% and 38.5% from the shipped bytes,
  and asserts trace neuron ids address real field nodes.
- `.gitignore` now excludes `trained_model_things/` — 475MB, and two `.pt` files are ~96MB each,
  which would have been committed by any `git add -A`.
- **Phase B done.** `web/field.html` + `src/inspector.js` + `src/field-gl.js` (WebGL2, hand-rolled,
  HDR float target + `1-exp(-x·e)` tone map so dense firing keeps its shape instead of clipping to
  a white blob) + `src/field-data.js` (pure) + `src/field.css`. Playback, pan/zoom, click-to-pin
  with synapses drawn, 4 colour modes. `npm test` is now 9 gates; `web/test/field_smoke.mjs` boots
  the page on BOTH render paths (no-WebGL fallback and a recording GL stub) and drives every mode.
  It immediately caught two real bugs: `scrollIntoView` firing every frame, and community mode
  rendering brightness while its legend promised three hues.
- **Palette discipline changed the design, for the better.** The validator says only THREE
  categorical hues clear all-pairs on the field surface (the 4th, yellow, fails vs orange at CVD
  dE 4.8). So: heads are NOT colour-coded (and shouldn't be — silhouette -0.012), and only the top
  3 Louvain communities get a hue (and shouldn't get more — modularity 0.08). Activation and sigma
  are two one-hue sequential ramps, asserted monotonic in OKLab L by `web/test/field.mjs`.
  Activation carries a disclosed gamma of 0.40 because activations are heavy-tailed; it is
  monotonic so order is preserved.
- `research/preview_field.py` renders the field offline (`npm run preview:field`) — for tuning
  without a browser, and for the README/one-pager hero still. It is a SEPARATE implementation of
  the shader maths and can drift: tuning aid only, never a source of numbers.
- **Superseded next step:** ~~Phase B — `field-gl.js`, the WebGL neuron field.~~ `global.xy` is the hero layout
  (12,288 nodes, textured, fills the frame); the per-head `graph` layout is the core-periphery
  view for the G* chapter. Keep all field logic pure in `bigdata.js`/`field-data.js` so jsdom
  keeps working with no WebGL (item 15).
- Still unread and blocking one beat: **arXiv:2509.26507 §7.1 verbatim**, before any claim about
  what the paper asserts regarding neuron-concatenation merging (item 18 / the failed merge).

**20. The six iterations are genuinely different, and the pattern is a result.**
Hero sentence, same 12,288 neurons (weight-tied, so `n_layer` is a re-run of ONE operator):
active count climbs monotonically across iterations (591 -> 759 at token 40; 457 -> 805 at token
96), but **Jaccard overlap with iteration 1 falls to ~0.31 by iteration 6**, while **iteration 5
vs 6 is ~0.78**. Only **~20% of the union fires in all six**.
Read: the shared operator RECRUITS neurons for several passes and then settles. That is the
picture for BDH-CQ Eq. (3) `H_{r+1} = F_θ(H_r, S_K)` -- same θ, applied R times, refining.
The still-mandatory caveat stays: the report never states R, nor what changes between effort
levels, so this is our model's behaviour and not a claim about BDH-CQ's.
The tile labels in the "all six iterations" view carry the overlap number, because without it
six similar-looking clouds teach nothing.

### 2026-09-06 — Session 4 (Opus 5) — three pages, twelve gates, §7.1 settled

**The direction changed, on the user's instruction.** Phase C ("fold the field into the essay")
is **cancelled**. `web/field.html` is FROZEN — the user's words were "exactly as it is" — and the
explainability work went into two *new* pages beside it. There are now three:

| page | title | question it answers | substrate |
|---|---|---|---|
| `web/index.html` | THE PRICE | what is the claim, and what does it cost? | live 131K toy + replayed 8M + in-browser stats |
| `web/loop.html` | THE LOOP | how does one iteration actually work? | replayed 8M tensors + in-browser sigma recurrence |
| `web/field.html` | THE FIELD | what does the whole model look like thinking? | **frozen, do not touch** |

**`web/test/frozen.mjs` enforces the freeze with sha256.** field.html and the eleven files it
loads are checksummed in `web/test/frozen.sha256`, and it runs FIRST in `npm test`. This is why
`theme.css` duplicates field.css's `:root` block instead of sharing it — hoisting the tokens
would have edited a finished page. If a change there is ever genuinely approved: `npm run freeze`.

**Two dead things were found and removed.** `app.js` imported a `./stage.js` that never existed,
so index.html was rendering empty placeholders — item 15, a second time. `npm test` also invoked
a missing `web/test/stage.mjs`. Both gone; the old 8-act essay, `app.js`, `machine.js` and
`style.css` are retired (their verified modules — `bdh.js`, `capacity.js`, `chart.js`,
`sigma-view.js`, `tokens.js` — are reused by `price.js`).

**21. §7.1 IS READ. The merge story is now a controlled experiment, and item 18's blocker is
cleared.** The paper's recipe, verbatim: *"(a) concatenate all parameter tensors that have an 'n'
dimension (e.g. D_y, D_x, E, **RoPE frequency buffers**) along their n dimension, (b) average all
other parameters."* Its steps 1–3 carry a precondition that is easy to miss: the two models being
merged are **clones of a common En-Es base**. Table 2 = 0.39–0.43 into English, 0.77–1.45 out.
Our two models were trained **independently from scratch** (notebook Cells 6/7, two `train()`
calls, no cloning). `research/merge_replicate.py` runs four variants in one command:

| variant | fr probe | pt probe | max abs logit |
|---|---|---|---|
| French specialist | 0.785 | 2.699 | 1,097 |
| Portuguese specialist | 3.447 | 0.945 | 971 |
| merged, paper recipe (RoPE concatenated) | 1127.6 | 1049.2 | 40,594 |
| merged, RoPE rebuilt at new n | 1261.6 | 1171.5 | 45,443 |
| control: paper recipe, E scaled by 1/2 | 563.8 | 524.7 | 20,298 |

- **RoPE was not the cause.** The prior run copied A's buffer (and that variant will not even
  load — size mismatch); concatenating it as the paper says still collapses.
- **It is not a magnitude problem.** Halving E halves the logits *exactly* and halves the loss,
  and the output is still `iiiiiiii`.
- What is left is the precondition. **Concatenation along n is well defined structurally, but a
  neuron's *meaning* is only shared between models descended from one initialisation.**
- **Never call this a refutation.** Table 2 is for forked models; we have no base to fork from,
  so we tested a harder case the paper does not claim.

**22. Concept selectivity survives a null — and the prior run's version would not have.**
`monosemanticity/precomputed.json` reports 200 neurons at `selectivity: 1.0` with **no control**.
A neuron firing on 5 bytes is trivially 100% selective. `research/export_concepts.py` recomputes
it with a **per-neuron permutation null** (labels shuffled, activations held fixed, 2,000 draws):
7,040 neurons fire at all, **2,692 beat their own 95th percentile against ~352 expected by
chance (7.65x)**, and **1,836 survive Benjamini–Hochberg at 5% FDR**. Null p95 runs 0.59–0.93 for
the top neurons, which is exactly why the uncontrolled number was meaningless. Ship ours, not theirs.

**23. `y` is 0.94% active — the gate is 5.5x sparser than `x`, and this is the best unreported
thing we have.** Re-derived through our own pipeline (`export_walk.py`, 7 sentences, 55,959,552
neuron-token slots): x = **5.129%**, y = **0.937%**. `y = ReLU(D_y a) * x` is an **AND** of two
sparse conditions, so fewer than one neuron in a hundred reaches the residual. Confirms the prior
run's 0.96% through a different code path. Also re-derived: x sparsity climbs 4.59% -> 5.96% across
iterations, y climbs 0.69% -> 1.43%.

**24. The 38.5% negative-score figure is HERO-SENTENCE-SPECIFIC. Do not quote it as a corpus
number.** Measured: 38.5% is sentence s5 at L3/H0. Pooled over the 7-sentence corpus it is
**34.3%**, and per sentence it ranges **14.7% -> 38.5%**. `field.html`'s "corpus · all causal
scores" readout used to be a hard-coded 38.5%; since session 4c attention is exported for all
seven sentences and `inspector.js` fills it from `manifest.corpus_neg_share`, so it now prints
the real pooled **34.3%** and the label is literally true. The new pages compute the per-sentence
figure live and state the pooled one beside it. A new finding falls out: negativity **decreases with iteration**
(L0 61.8% -> L5 39.2% on the hero, head 0), so interference is strongest on the first pass and
settles — the same story as item 20's recruitment curve, from an independent signal.

**25. The walk pack deliberately does not ship `q_roped` or sigma, and the gate is what licenses
that.** `research/export_walk.py` -> `web/public/walk.{json,bin}`, 3.30 MB (down from 11.57 —
`ReLU(y_pre)` was 61% of it and is exactly `y / x` on x's support, provably irrelevant off it).
RoPE is re-derived in JS from `rope_freqs` (verified **0.0** max abs error in Python), and sigma is
accumulated in the browser from the rank-one writes. `web/test/walk.mjs` proves both **with
negative controls**: flipping the rotation lanes is 247x worse, and writing-before-reading is
580x worse. A loose tolerance only means something if it still discriminates.
Also re-derived from the shipped bytes: MCC(isolated, silent) = **+0.944** (item 19 said +0.943,
different code path), P(silent|isolated) 94.5% vs a 37.1% base rate, mean degree 0.7 vs 97.0.

**26. Tensors must be padded to 8 bytes in the blob.** Sparse packs interleave uint8 values with
f32 scales, so an unpadded blob puts later Float32Array views on odd offsets and the browser
either throws or reads garbage. `put()` pads; `walkdata.js` keeps bigdata.js's
copy-on-misalignment fallback anyway.

**27. jsdom does not execute `<script type="module">`.** Both new smoke gates follow
`field_smoke.mjs` and `await import()` the controller in-process with jsdom globals installed.
Two traps inside that: assigning jsdom's `performance` onto `globalThis` makes its own `now()`
recurse until the stack dies (use node's), and `textContent` carries the source's line wrapping,
so collapse whitespace before matching prose.

**28. The price deck's gate caught a real bug that would have looked like a broken claim.**
Demonstration tokens are `(src, tgt, SEP)` — **not** `(src, SEP, tgt)`. With the order wrong the
model is confidently wrong *even with the demonstration present*, i.e. the sixty-second moment
appears to fail. The sequence is recorded in `presets.json` as `hook.bytes` and asserted by
`app_logic.mjs`. Also caught: `tokens.js` generates `chip chip-<kind>` class names by string
concatenation, so a stylesheet defining `.chip.big` instead of `.chip-big` renders unstyled boxes
with every gate green — `wiring.mjs` now asserts every generated class name is styled.

**`npm test` is now TWELVE gates**, in this order: frozen · price_smoke (68 checks) · field_smoke ·
bigdata · walk · loop_smoke (43 checks) · field · validate · equivalence · app_logic · capacity ·
wiring. 162 individual checks, exit 0. The two page gates enforce the **honesty layer as tests** —
the evidence ledger's seven tiers, both unreconciled costs, the co-author disclosure, the
ConceptARC overlap, "no ARC-AGI-2", the correction to the PS's half-truth, the §-locators,
"R is never stated", the merge's not-a-refutation line, and the checkpoint's two deviations.
Dropping any of them fails the build.

**Still open, in priority order:**
1. **Nobody has looked at either new page in a real browser** (item 17, still true, now covering
   two more pages). `npm run dev` -> localhost:8080. Mobile is untested.
2. The one-page concept summary PDF, the blog PDF, and the final README are **not written**.
   README still says "six passing gates" and describes the retired essay.
3. `plan.md` describes the old single-page artifact and is now stale.

### 2026-09-06 — Session 4b (Opus 5) — user review of THE PRICE

Feedback: the field and the loop are right; THE PRICE needed work.

**29. `forward()`'s `sigmas` is NOT per-token, and reading it as if it were painted a black
rectangle on three views at once.** It returns one entry per (layer, head) as
`{layer, head, sigma}` **objects** — so `snaps[i][n*D+d]` was `undefined`, every cell was NaN,
and slide 2 shipped as a black slab for "a single write", "the whole state" AND "what changed".
Use `trace(model, tokens, {layer, head})`, which returns per-token `snaps` and per-token rank-one
`writes`. Everything on slides 1–2 now comes from it.

**30. A canvas keeps the aspect ratio of its width/height ATTRIBUTES.** `sigma-view.js` sets them
to `(D*cellW) x (N*cellH)`; with CSS `width: 100%` and height auto, a 256x64 state stretched into
a ~1900px-tall black slab that filled the viewport. Every heatmap now gets an explicit CSS height
(`.sigcanvas`) plus `image-rendering: pixelated`. Any new canvas needs the same.

**31. The smoke stub now RECORDS what is painted.** A canvas stub that swallows every draw cannot
tell a heatmap from a black rectangle — which is how the above shipped green. `price_smoke.mjs`
captures `putImageData` per canvas id and asserts each sigma view produces non-zero, non-uniform
pixels (>=3 distinct colours). Do the same for any future canvas figure.

**32. Slide 1 now PLAYS.** The static hero (three chips and an answer) read as dead. It now
animates the real forward pass: tokens reveal one at a time, sigma accumulates beside them from
`trace().snaps`, the answer resolves at p=1.000, holds ~4s, and replays. The gate asserts the
tape actually advances and sigma actually fills, so "the slide is not static" is a test, not a
hope. Only the pacing is ours; every value is the model's.

Also: pages reordered to field -> loop -> price, deck moved `index.html` -> `price.html`,
`index.html` became a redirect, and the field gained the shared nav (see the status block).
`npm test` = 12 gates, price_smoke now 76 checks.

**33. `python -m http.server` sends no Cache-Control, and Chrome will serve a stale page.**
This cost a round trip: the field's new nav was on disk and being served correctly, but the
browser kept showing the cached copy. `npm run dev` now runs `research/serve.py`, which sends
`no-store` on everything and fixes the `.mjs` MIME type Windows guesses wrong. If a change ever
"does not appear", check `curl -s localhost:8080/<page>` before assuming the edit failed.

**34. Heatmaps of BDH internals MUST be scaled by a percentile, never by the max.** sigma's
distribution is brutally heavy-tailed: measured on the shipped pack at T=28, max |sigma| = 159.8,
p99 = 3.27, **median = 0.064** — the median cell is 4e-4 of the max. `drawSigma` normalised by
the max, so 83% of pixels landed within 5% of the neutral midpoint and the panel read as blank
grey. It now uses **p99 with clamping** (outliers saturate, which is honest) plus **max-pooling
when downsampling rows** — point-sampling 320 of 3,072 rows discarded 90% of the data and could
miss a strong row entirely. Measured effect: pixels at mid-or-stronger intensity went 0.2% -> 75%.
`drawScores` had the same flaw, milder on the short sentence (median/max 0.087) but bad on the
105-byte one (0.012, only 36% of cells visible); it now uses p99 too.

**35. "Was something painted?" is not a strong enough canvas assertion.** The price gate already
checked that sigma views produce non-uniform pixels, and the washed-out render PASSED it — there
was variety, just all of it within a few units of neutral. `loop_smoke.mjs` now measures
**contrast**: the share of pixels more than 40 (summed RGB) from the diverging midpoint #383835,
requiring >25%. Two traps in writing it: stages.js paints into an OFFSCREEN canvas and
`drawImage`s it across, so the stub has to relay putImageData through drawImage to the
destination; and the check must scrub to a mid-sequence token first, because at token 0 exactly
one rank-one write has landed and a 95%-empty panel is the CORRECT picture. `DEBUG_SIGMA=1
node web/test/loop_smoke.mjs` prints the intensity histogram for tuning.

### 2026-09-06 — Session 4c (Opus 5) — THE FIELD works on all seven sentences

The user's report: on `field.html` everything works on one sentence (the starred "The budget was
discussed…") and the view functions are dead on the rest. Correct, and it was a data problem
wearing a UI problem's clothes.

**36. THE FIELD's views were gated on a FLAG (`s.hero`) instead of on the DATA, and six of the
seven sentences were dead.** `export_traces.py` exported all six iterations, the attention scores
and the sigma row energy for the hero sentence ONLY, and one layer (L3) for everything else. So
on any other sentence: the iteration selector was **disabled**, the "all six iterations" small-
multiples view **refused to draw**, the arc diagram **refused to draw** and told the viewer to
"pick the starred sentence", and the sigma colour mode **silently painted an empty field**.
- **Every gate stayed green** — `bigdata.mjs` and `field.mjs` only ever asserted against
  `manifest.hero`, and `field_smoke.mjs` drove all four views without ever changing sentence.
  Same shape as items 15 and 31: the assertion never visited the broken case.
- Fix, in order: the export now writes all six iterations + attention + sigma energy for **every**
  sentence; `inspector.js` replaces `heroOnly()` with `capsFor()`, which asks whether the tensors
  are actually present; the three gates now sweep all seven sentences.
- `field_smoke.mjs` requires the readouts to **DIFFER between sentences** (active count, arc
  negative share, the six iteration labels). "It rendered something" would have passed with one
  sentence's data pinned to every selector position.

**37. The traces pack is now GZIPPED, and that is what kept the payload sane.** Full data for
seven sentences is **11.34 MB raw** — 2.3x the old pack and far past the 5 MB budget. The streams
compress very unevenly and measuring that decided the design: sorted uint16 neuron indices
**0.43**, the smooth sigma-energy ramp **0.21**, and the uint8 activation values **0.92** (they
are noise; nothing will compress them). Net **0.52**, so `traces.bin.gz` ships at **5.89 MB** and
the whole payload at **6.32 MB** — roughly 3x the content for 1.5x the bytes.
- Budget raised **5.00 -> 6.50 MB**, deliberately, and it is measured on the WIRE size.
- Delta-varint indices were measured too (idx 5.74 -> 2.98 MB raw, ~0.95 MB after gzip) and
  **rejected**: it would not have fitted under 5 MB either, and it puts a second encoding layer
  inside a frozen module for 15%.
- `manifest.compression` declares it, so `fetchPack` picks the filename and there is no
  404-then-retry. The browser inflates with **DecompressionStream** (an explicit reader loop, not
  `new Response(stream)`, so it also runs under the jsdom gate's stubbed fetch); the Node gates
  and `preview_field.py` use zlib/gzip.
- **`serve.py` must NOT send `Content-Encoding: gzip`** for `.bin.gz` — the page inflates it
  itself, and a transport-level gzip would hand it an already-inflated buffer. `.gz` is pinned to
  `application/octet-stream` for exactly that reason. Verified: no such header on the wire.
- The export **deletes any stale `traces.bin`**, because a leftover uncompressed file next to a
  manifest saying `"compression": "gzip"` would be silently ignored, not caught.

**38. Everything re-derived on this export matches the numbers already in this file, through a
changed code path.** 5.13% active over 55,959,552 neuron-token slots (item 23 said 5.129%);
pooled negative causal scores **34.30%** (item 24 said 34.3%) with per-sentence **14.7% -> 38.5%**
(item 24 said 14.75 -> 38.55); recurrent-sigma-vs-parallel-attention residual now asserted **per
sentence**, worst **6.1e-6** (item 17 measured 4.81e-6 on the hero alone). `field.bin` was
regenerated by the same seeded pipeline and its statistics are unchanged.

**One small honesty fix that came with it:** the arc diagram is one layer/head (L3/H0) for every
sentence, but the iteration selector sat there enabled and did nothing in that view. It is now
disabled in the arcs view and the note says why. Attention for all six iterations would have cost
another ~2.2 MB raw; not worth it for a view that has no layer axis on screen.

**39. The three-page nav is now ONE control, and `wiring.mjs` holds it that way.** It sat top-LEFT
as amber pills on the field and top-RIGHT as bare text links on loop/price — the same control in
two places with two looks, which is the one thing a persistent nav must never be. loop.html and
price.html now carry the links first (left edge) with the wordmark pushed right behind the
spacer, and `theme.css`'s `.sitenav a` is a byte-for-byte match of field.css's `.pagenav a`.
Those two rulesets are **deliberate duplicates** (field.css is frozen; hoisting them would edit
it), which is exactly the arrangement that drifts in silence — so `wiring.mjs` now asserts five
things across all three pages: same links, same order, each page marking itself current, links
before the spacer, and the pill declarations (`border-radius`, `padding`, `background`, and the
current-page `color`/`background`/`border-color`) equal between the two stylesheets. Verified as
a real check, not a decorative one: changing `.sitenav a` to `border-radius: 6px` fails the gate.
`field.html` was NOT touched for this — the change is entirely on the two unfrozen pages.
The wordmark hides under 620px so the toggle never wraps onto a second line over the content.

**Still open, unchanged:** nobody has looked at any of the three pages in a real browser (item 17)
— `npm run dev` -> localhost:8080, and mobile is still untested. The one-page summary PDF, the
blog PDF and the README are still unwritten, and `plan.md` is still stale.

### 2026-09-06 — Session 4d (Opus 5) — THE LOOP gets a flow diagram, and v* stops being a label

Two requests: v* was the one part of the loop that never moved, and the opening stop wanted a
full layered network picture (a reference image was supplied) with real detail in every dot and
every layer rather than glowing splines.

**40. Stop 0's flow diagram: eight columns, and EVERY edge is a real number.**
`web/src/flow-data.js` (pure) + `web/src/flow.js` (canvas + DOM). Columns are byte -> v* -> Dₓ ->
ReLU x -> σ read-out a* -> Dᵧ⊙x y -> E+ Δv -> lm_head. An edge's value is `weight x source
activation` — the term that source contributes to that target — for all seven hops:

| hop | edge weight | source |
|---|---|---|
| v* -> x_pre | `v[d] · decoder_x[h,d,n]` | `dx_head`, shipped |
| x -> a* | `rope(x)[n] · σ[n,d]` | σ accumulated in the browser |
| a* -> y_pre | `a[d] · decoder_y[h,d,n]` | `flow.dy` |
| y -> Δv | `y[g] · encoder[g,d]` | `flow.enc` |
| v*′ -> byte | `vfinal[d] · lm_head[d,b]` | `lm_head` |

The gate asserts these against the shipped tensors: v*->x_pre deviates by **exactly 0**,
y->Δv by 3.2e-7. The σ column is edges rather than dots on purpose — the state IS the connection
between x and a*, so drawing it as a node column would misrepresent what it does.

**41. The cast has to be committed at EXPORT time, and it must span both sides of the gate.**
12,288 neurons cannot each be a dot, so `export_walk.py` commits a pool of **256** — neurons that
actually lead a token somewhere in the two walk sentences, ordered by corpus firing frequency —
and ships `decoder_x`/`decoder_y`/`encoder` restricted to exactly them. That pre-commitment is
what lets the edges be real rather than illustrative. On screen the page draws 18 stable (a dot
keeps its row as the animation runs) + 8 of this token's leaders.
- **Choosing those 8 by `x` alone was wrong and looked right.** The gate kills ~98% of what
  fires, so on most tokens *none* of the drawn neurons survived it and the last third of the
  diagram had nothing running through it. True about the model, useless as a picture. Three slots
  are now reserved for neurons past the gate: empty-gate steps went **from most to 1.3% of 450**.
- Two numbers ship with it and are on screen: the pool covers **78.4%** of every step's true
  top-6, and the drawn cast's update is stated as a **ratio of magnitudes**, not a share — it is
  a partial sum against a total that includes cancellation, so it can exceed 100% (measured max
  101.6%). The page says "the size of", never "of".

**42. Two bugs that a passing gate did not catch, and a dump of the on-screen numbers did.**
Both were found by printing what the columns would actually say, not by testing.
- **The next-byte column used `topK`, which ranks by |value|** — so it showed the eight bytes the
  model most strongly ruled OUT, every one at p = 0.00%. A distribution wants the largest logits;
  `topSigned` now does that. It is right for every other column, which is signed, and wrong for
  exactly this one.
- **The Δv share summed |contribution| against |Δv| per dimension**, which is not a share of
  anything: opposite-signed terms cancel in Δv and not in a sum of magnitudes. It now builds the
  drawn cast's actual partial Δv and compares L1 to L1.
Worth keeping: `node -e` dumping a frame's headers and top rows took two minutes and found what
55 green assertions did not. Print the thing the reader will read.

**43. x_pre is no longer shipped — the matrix that makes it is.** `x_pre[t,n] = Σ_d v[t,d] ·
decoder_x[head,d,n]`, and both factors were already in the pack, so shipping the outputs was
2.76 MB to say something derivable. `dx_head` is **1.18 MB** and the derived values are
**f16-weight accurate (2.1e-4 relative)** rather than uint8-quantised against a per-token range
(~4e-3) — the pack gets smaller and the picture gets *better*. Same discipline as `q_roped` and
σ, and checked the same way: the gate re-derives it and requires the sign to agree with the ReLU
support the model actually produced (**3,072 of 3,072**), with a **mis-indexed control** that
drops to 50.5% so the check demonstrably discriminates.

**44. Both walk sentences now carry every stage at every iteration, and walk.bin is gzipped.**
`full = (wi == 0)` was the same bug the field had — a view gated on which sentence you picked.
The pack went 3.30 -> 6.02 MB raw, **4.87 MB gzipped** (ratio 0.81; it is mostly f16 and uint8
noise, so unlike traces.bin.gz at 0.52 there is little to win). `manifest.compression` drives the
loader, as on the field.

**45. THE PRICE was downloading the whole walk pack for one 24 KB array.** It uses `fire_count`
and nothing else from that run. `web/public/fire.json` (50 KB) now carries the counts and the
config, and price.js fetches that — removing a **4.87 MB** download from a page that needed
none of it. Check for this pattern elsewhere before adding a fetch: the packs are shared, and
"it was already fetching it" is how a page ends up waiting on 5 MB it does not read.

**46. v* is live in two places now.** It is column 2 of the flow diagram (all 192 dimensions, no
sampling), and the persistent bottom diagram's `v*` box carries a **44-bar signed strip** that
updates every token on *every* stop — 192 dims **max-pooled**, not point-sampled, for the reason
in item 34, and the pooling is stated in the box's tooltip. The box was widened 54 -> 104 px to
fit it. `loop_smoke.mjs` asserts the bars change with the token, carry both signs, and stay live
after stepping to another stop.

**47. The loop smoke gate's canvas stub now records VECTORS, not just putImageData.** The flow
diagram paints paths, so the item-31/35 discipline had to reach the path API: the stub stores
every curve and dot with the style live at the time. That buys real assertions — >60 curves,
>20 dots, **both hues present** (the signed-edge claim), and ≥4 distinct line widths (magnitude,
not a constant). One trap: classify edges by **hue** (`R > B` vs `B > R`), never by brightness —
`edgeColour` scales the whole triple by magnitude, so a first-cut threshold regex called 141 of
141 amber edges "negative" and passed for the wrong reason.

**48. The flow diagram shipped CLIPPED at both ends, and the fix is a clamp, not more padding.**
Column headers are 124 px wide and centred on their column; the end columns sat 18 px from the
frame, so "Input byte" rendered as "put byte" and lm_head's byte labels (drawn to the RIGHT of
their node) ran off the edge. Padding alone does not fix it -- it only moves the width at which
it breaks -- so the headers are clamped into the frame independently of the plot's padding, and
the padding now only has to clear the dots and their value labels. The inspector card flips to
the left of a node instead of clamping on top of the column it describes, and the pane fills the
stage (`height: 100%`) rather than stacking to its content height and putting a scrollbar over
the diagram. `loop_smoke.mjs` asserts every header's `left` is inside the frame; removing the
clamp fails it with 43 and 888 in a 960 px frame, so the check discriminates.

**The first attempt at that fix was backwards and the user caught it.** Clipped labels got padded
around, which stops the clipping by SHRINKING the diagram -- the opposite of the request. The
width was there to be taken from the narration rail: **372 -> 300 px** (with its padding 24 -> 20
and h1 27 -> 24 px; prose reads fine at 260 px of content), stage padding 24 -> 20, and the plot's
own inset back down to 22 left / 66-78 right -- the right side is set by lm_head's value labels,
"·213 100%" at about 51 px, and nothing else. Net at a 1456 px viewport: column span **998 ->
1026 px** and nothing cut. `loop_smoke.mjs` pins the trade by asserting the drawn columns span
**>88% of the frame**, so a future "just add padding" fails instead of quietly narrowing the
instrument. General rule for this page: the stage is the instrument, the rail is prose -- take
space from the rail.

**And width was only half of it.** The flow pane also carried its sampling disclosure and its four
counters BELOW the canvas, which cost the diagram ~130px of HEIGHT on every screen -- far more than
the horizontal inset ever cost it. Both moved into the rail (`#r-live`, shown only on stop 0 and
placed AFTER `.rail-nav` so the stepper never gets pushed below the fold), `PAD_B` went 34 -> 14,
`PAD_T` 64 -> 58, and the panes/diagram-host padding tightened. Net **+164px of plot height at
every window size** -- +32% at 1080px, +80% at 780px. Node radii and tick thickness now scale with
the row height they are given, so a bigger window shows a bigger diagram rather than the same
diagram with more gaps in it. `loop_smoke.mjs` pins the vertical trade too: the columns must span
**>82% of the frame's height**, alongside the >88% width check.

**49. Both side HUDs on THE FIELD collapse to a handle.** They float over a full-bleed canvas and
cover a real slice of it, so an unobstructed view had to be one click. Each panel's contents are
wrapped in `.hud-body` and **hidden**, not translated off-screen — a collapsed panel therefore
holds nothing focusable and cannot be tabbed into, and the panel is sized by its content so no JS
measurement and no `@media` special case is involved. `H` toggles both (ignored while a form
control has focus). The handle is the one thing that never disappears; `field_smoke.mjs` asserts
that, both directions of the toggle, the `aria-expanded` state, and that H does not fire from
inside a select.

**50. The `[hidden]` cascade check had a hole, and building item 49 walked straight into it.**
It only asked "if a `display` rule matches a hidden class, does the `[hidden]` reset exist?" — but
`.hud.is-collapsed .hud-body { display: block !important }` beats `[hidden] { display: none
!important }` on specificity, and no cascade order fixes that. My first cut at the responsive
rules contained exactly that line. The check now treats an `!important` display on a hidden class
as a **hard failure**, separate from the reset warning, and the scan covers classes that only get
`hidden` at RUNTIME (`RUNTIME_HIDDEN`) since the source scan cannot see those. Verified by adding
the offending rule back: it fails. jsdom cannot catch this at all — its `getComputedStyle` honours
`hidden` regardless of author CSS — so the static read of the stylesheet is the only guard.

**51. Clicking a node lit almost nothing, and the cause was structural, not visual.** The flow
pool spanned all four heads, but `x_pre` is derived from `decoder_x[head]` and sigma is
accumulated for ONE head -- so a neuron from any other head had no pre-activation and no state
row. It was a dot with nothing entering or leaving it, and 3 of 4 drawn neurons were like that.
Full up- and downstream closure from a drawn neuron reached **1 edge**.
The pool is now built from the traced head only (`--head`, default 0). Effects, all measured:
closure from a neuron went **1 -> up to 41 edges**, all 26 drawn neurons now reach past their
immediate edges, every one has a real pre-activation, and leader coverage *improved* **78.4% ->
89.9%** because the pool and the leaders it is scored against are finally the same population.
The page states the restriction in the first sentence of the note -- the counts above the columns
stay all-heads, because the sparsity claim is about all of them.

**52. And the highlight itself answered the wrong question.** It lit only the edges TOUCHING the
node, at the brightness they already had, and dropped everything else to 5% -- so a click read as
"the picture went dark", not "here is the route". It is now a proper **multi-hop trace**: both
directions to the ends of the graph, brightness by hop (0.98 / 0.62 / 0.34), a wide faint underlay
beneath the immediate connections so they read as lit rather than merely un-dimmed, rings on the
nodes the path passes through, and everything off it in a **hueless** grey so colour keeps meaning
sign. The card reports `N direct · M on its path, across K of 8 columns`.
Two things that had to change with it: `draw()` computes the trace, so it must run BEFORE
`showCard()` cites it; and the pin now remembers WHICH node, not which row -- the cast's visitors
change every token, so a slot-held pin silently jumped to a different neuron while still claiming
to be the one clicked. Every 192-tick column is selectable too ("click any node" has to mean any).

`npm test` is **228 checks** across the same twelve gates.

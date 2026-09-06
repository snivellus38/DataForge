# Defending this submission

Working aid for the live defense (15 points: *"whether the team understands every major
component, can trace the system, can predict the result of changes, and can distinguish real
behavior from precomputation or animation"*).

`CLAUDE.md` is the full build log. This is the short version, organised the way the questions come.

---

## 1. Every number on screen → the script that produces it

If a judge points at a figure, this is where it comes from. **Nothing on any page is typed in by
hand.** Numbers marked *cited* are quoted from a primary source and never recomputed.

| Number on screen | Where it comes from | Tier |
|---|---|---|
| The answer, its probability, the truth beside it | `web/src/bdh.js` `forward()`, in the browser | live |
| σ, Δσ per token, the before/after difference | `bdh.js` `trace()` → `snaps`, `writes` | live |
| `2.8e-14` float64 residual | `research/verify_equivalence.py` | ours |
| `1.2e+02` / `1.7e-5` / `1.6e-5` ablations | same script; JS twin in `web/test/equivalence.mjs` | ours + live |
| `5.2e-7` JS-vs-PyTorch logit error | `web/test/validate.mjs` against `web/test/fixture.json` | ours |
| Capacity curve, +0.319, 1/π | `research/sigma_capacity.py`; JS port re-run live on the k slider | ours + live |
| 5.13% x-sparsity, 0.94% y-sparsity | `research/export_walk.py` over 55,959,552 neuron-token slots | ours |
| 34.3% pooled negative scores (14.7–38.5% per sentence) | `research/export_traces.py`; page reads `manifest.corpus_neg_share` | ours |
| MCC +0.944, P(silent\|isolated) 94.5% | `research/export_walk.py`; recomputed in-browser by `price.js` | ours |
| G\* edge counts, modularity 0.080–0.085, 39% isolated | `research/export_field.py` | ours |
| Iteration overlap (Jaccard ~0.31 by iter 6) | `research/export_traces.py` | ours |
| 2,692 selective neurons, 1,836 at 5% FDR | `research/export_concepts.py`, permutation null, 2,000 draws | ours |
| Every merge-experiment figure | `research/merge_replicate.py` | ours |
| Every flow-diagram edge value | `research/export_walk.py` tensors × in-browser activations | ours + live |
| 29.5%, 118/400, $0.00070, $0.00265246, 59.38%, Table 3/8/9 | **arXiv:2608.09888**, via `research/bdh-cq-dossier.md` | cited |

**"Is anything animated?"** No. The only authored timing is the *pacing* of THE PRICE's opening
slide — the order tokens reveal in. Every value it shows is the model's, and `price_smoke.mjs`
asserts the tape advances and σ fills rather than a canned image being shown.

## 2. Tracing the system

**One token through one block** (this is exactly what THE LOOP draws, stop by stop):

| Step | What happens | Code |
|---|---|---|
| 1 | byte → embedding → residual `v*` (D=192) | `bdh.js` / `bdh_big.py` `embed` |
| 2 | `LN(v)` | layer norm before the shared operator |
| 3 | `x_pre = v · decoder_x[head]` → N=3,072 per head | `Dₓ`, shared across all layers |
| 4 | `x = ReLU(x_pre)` — **non-negative, 5.13% active** | the ReLU that buys readability |
| 5 | `Q = K = rope(x)` — same tensor, rotated | RoPE **inside** attention |
| 6 | `σ += rope(K)ₜ ⊗ Vₜ` ; `a = σᵀ rope(Q)ₜ` | the fast-weight write and read |
| 7 | `y = ReLU(D_y a) ⊙ x` — an AND of two sparse things, **0.94% active** | the multiplicative gate |
| 8 | `Δv = y · encoder` → added back to the residual | `E` |
| 9 | ×6 — **the same weights again**, not six layers | weight tying |
| 10 | `lm_head` → next-byte distribution | |

**Two things to be ready to say precisely:**

- **`n_layer` is not depth.** `encoder`, `encoder_v` and `decoder` are built once in `__init__`
  and reused inside `for level in range(n_layer)`. Total parameters = `3·nh·D·N + 2·vocab·D`, with
  no per-layer terms. So the 131K model is 131,072 parameters, not ~426K. This maps onto BDH-CQ's
  Eq. (3) `H_{r+1} = F_θ(H_r, S_K)` — the same θ applied R times.
- **The materialisable state is `N×D`, not the paper's conceptual `N×N`.** `V = x` (dimension D),
  and a LayerNorm between `scores @ V` and the projection back to neuron space blocks the exact
  reduction. Say this before a judge asks it.

## 3. Predicting the result of a change

Each live control maps to one real variable. Be able to say what will happen *before* clicking.

| Control | Where | What changes, and why |
|---|---|---|
| **Remove the demonstration** | PRICE slide 1 | Answer flips to **wrong at p≈0.972**, still inside the target alphabet. Weight hash unchanged. The rule lived in σ, which is rebuilt per forward pass. |
| **softmax on** | PRICE slide 3 | Equivalence residual jumps to `1.2e+02`. Softmax normalises across the row, so the sum no longer factors into a per-token outer product — you now need every key, i.e. a growing cache. |
| **Q ≠ K** | PRICE slide 3 | Residual stays ~`1.7e-5`. **Nothing breaks.** You lose the reading of σ as a Hebbian synapse matrix, not the constant memory. |
| **no ReLU** | PRICE slide 3 | Residual stays ~`1.6e-5`. Same story — readability, not memory. |
| **k slider** | PRICE slide 7 | Retrieval falls along the measured curve; the interference is real, not a fit. |
| **let keys go negative** | PRICE slide 7 | The curve flattens to ~100% past k=256. This is the counterfactual that proves overlap — not rank — is the binding constraint. |
| **layers 4 → 2** | wiring gate | Logits change (Δ ≈ 1.8e+1). At 4 the delta is exactly 0, because the model was trained at 4. |
| **sentence selector** | FIELD, LOOP | Different real data. Active counts, negative-score share and iteration labels all differ — `field_smoke.mjs` requires them to. |
| **iteration selector** | FIELD | The same operator's output after r passes. Active count climbs; Jaccard vs iteration 1 falls to ~0.31. Disabled in the arcs view, because that view has no layer axis. |
| **token slider** | LOOP, PRICE | Steps σ's accumulation. At token 0 a nearly-empty panel is the *correct* picture — one rank-one write has landed. |

## 4. The five challenges most likely to come

**"Isn't this just a state-space model?"**
No, and the track forbids the framing. The BDH paper says "ReLU-low-rank transformations with
linear attention" and never "state-space model." Mechanically: there is no learned state-transition
matrix and no decay term at all. What stands in for decay here is destructive interference from
RoPE phase — measured, 34.3% of causal scores are negative pooled over our corpus.

**"You said the merge fails — doesn't that refute §7.1?"**
No, and do not let it be characterised that way. The paper's recipe (concatenate tensors with an
`n` dimension, average the rest) carries a precondition: the models being merged are **clones of a
common base**. Ours were trained independently from scratch. We ran four variants including the
paper's exact recipe and a magnitude control; all collapse. That tests a *harder* case than the
paper claims. RoPE was not the cause (concatenating it as specified still collapses) and it is not
a magnitude problem (halving `E` halves the logits exactly and the output is still `iiiiiiii`).

**"Your accuracy curve shows the state filling up."**
It does not, and this is the sharpest trap in the project. The **model** curve is a
*demonstration-coverage* limit set by the k range seen in training. The **σ** curve, measured
directly on the same state, retrieves 100% at k=16 and 95% at k=32 — far past where the model
fails. Two different curves teaching two different lessons; we present them as such, on separate
slides. To probe capacity *through* the model you would have to train past the eval range.

**"You claim ~5% sparsity but your toy is 43%."**
Two different models, and we never conflate them. The 131K toy on a synthetic cipher is ~43%
active and the artifact says so. The 5.13% is our **8M** model on real French, measured through our
own pipeline over 55,959,552 slots. Separately: the BDH paper's own ~5% figure we have **not**
verified against the primary source, so we never quote it as support.

**"38.5% negative attention scores — is that the model or one sentence?"**
One sentence, at layer 3 head 0. Pooled over the seven-sentence corpus it is **34.3%**, and per
sentence it ranges 14.7%–38.5%. The field's readout is filled from `manifest.corpus_neg_share`, so
the label is literally true. It also *decreases* with iteration (61.8% → 39.2% on the hero),
which is an independent signal for the same recruitment story as the Jaccard curve.

## 5. Where we corrected the sources

Worth volunteering — it demonstrates the evidence discipline the track scores.

- **The problem statement itself contains a half-truth.** It says BDH-CQ's "state accumulates
  additively per demonstration." Eq. (1) is the general `Sₜ = U_θ(Sₜ₋₁, Dₜ)`; additive accumulation
  is explicitly named in §3.2 as *the special case* realised by linear attention, and BDH-CQ's
  actual rule is proprietary. The artifact corrects this on screen as a teaching beat.
- **The report gives two unreconciled costs for the same result.** $0.00070 (abstract/§5) and
  $0.00265246 (§6.6), both attached to 118/400. At the higher figure "57× cheaper" becomes ~15×.
  And $0.00070 is *computed* — 0.85 s × an assumed $3/H200-hour — not billed.
- **The "independent" audit was run by co-authors.** BDH-CQ is on no ARC Prize leaderboard.
- **ConceptARC is in the training mixture** (§4.2) and then used as evaluation (59.38%). The paper
  admits its control does not rule out training exposure.
- **A published selectivity claim does not survive a null.** `monosemanticity/precomputed.json` in
  the prior run reports 200 neurons at selectivity 1.0 with no control; a neuron firing on 5 bytes
  is trivially 100% selective. With a per-neuron permutation null, 2,692 beat their own p95 against
  ~352 expected by chance, and 1,836 survive Benjamini–Hochberg at 5% FDR. We ship ours.

## 6. Things we must never say

Each is a factual error a judge can catch. Full list with sources in
`research/bdh-cq-dossier.md` under MISCONCEPTION TRAPS.

- What **"CQ"** stands for — undefined in every source. Do not guess.
- That effort levels equal some number of latent iterations `R` — **`R` is never disclosed**, nor
  is what changes between effort levels.
- That BDH is a state-space model in the Mamba sense.
- That 29.5% is ARC-Prize-verified.
- That any **ARC-AGI-2** result exists for BDH-CQ. (TRM reports 8% on ARC-AGI-2 — that is TRM's,
  and the contrast is worth drawing, but do not let the two get attached to each other.)
- That the ConceptARC number is clean.
- Any number that is not in the dossier or produced by a script in this repo.

## 7. Known weak points — say them first

Volunteering these is stronger than being caught by them.

- 131K parameters on a synthetic task; 8M on real language. Neither is evidence about BDH at scale.
- The 8M checkpoints add a learned positional embedding on top of RoPE and have no decay term.
  They are faithful to `bdh.py` in every other load-bearing respect (weight-tied, `Q = K = ReLU(·)`,
  `V = LN(v)`, no softmax, strict-causal `tril(-1)`, RoPE inside attention).
- "Never fires" is measured over **7 sentences**. A wider corpus may wake some neurons.
- "Isolated" depends on the **p99 threshold**. Move the threshold, move the count.
- The 8M model is **reused prior work** by one of us, disclosed in the README.
- σ is dense; we draw Δσ, differences and row energy rather than pretending otherwise.
- No BDH-CQ result here is reproduced. It cannot be run.

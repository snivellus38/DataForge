# Demonstrations Are Weights

**DataForge 2026 — Pathway track, "Explain the Frontier."**
Concept: **Test-Time Adaptation — optimization versus context.**

> **Demonstrations are weights you write at inference time:** a frozen model adapts by adding
> rank-one updates to a fixed-size associative state, so changing only the demonstrations changes
> the answer — **but that state's capacity is set by how much its keys overlap, not by its size,
> so the non-negativity that makes BDH inspectable is also what makes it forget sooner.**

Both halves are falsifiable in the artifact. The first takes about sixty seconds.

| | |
|---|---|
| **Artifact** | *pending deployment — see [Deploying](#deploying)* |
| **Source** | https://github.com/snivellus38/DataForge |
| **One-page summary** | [`docs/concept-summary.pdf`](docs/concept-summary.pdf) |
| **Run locally** | `npm run dev` → http://localhost:8080 |
| **Tests** | `npm test` — 12 gates, 235 checks. `npm run test:mobile` for narrow viewports. |

---

## Who this is for

A data scientist or ML engineer who can read a Transformer diagram and has not yet met a
post-Transformer architecture. No BDH background is assumed, and nothing needs a GPU, an install,
or a sign-in — the models run in the browser tab.

**Prerequisites.** Attention as a matrix product (that `QKᵀV` is three multiplications — you need
not have implemented one); what a KV cache is and why it grows; that softmax normalises across a
whole row.

**Learning objectives.** After using the artifact a learner can:

1. Say **which** of BDH's unusual choices actually buys a fixed-size state, and which two only buy
   readability — most people guess wrong, and the artifact lets you test each in one click.
2. Predict **from the weights alone** which neurons in a trained model will never fire on real text.
3. State what non-negativity **costs**, and reproduce the constant that sets the ceiling: `1/π`.

## The journey

Three pages, in order. `index.html` is the front door and states the above on screen.

| # | Page | The question it answers | Substrate | Payload |
|---|---|---|---|---|
| 1 | `web/field.html` — **THE FIELD** | What does a whole model look like while it is thinking? | replayed 8M | 6.3 MB |
| 2 | `web/loop.html` — **THE LOOP** | What actually happens inside one iteration? | replayed 8M + σ accumulated in-browser | 4.9 MB |
| 3 | `web/price.html` — **THE PRICE** | What is the claim, and what does it cost? | **live 131K model** + replayed 8M + cited BDH-CQ | 0.9 MB |

THE PRICE is where the sixty-second test lives, so the door marks it "start here" for anyone
short of time.

## What is live, what is not

The track asks for this explicitly, and every number on every page carries a tag saying which
tier it belongs to.

- **Live, computed in your browser.** A **131,072-parameter BDH we trained** on an in-context
  substitution cipher, running as plain typed arrays (`web/public/model.bin`, 257 KB): the forward
  pass, σ, the parallel↔recurrent residual, the break-it toggles, and the capacity sweep on
  THE PRICE. Also σ on THE LOOP, which is accumulated in the browser from the model's own
  rank-one writes rather than shipped.
- **Ours, replayed.** An **8M-parameter BDH** that translates English into French, trained by one
  of us before this hackathon and reused with disclosure. Its per-token activations, attention
  scores, synapse graph and traces were exported offline by `research/export_*.py` and are replayed
  from `web/public/big/` and `web/public/walk.*`.
- **Replayed and cited.** Every **BDH-CQ** figure, quoted from arXiv:2608.09888 with a section or
  table locator. BDH-CQ has no public weights and no public API. Nothing here runs it, and nothing
  here is presented as its output.
- **Animated.** Nothing. The only authored timing is the pacing of THE PRICE's opening slide; every
  value it displays is the model's, and the gate asserts the tape actually advances.
- **Synthetic.** The cipher task (`research/cipher_task.py`) resamples a fresh bijection per
  example, so the mapping exists only in the prompt. That is the point: the only way to answer is
  to read it out of the recurrent state.

**Neither model is an official BDH model, and neither is BDH-CQ.**

## Verified results

Everything below is reproducible from this repo by the named script.

**The parallel↔recurrent equivalence is exact**, and only one of BDH's three unusual choices is
load-bearing for it (`research/verify_equivalence.py`, `npm run test:equivalence`):

| Check | Result |
|---|---|
| float64 residual, parallel vs recurrent | `2.8e-14` — exact |
| Browser JS vs PyTorch logits | `5.2e-7` worst relative, argmax 100% |
| Q ≠ K | equivalence **survives** (`1.7e-5`) |
| no ReLU | equivalence **survives** (`1.6e-5`) |
| **softmax on** | equivalence **breaks** (`1.2e+02`) |

Softmax-freeness buys the fixed-size state. `Q = K` and the ReLU buy readability — which is what
the rest of the artifact is about.

**Non-negativity costs associative capacity, at a rate of 1/π** (`research/sigma_capacity.py`,
ported to JS and checked against it by `npm run test:capacity`):

| k | 8 | 32 | 64 | 128 | 256 | 384 |
|---|---|---|---|---|---|---|
| non-negative keys (BDH) | 100% | 94% | 80% | 47% | 22% | 12% |
| signed keys (counterfactual) | 100% | 100% | 100% | 100% | 100% | 99% |

Mean pairwise cosine is **+0.319** for ReLU'd keys vs **−0.000** for signed, and +0.319 is
analytic: **1/π** for rectified Gaussians. Capacity is bounded by key overlap, not state size —
signed keys hold 384 bindings in a rank-256 state.

**Structure predicts function** (`research/export_field.py`, `research/export_walk.py`). `G* = Dₓᵀ Eᵀ`
is computed from the weights with no data at all, yet on 7 sentences of real French:

- 37.9% of the 12,288 neurons never fire; 38.7% have no synapse in `G*` at the p99 threshold.
- They are nearly the same set: **MCC +0.944**, P(silent | isolated) 94.5% against a 37.1% base rate.
- Mean total degree 0.7 for silent neurons vs 97.0 for firing ones.

*Caveats that travel with it:* the corpus is 7 Europarl-style sentences, so "never fires here" is
not "dead forever"; and "isolated" is defined by the p99 threshold, so the count moves if the
threshold moves.

**Sparsity, measured through our own pipeline.** `x` is **5.13%** active over 55,959,552
neuron-token slots; the gate `y = ReLU(D_y a) ⊙ x` is **0.94%** — an AND of two sparse conditions,
so fewer than one neuron in a hundred reaches the residual.

**Model accuracy** (131K toy, trained on k ∈ [2,12], 48-symbol alphabet, chance = 2.1%):

| k | 2–12 | 16 | 20 | 24 | 32 | 48 |
|---|---|---|---|---|---|---|
| accuracy | **100%** | 64% | 40% | 20% | 9% | 3% |

**This curve is a demonstration-coverage limit, not a capacity limit.** The same state retrieves
99% at k=16 when measured directly. Conflating the two is a factual error, and the artifact
teaches them as separate lessons on separate slides.

## Architecture

```
research/                  offline: train, measure, export
  bdh_big.py               8M architecture, transcribed from the training notebook
  cipher_task.py           the synthetic in-context task (fresh bijection per example)
  train.py                 trains the 131K model -> runs/final.pt
  export_weights.py        -> web/public/model.{json,bin}   (fp16, 257 KB)
  export_field.py          G*, Louvain, layouts     -> web/public/big/field.*
  export_traces.py         per-token activations, attention, sigma energy -> big/traces.bin.gz
  export_walk.py           the two walk sentences, every stage -> web/public/walk.bin.gz
  export_concepts.py       concept selectivity vs a permutation null
  export_synapses.py       synapse timelines
  merge_replicate.py       the neuron-concatenation merge experiment
  verify_equivalence.py    the PyTorch twin of the equivalence gate
  sigma_capacity.py        the 1/pi measurement and its signed counterfactual
  serve.py                 dev server (no-store; pins .gz to octet-stream)
  check_mobile.py          narrow-viewport check over the DevTools protocol

web/src/                   pure modules first, then the controllers that use them
  bdh.js                   the 131K forward pass + trace(), plain typed arrays
  bigdata.js walkdata.js field-data.js flow-data.js   pure decode/derive, no DOM, no GL
  capacity.js chart.js sigma-view.js tokens.js palette.js
  field-gl.js              hand-rolled WebGL2, HDR target + tone map
  inspector.js             THE FIELD's controller          (frozen)
  loop.js stages.js flow.js diagram.js                     THE LOOP
  price.js                                                 THE PRICE
  theme.css door.css loop.css price.css field.css

docs/
  concept-summary.md       the one-page summary, source of truth
  build_summary.py         renders it to HTML + PDF (npm run summary)
  defense.md               how to trace, defend and break every claim
```

Every module that touches data is **pure and separately gated** before a controller uses it. That
split is why the suite can test the science in Node without a browser.

### `web/field.html` is frozen

It is finished, and `web/test/frozen.mjs` sha256s it plus the eleven files it loads
(`web/test/frozen.sha256`); that gate runs first in `npm test`. This is why `theme.css` duplicates
field.css's `:root` block and nav rules instead of sharing them — hoisting would have edited a
finished page. `wiring.mjs` asserts the two copies stay identical. To change it deliberately:
edit, then `npm run freeze`.

## Reproducing

```bash
npm test                # 12 gates, 235 checks — no GPU, no network
npm run test:mobile     # narrow-viewport check (needs Edge/Chrome + websocket-client)
npm run dev             # serve the artifact at http://localhost:8080
npm run summary         # rebuild docs/concept-summary.{html,pdf}
```

Retraining the 131K model needs Python + PyTorch + a GPU (~65 min):

```bash
npm run train           # research/train.py -> research/runs/final.pt
npm run export          # weights + fixture + verified presets
npm run verify:torch    # the PyTorch twin of the equivalence test
```

Re-exporting the 8M data needs `trained_model_things/`, which is **not in this repo** (475 MB; see
Attribution). With it present: `npm run export:big` and `npm run export:walk`.

**After any retrain, run `python research/make_presets.py`** — it re-verifies the sixty-second
moment and prints `SIXTY-SECOND MOMENT HOLDS` or `DOES NOT HOLD`.

### The gates

In the order `npm test` runs them. Several exist because a specific bug shipped past everything
else, and the ones enforcing the honesty layer are not optional decoration — dropping a disclosure
fails the build.

| Gate | What it protects |
|---|---|
| `frozen` | field.html and its eleven files are byte-identical to the frozen checksums |
| `price_smoke` | boots THE PRICE in jsdom; asserts ~80 elements populate, canvases paint non-uniform pixels, and the evidence ledger's disclosures are present |
| `field_smoke` | boots THE FIELD on both render paths; readouts must **differ** between sentences |
| `bigdata` | shipped bytes round-trip within 0.5 quantisation steps; re-derives 5.11% and 38.5% |
| `walk` | RoPE re-derivation and σ accumulation, each with a negative control (247× and 580× worse) |
| `loop_smoke` | boots THE LOOP; measures canvas *contrast*, records vector draws, pins the diagram's span |
| `field` | palette monotonic in OKLab L; pure transforms |
| `validate` | browser JS vs PyTorch parity |
| `equivalence` | parallel↔recurrent, plus the three ablations |
| `app_logic` | the sixty-second moment, including the `(src, tgt, SEP)` token order |
| `capacity` | the JS capacity port matches PyTorch within 10 pp at every k |
| `wiring` | ids, generated class names, the shared nav, the door's framing, ≥3 papers cited beside claims |

## Honest limitations

- The live model is **131K parameters on a synthetic task**. It is an honest miniature, not
  evidence about BDH at scale. The 8M model is real language but still small.
- The toy runs **~43% active** neurons, nowhere near the ~5% BDH reports at scale. Our 8M model
  *does* reach **5.13%**, measured through our own pipeline — but that is our model, not theirs.
  *(The BDH paper's own ~5% figure has not been verified against the primary source by us, and is
  not quoted as ours.)*
- σ is **dense**, not sparse ridges — 100% of its cells are nonzero within a few tokens. What is
  legible is Δσ per token, a before/after difference, and row energy, which is what we draw.
- The **8M checkpoints deviate from reference BDH in two ways**: a learned positional embedding on
  top of RoPE, and no decay term. Stated wherever their numbers appear.
- The **merge experiment is not a refutation**. The paper's recipe assumes models forked from a
  common base; ours were trained independently from scratch, so we tested a harder case than it
  claims.
- The negative-attention-score share is **per sentence** (14.7%–38.5%, pooled 34.3%), not one
  number.
- **BDH-CQ figures are replayed, never reproduced.** We cannot run it and do not claim to.
- Mobile is verified for **layout** at 320–390 px. On phones THE LOOP's flow diagram keeps its
  design width and scrolls sideways, with a note on screen saying so.

## Primary sources

Cited beside the claims they support, on the pages and in the summary.

- Kosowski, Uznański, Chorowski, Stamirowska, Bartoszkiewicz (2025). *The Dragon Hatchling.*
  [arXiv:2509.26507](https://arxiv.org/abs/2509.26507) — the architecture, and the source of the
  claim that non-negativity buys interpretable sparse structure. That half is theirs; the price we
  put on it is ours.
- Engdahl et al. (2026). *BDH-CQ technical report.*
  [arXiv:2608.09888](https://arxiv.org/abs/2608.09888) — Eq. (1) and the additive special case
  named in §3.2, which is exactly what the public BDH code computes.
- Yang, Wang, Zhang, Shen, Kim (2024). *Parallelizing Linear Transformers with the Delta Rule over
  Sequence Length.* [arXiv:2406.06484](https://arxiv.org/abs/2406.06484) — replaces "the additive
  update in linear transformers" with the delta rule and reports it more effective at associative
  recall. That additive update is BDH's write; our ceiling is why the alternative exists.
- Sun et al. (2024). *Learning to (Learn at Test Time): RNNs with Expressive Hidden States.*
  [arXiv:2407.04620](https://arxiv.org/abs/2407.04620) — the update rule as a step of
  self-supervised learning; the contrast that places BDH's write in a design space.
- Wang et al. (2025). *Hierarchical Reasoning Model.*
  [arXiv:2506.21734](https://arxiv.org/abs/2506.21734), and Jolicoeur-Martineau (2025). *Less is
  More: Recursive Reasoning with Tiny Networks.* [arXiv:2510.04871](https://arxiv.org/abs/2510.04871)
  — the optimization route to test-time adaptation, and the fork this artifact takes a side in.

`research/bdh-cq-dossier.md` is our verbatim extraction of the BDH-CQ report with a locator and an
evidence level on every claim. It is canonical for every BDH-CQ number here.

## Attribution, sources and licenses

| Component | Source | License |
|---|---|---|
| `vendor/bdh.py` | [pathwaycom/bdh](https://github.com/pathwaycom/bdh) — **unmodified** | MIT (`vendor/LICENSE-bdh`) |
| 131K trained weights, cipher task, browser port | this repo | MIT |
| 8M En→Fr and En→Pt checkpoints | **prior solo hackathon work by one of this team**, reused with disclosure | author's own |
| All three pages, exports, tests, docs | this repo | MIT |
| Colour palette and its validation method | Claude Code `dataviz` skill reference palette | — |
| Fonts | none bundled — system font stacks only | — |
| Graphics | none bundled — everything is drawn at runtime from data | — |

**Reused prior work, disclosed.** `trained_model_things/` (475 MB, gitignored for size, not for
provenance) holds the 8M English–French and Portuguese checkpoints. They were trained by one of
this team at an earlier hackathon and are reused here. Everything derived from them in this repo —
`research/bdh_big.py` and every `export_*.py`, plus all analysis — is new work for this submission,
and our pipeline independently reproduces the prior run's 5.15% sparsity as 5.11%.

## AI assistance disclosure

Built with substantial assistance from **Claude (Opus 5)** via Claude Code: primary-source
extraction, experiment design, training and analysis code, the browser port, the three pages, the
test suite, and drafting of this README and the concept summary. Prompting, direction, review and
every design decision are the team's, as is the prior 8M model.

Every empirical claim in this repo is produced by a script in it, and the team must be able to
trace and defend each one — `docs/defense.md` is the working aid for that. No number appears
anywhere in the artifact that is not either produced by a script here or quoted from a primary
source with a locator.

## Deploying

Static, no build step. The site root is `web/`, and `vercel.json` pins it.

The one header that matters: `public/**.bin.gz` must be served as `application/octet-stream`
**with no `Content-Encoding: gzip`**. Both big packs are inflated *by the page* with
`DecompressionStream`, so a transport-level encoding hands the loader already-inflated bytes.
After deploying, verify on the live origin — not locally:

```bash
curl -sI <url>/public/big/traces.bin.gz | grep -i "content-type\|content-encoding"
curl -sI <url>/public/walk.bin.gz       | grep -i "content-type\|content-encoding"
```

```bash
vercel login && vercel --prod
```

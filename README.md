# Demonstrations Are Weights

**An interactive artifact about test-time adaptation, built on Pathway's Dragon Hatchling (BDH).**
Three pages, two models we can run, one claim that takes sixty seconds to falsify.

[![tests](https://github.com/snivellus38/DataForge/actions/workflows/tests.yml/badge.svg)](https://github.com/snivellus38/DataForge/actions/workflows/tests.yml)
[![live artifact](https://img.shields.io/badge/live-demonstrations--are--weights.vercel.app-e8a33d)](https://demonstrations-are-weights.vercel.app)
[![license](https://img.shields.io/badge/license-MIT-4c8fbd)](LICENSE)
[![primary sources](https://img.shields.io/badge/primary%20sources-6%20papers-6fbf9a)](#references-and-tooling)

![The neuron field: 12,288 neurons of an 8M BDH, rendered in WebGL while it translates a sentence](docs/media/field-full-view.gif)

<sup>**THE FIELD.** Every dot is one of the 12,288 neurons in an 8M-parameter BDH we trained,
laid out by its position in the model's own synapse graph. Amber is activation at the current
byte. The clip moves through the views: one iteration, then all six at once, recoloured by
synapse count and by graph community, then back. The dark region is not a rendering artifact —
it is the 38.7% of neurons that the weights alone predict will never fire.</sup>

---

> **Demonstrations are weights you write at inference time.** A frozen model adapts by adding
> rank-one updates to a fixed-size associative state, so changing only the demonstrations changes
> the answer — **but that state's capacity is set by how much its keys overlap, not by its size,
> so the non-negativity that makes BDH inspectable is also what makes it forget sooner.**

Both halves are falsifiable inside the artifact. The first takes about a minute:

1. Open **[THE PRICE](https://demonstrations-are-weights.vercel.app/price.html)**. A 131K-parameter
   BDH runs in your browser tab and answers a substitution cipher it has never seen — the mapping
   exists only in the prompt, resampled fresh for every example.
2. Press **"Remove the demonstration it needs"**. One demonstration disappears; the query bytes
   stay byte-identical.
3. The answer changes from correct at p = 1.000 to **wrong at p = 0.972**, still inside the target
   alphabet. No parameter moved. The model does not hedge when a binding is missing — it is
   confidently wrong and structurally plausible, which is the same failure signature BDH-CQ reports
   in its Table 9.

| | |
|---|---|
| **Artifact** | **https://demonstrations-are-weights.vercel.app** — opens without sign-in |
| **One-page summary** | [`docs/concept-summary.pdf`](docs/concept-summary.pdf) |
| **Defense sheet** | [`docs/defense.md`](docs/defense.md) — every on-screen number mapped to the script that makes it |
| **Run locally** | `npm run dev` → http://localhost:8080 |
| **Tests** | `npm test` — 12 gates, 236 checks, no GPU and no network |

Submitted to **DataForge 2026, Pathway track ("Explain the Frontier")**, on the concept of
**test-time adaptation: optimization versus context**.

## Contents

- [Who this is for](#who-this-is-for)
- [What you are looking at](#what-you-are-looking-at) — the three pages
- [The idea, in three steps](#the-idea-in-three-steps)
- [Where this sits](#where-this-sits) — against Transformers, and what constant memory costs
- [What we measured](#what-we-measured) — six results, each with its script and its caveat
- [The two models](#the-two-models)
- [How it is built](#how-it-is-built)
- [The test suite, and what each gate caught](#the-test-suite-and-what-each-gate-caught)
- [Reproducing](#reproducing)
- [Honest limitations](#honest-limitations)
- [References and tooling](#references-and-tooling)
- [Attribution and licenses](#attribution-and-licenses)

## Who this is for

A data scientist or ML engineer who can read a Transformer diagram and has not yet met a
post-Transformer architecture. No BDH background is assumed, and nothing needs a GPU, an install,
or a sign-in — the models run in the browser tab.

**Prerequisites.** Attention as a matrix product (that `QKᵀV` is three multiplications — you need
not have implemented one); what a KV cache is and why it grows; that softmax normalises across a
whole row.

**Learning objectives.** After using the artifact a learner can:

1. Say **which** of BDH's unusual choices actually buys a fixed-size state, and which two only buy
   readability. Most people guess wrong, and the artifact lets you test each in one click.
2. Predict **from the weights alone**, with no data, which neurons in a trained model will never
   fire on real text.
3. State what non-negativity **costs**, and reproduce the constant that sets the ceiling: `1/π`.

## What you are looking at

Three pages, in order. [`index.html`](web/index.html) is the front door and states the above on
screen.

### 1. THE FIELD — what does a whole model look like while it is thinking?

12,288 neurons in hand-rolled WebGL2, positioned by a force-directed layout over the model's own
standing synapse graph. Playback scrubs through the sentence; you can pan, zoom, and click any
neuron to pin it and draw its synapses.

![Six iterations of the same weights, side by side, each labelled with how much of iteration 1 it still shares](docs/media/field-six-iterations.png)

<sup>**Six iterations of one operator, not six layers.** BDH is weight-tied: `encoder`,
`decoder_x` and `decoder_y` are built once and reused, so `n_layer` is an iteration count. Here the
same weights are applied six times to the same sentence, at the same byte. The neuron count rises
across the tiles while the overlap with iteration 1 falls and then settles — the shared operator
*recruits*, then converges. The tile labels carry the overlap, because six similar-looking clouds
without it teach nothing.</sup>

![Causal attention drawn as arcs, with negative scores shown as a distinct hue](docs/media/field-attention-arcs.gif)

<sup>**Attention as arcs.** BDH has no decay term. What makes it forget is that RoPE rotates keys
before they meet, so scores between distant tokens can go negative and cancel earlier writes. Over
the corpus **34.3% of causal scores are negative** (14.7%–38.5% per sentence). The page computes
the per-sentence figure live and prints the pooled one beside it.</sup>

### 2. THE LOOP — what actually happens inside one iteration?

Eight stops through a single block, over a persistent architecture diagram that stays on screen
the whole way.

![The flow diagram: eight columns from input byte to next-byte distribution, every edge a real number](docs/media/loop-flow-diagram.gif)

<sup>**Stop 0: the whole block, as eight columns.** `byte → v* → Dₓ → ReLU x → σ read-out a* →
Dᵧ⊙x y → E + Δv → lm_head`. Every edge's width and colour is `weight × source activation` — the
actual term that source contributes to that target — read from the shipped tensors, not
illustrated. The column headers carry the live counts: `95.1% below zero` after `Dₓ`,
`626 firing · 5.09%` after the ReLU, `117 firing · 0.95%` after the gate. Fewer than one neuron in
a hundred reaches the residual.</sup>

![Clicking a neuron traces its path forward and backward across the diagram](docs/media/loop-neuron-trace.gif)

<sup>**Click any node and it traces.** Not just the edges touching it — the full path in both
directions to the ends of the graph, brightness by hop, everything off the path in a hueless grey
so colour keeps meaning sign. The card reports how far it reached: `3 direct · 10 on its path,
across 6 of 8 columns`. The clip also steps the iteration selector, which is why the sparsity
readouts move between frames.</sup>

![The eight stops of the walkthrough, with the persistent architecture diagram below](docs/media/loop-walkthrough.gif)

<sup>**The eight stops.** One of them is the RoPE stop, whose polar plot draws a single token's
neuron pairs before and after rotation: every un-rotated key sits inside the non-negative quadrant
because it just came out of a ReLU, and `111 of 3,072` rotated components have gone below zero.
σ on this page is not shipped — it is accumulated in your browser from the model's own rank-one
writes, and a gate proves the accumulation is right by breaking it two ways.</sup>

### 3. THE PRICE — what is the claim, and what does it cost?

A twelve-slide deck. The 131K model runs live here: the forward pass, σ, the
parallel↔recurrent residual, the three ablation toggles, and the capacity sweep are all computed
in the tab as you move the controls.

![THE LOOP, showing the persistent architecture diagram beneath the stage](docs/media/loop-overview.png)

## The idea, in three steps

**1. A Transformer's adaptation lives in a cache that grows.** Every new token appends a K and a V
to every layer's cache. Nothing about the model changes; the context does, and it costs memory
linearly.

**2. Softmax-free attention with `Q = K` has an exact recurrent form.** The same arithmetic,
reassociated, carries the whole history in a fixed-size associative state updated by rank-one
writes:

```
σₜ    =  σₜ₋₁ + rope(K)ₜ ⊗ Vₜ
outₜ  =  σₜᵀ rope(Q)ₜ
```

This is not an approximation. The official MIT `bdh.py` run both ways agrees to
**2.8 × 10⁻¹⁴ in float64** ([`research/verify_equivalence.py`](research/verify_equivalence.py)).
Each demonstration you put in the prompt is literally a rank-one update to a weight matrix — which
is why the claim is not a metaphor.

**3. The obvious intuition about *why* is wrong.** BDH makes three unusual choices — no softmax,
`Q = K`, and a ReLU that forces activations non-negative — and it is tempting to assume all three
buy the constant-size state. Ablate them one at a time and only one does. That is the hinge the
whole artifact turns on, and it is a one-click experiment on THE PRICE.

## Where this sits

The honest positioning first, then the measurement.

| | What adapts at inference | State per extra token | What you can inspect | Runs here? |
|---|---|---|---|---|
| **Decoder-only Transformer** (GPT-style) | the context window | K and V per layer — **grows linearly** | attention maps | — |
| **Encoder-only** (BERT-style) | nothing — one bidirectional pass, no causal decode | no decode-time cache at all | attention maps, probes | — |
| **Linear attention / DeltaNet** | a recurrent state | **constant**; DeltaNet replaces the additive write with a delta rule | the state, in principle | — |
| **BDH** (public, MIT) | a recurrent state σ, written rank-one per token | **constant** | sparse activations **and a standing synapse graph** | **yes — we train and run it** |
| **BDH-CQ** | a recurrent state, general update `U_θ` | constant; dimensions undisclosed | nothing published | no — no public weights or API |
| **HRM / TRM** | the **weights** — a backward pass per task | no context state of this kind | — | no — cited only |

BERT is in the table to mark where it is *not*. A masked-language encoder has no causal decode
state and no in-context write, so a head-to-head against it on either the memory axis or the
adaptation axis would compare things that do not correspond. The meaningful comparison is against
a decoder-only Transformer, and that one is exact.

### What "constant memory" actually costs

Constant is not the same as small, and the artifact would be dishonest if it left that out.
[`research/compare_memory.py`](research/compare_memory.py) computes it from the two configs this
repo ships — no weights and no torch required, so it runs on a clean clone.

| Model | KV cache, same D/L/H | BDH state σ | Crossover | At its own context |
|---|---|---|---|---|
| **8M translation BDH** (6 × 192, 4 heads, N = 3,072) | 4,608 B/token | **28.3 MB, flat** | **6,144 tokens** | σ is **12.0× larger** than the cache would be at 512 |
| **131K cipher BDH** (4 × 64, 2 heads, N = 256) | 1,024 B/token | **0.26 MB, flat** | **256 tokens** | σ is **1.7× larger** at its 147-byte prompt |

![Both models: KV cache growing linearly against BDH's flat state, on log axes, with the crossover marked](docs/media/memory-crossover.png)

Dividing one by the other, `D`, `L` and the dtype all cancel:

```
T*  =  (L·H·N·D·b) / (2·L·D·b)  =  H·N/2  =  N_total / 2
```

**BDH's state is the cheaper one exactly when the context is longer than half the model's neuron
count.** That is a property of the architecture, not of a hyperparameter or a dtype, and it holds
for both models here. The same cancellation on arithmetic per generated token puts the compute
crossover at `N·H = N_total`, twice as far out.

This is not "BDH loses". A KV cache grows without bound and σ does not; below the crossover you
are paying a fixed premium for that guarantee, and above it you are collecting on it. Saying which
side of the line a model is on is the honest version of the claim — and both of ours are trained
below their own line.

## What we measured

Everything below is produced by a named script in this repo. Nothing is quoted from memory, and
nothing on any page is typed in by hand.

### 1. The parallel↔recurrent equivalence is exact — and only one choice is load-bearing

`research/verify_equivalence.py` · `npm run test:equivalence` · live on THE PRICE

| Check | Result |
|---|---|
| float64 residual, parallel vs recurrent | `2.8e-14` — exact |
| Browser JS vs PyTorch logits | `5.2e-7` worst relative, argmax agreement 100% |
| `Q ≠ K` | equivalence **survives** (`1.7e-5`) |
| no ReLU | equivalence **survives** (`1.6e-5`) |
| **softmax on** | equivalence **breaks** (`1.2e+02`) |

**Softmax-freeness buys the fixed-size state. `Q = K` and the ReLU buy readability**, which is
what the rest of the artifact is about. An early draft of our own plan claimed otherwise and the
measurement corrected it. The script asserts the negative results too, so if someone later
"improves" the model and quietly breaks the story, the build fails.

### 2. Non-negativity costs associative capacity, at a rate of 1/π

`research/sigma_capacity.py`, ported to JS and checked against the PyTorch original by
`npm run test:capacity`

Write `k` rank-one bindings into a fixed `N × D` state and read them back. One command prints
both rows; the only difference between them is the ReLU on the keys:

| k | 8 | 16 | 32 | 64 | 128 | 256 | 384 | 512 |
|---|---|---|---|---|---|---|---|---|
| **non-negative keys** (BDH) | 100% | 99% | 94% | 75% | 48% | 23% | 14% | 10% |
| **signed keys** (counterfactual) | 100% | 100% | 100% | 100% | 100% | 100% | 99% | 96% |

The counterfactual is what makes this a mechanism rather than a curve — same state, same size,
same rank bound, one ReLU removed. Mean pairwise cosine is **+0.320** for ReLU'd keys against
**−0.000** for signed, and +0.320 is not noise: it is analytic, **1/π = 0.3183** for rectified
Gaussians. Non-negative vectors cannot be near-orthogonal, so they interfere. Signed keys hold
384 bindings in a rank-256 state; BDH's cannot reach half that.

Careful with the attribution, both directions: **that non-negativity buys interpretability is the
Dragon Hatchling paper's claim**, and it is cited to them. **That it caps capacity at 1/π is our
measurement.** The ceiling is also known to the field, which is the point — DeltaNet
([2406.06484](https://arxiv.org/abs/2406.06484)) exists because "the additive update in linear
transformers" is weak at associative recall, and that additive update is precisely BDH's write.

### 3. Structure predicts function: `G*` says which neurons stay silent, from the weights alone

`research/export_field.py`, `research/export_walk.py` · recomputed in-browser by `price.js`

`G* = Dₓᵀ Eᵀ` is the standing neuron-to-neuron synapse matrix. It is computed from the weights
with **no data whatsoever**. Yet on 7 sentences of real French:

- **37.1%** of the 12,288 neurons never fire once (4,561 of them); **38.7%** have no synapse in
  `G*` at the p99 threshold.
- They are nearly the same set: **Matthews correlation +0.944**, P(silent | isolated) **94.5%**
  against a **37.1%** base rate. The confusion matrix is 4,496 / 260 / 65 / 7,467.
- Mean total degree is **0.7** for silent neurons against **97.0** for firing ones.
- Firing is brutally concentrated: the top 1% of neurons account for **19.3%** of all firing.

This is the one picture a Transformer structurally cannot have, and it is why THE FIELD is laid
out on `G*` rather than on an embedding projection. It also explains the dark region in the hero
image: that is the disconnected half of the graph.

*Caveats that travel with it:* the corpus is 7 Europarl-style sentences, so "never fires here" is
not "dead forever"; and "isolated" is defined by the p99 threshold, so the count moves if the
threshold moves.

### 4. Sparsity is learned, not imposed

`research/plot_training.py`, from the training run's own telemetry

Nothing in the objective encourages sparsity — no L1 term, no k-winners-take-all, no threshold.
The only structural pressure is the ReLU. The curve that comes out has a shape worth reading:

![Neuron activity across training: 49.9% at init, collapsing to 3.11%, then climbing back to 5.09% as loss falls](docs/media/sparsity-emergence.png)

- At initialisation **49.9%** of neurons are active — exactly what a ReLU on roughly symmetric
  pre-activations gives you. Half of everything, on for no reason.
- By iteration 2,500 it has collapsed to **3.11%**. The network's first move is to switch most of
  itself off.
- Then it **climbs back, monotonically, to 5.09%** while the loss keeps falling. It *recruits*.
  The end state is not the sparsest one it ever visited; it is the sparsest one it can afford at
  that loss.
- The gate `y = ReLU(Dᵧ a) ⊙ x` tracks the same shape an order of magnitude lower, ending near
  **0.92%** — an AND of two sparse conditions.

The endpoint corroborates a number this repo measures independently. The training probe lands at
5.09% on three sentences through the prior run's own instrumentation; our export pipeline, a
different code path over a 7-sentence corpus and **55,959,552 neuron-token slots**, lands at
**5.129%**. The dashed line on the figure is read from `web/public/big/traces.json` at plot time,
not typed in.

*Caveat:* the training probe is three sentences, not a corpus, and was taken mid-training. It is
evidence about the *shape*. Only the endpoint is independently reproduced.

### 5. The merge experiment fails, and that is not a refutation

`research/merge_replicate.py`

The Dragon Hatchling paper's §7.1 merges two models by concatenating every tensor with an `n`
dimension and averaging the rest. We ran it on our two independently trained specialists:

| Variant | fr probe loss | pt probe loss | max abs logit |
|---|---|---|---|
| French specialist | 0.785 | 2.699 | 1,097 |
| Portuguese specialist | 3.447 | 0.945 | 971 |
| merged, paper recipe (RoPE concatenated) | 1127.6 | 1049.2 | 40,594 |
| merged, RoPE rebuilt at the new width | 1261.6 | 1171.5 | 45,443 |
| control: paper recipe, `E` scaled by ½ | 563.8 | 524.7 | 20,298 |

The merged model emits `iiiiiiii`. It is **not** a RoPE bug — concatenating the frequency buffer
exactly as the paper says still collapses. It is **not** a magnitude bug — halving `E` halves the
logits exactly, halves the loss, and changes nothing about the output.

What is left is a precondition that is easy to miss: the paper's Table 2 merges **clones of a
common base**. Ours were trained independently from scratch, so there is no shared
initialisation. **Concatenation along `n` is structurally well defined, but a neuron's *meaning* is
only shared between models descended from one initialisation.** We tested a harder case than the
paper claims, and we say so on the page.

### 6. Concept selectivity survives a null — the uncontrolled version would not have

`research/export_concepts.py`

The prior run's analysis reported 200 neurons at selectivity 1.0, with no control. A neuron that
fires on five bytes is trivially 100% selective. Recomputed against a **per-neuron permutation
null** (labels shuffled, activations held fixed, 2,000 draws): 7,040 neurons fire at all,
**2,692 beat their own 95th percentile** against ~352 expected by chance (7.65×), and **1,836
survive Benjamini–Hochberg at 5% FDR**. Null p95 runs 0.59–0.93 for the top neurons, which is
exactly why the uncontrolled number meant nothing. The artifact ships ours, and keeps theirs in
[`models/telemetry/`](models/) as the contrast.

## The two models

Neither is an official BDH model, and neither is BDH-CQ.

### 131K cipher BDH — trained here, runs in your browser

Trained by [`research/train.py`](research/train.py) on an in-context substitution cipher
([`research/cipher_task.py`](research/cipher_task.py)) that resamples a **fresh bijection per
example**, so the mapping exists only in the prompt. The only way to answer is to read it out of
the recurrent state. Source symbols render as shapes and targets as colours, which makes the task
a visual miniature of BDH-CQ §6.3 — a fresh colour permutation defined entirely by demonstrations.

Exported to fp16 (`web/public/model.bin`, 257 KB) and re-implemented as a hand-written forward
pass in plain typed arrays ([`web/src/bdh.js`](web/src/bdh.js)) — 20–85 ms per forward, no WebGPU
needed. Parity against PyTorch: **5.2e-7 worst relative logit error** with every argmax agreeing.

Accuracy, trained on k ∈ [2,12] with a 48-symbol alphabet (chance = 2.1%):

| k | 2–12 | 16 | 20 | 24 | 32 | 48 |
|---|---|---|---|---|---|---|
| accuracy | **100%** | 64% | 40% | 20% | 9% | 3% |

**This curve is a demonstration-coverage limit, not a capacity limit.** The same state retrieves
99% at k = 16 when measured directly. Conflating the two is a factual error, and the artifact
teaches them as separate lessons on separate slides.

### 8M translation BDH — prior work, reused with disclosure

An English→French byte-level BDH, 6 layers × D=192 × 4 heads, N=3,072/head → **12,288 neurons,
7,962,624 parameters**, trained on Europarl to **val loss 0.670**. There is an independent
Portuguese sibling and the failed merge of the two.

**These checkpoints were trained by one member of this team at an earlier hackathon**, before
DataForge 2026, and are reused here with disclosure. Everything derived from them in this repo is
new work for this submission. Two deviations from reference BDH are disclosed wherever their
numbers appear: a learned positional embedding on top of RoPE, and no decay term.

Full recipe, file inventory, download instructions and checksums:
**[`models/README.md`](models/README.md)**. The training notebook is committed at
[`models/notebooks/bdh-translation-training.ipynb`](models/notebooks/bdh-translation-training.ipynb);
the weights are GitHub Release assets, because 96 MB files do not belong in a clone.

```bash
gh release download v1.0-models -p 'french_best.pt' -D models/checkpoints/
```

## How it is built

No framework, no build step, no bundler. Every module that touches data is pure and separately
gated before a controller uses it — which is why the science can be tested in Node with no
browser at all.

```
research/                  offline: train, measure, export
  bdh_big.py               the 8M architecture, transcribed from the training notebook
  cipher_task.py           the synthetic in-context task (fresh bijection per example)
  train.py                 trains the 131K model -> runs/final.pt
  export_weights.py        -> web/public/model.{json,bin}   (fp16, 257 KB)
  export_field.py          G*, Louvain, three layouts       -> web/public/big/field.*
  export_traces.py         per-token activations, attention, sigma energy -> big/traces.bin.gz
  export_walk.py           the two walk sentences, every stage -> web/public/walk.bin.gz
  export_concepts.py       concept selectivity against a permutation null
  merge_replicate.py       the neuron-concatenation merge experiment
  verify_equivalence.py    the PyTorch twin of the equivalence gate
  sigma_capacity.py        the 1/pi measurement and its signed counterfactual
  compare_memory.py        KV cache vs sigma, and the N_total/2 crossover
  plot_training.py         the sparsity-emergence figure
  serve.py                 dev server: no-store, and pins .gz to octet-stream
  check_mobile.py          narrow-viewport measurement over the DevTools protocol

web/src/                   pure modules first, then the controllers that use them
  bdh.js                   the 131K forward pass + trace(), plain typed arrays
  bigdata.js walkdata.js field-data.js flow-data.js   pure decode/derive: no DOM, no GL
  capacity.js chart.js sigma-view.js tokens.js palette.js
  field-gl.js              hand-rolled WebGL2, HDR float target + a tone map
  inspector.js             THE FIELD's controller                        (frozen)
  loop.js stages.js flow.js diagram.js                                   THE LOOP
  price.js                                                               THE PRICE

models/                    the 8M checkpoints, notebook and telemetry  (see models/README.md)
docs/                      the one-page summary, the defense sheet, the media and its encoder
```

**The payload is a budget, and it is measured on the wire.** THE FIELD ships 6.37 MB, THE LOOP
4.90 MB, THE PRICE 0.50 MB. Two decisions kept that sane:

- **`traces.bin.gz` is gzipped and the streams compress very unevenly** — measuring that decided
  the design. Sorted uint16 neuron indices compress to **0.43**, the smooth sigma-energy ramp to
  **0.21**, and the uint8 activation values to **0.92**, because they are noise and nothing will
  compress them. Net 0.52, so seven sentences of full data cost 5.89 MB instead of 11.34 MB. The
  browser inflates it with `DecompressionStream`; the manifest declares the compression so there
  is no 404-then-retry.
- **Derivable tensors are not shipped.** `x_pre` was 2.76 MB to say something reconstructible from
  two matrices already in the pack; shipping the 1.18 MB matrix instead made the pack smaller
  *and* the values more accurate (f16-weight accurate at 2.1e-4, rather than uint8-quantised at
  ~4e-3). RoPE is re-derived in the browser from `rope_freqs`, and σ is accumulated from the
  rank-one writes. Each of those is licensed by a gate with a negative control, not by assertion.

**`web/field.html` is frozen and checksummed.** It is finished; `web/test/frozen.mjs` sha256s it
plus the eleven files it loads and runs first in `npm test`. This is why `theme.css` duplicates
field.css's `:root` block and nav rules instead of sharing them — hoisting would have edited a
finished page — and why `wiring.mjs` asserts the two copies stay byte-identical. To change it
deliberately: edit, then `npm run freeze`.

## The test suite, and what each gate caught

`npm test` runs twelve gates and 236 checks in this order. Several of them exist because a
specific bug shipped past everything else.

| Gate | What it protects |
|---|---|
| `frozen` | field.html and its eleven files are byte-identical to the frozen checksums |
| `price_smoke` | boots THE PRICE in jsdom; 77 checks; canvases must paint non-uniform pixels; the evidence ledger's disclosures must be present |
| `field_smoke` | boots THE FIELD on both render paths; readouts must **differ** between sentences |
| `bigdata` | shipped bytes round-trip within 0.5 quantisation steps; re-derives the sparsity and negative-score figures from them |
| `walk` | RoPE re-derivation and σ accumulation, each with a negative control (247× and 580× worse) |
| `loop_smoke` | boots THE LOOP; 74 checks; measures canvas *contrast*; records vector draws; pins the diagram's span |
| `field` | palette monotonic in OKLab L; the pure transforms |
| `validate` | browser JS against PyTorch, on a recorded fixture |
| `equivalence` | parallel↔recurrent, plus the three ablations including the ones that must **not** break |
| `app_logic` | the sixty-second moment, including the `(src, tgt, SEP)` token order |
| `capacity` | the JS capacity port matches PyTorch within 10 pp at every k |
| `wiring` | element ids, generated class names, the shared nav, the door's framing, ≥3 in-window papers cited beside claims |

The honesty layer is enforced as tests, not as a convention: the evidence ledger's seven tiers,
both unreconciled BDH-CQ costs, the co-author disclosure, the ConceptARC training overlap, "no
ARC-AGI-2 result exists", the correction to the problem statement's own half-truth, the section
locators, "R is never stated", the merge's not-a-refutation line, and the checkpoint's two
deviations. **Dropping any one of them fails the build.**

### What actually shipped broken, and the gate that now catches it

| What went wrong | Why every gate stayed green | The gate that catches it now |
|---|---|---|
| A patch dropped a `function` declaration; the module threw on load and the page rendered **completely empty placeholders** | no gate ever executed the entry point — they tested the model, the ports and the markup, and the glue between them | `price_smoke` boots the real page in jsdom and asserts ~80 elements populate |
| Three σ views shipped as **black rectangles** — `forward()` returns one entry per (layer, head), not per token | a canvas stub that swallows every draw cannot tell a heatmap from a black rectangle | the stub now records `putImageData` per canvas and requires ≥3 distinct colours |
| Heatmaps normalised by **max** instead of a percentile; σ's median cell is 4e-4 of its max, so **83% of pixels** landed within 5% of neutral and the panel read as blank grey | "was something painted?" passed — there was variety, all of it within a few units of neutral | `loop_smoke` measures *contrast*: the share of pixels far from the midpoint, requiring >25% |
| Six of seven sentences were **dead on THE FIELD** — views were gated on a flag rather than on whether the data was present | the gates only ever asserted against the hero sentence and never changed the selector | all three field gates sweep all seven sentences and require the readouts to **differ** |
| The next-byte column ranked by `|value|`, so it showed the eight bytes the model most strongly **ruled out**, every one at p = 0.00% | 55 assertions passed; nobody printed what the column would actually say | found by dumping a frame's headers to stdout — now a fixed `topSigned`, and the lesson is in the handoff |
| A stylesheet defined `.chip.big` where the code generates `chip chip-big`, so chips rendered as unstyled boxes | class names were built by string concatenation and never compared to the CSS | `wiring` asserts every generated class name is actually styled |
| **The first gate only passed on one machine.** Four frozen files were CRLF in a Windows working tree and LF in the repository, so the checksums matched nowhere else | nobody had ever run the suite on a clean clone on another platform | `.gitattributes` pins `eol=lf`; the checksums now describe the same bytes everywhere, and CI runs the suite on Linux on every push |

The general lesson, stated once: **a test that never visits the broken case is not testing
anything.** Every row above is an assertion that existed and passed while the product was wrong.

## Reproducing

Nothing here needs a GPU or a network unless you are retraining.

```bash
npm ci
npm test                # 12 gates, 236 checks
npm run dev             # serve the artifact at http://localhost:8080
```

| Command | What it does | Needs |
|---|---|---|
| `npm test` | the full suite | node only |
| `npm run test:mobile` | narrow-viewport measurement at 390 / 360 / 320 px | Edge or Chrome + `websocket-client` |
| `npm run figures` | rebuilds `docs/media/memory-crossover.png` and `sparsity-emergence.png` | python + matplotlib |
| `npm run summary` | rebuilds `docs/concept-summary.{html,pdf}` | python + Edge/Chrome |
| `npm run media` | re-encodes the screen captures from `docs/media/_originals/` | ffmpeg |
| `npm run train` | retrains the 131K model (~65 min) | python + torch + GPU |
| `npm run export` | weights, fixture and verified presets | python + torch |
| `npm run verify:torch` | the PyTorch twin of the equivalence gate | python + torch |
| `npm run export:big` / `export:walk` | re-exports everything the 8M pages read | the released checkpoints |

**After any retrain, run `python research/make_presets.py`.** It re-verifies the sixty-second
moment and prints `SIXTY-SECOND MOMENT HOLDS` or `DOES NOT HOLD`.

## Honest limitations

- The live model is **131K parameters on a synthetic task**. It is an honest miniature, not
  evidence about BDH at scale. The 8M model is real language, but still small.
- The toy runs **~43% active** neurons, nowhere near the ~5% BDH reports at scale. Our 8M model
  *does* reach **5.13%**, measured through our own pipeline — but that is our model, not theirs.
  The BDH paper's own ~5% figure has not been verified by us against the primary source and is
  never quoted as ours.
- **BDH's state is constant but large** — 12× an equal-dimension KV cache at the length this model
  was trained on. See [the crossover](#what-constant-memory-actually-costs).
- σ is **dense**, not sparse ridges: 100% of its cells are non-zero within a few tokens. What is
  legible is Δσ per token, a before/after difference, and row energy — which is what we draw.
- The **8M checkpoints deviate from reference BDH in two ways**: a learned positional embedding on
  top of RoPE, and no decay term.
- The **merge experiment is not a refutation.** The paper's recipe assumes models forked from a
  common base.
- The negative-attention-score share is **per sentence** (14.7%–38.5%, pooled 34.3%), not one
  number.
- The sparsity-emergence curve's intermediate points come from a **three-sentence probe**. Only
  its endpoint is independently reproduced.
- **BDH-CQ figures are replayed, never reproduced.** It has no public weights and no public API.
  We cannot run it and do not claim to.
- Mobile is verified for **layout** at 320–390 px. On phones THE LOOP's flow diagram keeps its
  design width and scrolls sideways, with a note on screen saying so.

## References and tooling

### Primary sources

Cited beside the claims they support, on the pages and in the summary — not collected at the end.

- Kosowski, Uznański, Chorowski, Stamirowska, Bartoszkiewicz (2025). **The Dragon Hatchling: The
  Missing Link between the Transformer and Models of the Brain.**
  [arXiv:2509.26507](https://arxiv.org/abs/2509.26507) — the architecture, and the source of the
  claim that non-negativity buys interpretable sparse structure. That half is theirs; the price we
  put on it is ours. §7.1 is the merge recipe we replicate.
- Engdahl et al. (2026). **BDH-CQ technical report.**
  [arXiv:2608.09888](https://arxiv.org/abs/2608.09888) — Eq. (1) `Sₜ = U_θ(Sₜ₋₁, Dₜ)` and the
  additive special case named in §3.2, which is exactly what the public BDH code computes. That
  correspondence is the bridge the whole artifact rests on, and it does not require running
  BDH-CQ — which is fortunate, because nobody outside Pathway can.
- Yang, Wang, Zhang, Shen, Kim (2024). **Parallelizing Linear Transformers with the Delta Rule
  over Sequence Length.** [arXiv:2406.06484](https://arxiv.org/abs/2406.06484) — replaces "the
  additive update in linear transformers" with the delta rule and reports it more effective at
  associative recall. That additive update is BDH's write; our 1/π ceiling is why the alternative
  exists.
- Sun et al. (2024). **Learning to (Learn at Test Time): RNNs with Expressive Hidden States.**
  [arXiv:2407.04620](https://arxiv.org/abs/2407.04620) — the update rule as a step of
  self-supervised learning; the contrast that places BDH's write in a design space.
- Wang et al. (2025). **Hierarchical Reasoning Model.**
  [arXiv:2506.21734](https://arxiv.org/abs/2506.21734) and Jolicoeur-Martineau (2025). **Less is
  More: Recursive Reasoning with Tiny Networks.**
  [arXiv:2510.04871](https://arxiv.org/abs/2510.04871) — the optimization route to test-time
  adaptation, and the fork this artifact takes a side in. TRM's 8% on ARC-AGI-2 is TRM's result
  and is never attached to BDH-CQ.

[`research/bdh-cq-dossier.md`](research/bdh-cq-dossier.md) is our verbatim extraction of the
BDH-CQ report with a section locator and an evidence level on every claim. It is canonical for
every BDH-CQ number in this repo, and it exists because a plausible-looking BDH-CQ figure that is
wrong is the failure mode nobody catches.

### Design references

The problem statement benchmarks against explanatory instruments rather than write-ups, and these
are the ones the three pages were built against — for what they do, not to imitate them:
[Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (a full architecture
diagram with data visibly flowing through it),
[TensorFlow Playground](https://playground.tensorflow.org/) and
[GAN Lab](https://poloclub.github.io/ganlab/) (the whole page as one live instrument),
[bbycroft's LLM visualisation](https://bbycroft.net/llm) (a walkthrough where the camera moves
with the narrative), [Distill](https://distill.pub/) (figures that respond to the reader),
[CNN Explainer](https://poloclub.github.io/cnn-explainer/) (hover anything, see it highlighted
everywhere else), and [Neuronpedia](https://www.neuronpedia.org/) (browsing individual units).

### Code, data and tools

- [`pathwaycom/bdh`](https://github.com/pathwaycom/bdh) — the reference implementation, vendored
  **unmodified** at [`vendor/bdh.py`](vendor/bdh.py) under MIT.
- [Europarl v7](https://www.statmt.org/europarl/) en–fr and en–pt — the 8M models' training data.
- PyTorch 2.6 + CUDA for training and export; no ML framework in the browser at all — the forward
  pass is hand-written typed arrays.
- jsdom for the page gates; the Chrome DevTools Protocol for the mobile measurement; ffmpeg for
  the media in this README.
- The colour palette follows the validation method in Claude Code's `dataviz` skill reference:
  every categorical pair is checked for contrast under both themes and under simulated colour
  vision deficiency. That check is why heads are *not* colour-coded and only the top three Louvain
  communities get a hue — only three categorical hues clear all pairs on the field surface.

### On building this

**Claude (Opus 5), via Claude Code, was used throughout** — to read and extract the BDH and BDH-CQ
primary sources into the dossier, to design and write the training, analysis and export code, to
write the browser port of BDH, the three pages and the test suite, and to draft this README and
the concept summary. Understanding BDH well enough to build an explanation of it was itself much
of the work, and that reading was done with Claude against the papers rather than from memory.

Direction, review and every design decision are the team's, as is the prior 8M model. The
discipline that matters here is not who typed what: **every empirical claim in this repo is
produced by a script in this repo**, and the team can trace and defend each one —
[`docs/defense.md`](docs/defense.md) is the working aid for exactly that. No number appears
anywhere in the artifact that is not either produced by a script here or quoted from a primary
source with a locator.

## Attribution and licenses

| Component | Source | License |
|---|---|---|
| `vendor/bdh.py` | [pathwaycom/bdh](https://github.com/pathwaycom/bdh) — **unmodified** | MIT (`vendor/LICENSE-bdh`) |
| 131K trained weights, cipher task, browser port | this repo | MIT |
| 8M En→Fr and En→Pt checkpoints | **prior solo hackathon work by one of this team**, reused with disclosure ([`models/README.md`](models/README.md)) | author's own |
| All three pages, exports, tests, docs, figures | this repo | MIT |
| Europarl v7 corpus | statmt.org | as published |
| Colour palette and its validation method | Claude Code `dataviz` skill reference | — |
| Fonts | none bundled — system font stacks only | — |
| Graphics | none bundled — everything is drawn at runtime from data | — |

The screen captures in [`docs/media/`](docs/media/) are recordings of this artifact running; the
originals are re-encoded by [`docs/media/encode.sh`](docs/media/encode.sh) from 138 MB down to
about 14 MB, because a README nobody can load explains nothing.

## Deploying

Static, no build step. The site root is `web/`, and `vercel.json` pins it.

The one header that matters: `public/**.bin.gz` must be served as `application/octet-stream`
**with no `Content-Encoding: gzip`**. Both big packs are inflated *by the page* with
`DecompressionStream`, so a transport-level encoding would hand the loader already-inflated bytes.
After deploying, verify on the live origin — not locally:

```bash
curl -sI <url>/public/big/traces.bin.gz | grep -i "content-type\|content-encoding"
curl -sI <url>/public/walk.bin.gz       | grep -i "content-type\|content-encoding"
```

Verified on the live origin: both packs arrive byte-identical to disk. Vercel applies Brotli on
top when the browser asks for it, which is transparent re-compression, not the mislabelling that
would break the loader.

Deployment Protection must stay **off** for Production — the track requires a URL that opens
without sign-in. Test it in a private window, not just with curl.

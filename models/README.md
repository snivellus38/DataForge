# The 8M translation BDH

Two byte-level BDH language models — English→French and English→Portuguese — plus the merge
experiment on the two, the notebook that trained them, and the telemetry the training run recorded
about itself.

Every "8M model" figure in the artifact traces back to `french_best.pt`: the 5.13% sparsity, the
synapse graph `G*`, the neuron field of 12,288 dots, the flow diagram's edge values, the negative
attention scores. This directory is where those numbers come from.

## What these are, and what they are not

They are **not** an official Pathway model. They are not BDH-CQ, which has no public weights and
cannot be run by anyone outside Pathway. The reference implementation
([`vendor/bdh.py`](../vendor/bdh.py), MIT) is vendored unmodified and is a separate thing again.
These are our own byte-level BDHs, trained on Europarl v7 with the notebook in
[`notebooks/`](notebooks/); [`research/bdh_big.py`](../research/bdh_big.py) transcribes the
architecture so the checkpoints load exactly.

**Two deviations from reference BDH**, inherited from the checkpoint and unfixable without
retraining. They are stated everywhere these models appear on screen:

1. a **learned positional embedding** (`pos_emb`, 4096 × D) added on top of RoPE;
2. **no decay term**.

Everything else is faithful in the respects the artifact depends on: weight-tied across layers,
`Q = K = ReLU(D_x · LN(v))`, `V = LN(v)` so the state is `N × D` and not neuron × neuron, no
softmax anywhere, strictly causal `tril(-1)`, and RoPE applied inside attention so the activations
feeding the multiplicative gate are never rotated.

## Architecture and training

| | |
|---|---|
| Parameters | **7,962,624** — and there are no per-layer weights |
| Layers (`n_layer`) | 6 — an **iteration count of one shared operator**, not depth |
| Residual width `D` | 192 |
| Heads | 4, with `N` = 3,072 neurons each → **12,288 neurons** |
| Vocabulary | 256 — raw bytes, no tokenizer |
| Data | Europarl v7 en–fr and en–pt, interleaved as `<F:en>{source}<T:fr>{target}` |
| Context | 512 bytes |
| Batch | 32 × 2 gradient-accumulation steps = 32,768 tokens/step |
| Iterations | 50,000 (French) / 40,000 (Portuguese) |
| Optimiser | AdamW, lr 1e-3 → 1e-4 cosine, 1,000 warm-up, weight decay 0.1, grad clip 1.0 |
| Hardware | one A100, `torch.compile` on, roughly 45–60 min |
| Final loss | **0.670 val**, 0.648 train (nats/byte, held-out Europarl) |

Weight tying is the part that surprises people, and it is why the parameter count is what it is:
`encoder`, `decoder_x` and `decoder_y` are built once and reused inside `for L in range(n_layer)`,
so the total is `3·H·D·N + 2·vocab·D` with no `n_layer` term anywhere. Six "layers" is the same
operator applied six times. That maps onto BDH-CQ's Eq. (3) `H_{r+1} = F_θ(H_r, S_K)`, and it is
why the artifact can show all six iterations of the *same* weights side by side.

It generates real French, accents and agreement included: `Le Parlement européen a voté contre
cette résolution`, `Je voudrais remercier le rapporteur pour son travail accompli.`, `Merci
beaucoup.` See the README's model section for the honest failure cases alongside these.

## What is here, and what is a download

Committed (about 600 KB — the experimental record, so the numbers are checkable without a
443 MB download):

| Path | What it is | Consumed by |
|---|---|---|
| `notebooks/bdh-translation-training.ipynb` | the training and analysis notebook, as run | reference; `research/bdh_big.py` is transcribed from Cell 2 |
| `telemetry/evolution/evolution_*.json` | loss and sparsity probed every 2,500 iterations | [`research/plot_training.py`](../research/plot_training.py) |
| `telemetry/graph/gstar_head*.json` | the notebook's `G*` edge lists | cross-check against `research/export_field.py` |
| `telemetry/monosemanticity/precomputed.json` | the notebook's concept selectivity — **uncontrolled**, 200 neurons at 1.0 | kept as the contrast for [`research/export_concepts.py`](../research/export_concepts.py), which redoes it against a permutation null |
| `telemetry/synapses/timeline.json` | Hebbian synapse traces | cross-check |
| `telemetry/{sparsity,merge,corpus,meta}` | global stats, merge evaluation, analysis corpus, config | [`research/compare_memory.py`](../research/compare_memory.py) reads `meta.json` |
| `checkpoints/SHA256SUMS` | checksums for the released weights | verification |

Released, not committed — `.pt` files are 60–96 MB each and would sit in every clone forever:

| Asset | Size | Why you would want it |
|---|---|---|
| `french_best.pt` | 96 MB | **the one that matters.** Every `research/export_*.py` defaults to it |
| `portuguese_best.pt` | 96 MB | the merge experiment's second parent |
| `merged_merged.pt` | 60 MB | the merged model that collapses — the negative result's evidence |
| `french_latest.pt`, `portuguese_latest.pt` | 96 MB each | last-iteration rather than best-val; kept for completeness |
| `prior-run-telemetry.tar.gz` | 33 MB | the training run's raw per-token dumps and hero-token frames |

```bash
gh release download v1.0-models -p 'french_best.pt' -D models/checkpoints/
cd models/checkpoints && sha256sum -c SHA256SUMS --ignore-missing
```

The release is created and uploaded by [`publish-release.sh`](publish-release.sh), which needs
`gh auth login` as the repository owner. `SHA256SUMS` is committed, so a download can be verified
against a checksum that travelled by a different route than the file did.

With `french_best.pt` present, the full export pipeline runs:

```bash
npm run export:big     # G*, layouts, per-token traces  -> web/public/big/
npm run export:walk    # the walkthrough packs, concepts, synapses, the merge replication
```

## What was measured from these weights

All of it by scripts in this repo, all reproducible, all with its caveat attached. The README's
"What we measured" section is the long form; in brief:

- **5.13% of neurons active** over 55,959,552 neuron-token slots, and the gate `y` at **0.94%** —
  an AND of two sparse conditions. The training run's own probe independently ends at 5.09%.
- **`G*` = `Dₓᵀ Eᵀ` predicts which neurons stay silent on real text, from the weights alone,
  with no data**: MCC **+0.944**, P(silent | isolated) 94.5% against a 37.1% base rate.
- **34.3% of causal attention scores are negative** pooled over the corpus (14.7%–38.5% per
  sentence). RoPE, not a decay term, is what makes BDH forget.
- **The §7.1 merge was run outside the precondition it assumes, and collapses there.** The recipe
  assumes two models forked from a common base; these two were trained independently from scratch,
  so [`research/merge_replicate.py`](../research/merge_replicate.py) tests a case the paper does
  not claim to cover. Concatenating neurons is structurally well defined but a neuron's *meaning* is
  only shared between models descended from one initialisation. Loss goes 0.785 → 1127.6 and the
  output becomes `iiiiiiii`. Rebuilding RoPE at the new width does not help, and halving `E`
  halves the logits exactly without fixing anything, so it is neither a RoPE bug nor a magnitude
  bug.

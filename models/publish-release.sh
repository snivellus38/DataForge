#!/usr/bin/env bash
# Publish the 8M checkpoints as GitHub Release assets.
#
# WHY A RELEASE AND NOT THE REPOSITORY
#   The five .pt files are 60-96 MB each, 443 MB together. GitHub's hard limit is 100 MB per file,
#   so they would squeak in -- and then sit in every clone of this repository forever, for the
#   sake of readers who mostly do not need them. Git LFS is worse here, not better: the free tier
#   is 1 GB of bandwidth a month, so a handful of clones exhausts it and the pointers start 404ing
#   for everyone. Release assets have no clone cost, no bandwidth cap, and a 2 GB per-file limit.
#
#   Everything needed to CHECK the numbers -- the notebook, the sparsity-evolution curves, the G*
#   graphs, the merge evaluation -- is committed, at about 600 KB. The weights are only needed to
#   re-run the exports.
#
# PREREQUISITES
#   gh auth login          (the GitHub CLI, authenticated as the repository owner)
#   the .pt files present in models/checkpoints/
#
# USAGE
#   bash models/publish-release.sh
set -euo pipefail
cd "$(dirname "$0")/.."

TAG=v1.0-models
CKPT=models/checkpoints

gh auth status >/dev/null 2>&1 || {
  echo "Not authenticated. Run: gh auth login" >&2; exit 1; }

for f in french_best.pt portuguese_best.pt merged_merged.pt french_latest.pt portuguese_latest.pt; do
  [ -f "$CKPT/$f" ] || { echo "missing $CKPT/$f" >&2; exit 1; }
done

# The prior run's raw per-token telemetry: 33 MB of JSON, kept as a cross-check rather than as a
# dependency (our own pipeline reproduces its headline sparsity figure through different code).
TARBALL=models/prior-run-telemetry.tar.gz
if [ ! -f "$TARBALL" ]; then
  tar -czf "$TARBALL" -C models/telemetry telemetry hero_tokens
  echo "packed $TARBALL ($(du -h "$TARBALL" | cut -f1))"
fi

gh release view "$TAG" >/dev/null 2>&1 || gh release create "$TAG" \
  --title "8M BDH translation checkpoints" \
  --notes "$(cat <<'NOTES'
The English->French and English->Portuguese byte-level BDH checkpoints behind every "8M model"
figure in the artifact, plus the merged model that collapses.

**These are prior work**, trained by one member of the team at an earlier Pathway hackathon and
reused in this submission with disclosure. They are not an official Pathway model and they are not
BDH-CQ. Two deviations from reference BDH, disclosed wherever their numbers appear: a learned
positional embedding on top of RoPE, and no decay term.

6 layers x D=192 x 4 heads, N=3,072/head, 12,288 neurons, 7,962,624 parameters, byte vocabulary,
Europarl v7, 512-byte context, val loss 0.670.

| Asset | What it is |
|---|---|
| `french_best.pt` | the one that matters -- every `research/export_*.py` defaults to it |
| `portuguese_best.pt` | the merge experiment's second parent |
| `merged_merged.pt` | the merged model that emits `iiiiiiii` -- the negative result's evidence |
| `french_latest.pt`, `portuguese_latest.pt` | last-iteration rather than best-val |
| `prior-run-telemetry.tar.gz` | the prior run's raw per-token dumps, kept as a cross-check |

```bash
gh release download v1.0-models -p 'french_best.pt' -D models/checkpoints/
cd models/checkpoints && sha256sum -c SHA256SUMS --ignore-missing
```

Recipe, file inventory and provenance: `models/README.md`.
NOTES
)"

gh release upload "$TAG" --clobber \
  "$CKPT/french_best.pt" \
  "$CKPT/portuguese_best.pt" \
  "$CKPT/merged_merged.pt" \
  "$CKPT/french_latest.pt" \
  "$CKPT/portuguese_latest.pt" \
  "$TARBALL"

echo
gh release view "$TAG"

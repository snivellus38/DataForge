#!/bin/sh
# Shrink N until the associative capacity cliff becomes visible.
# The artifact needs a k where the learner can SEE interference, per plan.md "visible state".
for m in 1 2 4; do
  python research/week1_gate.py --steps 2500 --mult $m \
    --out research/runs/sweep_mult$m.json 2>&1 | tail -18
  echo "================================================"
done

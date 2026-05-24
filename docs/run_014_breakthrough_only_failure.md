# Run 014: Breakthrough-Only Failure Test

## Motivation

This run tests the manuscript's central counterfactual: a generator can be selected because it preserves breakthrough behavior, while still failing to preserve the memories needed by dilution, pair separation, or encounter probability.

## Command

```bash
python3 scripts/objective_weight_sensitivity.py \
  --input data/processed/bentheimer_6um_downsample3_trajectories.npz \
  --n-outer-splits 4 \
  --n-repeats 3 \
  --grid-step 0.25 \
  --n-validation-generated 35 \
  --n-test-generated 60 \
  --pair-samples 1000 \
  --contrastive-epochs 300 \
  --contrastive-negative-ratio 6 \
  --hybrid-learned-weight 0.25 \
  --pair-rerank-weight 0.25 \
  --regimes balanced,breakthrough_only \
  --output outputs/bentheimer_6um_downsample3_breakthrough_only_failure.json
```

## Result

The breakthrough-only objective favors velocity memory as the best mean mechanism for arrival. When that same memory is evaluated against the full held-out diagnostic set, it is not universally adequate: relative to the best memory for each observable, it has larger dilution, pair-separation, and encounter errors.

This is the paper point. Breakthrough-only validation identifies a useful arrival memory, but it cannot certify memory adequacy for mixing or particle-pair organization.


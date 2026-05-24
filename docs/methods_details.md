# Methods Details

This note expands the manuscript methods into reproducible implementation details. It should eventually become either a Methods appendix or a computational supplement.

## Trajectory Data

The present prototype uses trajectories generated from a Bentheimer sandstone micro-CT volume because the original 2019 DNS trajectories are no longer available in this workspace. The 6 micrometer source image was downsampled by a factor of 3 to a 75^3 voxel grid. Pore space was segmented by an Otsu threshold, restricted to the inlet-outlet connected pore network, and used to compute an approximate pressure field by a graph-Laplace solve. Particle trajectories were then generated with advective-diffusive tracking on the voxel velocity field.

Important caveat: this is a bootstrap trajectory generator, not the final pore-scale solver. It is suitable for developing and testing the TTA-v2 methodology, but manuscript-quality physical validation should eventually use direct numerical simulation, lattice Boltzmann, OpenFOAM, or experimentally validated velocity fields.

Main processed file:

```text
data/processed/bentheimer_6um_downsample3_trajectories.npz
```

## Segment Archive

Each trajectory is split into overlapping fixed-length segments. For each segment, the implementation stores:

- the absolute path,
- the relative path with the first point removed as an origin,
- start and end velocity estimates,
- segment length and matching-window metadata.

The main runs used:

```text
segment_steps = 36
match_steps   = 20
dt            = 1.0
```

When fit on 210 training trajectories, this produces 6300 archive segments.

## Samplers

### Unconditional

The unconditional sampler draws the next segment uniformly from the archive. It is a memory-destroying baseline.

### kNN Conditional

The kNN sampler computes squared distance between the current segment end velocity and each candidate segment start velocity. It samples from the nearest candidates with a temperature-weighted softmax.

Main parameters:

```text
k           = 96
temperature = 0.8
```

### Gaussian/Bayes

The Gaussian/Bayes sampler is the closest implementation of the original training-trajectory transition kernel. It treats mismatch between end velocity and candidate start velocity as plausible if the mismatch can be explained by diffusion over the matching interval. Candidate likelihoods are Gaussian in velocity mismatch with a diffusive bandwidth.

Main parameters:

```text
diffusivity          = 0.001
bandwidth_multiplier = 0.25
candidate_limit      = 256
```

### Hybrid Contrastive/Gaussian

The hybrid sampler trains a contrastive transition model on observed adjacent segment pairs and negative pairs. The learned score is added to the Gaussian/Bayes score with a small learned weight, preserving the physics kernel while allowing the data to reshape transition probabilities.

Main parameters in the more expensive runs:

```text
negative_ratio       = 6
contrastive_epochs   = 400 to 700
hybrid_learned_weight = 0.25
```

### Pair-Aware Reranking

The pair-aware reranker begins from Gaussian/Bayes candidates and modifies candidate weights using archive descriptors intended to preserve short-horizon pair-separation behavior. Current results suggest that this proxy is not yet robust enough to win as a standalone sampler, but it remains useful as a component in selected mixtures.

### Mixture Sampler

The mixture sampler combines component transition distributions:

```text
p(next | state) = sum_i w_i p_i(next | state)
```

All components share the same segment archive. The weights are selected on held-out validation trajectories.

## Metrics

The evaluator computes four metric families.

### Breakthrough

For each control plane, the first crossing time is recorded for each trajectory. The error score combines quantile mismatch and a coverage deficit penalty so a sampler is not rewarded for failing to reach downstream planes.

### Dilution

Particle positions are binned at selected time indices. The dilution index is compared in log space.

### Pair Separation

Pairs of particles are sampled from the ensemble. Separation quantiles are compared over selected time indices.

### Reaction Encounter Proxy

Pairs are counted as encountering if their separation falls below a fixed reaction radius by a selected maximum time index. This is not a full reactive transport model, but it is a useful proxy for whether generated trajectories preserve encounter opportunities.

Default objective:

```text
objective = 1 * BTC score
          + 20 * pair MAE
          + 120 * dilution log MAE
          + 1000 * reaction absolute error
```

## Validation Workflows

### Run 004: Repeated Validation on One Test Split

One test split is held fixed. The remaining trajectories are repeatedly split into fit and validation subsets. The best mixture weights from each validation split are averaged, then components are refit on the full non-test pool and evaluated on the held-out test set.

### Run 005: Outer-Split Robustness

The full Run 004 workflow is repeated over multiple outer test splits. This evaluates whether a single held-out result is robust to test-set resampling.

### Run 006: Objective-Weight Sensitivity

The inner validation errors are reused to select mixture weights under several objective-weight regimes. This tests whether conclusions depend on the chosen scalarization of BTC, pair, dilution, and reaction metrics.

## Reproducibility Commands

```bash
python3 scripts/bootstrap_mixture_selection.py \
  --input data/processed/bentheimer_6um_downsample3_trajectories.npz
```

```bash
python3 scripts/outer_split_mixture_benchmark.py \
  --input data/processed/bentheimer_6um_downsample3_trajectories.npz
```

```bash
python3 scripts/objective_weight_sensitivity.py \
  --input data/processed/bentheimer_6um_downsample3_trajectories.npz
```

```bash
python3 scripts/make_selection_figures.py
```


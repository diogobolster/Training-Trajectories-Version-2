# Experiment Plan

## Experiment 1: Baseline TTA Rebuild

Goal: reproduce the essential cut-copy-paste behavior on synthetic data, then real DNS trajectories.

Inputs:

- trajectories shaped `(n_particles, n_steps, d)` or a list of `(n_steps, d)` arrays,
- segment length `lambda_v` represented first as a number of steps,
- matching length `lambda_alpha` represented first as a number of steps.

Outputs:

- generated trajectories of arbitrary length,
- BTC comparison at several control planes,
- velocity autocorrelation comparison,
- path smoothness diagnostics.

Success criteria:

- conditional sampling outperforms unconditional segment shuffling,
- generated paths have no obvious velocity discontinuities,
- BTC medians and tails are within useful tolerance on held-out synthetic/reference trajectories.

## Experiment 2: State-Dependent Memory

Hypothesis: slow trajectories require larger effective memory lengths than fast trajectories.

Implementation:

- estimate local speed quantiles,
- build multiple archives by speed class or latent state,
- allow `lambda_v` and `lambda_alpha` to vary by current state.

Metrics:

- late-time BTC tail,
- residence-time distribution,
- velocity autocorrelation by speed class.

Expected win:

- improved late arrivals without degrading early arrivals.

## Experiment 3: Learned Transition Kernel

Hypothesis: a learned transition score can outperform the original Gaussian velocity mismatch likelihood.

Simple version:

- feature vector for current segment end: velocity, acceleration proxy, speed quantile, recent displacement direction,
- feature vector for candidate segment start,
- train a contrastive model to score true observed continuations above random continuations.

Baselines:

- unconditional sampling,
- original Gaussian conditional sampling,
- hand-engineered kNN in velocity space.

Metrics:

- BTC,
- velocity autocorrelation,
- smoothness penalty,
- pair separation distribution.

## Experiment 4: Generative Segment Model

Hypothesis: generating segments, rather than retrieving them, improves interpolation across sparse archives.

Candidates:

- conditional diffusion over trajectory increments,
- conditional normalizing flow over segment increments,
- transformer decoder over increment sequences.

Conditioning:

- endpoint velocity and recent history,
- Peclet number,
- local geometry descriptor if available.

Important constraint:

- generated segments must obey physical continuity at the paste point.

## Experiment 5: Geometry Generalization

Hypothesis: geometry-conditioned TTA-v2 generalizes across statistically related porous media.

Train/test splits:

- train on one rock, test on another similar rock,
- train across several rocks, leave one out,
- train across Peclet numbers, interpolate/extrapolate.

Geometry descriptors:

- voxel image embedding,
- pore-network graph embedding,
- local porosity/tortuosity/connectivity features,
- local velocity-field features.

Success criteria:

- acceptable held-out BTCs,
- held-out pair separation/reaction potential,
- calibrated uncertainty when geometry is out of distribution.

## Metrics To Implement Early

- breakthrough time quantiles at control planes,
- longitudinal and transverse displacement moments,
- velocity autocorrelation as a function of lag,
- distribution of speed persistence times,
- dilution proxy from spatial occupancy entropy,
- pair separation density,
- reaction encounter probability below a distance threshold.


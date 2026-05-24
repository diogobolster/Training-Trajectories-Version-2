# Start Here

## One-Sentence Idea

Use resolved Lagrangian trajectories as the primitive training data for a physics-constrained generative model that can produce long, realistic, non-Fickian particle paths in complex porous media.

## Why This Is Worth Reopening

The 2019 TTA paper already had the right abstraction: do not reduce transport to a fitted transition matrix if the data already contains the high-order dependence structure. The method was a generative archive model before generative modeling became the obvious language.

What has changed since then:

- Conditional generative models are now strong enough to learn distributions over trajectory continuations.
- Diffusion and score-based models are natural fits for stochastic dynamics and rare-event statistics.
- Neural operators give a way to condition transport surrogates on geometry, parameters, and boundary conditions.
- Approximate nearest-neighbor search and learned embeddings make large segment archives practical.
- Scientific ML culture now values benchmark suites and hybrid physics-ML models, which fits TTA unusually well.

## Core Research Question

Can a learned, physics-constrained training-trajectory model generate pore-scale and upscaled Lagrangian transport trajectories that preserve:

- breakthrough curves,
- velocity autocorrelation and memory,
- dilution and scalar dissipation,
- pair separation statistics,
- reaction encounter probabilities,
- generalization across porous geometries and Peclet regimes?

## TTA-v2 Model Ladder

### Level 0: Original Baseline

Reimplement cut, conditional copy, paste with the original Gaussian/Bayes transition kernel.

Purpose: reproduce the 2019 method cleanly and define the reference floor.

### Level 1: Smarter Archive

Keep segment resampling, but replace brute-force conditional search with:

- learned or engineered segment embeddings,
- k-nearest-neighbor candidate filtering,
- state-dependent transition bandwidth,
- optional archive uncertainty diagnostics.

Purpose: make the old method faster and more robust without changing its character.

### Level 2: Learned Transition Kernel

Learn the transition probability

```text
p(next segment | current endpoint state, short history, Pe, local geometry)
```

with a neural classifier, energy model, mixture-density network, or contrastive embedding model.

Purpose: remove hand-tuned Gaussian likelihoods while preserving physical segment assembly.

### Level 3: Generative Segment Model

Generate new segments instead of only retrieving old ones:

```text
next segment ~ G(current state, latent transport state, geometry, Pe)
```

Candidate families:

- conditional diffusion model,
- normalizing flow,
- variational sequence model,
- transformer decoder over increments.

Purpose: interpolate between observed trajectory motifs and reduce dependence on archive density.

### Level 4: Geometry-Conditioned Transport Model

Condition the model on pore geometry, velocity fields, or pore-network graphs.

Purpose: move from "works on this rock" to "generalizes across rocks."

## First Sprint

1. Build the baseline archive and conditional sampler on synthetic trajectories.
2. Define metrics that mirror the 2019 paper: BTCs, velocity autocorrelation, dilution proxy, pair separation.
3. Swap in real DNS trajectories when available.
4. Compare original TTA, smarter kNN TTA, and one learned transition model.
5. Write the first internal note around one clear claim: learned transition kernels improve tailing and pair statistics without sacrificing physical smoothness.

## First Real Data Needed

Minimum useful dataset:

- particle trajectories as arrays of positions over time or arc length,
- corresponding advective velocities if available,
- diffusion coefficient and Peclet number,
- geometry or sample identifier,
- control-plane locations for BTC comparison.

Current local status:

- Core1 Subvol1 Bentheimer trajectories at three diffusivities (`D = 0.0003`, `0.001`, `0.003`),
- Core2 Subvol1 Bentheimer trajectories at `D = 0.001`,
- Core2 Subvol1 OpenFOAM finite-volume velocity field and OpenFOAM-derived trajectories.
- Core2 graph-flow versus OpenFOAM physics comparison in `docs/run_012_core2_flow_physics_comparison.md`,
- master evidence table in `docs/master_evidence_table.md`.
- tightened thesis-first manuscript draft in `docs/manuscript_v2_tightened.md`.

Nice-to-have:

- multiple Peclet numbers,
- multiple statistically similar rocks,
- train/test split by geometry,
- particle-pair initial separations.

## Working Name

**Trajectory Foundation Models for Non-Fickian Transport**

Conservative version:

**Physics-Constrained Generative Training Trajectories for Upscaled Advective-Diffusive Transport**

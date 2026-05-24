# Paper Outline

## Working Title

Physics-Constrained Generative Training Trajectories for Non-Fickian Transport in Porous Media

## Central Claim

Resolved Lagrangian trajectories can be used as a physically structured training set for modern conditional generative models, yielding long synthetic trajectories that preserve non-Fickian memory, dilution, and reaction-relevant pair statistics more faithfully than transition-matrix or hand-kernel spatial Markov models.

## Abstract Skeleton

Non-Fickian transport in porous media arises from persistent velocity and direction memory induced by complex pore-scale flow. Existing spatial Markov models encode this memory through transition matrices or hand-designed transition kernels, which become difficult to parameterize in three-dimensional finite-Peclet settings. Building on the training-trajectory approach of Most et al. (2019), we introduce a physics-constrained generative framework that treats resolved particle trajectories as the fundamental training data. Trajectories are decomposed into local segments, embedded into a transport-state space, and assembled through learned conditional transition rules that enforce smoothness and physical continuity. We evaluate the method against direct numerical simulation using breakthrough curves, velocity autocorrelation, dilution, pair separation, and reaction encounter statistics. The proposed framework improves late-time tailing and particle-pair metrics while preserving interpretability and providing a path toward geometry-conditioned upscaling.

## Narrative Arc

1. Non-Fickian transport is a memory problem.
2. The 2019 TTA solved the right problem with a nonparametric archive, but relied on hand-built transition kernels and fixed memory scales.
3. Modern ML can learn the transition structure while keeping the physical trajectory segment as the unit of generation.
4. The correct benchmark is not only BTCs; it is also dilution, pair separation, and reaction potential.
5. Naive learned transition rules can improve one metric while damaging another; in particular, learned context can improve particle-pair structure while trading off BTC fidelity.
6. Therefore, the right ML framing is validation-driven, multi-objective mixture selection rather than wholesale replacement of the physics-informed transition kernel.
7. Repeated validation stabilizes that selection: in Run 004, the averaged physics/learned mixture beats each individual component on a held-out multi-objective transport score.
8. Outer-split robustness is more nuanced: in Run 005, the pooled validation mixture has the best mean objective and beats Gaussian/Bayes and hybrid on 3 of 5 held-out test splits, but Gaussian/Bayes remains highly competitive.
9. Objective-weight sensitivity is essential: in Run 006, changing the metric priorities changes the preferred sampler, with Gaussian/Bayes most stable on mean score and learned/mixture samplers winning selected priorities and splits.
10. A hybrid physics-ML TTA-v2 is more defensible than a black-box neural PDE surrogate because it operates directly on Lagrangian transport objects and makes metric tradeoffs visible.

## Candidate Figures

1. Concept figure: original TTA versus TTA-v2.
2. Segment archive embedding colored by speed, residence time, or pore-region class.
3. Conditional transition examples showing physically plausible continuations.
4. BTC comparison across DNS, original TTA, learned TTA.
5. Dilution and pair separation benchmarks.
6. Metric tradeoff figure showing Gaussian/Bayes versus learned/contextual variants.
7. Validation-selected sampler mixture weights and held-out performance.
8. Objective-weight sensitivity or Pareto-front figure.
9. Outer-split robustness and objective-sensitivity figures from Run 007.
10. Generalization across Peclet number or geometry.

## Likely Reviewer Concerns

- Is this just data augmentation?
- Does it obey physics at segment interfaces?
- Does it generalize beyond the training rock?
- How much DNS data is needed?
- Why not use a neural operator directly?
- Are rare slow trajectories captured or hallucinated?

## Planned Responses

- The model is a Lagrangian stochastic closure, not merely augmentation.
- Interface constraints are enforced explicitly and audited with smoothness metrics.
- Geometry and Peclet conditioning are tested with held-out splits.
- Data requirements are studied through archive convergence curves.
- Neural operators solve complementary Eulerian surrogate problems; this model targets Lagrangian memory and reaction-relevant particle statistics.
- Tail and rare-event behavior is benchmarked directly.

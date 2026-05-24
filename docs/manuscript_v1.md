# Physics-Constrained Generative Training Trajectories for Non-Fickian Transport in Porous Media

## Authors

Sebastian Most, Diogo Bolster, Branko Bijeljic, Wolfgang Nowak, and collaborators to be determined

## Abstract

Non-Fickian transport in porous media reflects persistent memory in the Lagrangian motion of solute particles through complex pore-scale flow fields. Continuous time random walks and spatial Markov models have provided powerful descriptions of anomalous breakthrough and dispersion, but they often require reduced state spaces, hand-designed transition rules, or asymptotic assumptions that become difficult to defend for finite-Peclet, three-dimensional, advective-diffusive transport. Building on the training-trajectory approach of Most et al. (2019), we revisit resolved particle trajectories as the primitive data object for a physics-constrained generative model. Trajectories are decomposed into short segments, archived, and reassembled into long synthetic paths through transition rules that enforce local kinematic continuity. Rather than replacing the original Gaussian/Bayes transition kernel with a black-box neural sampler, we introduce a validation-driven mixture framework that combines physics-informed and learned transition samplers according to held-out transport objectives. We evaluate candidate samplers on breakthrough curves, dilution, pair separation, and reaction-encounter proxies using trajectories generated from segmented Bentheimer sandstone volumes, including a second subvolume and an OpenFOAM finite-volume velocity field. Repeated validation shows that selected physics/learned mixtures can outperform individual components on held-out multi-objective scores, while outer-split, objective-weight, Peclet-regime, geometry, and flow-fidelity tests show that no sampler dominates universally. The central result is methodological: modern machine learning is most useful here not as a wholesale replacement for the physics kernel, but as a way to expose, validate, and navigate tradeoffs among transport metrics that matter for non-Fickian upscaling.

## 1. Introduction

Solute transport in porous and fractured media is often non-Fickian: breakthrough curves exhibit early arrival and late-time tailing, plume moments need not scale linearly with time, and particle velocities retain memory of the pore-scale structures they have sampled. These behaviors are not pathological edge cases. They have been observed and modeled across laboratory, field, and pore-scale settings, motivating alternatives to the classical advection-dispersion equation. Continuous time random walks (CTRW) provide one influential framework for representing anomalous waiting times and broad transition statistics [Berkowitz2006]. Spatial Markov and correlated CTRW models pushed this view further by recognizing that Lagrangian velocity histories may be more naturally Markovian in distance than in time [LeBorgne2008a; LeBorgne2008b; Dentz2016; Sherman2021].

The attraction of these approaches is that they begin from particles rather than from an Eulerian concentration closure. In heterogeneous flow, the Lagrangian velocity sequence carries information about channeling, trapping, tortuosity, and exchange between fast and slow regions. Pore-scale studies have shown that pre-asymptotic dispersion can persist over experimentally relevant distances, even in media that are statistically homogeneous at larger scales [Bijeljic2006]. In three-dimensional porous media, the relevant memory is not only a scalar velocity correlation; it may include direction changes, diffusive exchange across streamlines, local stretching, particle-pair separation, and the reaction-relevant proximity of initially distinct particles. A model that reproduces only the breakthrough curve can therefore be misleading if it fails to reproduce dilution, mixing, or pair statistics.

Most et al. (2019) proposed a particularly direct way to preserve this Lagrangian information. Instead of estimating a transition matrix over a reduced set of states, they treated highly resolved particle trajectories as "training images" for transport. The method cuts direct numerical simulation trajectories into segments, conditionally copies new segments based on velocity continuity and diffusive plausibility, and pastes them into long synthetic trajectories. The original paper framed this as a spatial Markov model designed to avoid three common simplifications: finite-Peclet three-dimensional transport should not require dimensionality reduction or neglect of diffusion; dependence should not have to be parameterized by a high-dimensional transition matrix; and generated transport should retain the resolution needed for mixing and reaction metrics [Most2019].

That idea was ahead of its time. In current language, the 2019 method is a nonparametric, physics-constrained generative model over Lagrangian trajectory segments. What has changed since then is not the transport problem, but the surrounding computational ecosystem. Physics-informed machine learning has matured as a way to constrain learning by differential equations, conservation laws, symmetries, and measurement operators [Karniadakis2021; Raissi2019]. Neural operators, including Fourier neural operators, have shown how learned maps can approximate families of PDE solutions and handle parametric input fields such as Darcy-flow coefficients [Li2021]. Score-based and diffusion generative models have normalized the idea of learning stochastic data distributions through noise-conditioned transitions [Song2021]. In porous-media science specifically, machine learning is now used for image-based prediction, surrogate modeling, and generative reconstruction of pore geometries, from early GAN-based three-dimensional reconstructions [Mosser2017] to recent diffusion models for multiphase pore-scale images [Zhu2025] and broader reviews of data-driven flow and transport methods [Yang2024].

These advances do not imply that a black-box neural simulator should replace a physically interpretable Lagrangian closure. For non-Fickian transport, the essential object is not merely a concentration field or a pressure solution; it is the correlated sequence of particle displacements that determines arrival times, spreading, dilution, and reaction opportunity. A neural PDE surrogate may accelerate an Eulerian solve, but it does not automatically preserve the particle-level memory needed for reactive upscaling. Conversely, the training-trajectory framework already encodes useful invariants: segments come from resolved trajectories, interfaces are constrained by velocity continuity and finite-diffusion tolerance, and generated paths can be evaluated with the same metrics as direct particle tracking.

The question we ask here is therefore deliberately narrower and more testable than "Can machine learning simulate transport?" We ask whether modern learned transition rules can complement the original physics-informed training-trajectory sampler while preserving transparent validation against multiple transport objectives. Our answer is nuanced. Learned and contextual samplers improve some metrics on some splits but can degrade others. Validation-selected mixtures are competitive with the strongest hand-designed Gaussian/Bayes kernel and sometimes outperform it, but neither learned samplers nor mixtures dominate universally. This motivates our central claim: TTA-v2 should be framed as validation-driven physics/ML mixture selection, not as black-box replacement of the original transition kernel.

The contributions are:

1. We recast training trajectories as a physics-constrained generative modeling problem over Lagrangian path segments.
2. We implement a reproducible baseline ladder including unconditional resampling, k-nearest-neighbor conditional resampling, the original Gaussian/Bayes transition kernel, learned contrastive hybrids, pair-aware reranking, and sampler mixtures.
3. We evaluate samplers using breakthrough, dilution, pair-separation, and reaction-encounter metrics rather than breakthrough curves alone.
4. We introduce validation-driven mixture selection and show that repeated validation can identify physics/learned mixtures that are competitive with, and sometimes better than, the strongest fixed kernel.
5. We show through outer-split and objective-weight sensitivity that the scientifically honest result is not universal dominance but an exposed Pareto tradeoff among transport objectives.
6. We test the framework across Peclet regime, a second Bentheimer subvolume, and an OpenFOAM-derived finite-volume velocity field.

## 2. Methods

### 2.1 Training-Trajectory Archive

Let a resolved particle trajectory be a sequence

```text
x_i(t_0), x_i(t_1), ..., x_i(t_T)
```

in two or three spatial dimensions. Following the training-trajectory idea, each trajectory is divided into overlapping segments of length `segment_steps`. For each archived segment, we store the relative path, the initial and final positions, and estimates of the starting and ending velocities over a shorter `match_steps` window. The archive is therefore a set of local transport motifs that retain the spatial resolution of the direct particle tracking simulation.

Generation begins from an archive segment and recursively chooses a next segment. The candidate is shifted so that its matching point joins the previous endpoint, and the overlapping portion is discarded to avoid duplicating time samples. This cut-copy-paste construction preserves observed segment shapes while allowing arbitrary-length synthetic trajectories.

### 2.2 Transition Samplers

We evaluate a sampler ladder.

The unconditional sampler draws the next segment uniformly from the archive and provides a memory-destroying baseline. The kNN conditional sampler selects candidate starts whose initial velocity is near the current ending velocity and samples using a distance-weighted distribution. The Gaussian/Bayes sampler follows the original TTA logic: it interprets a velocity mismatch across a segment interface as plausible if it can be explained by diffusion over the matching interval, yielding a physics-informed transition likelihood with bandwidth proportional to a diffusive velocity scale.

The learned hybrid sampler trains a contrastive transition scorer from observed adjacent archive segments and negative samples. In the present prototype, this learned score is combined with the Gaussian/Bayes likelihood rather than used alone. The pair-aware reranking sampler modifies Gaussian/Bayes candidates using archive descriptors designed to preserve short-horizon pair behavior. Finally, a mixture sampler combines component transition distributions:

```text
p(next | state) = sum_i w_i p_i(next | state)
```

where the nonnegative weights are selected by validation.

### 2.3 Validation-Driven Mixture Selection

For a fixed archive and a fixed set of component samplers, we search a simplex grid over mixture weights. Each candidate mixture generates an ensemble from training-origin initial positions. Its metrics are compared with a held-out validation ensemble. The objective is a weighted score:

```text
J = a_BTC * E_BTC
  + a_pair * E_pair
  + a_dilution * E_dilution
  + a_reaction * E_reaction
```

where `E_BTC` is a breakthrough quantile and coverage score, `E_pair` is a pair-separation quantile error, `E_dilution` is a log-error in dilution index, and `E_reaction` is an absolute error in encounter probability.

We evaluate two aggregation rules. The bootstrap-mean mixture averages the best weights selected across repeated inner validation splits. The pooled-validation mixture chooses the grid point with the best mean validation score across repeats.

### 2.4 Data and Particle Tracking

The current prototype uses publicly available Bentheimer sandstone micro-CT volumes because the original DNS trajectory set is not available in this workspace. The initial workflow thresholds pore space, extracts the inlet-outlet connected pore network, solves an approximate graph-Laplace pressure problem, computes voxel-scale velocities, and tracks advective-diffusive particles. This graph-flow trajectory generator is used for reproducible method development. We then add two validation extensions: a second Bentheimer subvolume, and an OpenFOAM finite-volume solve on the connected pore voxels of that second subvolume.

The main benchmark uses a 6 micrometer Bentheimer volume downsampled by a factor of 3 to a 75^3 grid. The first trajectory set contains 300 particle paths; subsequent Peclet, second-geometry, and OpenFOAM runs use 500 paths. Unless stated otherwise, archives use `segment_steps = 36` and `match_steps = 20`.

For the OpenFOAM case, each connected pore voxel is exported as one hexahedral finite-volume cell. Pore-solid faces are no-slip walls, and inlet/outlet faces are fixed kinematic pressure patches. The Core2 voxel case contains 98,270 cells, passes `checkMesh`, and converges with `simpleFoam` in 103 SIMPLE iterations. The resulting velocity field is mapped back onto the connected voxel mask and normalized to the same mean advective speed as the graph-flow trajectory sets before particle tracking.

### 2.5 Evaluation Metrics

Generated ensembles are evaluated against held-out reference trajectories using breakthrough curves at multiple control planes, dilution index over selected time indices, particle-pair separation quantiles, and reaction-encounter probability for a fixed encounter radius. The pair and reaction metrics are essential because a sampler can match arrival statistics while corrupting spatial organization, mixing, or encounter structure.

## 3. Results

### 3.1 Repeated Validation Can Find Useful Physics/ML Mixtures

In the first selection experiment, one held-out test set was fixed and the remaining trajectories were repeatedly split into fit and validation subsets. Averaging the validation-selected weights produced a mixture with substantial Gaussian/Bayes, kNN, and hybrid learned contributions:

```text
gaussian_bayes:    0.35
knn_conditional:   0.25
hybrid:            0.35
pair_rerank:       0.05
```

On that held-out split, the bootstrap-mean mixture achieved the best multi-objective score, outperforming both the pure hybrid sampler and the Gaussian/Bayes baseline:

```text
sampler                    objective  btc_score  pair_mae  dilution_log  reaction_abs
bootstrap_mean_mixture         84.91      36.64      1.39         0.065         0.013
hybrid                         94.40      43.69      1.42         0.070         0.014
gaussian_bayes                 98.86      41.77      1.68         0.096         0.012
```

This result demonstrates that repeated validation can combine complementary transition rules into a better held-out generator. It is not by itself enough to claim universal improvement, so we next tested robustness over multiple outer test splits.

### 3.2 Outer-Split Robustness Shows Competitive, Not Universal, Improvement

The full selection workflow was repeated over five independent held-out test splits. The pooled-validation mixture had the best mean objective, tied Gaussian/Bayes on mean rank, and beat both Gaussian/Bayes and hybrid on three of five splits (Figure 1). Gaussian/Bayes remained extremely competitive.

```text
sampler                    mean_obj   std_obj  mean_rank  wins  beats_g  beats_h
pooled_validation_mixture    121.64     47.75       2.00     2        3        3
gaussian_bayes               125.48     50.22       2.00     2        0        4
hybrid                       127.71     57.25       3.20     1        1        0
bootstrap_mean_mixture       132.05     50.44       3.20     0        1        3
knn_conditional              142.47     38.49       4.80     0        0        1
pair_rerank                  167.72     60.16       5.80     0        0        0
```

The important result is therefore not that a learned mixture always wins. It does not. The robust result is that validation-selected mixtures are competitive with the strongest hand-designed kernel and often outperform it on held-out multi-objective scores.

![Figure 1. Outer-split robustness](../figures/run_005_outer_split_summary.svg)

**Figure 1. Outer-split robustness of validation-selected samplers.** Mean held-out multi-objective transport error over five independent outer train/test splits. Error bars show the standard deviation across outer splits. Labels above each bar show mean rank and number of split wins. Bottom labels report how often each sampler beats Gaussian/Bayes (`G`) and hybrid (`H`).

### 3.3 Objective Weights Change the Preferred Sampler

A single scalar objective is convenient, but it can hide scientific priorities. We therefore repeated selection under seven objective-weight regimes. Gaussian/Bayes was the most stable mean performer in this faster four-split sensitivity sweep, but the preferred sampler changed with the metric priorities (Figure 2):

```text
regime           best_mean_sampler            mean_obj  mean_rank  wins
balanced         gaussian_bayes                 145.26       2.25     1
btc_heavy        gaussian_bayes                 188.73       2.25     0
pair_heavy       knn_conditional                204.41       2.75     0
dilution_heavy   hybrid                         170.07       2.50     2
reaction_light   gaussian_bayes                 140.53       2.25     1
reaction_heavy   gaussian_bayes                  85.75       1.75     1
no_reaction      gaussian_bayes                 140.01       2.00     1
```

The selected pooled mixture weights also shifted with the objective (Figure 3). Pair-, dilution-, and reaction-sensitive objectives generally moved weight toward the learned hybrid component, while balanced and no-reaction objectives retained substantial Gaussian/Bayes mass. This supports the central methodological claim: validation-driven mixture selection exposes tradeoffs that would be hidden by reporting only one metric or one arbitrary objective.

![Figure 2. Objective-weight sensitivity](../figures/run_006_weight_sensitivity_heatmap.svg)

**Figure 2. Objective-weight sensitivity.** Mean held-out rank of each sampler under seven multi-objective scoring regimes. Rows correspond to objective-weight choices and columns correspond to candidate samplers. Green cells indicate lower mean rank, red cells indicate higher mean rank.

![Figure 3. Objective-dependent mixture weights](../figures/run_006_selected_weights.svg)

**Figure 3. Objective-dependent mixture weights.** Pooled-validation mixture weights selected under each objective-weight regime. Each stacked bar decomposes the selected transition distribution into Gaussian/Bayes, kNN conditional, hybrid learned, and pair-aware reranking components.

### 3.4 Metric Tradeoffs Are Visible in Pareto Space

The balanced objective can be unpacked into individual metric errors. Figure 4 shows a Pareto-style view in which each sampler is represented by its mean breakthrough error, pair-separation error, and dilution error. No sampler simultaneously minimizes all metrics. Gaussian/Bayes has strong breakthrough performance, hybrid improves some pair/dilution behavior, and selected mixtures occupy intermediate tradeoff regions. This is why model selection should remain multi-objective.

![Figure 4. Balanced-objective Pareto tradeoff](../figures/run_006_pareto_tradeoff.svg)

**Figure 4. Balanced-objective Pareto tradeoff.** Each point is averaged over four outer held-out splits. The x-axis is breakthrough-curve error, the y-axis is pair-separation error, and marker size indicates dilution log-error. Lower-left is better.

### 3.5 Peclet-Regime Sensitivity

To test whether validation-driven selection is tied to one transport condition, we regenerated trajectories on the same Bentheimer geometry at two additional diffusivities: a high-Peclet case (`D = 0.0003`) and a low-Peclet case (`D = 0.003`), compared with the baseline (`D = 0.001`). These runs used the same segmentation and approximate pressure field but new advective-diffusive particle tracking ensembles with 500 trajectories. The outer-split benchmark was repeated with four outer splits and three inner validation repeats for the new Peclet regimes.

The selected weights changed systematically with diffusivity:

```text
condition          Gaussian/Bayes   kNN      hybrid   pair
D = 0.0003             0.4375      0.1875   0.3333   0.0417
D = 0.0010             0.4125      0.1500   0.4125   0.0250
D = 0.0030             0.2708      0.0000   0.7083   0.0208
```

At high Peclet, the pooled-validation mixture remained the best mean-objective sampler and kNN conditional matching became much more competitive. At low Peclet, the hybrid sampler became the best mean-objective sampler:

```text
D = 0.0003: pooled mixture mean rank 2.00; kNN mean rank 2.50
D = 0.0010: pooled mixture mean rank 2.00; Gaussian/Bayes mean rank 2.00
D = 0.0030: hybrid mean rank 2.50; Gaussian/Bayes mean rank 2.50
```

This is physically interpretable. At lower diffusivity, local velocity continuity and nearest-neighbor velocity matching carry more information about future displacement. As diffusion increases, strict velocity matching becomes less dominant, and validation shifts weight toward the learned hybrid transition rule. The result strengthens the regime-sensitivity story: validation does not select a fixed universal mixture, but responds to the transport condition.

![Figure 5. Peclet-regime sampler ranks](../figures/run_009_peclet_sampler_ranks.svg)

**Figure 5. Peclet-regime sampler ranks.** Mean held-out sampler rank across high-Peclet, baseline, and low-Peclet trajectory ensembles. Lower rank is better. The preferred sampler changes with diffusivity.

![Figure 6. Peclet-regime selected mixture weights](../figures/run_009_peclet_selected_weights.svg)

**Figure 6. Peclet-regime selected mixture weights.** Mean validation-selected mixture weights across outer splits. Increasing diffusivity shifts weight away from kNN velocity matching and toward the hybrid learned transition rule.

### 3.6 Generalization Across Geometry and Flow Fidelity

We next tested whether the same conclusions survive a geometry and solver change. The second geometry uses `Core2_Subvol1_6micron_225cube_16bit_LE.raw`, downsampled to the same 75^3 grid. The connected porosity is 0.23294, compared with 0.22543 for the Core1 baseline. With the graph-flow trajectory generator, Gaussian/Bayes becomes the best mean sampler:

```text
sampler                    mean_obj   std_obj  mean_rank  wins
gaussian_bayes               255.40     39.43       1.75     2
hybrid                       268.50     38.71       3.25     1
pooled_validation_mixture    269.00     40.17       3.25     0
bootstrap_mean_mixture       272.22     33.19       3.50     1
knn_conditional              273.90     45.61       3.50     0
pair_rerank                  320.67     43.86       5.75     0
```

The selected mixture still assigns nonzero weight to all mechanisms:

```text
Gaussian/Bayes 0.4792, kNN 0.1875, hybrid 0.2708, pair rerank 0.0625
```

The OpenFOAM-derived trajectory set gives a sharper result. The finite-volume mesh has 98,270 cells and the solved field has a mean pore speed of `2.28e-08 m/s`, an outlet flux of `5.77e-15 m^3/s`, and an apparent permeability of `4.28e-12 m^2` under the imposed kinematic pressure drop. After normalizing the OpenFOAM velocity field to the same mean advective scale used in the particle tracker, Gaussian/Bayes wins three of four outer splits:

```text
sampler                    mean_obj   std_obj  mean_rank  wins
gaussian_bayes               261.39     28.17       1.25     3
hybrid                       273.63     28.38       2.50     1
bootstrap_mean_mixture       285.49     17.76       3.75     0
pooled_validation_mixture    299.46     33.33       3.50     0
knn_conditional              311.05     19.82       4.50     0
pair_rerank                  317.19     31.80       5.50     0
```

This does not weaken the argument for validation-driven mixtures. It sharpens it. In the higher-fidelity velocity field, the original physics-informed Gaussian/Bayes interface kernel is the strongest fixed sampler, while validation still selects a mixture with substantial hybrid contribution:

```text
Gaussian/Bayes 0.3333, kNN 0.1667, hybrid 0.4167, pair rerank 0.0833
```

Repeating the objective-weight sensitivity analysis on the OpenFOAM-derived trajectories confirms this interpretation. Gaussian/Bayes is the best mean sampler in six of seven objective regimes, including balanced, breakthrough-heavy, dilution-heavy, reaction-heavy, and no-reaction scores. The exception is the pair-heavy regime, where the hybrid sampler has the best mean objective and beats Gaussian/Bayes on two of four outer splits. The bootstrap-mean mixture remains competitive across regimes, while the pooled-validation mixture is less stable in this smaller OpenFOAM sensitivity sweep. Thus, higher-fidelity flow does not erase the value of learned transition context; it clarifies where that context matters most.

![Figure 7. Geometry and flow sampler ranks](../figures/run_010_generalization_sampler_ranks.svg)

**Figure 7. Geometry and flow-fidelity sampler ranks.** Mean held-out sampler rank for the Core1 graph-flow baseline, Core2 graph-flow trajectories, and Core2 OpenFOAM-derived trajectories. Lower rank is better.

![Figure 8. Geometry and flow selected weights](../figures/run_010_generalization_selected_weights.svg)

**Figure 8. Geometry and flow-fidelity selected mixture weights.** Mean validation-selected weights across outer splits. The selected mechanism shifts with both geometry and velocity-field fidelity.

![Figure 9. OpenFOAM objective sensitivity](../figures/run_011_openfoam_weight_sensitivity_heatmap.svg)

**Figure 9. OpenFOAM objective-weight sensitivity.** Mean held-out rank of each sampler under seven multi-objective scoring regimes using OpenFOAM-derived Core2 trajectories. Gaussian/Bayes is most stable overall, while the pair-heavy regime favors the hybrid sampler.

## 4. Discussion

The strongest lesson is that the original training-trajectory idea remains scientifically current. It already uses resolved Lagrangian data as a generative archive, preserves physically meaningful path increments, and evaluates full particle histories. Modern machine learning adds value when it is used to learn or weight transition rules inside this physically constrained structure.

The experiments also warn against a simple "learned beats physics" narrative. The Gaussian/Bayes sampler remains hard to beat because it encodes the right interface physics. Learned transition scores can improve contextual or pair-related behavior, but they can also trade away breakthrough fidelity. Pair-aware reranking did not yet produce robust gains. The mixture framework makes these outcomes useful: each component becomes a candidate mechanism whose value is judged by held-out transport metrics.

This matters for reactive transport. A breakthrough curve can be right for the wrong reason if particle-pair separation, dilution, or encounter rates are wrong. The training-trajectory framework is naturally positioned to address this because it generates particle histories rather than only concentration moments. The validation objective can be changed to reflect the scientific target: conservative plume prediction, dilution-limited reaction, mixing-controlled reaction, or rare-event tailing.

The current prototype has important limitations. The graph-flow trajectory sets were generated with an approximate pressure solver and simple particle tracker, and the OpenFOAM validation uses a stair-step voxel mesh on a 75^3 downsampled geometry rather than a smoothed high-resolution DNS/LBM calculation. The geometry tests still use Bentheimer samples, so they are not yet cross-lithology validation. The learned transition model is intentionally lightweight and does not yet use geometry-conditioned embeddings, neural sequence models, diffusion segment generators, or uncertainty-calibrated classifiers. The objective-weight, Peclet-sensitivity, second-geometry, and OpenFOAM sweeps use fewer outer splits than a final production benchmark, so they should be treated as sensitivity evidence rather than definitive rankings.

These limitations define the next stage. The most important physical follow-up is stronger external validation: a less downsampled or smoothed OpenFOAM mesh, an LBM velocity field, or a different rock type. The second is to replace the contrastive transition scorer with a more expressive conditional model while preserving the same held-out multi-objective validation protocol. The third is to develop archive diagnostics that say when a generated path is interpolating within observed trajectory support and when it is extrapolating.

## 5. Conclusions

The 2019 training-trajectory method anticipated a key idea of modern scientific generative modeling: use physically resolved examples as the primitive data object and generate new dynamics by recombining them under constraints. In revisiting this idea, we find that modern ML is best used as a disciplined extension of the original physics-informed sampler. Validation-selected physics/learned mixtures can be competitive with, and sometimes better than, the Gaussian/Bayes transition kernel, but no sampler dominates across all splits, metric priorities, Peclet regimes, geometries, or flow solvers. This is not a weakness of the framework. It is the point. Non-Fickian transport is a multi-objective Lagrangian memory problem, and a useful generative model should make the metric tradeoffs visible, testable, and selectable.

## Acknowledgments

Placeholder for data sources, software, and funding. The current prototype uses the Zenodo multi-resolution Bentheimer sandstone image data and locally developed NumPy-based trajectory-generation and sampler scripts.

## Data and Code Availability

Placeholder. Current code, processed trajectories, run outputs, and SVG figures are organized in this workspace. Before submission, archive the exact scripts, processed trajectory data, JSON outputs, and figure-generation workflow with a DOI-bearing repository.

## References

[Berkowitz2006] Berkowitz, B., Cortis, A., Dentz, M., & Scher, H. (2006). Modeling non-Fickian transport in geological formations as a continuous time random walk. Reviews of Geophysics, 44. https://doi.org/10.1029/2005RG000178

[Bijeljic2006] Bijeljic, B., & Blunt, M. J. (2006). Pore-scale modeling and continuous time random walk analysis of dispersion in porous media. Water Resources Research, 42. https://doi.org/10.1029/2005WR004578

[Dentz2016] Dentz, M., Kang, P. K., Comolli, A., Le Borgne, T., & Lester, D. R. (2016). Continuous time random walks for the evolution of Lagrangian velocities. Physical Review Fluids, 1, 074004. https://doi.org/10.1103/PhysRevFluids.1.074004

[Karniadakis2021] Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). Physics-informed machine learning. Nature Reviews Physics, 3, 422-440. https://doi.org/10.1038/s42254-021-00314-5

[LeBorgne2008a] Le Borgne, T., Dentz, M., & Carrera, J. (2008). Lagrangian statistical model for transport in highly heterogeneous velocity fields. Physical Review Letters, 101, 090601. https://doi.org/10.1103/PhysRevLett.101.090601

[LeBorgne2008b] Le Borgne, T., Dentz, M., & Carrera, J. (2008). Spatial Markov processes for modeling Lagrangian particle dynamics in heterogeneous porous media. Physical Review E, 78, 026308. https://doi.org/10.1103/PhysRevE.78.026308

[Li2021] Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2021). Fourier neural operator for parametric partial differential equations. International Conference on Learning Representations. https://arxiv.org/abs/2010.08895

[Mosser2017] Mosser, L., Dubrule, O., & Blunt, M. J. (2017). Reconstruction of three-dimensional porous media using generative adversarial neural networks. Physical Review E, 96, 043309. https://doi.org/10.1103/PhysRevE.96.043309

[Most2019] Most, S., Bolster, D., Bijeljic, B., & Nowak, W. (2019). Trajectories as training images to simulate advective-diffusive, non-Fickian transport. Water Resources Research, 55, 3465-3480. https://doi.org/10.1029/2018WR023552

[Raissi2019] Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707. https://doi.org/10.1016/j.jcp.2018.10.045

[Sherman2021] Sherman, T., Engdahl, N. B., Porta, G., & Bolster, D. (2021). A review of spatial Markov models for predicting pre-asymptotic and anomalous transport in porous and fractured media. Journal of Contaminant Hydrology, 236, 103734. https://doi.org/10.1016/j.jconhyd.2020.103734

[Song2021] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-based generative modeling through stochastic differential equations. International Conference on Learning Representations. https://openreview.net/forum?id=PxTIG12RRHS

[Yang2024] Yang, G., Xu, R., Tian, Y., Guo, S., Wu, J., & Chu, X. (2024). Data-driven methods for flow and transport in porous media: A review. International Journal of Heat and Mass Transfer, 235, 126149. https://doi.org/10.1016/j.ijheatmasstransfer.2024.126149

[Zhu2025] Zhu, L., Bijeljic, B., & Blunt, M. J. (2025). Diffusion model-based generation of three-dimensional multiphase pore-scale images. Transport in Porous Media, 152. https://doi.org/10.1007/s11242-025-02158-4

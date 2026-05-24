# Reviewer Risk Register

This document lists the main reasons a reviewer could reject or weaken the current manuscript and the concrete action that would reduce each risk.

## Risk Summary

| Risk | Severity | Current Status | Mitigation |
|---|---:|---|---|
| Trajectory data are from an approximate graph-based flow solver, not DNS | High | OpenFOAM voxel-flow validation added for Core2 | Present as finite-volume validation; later add smoothed DNS/LBM or higher-resolution mesh |
| Only one geometry and limited transport conditions | High | Core1 tested at three diffusivities; Core2 tested with graph flow and OpenFOAM flow | Add cross-lithology geometry if time permits |
| Objective weights are subjective | Medium | Run 006 sensitivity exists | Present objective-weight heatmap and Pareto figure; emphasize tradeoff exposure rather than one scalar optimum |
| Learned sampler is lightweight | Medium | Contrastive hybrid only | Frame as proof of validation-driven coupling; add stronger learned transition model later |
| Mixture does not universally beat Gaussian/Bayes | Medium | Honest result | Make this the methodological point: no universal winner; validation chooses metric-specific blend |
| Pair-aware reranker performs poorly | Low/Medium | Included in results | Treat as negative result; keep as mixture component but avoid highlighting as final model |
| Original 2019 DNS data are unavailable | Medium | Rebuilt from public Bentheimer CT | State clearly; use same conceptual benchmark metrics; seek original or comparable DNS data |
| Reaction metric is a proxy, not full reactive transport | Medium | Encounter probability only | Present as reaction-relevant proxy; later add mixing-limited reaction simulation |
| Figures are SVG prototype figures | Low | Manuscript-facing but not journal final | Later convert to publication style with consistent typography and final labels |
| References are good but not exhaustive | Low/Medium | Strong spine exists | Add recent pore-scale transport ML and reactive transport references before submission |

## Highest-Priority Risks

### 1. High-Fidelity Physical Validation

The largest vulnerability is that the current trajectory data are generated from an approximate graph-Laplace pressure solve and voxel particle tracker. This is acceptable for method development, but a Water Resources Research or transport-journal reviewer will likely ask whether the result survives realistic pore-scale flow.

**Progress:** Run 010 adds an OpenFOAM finite-volume velocity field on the connected Core2 pore space. The voxel mesh has 98,270 cells, passed `checkMesh`, and `simpleFoam` converged in 103 SIMPLE iterations. OpenFOAM-derived trajectories were then benchmarked with the same validation-driven selection protocol. Run 011 adds objective-weight sensitivity on those OpenFOAM trajectories.

**Remaining mitigation:** the current OpenFOAM mesh is a stair-step voxel mesh at 75^3. A stronger final validation would use a smoothed `snappyHexMesh` case, a less aggressive downsample, LBM, GeoChemFoam, or a public benchmark velocity field.

### 2. Generalization Beyond One Rock

The current result could still be a Bentheimer-specific story. Runs 008 and 009 add high- and low-Peclet diffusivity conditions, and Run 010 adds Core2 Subvol1 as a second geometry. This is a meaningful geometry check, though still within the Bentheimer family.

**Best mitigation:** evaluate on a second lithology or a public benchmark with a different pore structure.

**Progress:** the same Core1 geometry has now been evaluated at `D = 0.0003`, `D = 0.001`, and `D = 0.003`; selected mixtures shift in a physically interpretable way with diffusivity. Core2 Subvol1 has now been evaluated with both graph-flow and OpenFOAM-derived trajectories.

### 3. Objective-Weight Subjectivity

Run 006 already reduces this risk. The paper should not present one weighted objective as a universal truth. It should present the weight sensitivity as a central result.

**Mitigation already in place:** Figure 2 and Figure 3 show that sampler preference and selected mixture weights change with metric priorities. Run 011 repeats this test on OpenFOAM-derived trajectories and shows that Gaussian/Bayes is the most stable fixed sampler, while pair-heavy weighting favors the hybrid sampler.

**Remaining action:** add a short paragraph in Methods explaining why the balanced weights are not calibrated physical constants, but a transparent scalarization used for model selection.

### 4. Learned Model Strength

The current contrastive hybrid is a modest learned component. A reviewer may ask why this counts as an AI-era update.

**Mitigation:** describe the learned component as a first validation-compatible learned transition rule, not the final model. The real contribution is the validation-driven architecture that can accept stronger learned samplers later.

**Future improvement:** add a learned segment embedding, transformer/sequence transition scorer, or diffusion segment generator, but keep the same validation protocol.

## Suggested Framing

Do not claim:

```text
The learned sampler outperforms the physics sampler.
```

Claim instead:

```text
The original training-trajectory model is a natural scaffold for modern
physics-constrained generative modeling. Learned transition rules and physics
kernels capture different transport statistics, and held-out multi-objective
validation is needed to select or weight them.
```

That claim is well supported by the current evidence.

## Pre-Submission Checklist

- Add one second-geometry result; done in Run 010.
- Add one high-fidelity flow result; first OpenFOAM voxel-flow version done in Run 010.
- Decide whether the paper is a methods/prototype paper or a full physical validation paper.
- Convert `manuscript_v1.md` to journal LaTeX or Word format.
- Add a data/code availability statement with a DOI-bearing archive.
- Verify all references and add missing recent porous-media ML papers.
- Add a limitations paragraph in the Results, not only in Discussion.
- Make all figure captions self-contained.

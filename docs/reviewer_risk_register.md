# Reviewer Risk Register

This document records the main submission risks after the May 24 review-driven revision and the action that would reduce each risk.

## Risk Summary

| Risk | Severity | Current Status | Mitigation |
|---|---:|---|---|
| Crisp winners are not statistically separated | High | Main text and SI now frame winners as favored means with split-to-split uncertainty | Add paired tests, win-count intervals, and reference-vs-reference uncertainty envelopes |
| Dilution gap may be archive/recombination artifact | High | Main text now states this ambiguity explicitly | Run archive-size and segment-length convergence tests |
| Trajectory recombination is opaque to WRR readers | High | Algorithm 1 and a reader-facing mechanism table added | Add one transition-event schematic when figures are revised |
| Learned-context mechanism may look like a black box | Medium/High | Exact feature set and physical interpretation added to Methods | Add a diagnostic showing what the learned score captures beyond the Gaussian kernel |
| Objective weights are subjective | Medium/High | Methods now explains weights as empirical scale normalizers and results emphasize sensitivity | Add raw unweighted error scales or uncertainty-normalized objectives |
| Dilution and encounter metrics are proxies | Medium/High | Methods and SI now define dilution as particle entropy and encounter as reaction-opportunity proxy | Add bin-size and encounter-radius sensitivity |
| No external reduced-model benchmark | Medium/High | Identified as next test in Discussion | Compare against correlated CTRW or spatial-Markov closure on the same observables |
| Scope is still Bentheimer-only | Medium | Core1, Core2, Peclet sweep, graph flow, and OpenFOAM ladder included | Add cross-lithology or public benchmark pore geometry |
| OpenFOAM mesh is stair-step voxel CFD | Medium | 18, 12, and 6 um ladder now included; velocity autocorrelation converges between 12 and 6 um | Add smoothed snappyHexMesh, LBM, or GeoChemFoam reference case |
| Open Research archive incomplete for submission | High | GitHub package and manifest exist; language now requires DOI archive before submission/review | Deposit large trajectory/OpenFOAM artifacts and figure data in Zenodo, OSF, HydroShare, or equivalent |
| Figures still need publication-grade pass | Medium | Current figures carry the story but some are infographic-heavy | Revise figures one by one with quantitative uncertainty and consistent typography |

## Most Important Remaining Analyses

1. Breakthrough-only failure is already in the manuscript; strengthen it with exact split-wise uncertainty and, if possible, paired tests.
2. Run archive-size and segment-length convergence for the dilution gap. This is the cleanest way to separate missing retained state from a finite-archive or join-operator artifact.
3. Add bin-size and encounter-radius sensitivity for the particle-entropy dilution and reaction-opportunity proxies.
4. Add reference-vs-reference uncertainty envelopes so "adequacy" can eventually mean absolute adequacy, not only relative performance among candidates.
5. Compare against a classical correlated-CTRW or spatial-Markov closure on breakthrough, dilution, pair separation, and encounter metrics.

## Current Framing

Do not claim:

```text
One trajectory sampler wins.
```

Claim instead:

```text
Different hydrologic observables require different retained Lagrangian state
information, and held-out multi-objective validation reveals which state is
relatively adequate for the prediction being made.
```

That claim is now supported by the manuscript text, SI uncertainty language, OpenFOAM resolution ladder, and the breakthrough/dilution asymmetry.

## Pre-Submission Checklist

- Deposit a DOI-bearing review archive for large trajectory arrays, OpenFOAM meshes/fields, processed figure data, and code snapshot.
- Run or explicitly defer archive-size/segment-length convergence for the dilution gap.
- Run or explicitly defer bin-size and encounter-radius sensitivity.
- Decide whether to add a CTRW/spatial-Markov benchmark before first submission or frame it as future work.
- Revise every main-text figure for publication-grade typography, uncertainty display, and data-first hierarchy.
- Replace development repository language with final DOI information before formal submission; repository licensing is now MIT.

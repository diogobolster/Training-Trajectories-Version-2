# Run 021: Core1 Particle-Count Convergence

Purpose: quantify whether the small Core1 baseline trajectory ensembles used during method development are adequate for manuscript-facing breakthrough, dilution, pair-separation, and encounter metrics.

## Design

- Geometry: Core1 Bentheimer subvolume, 6 um raw image, downsampled by 3 to the 18 um / 75^3 graph-flow baseline.
- Flow: graph-Laplace pressure solve, target mean speed 0.06 grid cells per time unit.
- Transport: D = 0.001, dt = 0.5, 500 saved steps.
- Reference: one 20,000-particle trajectory ensemble.
- Subsample sizes: N = 500, 2500, 5000, 10000, 20000.
- Repeats: 30 without-replacement subsamples for N < 20000; full reference for N = 20000.
- Metrics: breakthrough at x = 6, 10, 14; dilution entropy at t = 100, 200, 300, 400; pair separation; encounter probability with radius 3 cells.
- Pair samples: 10,000 for subsample metrics; 50,000 for the 20k reference.

Outputs:

- `data/processed/bentheimer_6um_downsample3_D001_n20000_trajectories.npz`
- `outputs/core1_baseline_particle_count_convergence.json`
- `outputs/core1_baseline_particle_count_convergence.csv`
- `figures/run_021_core1_particle_count_convergence.png`

## Main Result

The old small Core1 baseline ensembles are too small for the final manuscript evidence, especially for entropy-based dilution. Relative to the 20k reference, the mean dilution log error falls from 0.69 at N = 500 to 0.18 at N = 2500, 0.080 at N = 5000, and 0.027 at N = 10000. In physical terms, the t = 400 dilution index rises from about 7,742 at N = 500 to 19,068 at N = 20000. The 500-particle ensemble is therefore not merely noisy; it systematically underestimates the occupied-volume dilution proxy.

| N | BTC score error | Dilution log error | Pair quantile error | Encounter abs. error | Dilution index t400 |
|---:|---:|---:|---:|---:|---:|
| 500 | 10.21 +/- 4.36 | 0.691 +/- 0.025 | 0.590 +/- 0.287 | 0.00174 +/- 0.00121 | 7,742 +/- 216 |
| 2500 | 4.60 +/- 1.35 | 0.175 +/- 0.018 | 0.371 +/- 0.185 | 0.00145 +/- 0.00100 | 14,853 +/- 324 |
| 5000 | 3.13 +/- 1.05 | 0.080 +/- 0.012 | 0.263 +/- 0.108 | 0.00120 +/- 0.00080 | 17,000 +/- 244 |
| 10000 | 1.78 +/- 0.68 | 0.027 +/- 0.006 | 0.231 +/- 0.095 | 0.00151 +/- 0.00123 | 18,310 +/- 137 |
| 20000 | 0.00 | 0.000 | 0.282 | 0.00260 | 19,068 |

The nonzero pair and encounter errors at N = 20000 are not particle-count error; they are the remaining pair-sampling noise from comparing 10,000 sampled pairs to the 50,000-pair reference. This means the pair and encounter convergence curves are conservative. A submission-level version should either increase pair samples further or use a paired/fixed pair-index diagnostic to isolate particle-count effects from pair-sampling effects.

## Interpretation

For manuscript-facing reference trajectory ensembles, N = 500 is inadequate and N = 2500 is still marginal for dilution. N = 5000 is a reasonable lower bound for exploratory physical trends, but N = 10000 is the better target for final Core1 graph-flow evidence if dilution remains central to the paper. The OpenFOAM ladder already uses N = 5000, which is defensible for a costly high-fidelity run, but the graph-flow baseline, Peclet sweep, and Core2 graph-flow cases should be rerun at 10000 particles if we want the Core1/Core2 graph-flow evidence to carry the same authority as the tightened OpenFOAM results.

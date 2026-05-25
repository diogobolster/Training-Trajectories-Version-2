# Run 022: 20k Manuscript-Facing Archives and Memory Benchmarks

## Purpose

Run 022 replaces the small development trajectory ensembles with `N = 20000`
reference archives for every case used in the manuscript. The goal is to remove
particle-count vulnerability from breakthrough, dilution, pair-separation, and
encounter metrics before revising the figures and manuscript text.

## Trajectory Archives

All seven manuscript-facing archives were generated successfully.

```text
case                  particles   saved steps   note
Core1 high Pe graph   20000       800           D = 0.0003
Core1 baseline graph  20000       800           D = 0.001
Core1 low Pe graph    20000       800           D = 0.003
Core2 graph           20000       800           D = 0.001
OpenFOAM 18 um        20000       4000          D = 0.001, dt = 0.1
OpenFOAM 12 um        20000       4000          D = 0.00225, dt = 0.1
OpenFOAM 6 um         20000       4000          D = 0.009, dt = 0.1
```

OpenFOAM tracking reused the solved velocity fields; no flow solve was rerun.
The tight particle integrator did not hit the internal substep cap in any
OpenFOAM archive:

```text
case            mean length   max substeps   cap hits
OpenFOAM 18 um  4000.9976     4              0
OpenFOAM 12 um  4000.9358     5              0
OpenFOAM 6 um   4000.8048     10             0
```

## Benchmark Protocol

The held-out reference sets use all `20000` trajectories. To keep the transition
archives from being dominated by nearly redundant overlapping motifs, the
segment archives were strided:

```text
graph-flow cases: segment_steps = 36, match_steps = 20, segment_stride = 400
OpenFOAM cases:   segment_steps = 160, match_steps = 60-80, segment_stride = 1600
```

These settings keep the archive search problem in a comparable size class while
using the large particle ensembles for the validation and test observables.

## Balanced Benchmark Summary

```text
case              best mean sampler       mean obj.   mean rank   wins   mean selected weights
Core1 high Pe     hybrid                   316.35      2.00        1      0.229, 0.333, 0.313, 0.125
Core1 baseline    Gaussian/Bayes           319.51      3.00        0      0.250, 0.229, 0.271, 0.250
Core1 low Pe      pooled mixture           319.34      1.50        2      0.375, 0.229, 0.313, 0.083
Core2 graph       pooled mixture           325.98      3.00        2      0.271, 0.125, 0.396, 0.208
OpenFOAM 18 um    kNN/archive proximity   1570.01      3.75        1      0.083, 0.396, 0.250, 0.271
OpenFOAM 12 um    Gaussian/Bayes          1647.69      2.00        2      0.125, 0.500, 0.125, 0.250
OpenFOAM 6 um     pair-rerank             1657.02      2.50        0      0.313, 0.396, 0.083, 0.208
```

Weights are ordered as velocity memory, archive proximity, learned context, and
pair organization.

## Objective Sensitivity Summary

The objective sweeps reinforce the main paper thesis: winner labels are less
important than observable-conditioned memory requirements.

```text
case          main objective-sensitivity result
Core1         Breakthrough and dilution-heavy regimes favor Gaussian/Bayes by mean,
              while pair-heavy and reaction-heavy regimes favor bootstrap mixtures.
OpenFOAM 18   Hybrid is best by mean for most regimes; dilution-heavy favors
              Gaussian/Bayes.
OpenFOAM 12   Gaussian/Bayes is best by mean across all tested objective regimes.
OpenFOAM 6    Pooled mixtures are best for most regimes; pair-heavy and
              dilution-heavy favor bootstrap mixtures.
```

## Interpretation

The 20k results strengthen the manuscript but change the emphasis. The evidence
does not support crisp winner language. It supports a memory map:

- large-N Core1 baseline has very close Gaussian/Bayes, kNN, and hybrid scores;
- low-Pe Core1 gives the cleanest validation-mixture improvement;
- Core2 graph-flow remains non-universal and learned-context rich;
- OpenFOAM 12 um is the strongest velocity-memory case;
- full-resolution OpenFOAM keeps velocity, archive proximity, and pair
  organization all active.

The revised manuscript and figures should therefore emphasize uncertainty,
win counts, split-level variability, and objective-conditioned retained state
information rather than declaring a single preferred sampler per condition.

## Key Outputs

```text
scripts/run_stage2_20k_trajectory_archives.py
scripts/run_stage3_20k_memory_benchmarks.py

outputs/bentheimer_6um_downsample3_D0003_n20000_stride400_outer_split_mixture_benchmark.json
outputs/bentheimer_6um_downsample3_D001_n20000_stride400_outer_split_mixture_benchmark.json
outputs/bentheimer_6um_downsample3_D003_n20000_stride400_outer_split_mixture_benchmark.json
outputs/bentheimer_core2_subvol1_6um_downsample3_D001_n20000_stride400_outer_split_mixture_benchmark.json
outputs/bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_dt010_n20000_stride1600_outer_split_mixture_benchmark.json
outputs/bentheimer_core2_subvol1_6um_downsample2_D00225_openfoam_phys_scaled_dt010_n20000_stride1600_outer_split_mixture_benchmark.json
outputs/bentheimer_core2_subvol1_6um_fullres_D009_openfoam_strict_dt010_n20000_stride1600_outer_split_mixture_benchmark.json

outputs/bentheimer_6um_downsample3_D001_n20000_stride400_objective_weight_sensitivity.json
outputs/bentheimer_6um_downsample3_D001_n20000_stride400_breakthrough_only_failure.json
outputs/bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_dt010_n20000_stride1600_objective_weight_sensitivity.json
outputs/bentheimer_core2_subvol1_6um_downsample2_D00225_openfoam_phys_scaled_dt010_n20000_stride1600_objective_weight_sensitivity.json
outputs/bentheimer_core2_subvol1_6um_fullres_D009_openfoam_strict_dt010_n20000_stride1600_objective_weight_sensitivity.json
```

# Run 020: Tight OpenFOAM Memory Benchmarks

## Purpose

Run 020 reruns the manuscript-facing OpenFOAM memory-selection analyses on the tight trajectory archives from Runs 018--019. These archives use:

```text
particles:          5000
saved steps:        4000
dt:                 0.1
total duration:     400 step-units
segment steps:      160
segment stride:     400
generated segments: 8
```

The segment stride is intentional. It samples motifs across all 5000 particles while avoiding an over-dense set of nearly overlapping windows from the same path. Without this stride, the candidate archive exceeded 100,000 segments per split and made all-to-all transition matching unnecessarily slow.

## Outputs

Balanced outer-split benchmarks:

```text
outputs/bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_dt010_n5000_stride400_outer_split_mixture_benchmark.json
outputs/bentheimer_core2_subvol1_6um_downsample2_D00225_openfoam_phys_scaled_dt010_n5000_stride400_outer_split_mixture_benchmark.json
outputs/bentheimer_core2_subvol1_6um_fullres_D009_openfoam_strict_dt010_n5000_stride400_outer_split_mixture_benchmark.json
```

Objective-weight sensitivity:

```text
outputs/bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_dt010_n5000_stride400_objective_weight_sensitivity.json
outputs/bentheimer_core2_subvol1_6um_downsample2_D00225_openfoam_phys_scaled_dt010_n5000_stride400_objective_weight_sensitivity.json
outputs/bentheimer_core2_subvol1_6um_fullres_D009_openfoam_strict_dt010_n5000_stride400_objective_weight_sensitivity.json
```

Regenerated summaries and figures:

```text
outputs/openfoam_resolution_ladder_summary.json
outputs/master_evidence_table.json
figures/run_016_openfoam_resolution_ladder.png
figures/figure6_flow_fidelity_velocity_memory.png
figures/figure7_memory_adequacy_atlas.png
```

## Balanced Benchmarks

```text
case    best mean memory          mean obj.   mean rank   wins   mean selected weights
18 um   archive proximity         1567.42     2.75        1      0.167, 0.417, 0.083, 0.333
12 um   validation mixture        1593.69     1.50        3      0.063, 0.479, 0.125, 0.333
6 um    validation mixture        1695.18     2.50        2      0.271, 0.417, 0.104, 0.208
```

Weights are ordered as velocity memory, archive proximity, learned context, and pair organization.

The tight benchmarks change the emphasis of the OpenFOAM story. The earlier 500-particle archives made the 18 um OpenFOAM case look like a strong Gaussian/Bayes victory. With larger held-out ensembles and tighter particle tracking, the stronger result is that no single memory closes the problem. Archive proximity is safest at 18 um, while validation-selected mixtures are safest at 12 um and 6 um.

## Objective Sensitivity

Best mean sampler by objective regime:

```text
regime             18 um                 12 um                    6 um
balanced           archive proximity     validation mixture       validation mixture
breakthrough only  archive proximity     validation mixture       validation mixture
btc heavy          archive proximity     validation mixture       validation mixture
pair heavy         archive proximity     validation mixture       velocity memory
dilution heavy     archive proximity     validation mixture       validation mixture
reaction light     archive proximity     validation mixture       validation mixture
reaction heavy     archive proximity     validation mixture       validation mixture
no reaction        archive proximity     validation mixture       validation mixture
```

The 6 um pair-heavy exception is scientifically useful. It shows that full-resolution flow does not make velocity memory obsolete; instead, validation decides when that memory is the one carrying the target observable.

## Interpretation

The revised story is stronger and less brittle:

```text
OpenFOAM resolution preserves velocity memory, but tight multi-objective validation
does not crown velocity memory as a universal closure. The adequate memory changes
with resolution and objective: archive proximity at coarse finite-volume resolution,
validation-selected mixtures at finer resolution, and velocity memory for the
full-resolution pair-heavy objective.
```

This supports the central manuscript thesis. Better physics gives the archive more meaningful memories, but validation still has to decide which memories the prediction is not allowed to forget.

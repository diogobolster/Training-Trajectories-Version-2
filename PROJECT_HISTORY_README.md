# Training Trajectories Neural Net

This workspace is a launchpad for revisiting the 2019 training-trajectory idea in the modern AI era.

The original method used resolved particle trajectories as "training images": cut them into physically meaningful trajectory segments, conditionally copy likely next segments, and paste them into long synthetic trajectories that preserve advective-diffusive memory. The near-term goal here is to build a **physics-constrained generative model for non-Fickian Lagrangian transport**.

## Starting Point

- Read [docs/start_here.md](docs/start_here.md) for the research framing.
- Read [docs/experiment_plan.md](docs/experiment_plan.md) for the first experiments.
- Read [docs/paper_outline.md](docs/paper_outline.md) for a draft manuscript arc.
- Read [docs/data_format.md](docs/data_format.md) for the trajectory input format.
- Read [docs/dataset_candidates.md](docs/dataset_candidates.md) for real porous-media data options.
- Read [docs/simulation_pipeline.md](docs/simulation_pipeline.md) for the first CT-to-trajectory workflow.
- Read [docs/first_real_rock_run.md](docs/first_real_rock_run.md) for the first Bentheimer smoke-test result.
- Read [docs/second_real_rock_run.md](docs/second_real_rock_run.md) for the 6 micrometer source run and tuned sampler result.
- Read [docs/full_metric_baseline.md](docs/full_metric_baseline.md) for BTC, dilution, pair-separation, and reaction-proxy results.
- Read [docs/learned_transition_attempts.md](docs/learned_transition_attempts.md) for the first learned-sampler attempts and what failed.
- Read [docs/run_001_report.md](docs/run_001_report.md) for the generated report with SVG figures.
- Read [docs/short_horizon_reranking.md](docs/short_horizon_reranking.md) for the first reranking attempt.
- Read [docs/run_002_report.md](docs/run_002_report.md) and [docs/pair_aware_reranking.md](docs/pair_aware_reranking.md) for the pair-aware reranking attempt.
- Read [docs/run_003_report.md](docs/run_003_report.md) and [docs/validation_driven_mixture_selection.md](docs/validation_driven_mixture_selection.md) for single-split validation-driven mixture selection.
- Read [docs/run_004_report.md](docs/run_004_report.md) and [docs/bootstrap_mixture_selection.md](docs/bootstrap_mixture_selection.md) for repeated validation mixture selection.
- Read [docs/run_005_report.md](docs/run_005_report.md) and [docs/outer_split_mixture_benchmark.md](docs/outer_split_mixture_benchmark.md) for outer-split robustness results.
- Read [docs/run_006_report.md](docs/run_006_report.md) and [docs/objective_weight_sensitivity.md](docs/objective_weight_sensitivity.md) for objective-weight sensitivity.
- Read [docs/run_014_breakthrough_only_failure.md](docs/run_014_breakthrough_only_failure.md) for the breakthrough-only counterfactual validation test.
- Read [docs/run_015_high_resolution_openfoam.md](docs/run_015_high_resolution_openfoam.md) for the less-downsampled OpenFOAM resolution test.
- Read [docs/run_016_full_resolution_openfoam.md](docs/run_016_full_resolution_openfoam.md) for the full-resolution OpenFOAM resolution-ladder result.
- Read [docs/run_017_strict_openfoam_convergence.md](docs/run_017_strict_openfoam_convergence.md) for the strict full-resolution OpenFOAM convergence audit.
- Read [docs/run_018_strict_particle_tracking.md](docs/run_018_strict_particle_tracking.md) for the strict-field, tight-step, 5000-particle trajectory archive.
- Read [docs/run_019_tight_particle_tracking_resolution_ladder.md](docs/run_019_tight_particle_tracking_resolution_ladder.md) for the matching tight-step, 5000-particle archives at 18, 12, and 6 micrometers.
- Read [docs/run_020_tight_openfoam_memory_benchmarks.md](docs/run_020_tight_openfoam_memory_benchmarks.md) for the updated memory-selection benchmarks on the tight OpenFOAM archives.
- Read [docs/run_007_figures.md](docs/run_007_figures.md) for manuscript-facing SVG figures from Runs 005 and 006.
- Read [docs/run_008_peclet_generalization.md](docs/run_008_peclet_generalization.md) for the first second-Peclet generalization result.
- Read [docs/run_009_peclet_sweep.md](docs/run_009_peclet_sweep.md) for the three-regime Peclet sweep and new Peclet figures.
- Read [docs/manuscript_v0.md](docs/manuscript_v0.md), [docs/figure_captions.md](docs/figure_captions.md), [docs/methods_details.md](docs/methods_details.md), and [docs/reference_notes.md](docs/reference_notes.md) for the first paper draft.
- Read [docs/manuscript_v1.md](docs/manuscript_v1.md), [docs/reviewer_risk_register.md](docs/reviewer_risk_register.md), and [docs/next_experiment_generalization.md](docs/next_experiment_generalization.md) for the submission-shaped draft and credibility plan.

Select sampler mixtures on a validation split:

```bash
python3 scripts/select_sampler_mixture.py \
  --input data/processed/bentheimer_6um_downsample3_trajectories.npz
```

Average validation-selected sampler mixtures over repeated splits:

```bash
python3 scripts/bootstrap_mixture_selection.py \
  --input data/processed/bentheimer_6um_downsample3_trajectories.npz
```

Benchmark mixture selection over multiple held-out test splits:

```bash
python3 scripts/outer_split_mixture_benchmark.py \
  --input data/processed/bentheimer_6um_downsample3_trajectories.npz
```

Check sensitivity to the multi-objective score:

```bash
python3 scripts/objective_weight_sensitivity.py \
  --input data/processed/bentheimer_6um_downsample3_trajectories.npz
```

Generate the selection and sensitivity figures:

```bash
python3 scripts/make_selection_figures.py
```

Generate the Peclet-regime figures:

```bash
python3 scripts/make_peclet_figures.py
```
- Run the toy baseline:

```bash
python3 scripts/run_synthetic_tta_demo.py
```

The demo generates synthetic advective-diffusive trajectories, builds a segment archive, samples new trajectories with conditional segment matching, and compares simple breakthrough-time statistics.

Run a dependency-light smoke test:

```bash
python3 scripts/smoke_test.py
```

Compare the current baseline ladder:

```bash
python3 scripts/compare_samplers.py
```

Scan TTA memory and bandwidth parameters:

```bash
python3 scripts/tune_tta_parameters.py --input data/processed/bentheimer_trajectories.npz
```

Evaluate BTC, dilution, pair-separation, and reaction-proxy metrics:

```bash
python3 scripts/evaluate_transport_metrics.py \
  --input data/processed/bentheimer_6um_downsample3_trajectories.npz
```

## Near-Term Thesis

TTA-v2 should not begin as a fully black-box neural simulator. The best first step is a hybrid:

1. Keep the physical data object: resolved Lagrangian trajectory segments.
2. Keep the physical constraints: continuity of velocity/direction, finite-Peclet diffusion tolerance, benchmark metrics beyond BTCs.
3. Learn the parts the 2019 method had to hand-design: transition weights, latent transport state, state-dependent memory length, and geometry conditioning.

## Repository Layout

```text
docs/
  start_here.md          Research framing and first sprint
  experiment_plan.md     Concrete experiments and success criteria
  paper_outline.md       Manuscript narrative skeleton
scripts/
  run_synthetic_tta_demo.py
src/tta_v2/
  segments.py            Segment archive construction
  sampler.py             Conditional segment sampler
  synthetic.py           Synthetic trajectory generator
  metrics.py             Basic transport metrics
tests/
  test_segments.py
```

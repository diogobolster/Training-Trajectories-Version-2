# Reproducibility Notes

## Environment

Install the local package and Python dependencies from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e . -r requirements-simulation.txt
python -m pytest tests
```

OpenFOAM was run through the local Docker helper in the development workspace. The GitHub package stores the OpenFOAM case templates, but full mesh and field directories should be regenerated or retrieved from the DOI-bearing review archive described in `DATA_MANIFEST.md`.

## Tight OpenFOAM Trajectory Protocol

The manuscript-facing OpenFOAM trajectory archives use:

```text
particles:      5000
saved steps:    4000
dt:             0.1
segment steps:  160
segment stride: 400
seed:           20260524
```

The three tight archives are:

```text
18 um: data/processed/bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_dt010_n5000_trajectories.npz
12 um: data/processed/bentheimer_core2_subvol1_6um_downsample2_D00225_openfoam_phys_scaled_dt010_n5000_trajectories.npz
6 um:  data/processed/bentheimer_core2_subvol1_6um_fullres_D009_openfoam_strict_dt010_n5000_trajectories.npz
```

The summary JSON files are included in `data/processed_summaries/`; the binary `.npz` files should be deposited in the companion DOI-bearing review archive.

## Manuscript-Facing Analyses

Balanced OpenFOAM memory benchmarks:

```bash
python scripts/outer_split_mixture_benchmark.py \
  --input data/processed/bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_dt010_n5000_trajectories.npz \
  --segment-steps 160 --segment-stride 400

python scripts/outer_split_mixture_benchmark.py \
  --input data/processed/bentheimer_core2_subvol1_6um_downsample2_D00225_openfoam_phys_scaled_dt010_n5000_trajectories.npz \
  --segment-steps 160 --segment-stride 400

python scripts/outer_split_mixture_benchmark.py \
  --input data/processed/bentheimer_core2_subvol1_6um_fullres_D009_openfoam_strict_dt010_n5000_trajectories.npz \
  --segment-steps 160 --segment-stride 400
```

After benchmark outputs exist, regenerate the manuscript summary products:

```bash
python scripts/make_master_evidence.py
python scripts/summarize_openfoam_resolution_ladder.py
python scripts/make_memory_adequacy_figures.py
```

## Manuscript Build

From `paper/`, compile:

```bash
tectonic wrr_manuscript.tex
tectonic wrr_supporting_information.tex
```

The current package includes compiled PDFs for convenience, but the TeX sources and bibliography should be treated as the archival manuscript source.

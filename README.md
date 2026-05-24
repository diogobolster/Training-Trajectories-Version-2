# Memory Adequacy in Non-Fickian Transport

This repository is the GitHub-ready Open Research companion for the manuscript:

**What Must a Pore-Scale Transport Model Remember? Memory Adequacy in Non-Fickian Transport**

It contains the analysis code, manuscript sources, figure scripts, small run outputs, raw Bentheimer subvolume inputs, OpenFOAM case templates, and data manifests needed to reproduce the paper workflow. Large generated files are intentionally not stored in this GitHub package because they exceed normal repository limits; they are listed in `DATA_MANIFEST.md` for DOI-bearing archival deposit.

## Contents

- `src/tta_v2/`: trajectory archive, sampler, tracking, OpenFOAM, and metric utilities.
- `scripts/`: simulation, benchmarking, summary, and figure-generation scripts.
- `data/raw/`: raw Bentheimer subvolume inputs used by the workflow.
- `data/processed_summaries/`: summary JSON files for generated trajectory archives.
- `outputs/`: JSON and CSV run outputs used by the manuscript and SI.
- `figures/` and `paper/figures/`: generated manuscript figures.
- `paper/`: manuscript, supporting information, bibliography, and compiled PDFs.
- `openfoam_case_templates/`: OpenFOAM initial fields and solver settings without large meshes or time directories.
- `docs/`: run notes and methodological audit trail.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e . -r requirements-simulation.txt
python -m pytest tests
```

The manuscript-facing tables and figures are regenerated from the JSON outputs with:

```bash
python scripts/make_master_evidence.py
python scripts/summarize_openfoam_resolution_ladder.py
python scripts/make_memory_adequacy_figures.py
```

The LaTeX manuscript and supporting information can be compiled from `paper/` with a local LaTeX engine such as `tectonic`.

## Large Artifacts

The tight 5000-particle trajectory archives are approximately 409-410 MB each, and the full OpenFOAM mesh/field directories are several GB. Those files should be archived with the final publication in Zenodo, OSF, HydroShare, or another DOI-bearing repository. The exact filenames, roles, and included checksums for smaller files are recorded in `DATA_MANIFEST.md`.

## Status

This package is prepared for manuscript development and Open Research review. Before public release, choose and add final code/data licenses and replace draft manuscript language with the accepted citation and repository DOI.

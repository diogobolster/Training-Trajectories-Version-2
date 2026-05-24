# Data Manifest

This manifest separates files included in the GitHub-ready package from large generated artifacts that should be deposited in a DOI-bearing archive.

## Included Raw Inputs

These raw subvolume files are included because each is below GitHub's single-file size limit.

| File | Description | SHA-256 |
|---|---|---|
| `data/raw/Core1_Subvol1_18micron_75cube_16bit_LE.raw` | Core1, 18 um, 75-cube 16-bit little-endian raw volume | `51a42864ddabcc39f7dd7123deefaa896cae985e6efa1cc71acc54daf2d6c98a` |
| `data/raw/Core1_Subvol1_6micron_225cube_16bit_LE.raw` | Core1, 6 um, 225-cube 16-bit little-endian raw volume | `b73d787df9a9b9f2b9ec38820e02d6219ea7fabd71a2b9601a51ec87f9b9cfe4` |
| `data/raw/Core2_Subvol1_6micron_225cube_16bit_LE.raw` | Core2, 6 um, 225-cube 16-bit little-endian raw volume | `5bb9ec014332270ed524959b39ebaa8f72755fdc31e0da933ffb9c77dda621e0` |

## Included Processed Summaries

The JSON summaries for the tight OpenFOAM trajectory archives are included in `data/processed_summaries/`. They record trajectory length, tolerance settings, particle counts, and tracking diagnostics.

| Source file | SHA-256 |
|---|---|
| `data/processed/bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_dt010_n5000_trajectories.summary.json` | `8f8d46fa676aaed02fc32d6a4f916cb4c3f7013d05116abd04ef3dfede49eb9a` |
| `data/processed/bentheimer_core2_subvol1_6um_downsample2_D00225_openfoam_phys_scaled_dt010_n5000_trajectories.summary.json` | `5e6d64e4b062dbed1fbe0fbe1d630a168083b3ca344b92fb81b26cb74c7e9534` |
| `data/processed/bentheimer_core2_subvol1_6um_fullres_D009_openfoam_strict_dt010_n5000_trajectories.summary.json` | `55a74228f1e705e4f43636303d3f4eeddfd232c69a9274947e3d979653d56f0f` |

## Included Manuscript Outputs

The small JSON/CSV outputs in `outputs/` are included. Two manuscript-facing summary files are:

| File | SHA-256 |
|---|---|
| `outputs/openfoam_resolution_ladder_summary.json` | `eeb16b3d126185ad4da471604d8378f46f1f3ce905ac6e7c238c4dee0fa03da3` |
| `outputs/master_evidence_table.json` | `379cceed6b99ee6ca666b6a37af36a5e1a83878e5026e80beb104ca59e7158ae` |

## Large Generated Artifacts To Archive Separately

These files are required for full reruns without regeneration, but they are too large for ordinary GitHub storage.

| File or directory | Approx. size | Role |
|---|---:|---|
| `data/processed/bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_dt010_n5000_trajectories.npz` | 409 MB | 18 um tight OpenFOAM trajectory archive |
| `data/processed/bentheimer_core2_subvol1_6um_downsample2_D00225_openfoam_phys_scaled_dt010_n5000_trajectories.npz` | 409 MB | 12 um tight OpenFOAM trajectory archive |
| `data/processed/bentheimer_core2_subvol1_6um_fullres_D009_openfoam_strict_dt010_n5000_trajectories.npz` | 410 MB | 6 um strict OpenFOAM trajectory archive |
| `openfoam_cases/bentheimer_core2_subvol1_6um_downsample3_voxel_flow/constant/polyMesh/` and time directories | large | 18 um OpenFOAM mesh and fields |
| `openfoam_cases/bentheimer_core2_subvol1_6um_downsample2_voxel_flow/constant/polyMesh/` and time directories | large | 12 um OpenFOAM mesh and fields |
| `openfoam_cases/bentheimer_core2_subvol1_6um_fullres_voxel_flow_strict/constant/polyMesh/` and time directories | several GB | 6 um strict full-resolution OpenFOAM mesh and fields |

The OpenFOAM case templates in this package contain the initial fields, solver controls, transport properties, turbulence settings, case metadata, and selected flow summaries, but not the large mesh or solved time directories.

# Data Manifest

This manifest separates files included in the GitHub-ready package from large generated artifacts covered by the DOI-bearing archive or reviewer-access record cited in the manuscript Open Research section.

## Included Raw Inputs

These raw subvolume files are included because each is below GitHub's single-file size limit.

| File | Description | SHA-256 |
|---|---|---|
| `data/raw/Core1_Subvol1_18micron_75cube_16bit_LE.raw` | Core1, 18 um, 75-cube 16-bit little-endian raw volume | `51a42864ddabcc39f7dd7123deefaa896cae985e6efa1cc71acc54daf2d6c98a` |
| `data/raw/Core1_Subvol1_6micron_225cube_16bit_LE.raw` | Core1, 6 um, 225-cube 16-bit little-endian raw volume | `b73d787df9a9b9f2b9ec38820e02d6219ea7fabd71a2b9601a51ec87f9b9cfe4` |
| `data/raw/Core2_Subvol1_6micron_225cube_16bit_LE.raw` | Core2, 6 um, 225-cube 16-bit little-endian raw volume | `5bb9ec014332270ed524959b39ebaa8f72755fdc31e0da933ffb9c77dda621e0` |

## Included Processed Summaries

The JSON summaries for the manuscript-facing graph-flow and OpenFOAM trajectory archives are included in `data/processed_summaries/`. They record trajectory length, tolerance settings, particle counts, and tracking diagnostics. Large `.npz` trajectory archives are listed separately below.

| Included file | SHA-256 |
|---|---|
| `data/processed_summaries/bentheimer_6um_downsample3_D0003_n20000_steps800_trajectories.summary.json` | `92dd7083a0c5adcd9b958d739200a4739a8aecd266ef776085ed9d04a55a4148` |
| `data/processed_summaries/bentheimer_6um_downsample3_D001_n20000_steps800_trajectories.summary.json` | `70dd5f6ac542ccfdd0ddf7db7a317a5ff5e249c94f147d48d4f77f045426d198` |
| `data/processed_summaries/bentheimer_6um_downsample3_D001_n20000_trajectories.summary.json` | `92ca04a0513f85247547913b670a8308b0af1593fb1c53e28bcebba29e8c2fce` |
| `data/processed_summaries/bentheimer_6um_downsample3_D003_n20000_steps800_trajectories.summary.json` | `aab318b7679955a773509a3cca5e269aa195466bf62b0812b3141f0458115d48` |
| `data/processed_summaries/bentheimer_core2_subvol1_6um_downsample3_D001_n20000_steps800_trajectories.summary.json` | `b4ffac4b15940b4343684cc4ca1a62b1ab82694e0c7638a60fc8c808a8c7f01b` |
| `data/processed_summaries/bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_dt010_n20000_trajectories.summary.json` | `c66796430823dc369ada7a0ba642cfb13dc11cad5bf9ee3e763e4ce1932a3b1d` |
| `data/processed_summaries/bentheimer_core2_subvol1_6um_downsample2_D00225_openfoam_phys_scaled_dt010_n20000_trajectories.summary.json` | `404a366a2e64dbc9d339864b891e5b61ac0e883a3b412a76e2008ddf64ee14ae` |
| `data/processed_summaries/bentheimer_core2_subvol1_6um_fullres_D009_openfoam_strict_dt010_n20000_trajectories.summary.json` | `6efa880fc60010ff8f32d9600527cbe4700b584a5b7fc2d931798ebf463dfdb0` |

## Included Manuscript Outputs

The small JSON/CSV/NPZ outputs in `outputs/` are included. Key manuscript-facing files are:

| File | SHA-256 |
|---|---|
| `outputs/openfoam_resolution_ladder_summary.json` | `eeb16b3d126185ad4da471604d8378f46f1f3ce905ac6e7c238c4dee0fa03da3` |
| `outputs/master_evidence_table.json` | `379cceed6b99ee6ca666b6a37af36a5e1a83878e5026e80beb104ca59e7158ae` |
| `outputs/core1_baseline_particle_count_convergence.json` | `719ed6ef54d2c9b2ab32eaa87cc74f653923e6605fb026682588491c1e134988` |
| `outputs/core1_baseline_particle_count_convergence.csv` | `e066d58e88108af002ea34e09e3cb75f12f92152a7b476a828e4d2524d99f4c6` |
| `outputs/recombined_geometric_support.json` | `f18d98f4d592f4d86d23c5d04413b14cd3ae01688f2480187868d72df2c08248` |
| `outputs/core1_inlet_pair_sensitivity.json` | `bb144b4bacff7904749b2ec4e6f72286a23a5c79df801209ff91bc4ea30ae4eb` |
| `outputs/core1_archive_convergence.json` | `9b08ce8ca8856c4c65e531c73f9055ea528151bbe66b887bc63ea39134bb6e71` |
| `outputs/core1_proxy_sensitivity.json` | `03356e22b8260a2c40ea380073b01af6eac11edc593eaf6ffda8aa1c207398a5` |
| `outputs/figure3_core1_baseline_visual_diagnostics.npz` | `730a834f05e949d2bdeffed130d81fd69a31e2d5abff99613bf2c0b4c98ef7c6` |
| `outputs/figure5_peclet_reference_diagnostics.npz` | `944dc5117d31a0274d0ce18729eaff76d381e9c993358137da9215f5b85880bd` |
| `outputs/figure6_openfoam_physical_diagnostics.npz` | `7a018a0ee44c4a591e1ede6d9a2d86c4343a9ad34ccf5db17d5d07d2d87445bf` |

## Large Generated Artifacts In The Preserved Archive

These files are required for full reruns without regeneration, but they are too large for ordinary GitHub storage. They are listed for inclusion in the DOI-bearing review archive rather than tracked directly in GitHub.

| File or directory | Approx. size | Role |
|---|---:|---|
| `data/processed/bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_dt010_n5000_trajectories.npz` | 409 MB | 18 um tight OpenFOAM trajectory archive |
| `data/processed/bentheimer_core2_subvol1_6um_downsample2_D00225_openfoam_phys_scaled_dt010_n5000_trajectories.npz` | 409 MB | 12 um tight OpenFOAM trajectory archive |
| `data/processed/bentheimer_core2_subvol1_6um_fullres_D009_openfoam_strict_dt010_n5000_trajectories.npz` | 410 MB | 6 um strict OpenFOAM trajectory archive |
| `data/processed/bentheimer_6um_downsample3_D001_n20000_trajectories.npz` | 206 MB | Core1 baseline 20k particle-count convergence reference |
| `openfoam_cases/bentheimer_core2_subvol1_6um_downsample3_voxel_flow/constant/polyMesh/` and time directories | large | 18 um OpenFOAM mesh and fields |
| `openfoam_cases/bentheimer_core2_subvol1_6um_downsample2_voxel_flow/constant/polyMesh/` and time directories | large | 12 um OpenFOAM mesh and fields |
| `openfoam_cases/bentheimer_core2_subvol1_6um_fullres_voxel_flow_strict/constant/polyMesh/` and time directories | several GB | 6 um strict full-resolution OpenFOAM mesh and fields |

The OpenFOAM case templates in this package contain the initial fields, solver controls, transport properties, turbulence settings, case metadata, and selected flow summaries, but not the large mesh or solved time directories.

# Dataset Candidates

## Recommendation

Start with **Bentheimer sandstone**.

It gives us three useful rungs:

1. A tiny 18 micrometer, 75^3 Zenodo volume for end-to-end smoke tests.
2. Larger 6 micrometer, 225^3 Zenodo volumes for more realistic first simulations.
3. Digital Porous Media Portal / former Digital Rocks Portal Bentheimer data with segmentation, pressure, and velocity fields for a faster path to particle trajectories.

This keeps the first stage focused: build the trajectory-generation machinery before we spend weeks perfecting the flow solver.

## Candidate A: Zenodo Multi-Resolution Bentheimer Sandstones

Link: https://zenodo.org/records/5542624

Why it is useful:

- Open dataset under CC-BY-4.0.
- Two distinct Bentheimer cores, each with two subvolumes.
- Multiple resolutions: 18, 6, and 2 micrometers.
- The 18 micrometer raw files are only 75^3 voxels and under 1 MB, ideal for our first real-rock smoke test.
- The 6 micrometer raw files are 225^3 voxels and about 23 MB, a practical next step.

Limitations:

- These are CT images, not ready particle trajectories.
- We need to segment pore space and solve an approximate flow, or use another source for velocity fields.

First target file:

```text
Core1_Subvol1_18micron_75cube_16bit_LE.raw
```

Next target file:

```text
Core1_Subvol1_6micron_225cube_16bit_LE.raw
```

## Candidate B: Bentheimer Project With Velocity Fields

RockVerse documentation:

https://rockverse.readthedocs.io/en/v1.2.1/tutorials/digitalrock/orthogonal_viewer/import_bentheimer_sandstone.html

Why it is useful:

- 500^3 Bentheimer sandstone subset.
- Documentation names raw CT, segmentation, pressure, and velocity components.
- If the files are accessible through the new Digital Porous Media Portal, this is the fastest route to particle trajectories because we can skip the first flow solve.

Known filenames from the documentation:

```text
BE_CT_Oxyz_0001_0001_0001
BE_Seg_Oxyz_0001_0001_0001
BE_Pressure_Oxyz_0001_0001_0001
BE_Velocity_Oxyz_0001_0001_0001
```

Data details from the documentation:

```text
shape: (500, 500, 500)
voxel length: 5 micrometers
raw file order: Fortran
segmentation dtype: uint8
pressure/velocity dtype: little-endian float32
```

## Candidate C: DRP-372 / Large Simulation Dataset

Paper:

https://www.nature.com/articles/s41597-022-01664-0

GitHub:

https://github.com/je-santos/Large-simulation-dataset

Why it is useful:

- 217 samples, standardized 256^3 and 480^3 domains.
- Includes binary geometries, LBM flow simulations, electrical simulations, and geometry features.
- Designed explicitly as a benchmark and ML dataset.

Limitations:

- Large, potentially overkill for first week.
- Velocity fields exist, but the portal workflow may require more data-management setup than Zenodo.

Best use:

- Later ML generalization experiments across lithologies/geometries.
- Geometry-conditioned TTA-v2.

## Candidate D: Fast Micro-CT Solute Transport Experiments

Paper:

https://www.nature.com/articles/s41597-021-00803-3

Why it is useful:

- Experimental solute transport image sequences in sintered glass, Bentheimer sandstone, and Savonnieres limestone.
- Includes segmented pore-space masks.
- Valuable later for validating concentration/BTC behavior against experiments.

Limitations:

- This is concentration imaging, not directly resolved particle trajectories.
- Better as validation data than as the first trajectory generator.

## Practical Decision

Use this order:

1. Zenodo 18 micrometer Bentheimer: tiny end-to-end pipeline.
2. Zenodo 6 micrometer Bentheimer: realistic enough first simulation.
3. DPMP Bentheimer with precomputed velocities if accessible.
4. DRP-372 for geometry-conditioned ML once our model works on one sample.


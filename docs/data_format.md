# Trajectory Data Format

The comparison script can run immediately on synthetic data, but it is ready for DNS trajectories once they are available.

## Preferred Format

Use `.npz` with a key named `trajectories`.

Accepted shapes:

```text
(n_particles, n_steps, d)
```

or an object array containing variable-length arrays:

```text
trajectories[i].shape == (n_steps_i, d)
```

where `d` is usually 2 or 3.

## CSV Format

CSV files should contain:

```text
particle_id,time,x,y,z
0,0.000,0.0,0.1,0.2
0,0.001,0.1,0.1,0.2
1,0.000,0.0,0.3,0.4
```

Recognized particle id columns:

- `particle_id`
- `trajectory_id`
- `traj_id`
- `id`

Recognized time columns:

- `time`
- `t`
- `step`

Coordinate columns are inferred from `x,y,z` or `x,y`.

## First Real-Data Run

```bash
python3 scripts/compare_samplers.py \
  --input path/to/trajectories.npz \
  --key trajectories \
  --segment-steps 80 \
  --match-steps 16 \
  --dt 0.001 \
  --diffusivity 1e-9 \
  --planes 0.0008,0.0016,0.0027
```

If you have separate train and validation DNS trajectory sets:

```bash
python3 scripts/compare_samplers.py \
  --input path/to/train_trajectories.npz \
  --reference-input path/to/reference_trajectories.npz
```

## Notes

- `segment_steps` is the current discrete proxy for the 2019 paper's `lambda_v`.
- `match_steps` is the current discrete proxy for `lambda_alpha`.
- `dt` converts path increments into endpoint velocity descriptors.
- `diffusivity` controls the original TTA-style Gaussian transition bandwidth.


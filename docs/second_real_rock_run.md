# Second Real-Rock Run

## Dataset

Bentheimer sandstone from the 6 micrometer Zenodo volume:

```text
Core1_Subvol1_6micron_225cube_16bit_LE.raw
```

Source:

https://zenodo.org/records/5542624

For this run, the 225^3 CT volume was block-averaged by a factor of 3 to create a 75^3 bootstrap simulation. This keeps the first solver cheap while using the higher-resolution CT as the source image.

## Commands

```bash
python3 scripts/download_bentheimer_zenodo.py --file core1_subvol1_6um

python3 scripts/build_bentheimer_trajectories.py \
  --raw data/raw/Core1_Subvol1_6micron_225cube_16bit_LE.raw \
  --shape 225,225,225 \
  --voxel-size 6e-6 \
  --downsample-factor 3 \
  --particles 300 \
  --steps 500 \
  --pressure-iters 800 \
  --output data/processed/bentheimer_6um_downsample3_trajectories.npz

python3 scripts/tune_tta_parameters.py \
  --input data/processed/bentheimer_6um_downsample3_trajectories.npz \
  --planes 6,10,14 \
  --segment-steps 16,20,24,30,36 \
  --match-steps 3,4,5,6 \
  --gaussian-bandwidths 0.25,0.5,1,2,4,8 \
  --n-generated 70 \
  --n-segments 32 \
  --diffusivity 0.001 \
  --output outputs/bentheimer_6um_downsample3_parameter_scan.json

python3 scripts/compare_samplers.py \
  --input data/processed/bentheimer_6um_downsample3_trajectories.npz \
  --planes 6,10,14 \
  --segment-steps 20 \
  --match-steps 5 \
  --n-generated 90 \
  --n-segments 32 \
  --diffusivity 0.001 \
  --gaussian-bandwidth 0.25 \
  --output outputs/bentheimer_6um_downsample3_best_comparison.json
```

## Geometry And Trajectory Summary

```text
input shape:          225^3
simulation shape:      75^3
effective voxel size:  18 micrometers
raw porosity:           0.22794
connected porosity:     0.22543
trajectories:           300
mean path length:       501 samples
```

## Initial Parameter Scan Result

Top setting from the short-match scan:

```text
sampler:              gaussian_bayes
segment_steps:        20
match_steps:          5
bandwidth_multiplier: 0.25
score:                42.83
quantile MAE:         32.51
coverage deficit:     0.041
```

Clean rerun with 90 generated trajectories:

```text
unconditional       score 113.98   coverage deficit 0.10
knn_conditional     score  74.06   coverage deficit 0.10
gaussian_bayes      score  38.93   coverage deficit 0.05
```

The tuned original-style Gaussian/Bayes transition kernel is now the best performer. This is encouraging because it means the 2019 physics-informed transition idea is still valuable; the immediate ML opportunity is to learn that transition metric and bandwidth adaptively rather than hand-scan it.

## Autocorrelation-Guided Scan

The adjacent-segment velocity autocorrelation scan suggested:

```text
match_steps:   16
segment_steps: 24
```

Running a wider scan around those longer matching intervals produced a better setting:

```text
sampler:              gaussian_bayes
segment_steps:        36
match_steps:          20
bandwidth_multiplier: 0.25
scan score:           32.41
quantile MAE:         32.41
coverage deficit:     0.000
```

Clean rerun with 90 generated trajectories:

```text
unconditional       score 125.94   coverage deficit 0.07
knn_conditional     score  47.09   coverage deficit 0.10
gaussian_bayes      score  33.80   coverage deficit 0.00
```

This is now the best baseline result. The important lesson is that the autocorrelation diagnostic pushed us toward longer memory and matching scales than the first hand scan.

## Interpretation

The tuned Gaussian kernel prefers a narrow bandwidth and a relatively long matching interval. In this bootstrap voxel-flow model, that means generated paths improve when segment starts are required to match endpoint velocities tightly over a persistent local trajectory window. The kNN conditional sampler is more robust than unconditional shuffling, but the physically scaled Gaussian score is now winning once tuned.

## Next Step

See `full_metric_baseline.md` for the next benchmark layer: BTC, dilution, pair-separation, reaction proxy, and the first learned contrastive transition sampler.

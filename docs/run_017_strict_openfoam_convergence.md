# Run 017: Strict Full-Resolution OpenFOAM Convergence

## Purpose

Run 017 rechecked the full-resolution Core2 OpenFOAM flow solve after noticing that the previous full-resolution run stopped under loose outer SIMPLE residual controls. The earlier calculation was useful for the memory-selection story, but it was not a strict CFD convergence check: the printed OpenFOAM residuals are normalized algebraic residuals, not dimensional pressure or velocity errors, and the original run allowed nonzero relative solver tolerances and relatively loose SIMPLE residual control.

The goal here was to keep the same full-resolution voxel mesh and push the flow solve to a defensible fixed point.

## Case

```text
case:       openfoam_cases/bentheimer_core2_subvol1_6um_fullres_voxel_flow_strict
mesh:       symlink to the Run 016 full-resolution polyMesh
shape:      225 x 225 x 225
voxel size: 6 micrometers
cells:      2,650,688
points:     3,509,487
faces:      8,779,164
```

`checkMesh` passed with `Mesh OK`. The mesh is an orthogonal stair-step voxel mesh, not a smoothed pore-surface mesh.

## Strict Solver Controls

The strict case used the same Stokes/laminar OpenFOAM model, but changed the pressure scale and residual controls:

```text
inlet pressure: 1
outlet pressure: 0
p solver tolerance: 1e-12
U solver tolerance: 1e-12
p relTol: 0
U relTol: 0
SIMPLE residualControl p: 1e-9
SIMPLE residualControl U: 1e-9
relaxation: p = 0.3, U = 0.7
```

Using `p_inlet = 1` instead of `1e-6` improves numerical scale. Because the selected model is Stokes/laminar, the velocity field scales linearly with pressure drop; the apparent permeability is unchanged by this pressure scaling when computed with the matching pressure difference.

A brief attempt to increase relaxation to speed convergence caused a residual jump, so it was abandoned. The final accepted run restarted cleanly from the saved time-100 field using the conservative relaxation factors above.

## Convergence

The accepted strict solve reached formal OpenFOAM convergence:

```text
SIMPLE solution converged in 604 iterations
latest saved time: 604
```

The final residual-controlling step had all monitored outer residuals below `1e-9`:

```text
Ux initial residual: 4.53e-10
Uy initial residual: 7.55e-10
Uz initial residual: 9.94e-10
p initial residual:  1.03e-10
```

## Flow Summary

Final converged flow summary:

```text
time:                         604
mean speed:                   1.7482601e-02
median speed:                 9.5054135e-03
95th percentile speed:        6.0797975e-02
maximum speed:                9.3733257e-01
mean axial velocity:          1.1287675e-02
inlet flux:                  -4.7141658e-09
outlet flux:                  4.7141658e-09
net boundary flux:           -2.94e-18
bulk Darcy velocity:          2.5866479e-03
apparent permeability:        3.4919747e-12 m^2
```

The pressure drop in this strict case is one, so the listed Darcy velocity is correspondingly scaled. Under the original `1e-6` pressure drop convention, the Darcy velocity would be `2.5866479e-09`, with the same apparent permeability.

## Stability Of The Hydraulic Result

The hydraulic quantities stabilized long before the final residual criterion. Apparent permeability changed by less than one part in ten thousand after time 300:

```text
time     apparent permeability (m^2)
50       3.4856894e-12
100      3.4893498e-12
150      3.4910547e-12
200      3.4916543e-12
250      3.4918629e-12
300      3.4919357e-12
350      3.4919611e-12
400      3.4919700e-12
450      3.4919731e-12
500      3.4919741e-12
550      3.4919745e-12
604      3.4919747e-12
```

The earlier loose full-resolution summary reported `3.6109191e-12 m^2`. The strict converged value is therefore about `3.3%` lower:

```text
loose full-resolution k:   3.6109191e-12 m^2
strict converged k:        3.4919747e-12 m^2
relative correction:      -3.29%
```

## Interpretation

This correction does not change the qualitative memory-adequacy result. The full-resolution OpenFOAM case still supports the claim that better-resolved finite-volume flow gives velocity memory a real physical signal while balanced transport validation still selects mixed memories.

It does tighten the hydraulic statement. The permeability ladder should now be read as:

```text
18 um voxel OpenFOAM: 4.277e-12 m^2
12 um voxel OpenFOAM: 3.690e-12 m^2
6 um strict OpenFOAM: 3.492e-12 m^2
```

The finer two cases are still much closer to each other than either is to the coarsest case, but the full-resolution value is not quite as close to the 12 micrometer value as the loose solve suggested. That is a useful correction for the manuscript: the resolution ladder strengthens velocity-memory interpretation, but the hydraulic convergence claim should be stated as "tightening" rather than complete convergence.

## Outputs

```text
outputs/bentheimer_core2_subvol1_6um_fullres_openfoam_strict_converged_flow_summary.json
outputs/bentheimer_core2_subvol1_6um_fullres_openfoam_strict_time50_flow_summary.json
outputs/bentheimer_core2_subvol1_6um_fullres_openfoam_strict_time100_flow_summary.json
outputs/bentheimer_core2_subvol1_6um_fullres_openfoam_strict_time150_flow_summary.json
outputs/bentheimer_core2_subvol1_6um_fullres_openfoam_strict_time200_flow_summary.json
outputs/bentheimer_core2_subvol1_6um_fullres_openfoam_strict_time250_flow_summary.json
outputs/bentheimer_core2_subvol1_6um_fullres_openfoam_strict_time300_flow_summary.json
outputs/bentheimer_core2_subvol1_6um_fullres_openfoam_strict_time350_flow_summary.json
outputs/bentheimer_core2_subvol1_6um_fullres_openfoam_strict_time400_flow_summary.json
outputs/bentheimer_core2_subvol1_6um_fullres_openfoam_strict_time450_flow_summary.json
outputs/bentheimer_core2_subvol1_6um_fullres_openfoam_strict_time500_flow_summary.json
outputs/bentheimer_core2_subvol1_6um_fullres_openfoam_strict_time550_flow_summary.json
```

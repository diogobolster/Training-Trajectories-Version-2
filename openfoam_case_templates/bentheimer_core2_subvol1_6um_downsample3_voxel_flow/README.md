# Voxel OpenFOAM Case

This case was generated directly from the connected pore voxels. Each pore voxel is
one finite-volume cell; pore-solid faces are no-slip walls, and the two faces normal
to the selected flow axis are pressure inlet/outlet patches.

Recommended checks:

```bash
checkMesh
simpleFoam
foamToVTK
```

The current mesh is a first high-fidelity bridge: it removes the in-house Jacobi
pressure approximation, but it still uses a stair-step voxel geometry. A later
surface-smoothed or LBM-resolved field can be added as an even stronger validation
condition.

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tta_v2.geometry import (  # noqa: E402
    block_average,
    connected_pore_network,
    load_raw_volume,
    porosity,
    segment_pore_space,
)


LOCAL_CORNERS = (
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
)

FACE_VERTICES = {
    (-1, 0, 0): (3, 0, 4, 7),
    (1, 0, 0): (1, 2, 6, 5),
    (0, -1, 0): (0, 1, 5, 4),
    (0, 1, 0): (2, 3, 7, 6),
    (0, 0, -1): (0, 3, 2, 1),
    (0, 0, 1): (4, 5, 6, 7),
}

FACE_DIRECTIONS = tuple(FACE_VERTICES)
PATCH_ORDER = ("inlet", "outlet", "walls")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a connected voxel pore space as an OpenFOAM finite-volume case."
    )
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--shape", default="225,225,225")
    parser.add_argument("--dtype", default="<u2")
    parser.add_argument("--order", choices=["C", "F"], default="C")
    parser.add_argument("--voxel-size", type=float, default=6e-6)
    parser.add_argument("--downsample-factor", type=int, default=3)
    parser.add_argument(
        "--trim-to-factor",
        action="store_true",
        help="Trim trailing voxels so each dimension is divisible by --downsample-factor.",
    )
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--flow-axis", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--pressure-inlet", type=float, default=1e-6)
    parser.add_argument("--pressure-outlet", type=float, default=0.0)
    parser.add_argument("--nu", type=float, default=1e-6)
    parser.add_argument("--end-time", type=int, default=500)
    parser.add_argument("--write-interval", type=int, default=100)
    parser.add_argument(
        "--max-cells",
        type=int,
        default=150_000,
        help="Refuse to write larger cases unless set to 0.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "openfoam_cases" / "bentheimer_voxel_flow",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = absolutize_output_dir(args.output_dir)
    shape = parse_shape(args.shape)
    volume = load_raw_volume(args.raw, shape=shape, dtype=args.dtype, order=args.order)
    effective_voxel_size = args.voxel_size
    if args.downsample_factor > 1:
        if args.trim_to_factor:
            volume = trim_to_factor(volume, args.downsample_factor)
        volume = block_average(volume, args.downsample_factor)
        effective_voxel_size *= args.downsample_factor

    pore_mask = segment_pore_space(volume, threshold=args.threshold, pore_is_dark=True)
    connected = connected_pore_network(pore_mask, axis=args.flow_axis)
    n_cells = int(np.count_nonzero(connected))
    if n_cells == 0:
        raise RuntimeError("connected pore network is empty")
    if args.max_cells and n_cells > args.max_cells:
        raise RuntimeError(
            f"case would contain {n_cells} cells; rerun with --max-cells 0 to override"
        )

    mesh_info = build_poly_mesh(
        connected,
        flow_axis=args.flow_axis,
        voxel_size=effective_voxel_size,
    )
    write_case(
        args.output_dir,
        mesh_info=mesh_info,
        pressure_inlet=args.pressure_inlet,
        pressure_outlet=args.pressure_outlet,
        nu=args.nu,
        end_time=args.end_time,
        write_interval=args.write_interval,
    )

    summary = {
        "raw": str(args.raw),
        "input_shape": shape,
        "case_shape": tuple(int(item) for item in connected.shape),
        "voxel_size": effective_voxel_size,
        "downsample_factor": args.downsample_factor,
        "flow_axis": args.flow_axis,
        "raw_porosity": porosity(pore_mask),
        "connected_porosity": porosity(connected),
        "n_cells": n_cells,
        "n_points": len(mesh_info["points"]),
        "n_faces": len(mesh_info["faces"]),
        "n_internal_faces": len(mesh_info["neighbour"]),
        "patches": {
            name: int(mesh_info["patch_counts"].get(name, 0))
            for name in PATCH_ORDER
        },
        "pressure_inlet": args.pressure_inlet,
        "pressure_outlet": args.pressure_outlet,
        "nu": args.nu,
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "case_info.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print()
    print("Run with:")
    volume_arg = shlex.quote(f"{ROOT}:/work")
    workdir_arg = shlex.quote(f"/work/{args.output_dir.relative_to(ROOT)}")
    print(
        "docker run --rm --entrypoint /bin/bash "
        f"-v {volume_arg} -w {workdir_arg} "
        "openeuler/openfoam:2506-oe2403sp2 "
        "-lc 'source /opt/OpenFOAM-v2506/etc/bashrc && checkMesh'"
    )
    print(
        "docker run --rm --entrypoint /bin/bash "
        f"-v {volume_arg} -w {workdir_arg} "
        "openeuler/openfoam:2506-oe2403sp2 "
        "-lc 'source /opt/OpenFOAM-v2506/etc/bashrc && simpleFoam'"
    )


def parse_shape(value: str) -> tuple[int, int, int]:
    shape = tuple(int(item) for item in value.split(","))
    if len(shape) != 3:
        raise ValueError("--shape must contain three comma-separated integers")
    return shape


def absolutize_output_dir(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def trim_to_factor(volume: np.ndarray, factor: int) -> np.ndarray:
    trimmed_shape = tuple((size // factor) * factor for size in volume.shape)
    if any(size <= 0 for size in trimmed_shape):
        raise ValueError(f"cannot trim shape {volume.shape} to factor {factor}")
    slices = tuple(slice(0, size) for size in trimmed_shape)
    return volume[slices]


def build_poly_mesh(mask: np.ndarray, *, flow_axis: int, voxel_size: float) -> dict[str, object]:
    cell_coords = [tuple(int(item) for item in coord) for coord in np.argwhere(mask)]
    cell_ids = {coord: idx for idx, coord in enumerate(cell_coords)}
    point_ids: dict[tuple[int, int, int], int] = {}
    points: list[tuple[float, float, float]] = []
    internal_faces: list[tuple[int, int, tuple[int, ...]]] = []
    boundary_faces: dict[str, list[tuple[tuple[int, ...], int]]] = {
        name: [] for name in PATCH_ORDER
    }
    shape = mask.shape

    def point_id(coord: tuple[int, int, int]) -> int:
        if coord not in point_ids:
            point_ids[coord] = len(points)
            points.append(tuple(float(item) * voxel_size for item in coord))
        return point_ids[coord]

    for coord in cell_coords:
        cid = cell_ids[coord]
        corners = tuple(
            (
                coord[0] + offset[0],
                coord[1] + offset[1],
                coord[2] + offset[2],
            )
            for offset in LOCAL_CORNERS
        )
        for direction in FACE_DIRECTIONS:
            neighbour_coord = (
                coord[0] + direction[0],
                coord[1] + direction[1],
                coord[2] + direction[2],
            )
            face = tuple(point_id(corners[idx]) for idx in FACE_VERTICES[direction])
            if neighbour_coord in cell_ids:
                nid = cell_ids[neighbour_coord]
                if cid < nid:
                    internal_faces.append((cid, nid, face))
                continue

            patch = classify_patch(coord, direction, shape=shape, flow_axis=flow_axis)
            boundary_faces[patch].append((face, cid))

    internal_faces.sort(key=lambda item: (item[0], item[1]))
    faces = [face for _, _, face in internal_faces]
    owner = [cid for cid, _, _ in internal_faces]
    neighbour = [nid for _, nid, _ in internal_faces]
    boundary_start = len(faces)
    patch_starts: dict[str, int] = {}
    patch_counts: dict[str, int] = {}
    for patch in PATCH_ORDER:
        patch_starts[patch] = len(faces)
        patch_counts[patch] = len(boundary_faces[patch])
        for face, cid in boundary_faces[patch]:
            faces.append(face)
            owner.append(cid)

    if boundary_start != len(neighbour):
        raise RuntimeError("internal face count and neighbour list length diverged")

    return {
        "points": points,
        "faces": faces,
        "owner": owner,
        "neighbour": neighbour,
        "patch_starts": patch_starts,
        "patch_counts": patch_counts,
    }


def classify_patch(
    coord: tuple[int, int, int],
    direction: tuple[int, int, int],
    *,
    shape: tuple[int, int, int],
    flow_axis: int,
) -> str:
    if direction[flow_axis] < 0 and coord[flow_axis] == 0:
        return "inlet"
    if direction[flow_axis] > 0 and coord[flow_axis] == shape[flow_axis] - 1:
        return "outlet"
    return "walls"


def write_case(
    case_dir: Path,
    *,
    mesh_info: dict[str, object],
    pressure_inlet: float,
    pressure_outlet: float,
    nu: float,
    end_time: int,
    write_interval: int,
) -> None:
    poly_dir = case_dir / "constant" / "polyMesh"
    zero_dir = case_dir / "0"
    system_dir = case_dir / "system"
    poly_dir.mkdir(parents=True, exist_ok=True)
    zero_dir.mkdir(parents=True, exist_ok=True)
    system_dir.mkdir(parents=True, exist_ok=True)

    write_points(poly_dir / "points", mesh_info["points"])
    write_faces(poly_dir / "faces", mesh_info["faces"])
    write_label_list(poly_dir / "owner", "owner", mesh_info["owner"])
    write_label_list(poly_dir / "neighbour", "neighbour", mesh_info["neighbour"])
    write_boundary(poly_dir / "boundary", mesh_info["patch_starts"], mesh_info["patch_counts"])
    write_field_files(zero_dir, pressure_inlet=pressure_inlet, pressure_outlet=pressure_outlet)
    write_constant_files(case_dir / "constant", nu=nu)
    write_system_files(system_dir, end_time=end_time, write_interval=write_interval)
    write_readme(case_dir)


def foam_header(class_name: str, object_name: str, *, location: str | None = None) -> str:
    location_line = f'    location    "{location}";\n' if location else ""
    return (
        "FoamFile\n"
        "{\n"
        "    version     2.0;\n"
        "    format      ascii;\n"
        f"    class       {class_name};\n"
        f"{location_line}"
        f"    object      {object_name};\n"
        "}\n\n"
    )


def write_points(path: Path, points: list[tuple[float, float, float]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(foam_header("vectorField", "points", location="constant/polyMesh"))
        handle.write(f"{len(points)}\n(\n")
        for x, y, z in points:
            handle.write(f"({x:.12g} {y:.12g} {z:.12g})\n")
        handle.write(")\n")


def write_faces(path: Path, faces: list[tuple[int, ...]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(foam_header("faceList", "faces", location="constant/polyMesh"))
        handle.write(f"{len(faces)}\n(\n")
        for face in faces:
            handle.write(f"{len(face)}({' '.join(str(item) for item in face)})\n")
        handle.write(")\n")


def write_label_list(path: Path, object_name: str, values: list[int]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(foam_header("labelList", object_name, location="constant/polyMesh"))
        handle.write(f"{len(values)}\n(\n")
        for value in values:
            handle.write(f"{value}\n")
        handle.write(")\n")


def write_boundary(path: Path, patch_starts: dict[str, int], patch_counts: dict[str, int]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(foam_header("polyBoundaryMesh", "boundary", location="constant/polyMesh"))
        handle.write(f"{len(PATCH_ORDER)}\n(\n")
        for patch in PATCH_ORDER:
            patch_type = "wall" if patch == "walls" else "patch"
            handle.write(
                f"    {patch}\n"
                "    {\n"
                f"        type            {patch_type};\n"
                f"        nFaces          {patch_counts[patch]};\n"
                f"        startFace       {patch_starts[patch]};\n"
                "    }\n"
            )
        handle.write(")\n")


def write_field_files(zero_dir: Path, *, pressure_inlet: float, pressure_outlet: float) -> None:
    (zero_dir / "p").write_text(
        foam_header("volScalarField", "p", location="0")
        + f"""dimensions      [0 2 -2 0 0 0 0];
internalField   uniform {pressure_outlet:.12g};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {pressure_inlet:.12g};
    }}
    outlet
    {{
        type            fixedValue;
        value           uniform {pressure_outlet:.12g};
    }}
    walls
    {{
        type            zeroGradient;
    }}
}}
""",
        encoding="utf-8",
    )
    (zero_dir / "U").write_text(
        foam_header("volVectorField", "U", location="0")
        + """dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);

boundaryField
{
    inlet
    {
        type            zeroGradient;
    }
    outlet
    {
        type            zeroGradient;
    }
    walls
    {
        type            noSlip;
    }
}
""",
        encoding="utf-8",
    )


def write_constant_files(constant_dir: Path, *, nu: float) -> None:
    (constant_dir / "transportProperties").write_text(
        foam_header("dictionary", "transportProperties", location="constant")
        + f"""transportModel  Newtonian;

nu              [0 2 -1 0 0 0 0] {nu:.12g};
""",
        encoding="utf-8",
    )
    (constant_dir / "turbulenceProperties").write_text(
        foam_header("dictionary", "turbulenceProperties", location="constant")
        + """simulationType  laminar;
""",
        encoding="utf-8",
    )


def write_system_files(system_dir: Path, *, end_time: int, write_interval: int) -> None:
    (system_dir / "controlDict").write_text(
        foam_header("dictionary", "controlDict", location="system")
        + f"""application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time};
deltaT          1;
writeControl    timeStep;
writeInterval   {write_interval};
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
""",
        encoding="utf-8",
    )
    (system_dir / "fvSchemes").write_text(
        foam_header("dictionary", "fvSchemes", location="system")
        + """ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}
""",
        encoding="utf-8",
    )
    (system_dir / "fvSolution").write_text(
        foam_header("dictionary", "fvSolution", location="system")
        + """solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-07;
        relTol          0.05;
        smoother        GaussSeidel;
    }

    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-08;
        relTol          0.1;
    }
}

SIMPLE
{
    consistent              yes;
    nNonOrthogonalCorrectors 1;
    residualControl
    {
        p               1e-05;
        U               1e-06;
    }
}

relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
    }
}
""",
        encoding="utf-8",
    )


def write_readme(case_dir: Path) -> None:
    (case_dir / "README.md").write_text(
        """# Voxel OpenFOAM Case

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
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tta_v2.openfoam import (  # noqa: E402
    latest_time_dir,
    read_internal_vector_field,
    read_patch_scalar_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a solved OpenFOAM pore-flow case.")
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--time", default="latest")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_dir = args.case_dir
    info = json.loads((case_dir / "case_info.json").read_text(encoding="utf-8"))
    time_dir = latest_time_dir(case_dir) if args.time == "latest" else case_dir / args.time

    u = read_internal_vector_field(time_dir / "U")
    speed = np.linalg.norm(u, axis=1)
    phi_path = time_dir / "phi"
    inlet_phi = read_patch_scalar_values(
        phi_path,
        "inlet",
        expected_count=info["patches"]["inlet"],
    )
    outlet_phi = read_patch_scalar_values(
        phi_path,
        "outlet",
        expected_count=info["patches"]["outlet"],
    )
    wall_phi = read_patch_scalar_values(
        phi_path,
        "walls",
        expected_count=info["patches"]["walls"],
    )

    voxel_size = float(info["voxel_size"])
    flow_axis = int(info["flow_axis"])
    case_shape = tuple(int(item) for item in info["case_shape"])
    transverse_axes = [axis for axis in range(3) if axis != flow_axis]
    bulk_area = float(np.prod([case_shape[axis] * voxel_size for axis in transverse_axes]))
    pore_inlet_area = float(info["patches"]["inlet"] * voxel_size**2)
    pore_outlet_area = float(info["patches"]["outlet"] * voxel_size**2)
    length = float(case_shape[flow_axis] * voxel_size)
    delta_p = float(info["pressure_inlet"]) - float(info["pressure_outlet"])
    nu = float(info["nu"])
    outlet_flux = float(np.sum(outlet_phi))
    inlet_flux = float(np.sum(inlet_phi))
    wall_flux = float(np.sum(wall_phi))
    darcy_velocity = outlet_flux / bulk_area
    apparent_permeability = darcy_velocity * nu * length / delta_p if delta_p else float("nan")

    summary = {
        "case_dir": str(case_dir),
        "time": time_dir.name,
        "n_cells": int(u.shape[0]),
        "speed_mean": float(np.mean(speed)),
        "speed_median": float(np.median(speed)),
        "speed_p95": float(np.quantile(speed, 0.95)),
        "speed_max": float(np.max(speed)),
        "axial_velocity_mean": float(np.mean(u[:, flow_axis])),
        "axial_velocity_median": float(np.median(u[:, flow_axis])),
        "inlet_flux": inlet_flux,
        "outlet_flux": outlet_flux,
        "wall_flux": wall_flux,
        "net_boundary_flux": inlet_flux + outlet_flux + wall_flux,
        "bulk_cross_section_area": bulk_area,
        "pore_inlet_area": pore_inlet_area,
        "pore_outlet_area": pore_outlet_area,
        "darcy_velocity_bulk_area": darcy_velocity,
        "outlet_interstitial_velocity": outlet_flux / pore_outlet_area,
        "apparent_permeability": apparent_permeability,
    }
    output = args.output or case_dir / "flow_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
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
from tta_v2.io import save_trajectories_npz  # noqa: E402
from tta_v2.openfoam import latest_time_dir, read_internal_vector_field  # noqa: E402
from tta_v2.tracking import trace_particles  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace particles through a solved OpenFOAM pore-flow field.")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--time", default="latest")
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
    parser.add_argument("--target-mean-speed", type=float, default=0.06)
    parser.add_argument("--particles", type=int, default=500)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--diffusivity", type=float, default=0.001)
    parser.add_argument(
        "--max-advective-step",
        type=float,
        help="Maximum advective displacement, in voxels, allowed within an internal substep.",
    )
    parser.add_argument(
        "--max-diffusive-step",
        type=float,
        help="Maximum three-dimensional RMS diffusive displacement, in voxels, allowed within an internal substep.",
    )
    parser.add_argument("--max-substeps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "openfoam_trajectories.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shape = tuple(int(item) for item in args.shape.split(","))
    if len(shape) != 3:
        raise ValueError("--shape must contain three comma-separated integers")

    volume = load_raw_volume(args.raw, shape=shape, dtype=args.dtype, order=args.order)
    effective_voxel_size = args.voxel_size
    if args.downsample_factor > 1:
        if args.trim_to_factor:
            volume = trim_to_factor(volume, args.downsample_factor)
        volume = block_average(volume, args.downsample_factor)
        effective_voxel_size *= args.downsample_factor

    pore_mask = segment_pore_space(volume, threshold=args.threshold, pore_is_dark=True)
    connected = connected_pore_network(pore_mask, axis=args.flow_axis)
    time_dir = latest_time_dir(args.case_dir) if args.time == "latest" else args.case_dir / args.time
    cell_velocities_m_per_s = read_internal_vector_field(time_dir / "U")
    cell_coords = np.argwhere(connected)
    if len(cell_coords) != len(cell_velocities_m_per_s):
        raise ValueError(
            f"OpenFOAM velocity count {len(cell_velocities_m_per_s)} does not match "
            f"connected pore cell count {len(cell_coords)}"
        )

    velocity = np.zeros(connected.shape + (3,), dtype=float)
    velocity[connected] = cell_velocities_m_per_s / effective_voxel_size
    physical_mean_speed_voxels_per_second = mean_nonzero_speed(velocity, connected)
    if args.target_mean_speed > 0:
        velocity *= args.target_mean_speed / physical_mean_speed_voxels_per_second
    simulation_mean_speed = mean_nonzero_speed(velocity, connected)

    tracking_diagnostics: dict[str, float] = {}
    trajectories = trace_particles(
        velocity,
        connected,
        n_particles=args.particles,
        n_steps=args.steps,
        dt=args.dt,
        diffusivity=args.diffusivity,
        axis=args.flow_axis,
        seed=args.seed,
        max_advective_step=args.max_advective_step,
        max_diffusive_step=args.max_diffusive_step,
        max_substeps=args.max_substeps,
        diagnostics=tracking_diagnostics,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_trajectories_npz(args.output, trajectories)
    summary = {
        "raw": str(args.raw),
        "case_dir": str(args.case_dir),
        "time": time_dir.name,
        "input_shape": shape,
        "simulation_shape": tuple(int(item) for item in connected.shape),
        "voxel_size": effective_voxel_size,
        "downsample_factor": args.downsample_factor,
        "raw_porosity": porosity(pore_mask),
        "connected_porosity": porosity(connected),
        "physical_mean_speed_voxels_per_second": physical_mean_speed_voxels_per_second,
        "target_mean_speed": args.target_mean_speed,
        "simulation_mean_speed": simulation_mean_speed,
        "particles_requested": args.particles,
        "steps_requested": args.steps,
        "dt": args.dt,
        "diffusivity": args.diffusivity,
        "max_advective_step": args.max_advective_step,
        "max_diffusive_step": args.max_diffusive_step,
        "max_substeps": args.max_substeps,
        "seed": args.seed,
        "n_trajectories": len(trajectories),
        "mean_trajectory_length": float(np.mean([len(t) for t in trajectories])) if trajectories else 0.0,
        "tracking_diagnostics": tracking_diagnostics,
        "output": str(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def mean_nonzero_speed(velocity: np.ndarray, mask: np.ndarray) -> float:
    speed = np.linalg.norm(velocity[mask], axis=1)
    positive = speed[speed > 0]
    if len(positive) == 0:
        raise ValueError("velocity field has no nonzero pore speeds")
    return float(np.mean(positive))


def trim_to_factor(volume: np.ndarray, factor: int) -> np.ndarray:
    trimmed_shape = tuple((size // factor) * factor for size in volume.shape)
    if any(size <= 0 for size in trimmed_shape):
        raise ValueError(f"cannot trim shape {volume.shape} to factor {factor}")
    slices = tuple(slice(0, size) for size in trimmed_shape)
    return volume[slices]


if __name__ == "__main__":
    main()

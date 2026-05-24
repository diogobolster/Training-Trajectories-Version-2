from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tta_v2.flow import solve_pressure_jacobi, velocity_from_pressure  # noqa: E402
from tta_v2.geometry import (  # noqa: E402
    block_average,
    connected_pore_network,
    load_raw_volume,
    porosity,
    segment_pore_space,
)
from tta_v2.io import save_trajectories_npz  # noqa: E402
from tta_v2.tracking import trace_particles  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build approximate particle trajectories from Bentheimer CT data.")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--shape", default="75,75,75")
    parser.add_argument("--dtype", default="<u2")
    parser.add_argument("--order", choices=["C", "F"], default="C")
    parser.add_argument("--voxel-size", type=float, default=18e-6)
    parser.add_argument("--downsample-factor", type=int, default=1)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--pressure-iters", type=int, default=1200)
    parser.add_argument("--particles", type=int, default=500)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--diffusivity", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "bentheimer_trajectories.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shape = tuple(int(item) for item in args.shape.split(","))
    if len(shape) != 3:
        raise ValueError("--shape must contain three comma-separated integers")

    volume = load_raw_volume(args.raw, shape=shape, dtype=args.dtype, order=args.order)
    effective_voxel_size = args.voxel_size
    if args.downsample_factor > 1:
        volume = block_average(volume, args.downsample_factor)
        effective_voxel_size *= args.downsample_factor

    pore_mask = segment_pore_space(volume, threshold=args.threshold, pore_is_dark=True)
    connected = connected_pore_network(pore_mask, axis=0)
    pressure, solver_info = solve_pressure_jacobi(
        connected,
        n_iters=args.pressure_iters,
        tolerance=1e-6,
    )
    velocity = velocity_from_pressure(pressure, connected, target_mean_speed=0.06)
    trajectories = trace_particles(
        velocity,
        connected,
        n_particles=args.particles,
        n_steps=args.steps,
        dt=args.dt,
        diffusivity=args.diffusivity,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_trajectories_npz(args.output, trajectories)

    summary = {
        "raw": str(args.raw),
        "input_shape": shape,
        "simulation_shape": tuple(int(item) for item in volume.shape),
        "voxel_size": effective_voxel_size,
        "downsample_factor": args.downsample_factor,
        "raw_porosity": porosity(pore_mask),
        "connected_porosity": porosity(connected),
        "pressure_solver": solver_info,
        "particles_requested": args.particles,
        "steps_requested": args.steps,
        "dt": args.dt,
        "diffusivity": args.diffusivity,
        "seed": args.seed,
        "n_trajectories": len(trajectories),
        "mean_trajectory_length": float(np.mean([len(t) for t in trajectories])) if trajectories else 0.0,
        "output": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

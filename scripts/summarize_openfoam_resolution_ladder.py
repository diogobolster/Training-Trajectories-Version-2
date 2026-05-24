from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tta_v2 import load_trajectories, velocity_autocorrelation  # noqa: E402


CASES = [
    {
        "name": "downsample3",
        "label": "18 um voxel OpenFOAM",
        "case_dir": ROOT / "openfoam_cases" / "bentheimer_core2_subvol1_6um_downsample3_voxel_flow",
        "flow_summary": ROOT / "openfoam_cases" / "bentheimer_core2_subvol1_6um_downsample3_voxel_flow" / "flow_summary.json",
        "trajectory": ROOT
        / "data"
        / "processed"
        / "bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_dt010_n5000_trajectories.npz",
        "balanced": ROOT
        / "outputs"
        / "bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_dt010_n5000_stride400_outer_split_mixture_benchmark.json",
        "sensitivity": ROOT
        / "outputs"
        / "bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_dt010_n5000_stride400_objective_weight_sensitivity.json",
    },
    {
        "name": "downsample2",
        "label": "12 um voxel OpenFOAM",
        "case_dir": ROOT / "openfoam_cases" / "bentheimer_core2_subvol1_6um_downsample2_voxel_flow",
        "flow_summary": ROOT / "openfoam_cases" / "bentheimer_core2_subvol1_6um_downsample2_voxel_flow" / "flow_summary.json",
        "trajectory": ROOT
        / "data"
        / "processed"
        / "bentheimer_core2_subvol1_6um_downsample2_D00225_openfoam_phys_scaled_dt010_n5000_trajectories.npz",
        "balanced": ROOT
        / "outputs"
        / "bentheimer_core2_subvol1_6um_downsample2_D00225_openfoam_phys_scaled_dt010_n5000_stride400_outer_split_mixture_benchmark.json",
        "sensitivity": ROOT
        / "outputs"
        / "bentheimer_core2_subvol1_6um_downsample2_D00225_openfoam_phys_scaled_dt010_n5000_stride400_objective_weight_sensitivity.json",
    },
    {
        "name": "fullres",
        "label": "6 um voxel OpenFOAM",
        "case_dir": ROOT / "openfoam_cases" / "bentheimer_core2_subvol1_6um_fullres_voxel_flow_strict",
        "flow_summary": ROOT / "outputs" / "bentheimer_core2_subvol1_6um_fullres_openfoam_strict_converged_flow_summary.json",
        "trajectory": ROOT
        / "data"
        / "processed"
        / "bentheimer_core2_subvol1_6um_fullres_D009_openfoam_strict_dt010_n5000_trajectories.npz",
        "balanced": ROOT
        / "outputs"
        / "bentheimer_core2_subvol1_6um_fullres_D009_openfoam_strict_dt010_n5000_stride400_outer_split_mixture_benchmark.json",
        "sensitivity": ROOT
        / "outputs"
        / "bentheimer_core2_subvol1_6um_fullres_D009_openfoam_strict_dt010_n5000_stride400_objective_weight_sensitivity.json",
    },
]


def main() -> None:
    output = ROOT / "outputs" / "openfoam_resolution_ladder_summary.json"
    summary = {
        "description": "Resolution ladder for Core2 OpenFOAM trajectory-memory tests using tight dt=0.1, n=5000 particle archives.",
        "trajectory_protocol": {
            "particles": 5000,
            "dt": 0.1,
            "steps": 4000,
            "segment_steps": 160,
            "segment_stride": 400,
        },
        "autocorrelation_lags": [10, 20, 40, 80],
        "cases": {},
    }
    for item in CASES:
        flow = read_json(item["flow_summary"])
        balanced = read_json(item["balanced"])
        sensitivity = read_json(item["sensitivity"])
        trajectories = load_trajectories(item["trajectory"])
        corr = velocity_autocorrelation(trajectories, max_lag=80)
        lag_values = {
            str(lag): float(corr[lag])
            for lag in summary["autocorrelation_lags"]
            if lag < len(corr)
        }
        summary["cases"][item["name"]] = {
            "label": item["label"],
            "case_dir": rel(item["case_dir"]),
            "flow_summary": rel(item["flow_summary"]),
            "trajectory": rel(item["trajectory"]),
            "n_cells": flow["n_cells"],
            "solve_time_step": flow.get("time"),
            "apparent_permeability": flow["apparent_permeability"],
            "darcy_velocity_bulk_area": flow["darcy_velocity_bulk_area"],
            "autocorrelation": lag_values,
            "balanced": summarize_balanced(balanced),
            "objective_winners": summarize_sensitivity(sensitivity),
        }
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {rel(output)}")
    print()
    print("Resolution ladder")
    for name, case in summary["cases"].items():
        balanced = case["balanced"]
        print(
            f"{name:11s} cells={case['n_cells']:,} "
            f"k={case['apparent_permeability']:.3e} "
            f"best={balanced['best_sampler']} obj={balanced['best_mean_objective']:.2f} "
            f"weights={balanced['mean_selected_weights']}"
        )
    print()
    print("Objective winners")
    for name, case in summary["cases"].items():
        winners = ", ".join(
            f"{regime}:{entry['best_sampler']}"
            for regime, entry in case["objective_winners"].items()
        )
        print(f"{name:11s} {winners}")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def summarize_balanced(payload: dict) -> dict:
    samplers = payload["summary"]["samplers"]
    best_name, best = min(
        samplers.items(),
        key=lambda pair: pair[1]["mean_objective"],
    )
    return {
        "best_sampler": best_name,
        "best_mean_objective": best["mean_objective"],
        "best_mean_rank": best["mean_rank"],
        "best_wins": best["wins"],
        "samplers": samplers,
        "mean_selected_weights": payload["summary"]["mean_selected_weights"],
    }


def summarize_sensitivity(payload: dict) -> dict:
    output = {}
    for regime, regime_payload in payload["summary"].items():
        best_name, best = min(
            regime_payload["samplers"].items(),
            key=lambda pair: pair[1]["mean_objective"],
        )
        output[regime] = {
            "best_sampler": best_name,
            "mean_objective": best["mean_objective"],
            "mean_rank": best["mean_rank"],
            "wins": best["wins"],
        }
    return output


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BenchmarkJob:
    name: str
    output: Path
    command: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run manuscript-facing memory benchmarks on the 20,000-particle trajectory archives."
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="Optional job names to run. Defaults to every benchmark whose output is missing.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate outputs even if they already exist.")
    parser.add_argument("--list", action="store_true", help="List benchmark jobs and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = build_jobs()
    if args.list:
        for job in jobs:
            print(job.name)
        return

    selected = set(args.only or [job.name for job in jobs])
    unknown = selected.difference(job.name for job in jobs)
    if unknown:
        raise ValueError(f"unknown job names: {', '.join(sorted(unknown))}")

    for job in jobs:
        if job.name not in selected:
            continue
        if not args.force and job.output.exists() and output_complete(job.output):
            print(f"[skip] {job.name}: {job.output}", flush=True)
            continue
        print(f"[run] {job.name}", flush=True)
        print("      " + " ".join(job.command), flush=True)
        subprocess.run(job.command, cwd=ROOT, check=True)
        if not output_complete(job.output):
            raise RuntimeError(f"{job.output} was not written as a complete JSON output")
        print(f"[done] {job.name}: {job.output}", flush=True)


def output_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return "summary" in payload


def build_jobs() -> list[BenchmarkJob]:
    outputs = ROOT / "outputs"
    processed = ROOT / "data" / "processed"
    jobs: list[BenchmarkJob] = []

    graph_balanced = [
        (
            "core1_high_pe_graph_balanced_20k",
            processed / "bentheimer_6um_downsample3_D0003_n20000_steps800_trajectories.npz",
            0.0003,
            outputs / "bentheimer_6um_downsample3_D0003_n20000_stride400_outer_split_mixture_benchmark.json",
        ),
        (
            "core1_baseline_graph_balanced_20k",
            processed / "bentheimer_6um_downsample3_D001_n20000_steps800_trajectories.npz",
            0.001,
            outputs / "bentheimer_6um_downsample3_D001_n20000_stride400_outer_split_mixture_benchmark.json",
        ),
        (
            "core1_low_pe_graph_balanced_20k",
            processed / "bentheimer_6um_downsample3_D003_n20000_steps800_trajectories.npz",
            0.003,
            outputs / "bentheimer_6um_downsample3_D003_n20000_stride400_outer_split_mixture_benchmark.json",
        ),
        (
            "core2_graph_balanced_20k",
            processed / "bentheimer_core2_subvol1_6um_downsample3_D001_n20000_steps800_trajectories.npz",
            0.001,
            outputs
            / "bentheimer_core2_subvol1_6um_downsample3_D001_n20000_stride400_outer_split_mixture_benchmark.json",
        ),
    ]
    for name, input_path, diffusivity, output in graph_balanced:
        jobs.append(
            BenchmarkJob(
                name=name,
                output=output,
                command=outer_split_command(
                    input_path=input_path,
                    output=output,
                    diffusivity=diffusivity,
                    dt=0.5,
                    segment_steps=36,
                    match_steps=20,
                    segment_stride=400,
                    n_segments=32,
                    planes="6,10,14",
                    time_indices="100,200,300,400",
                    bin_size=3.0,
                    reaction_radius=3.0,
                ),
            )
        )

    openfoam_balanced = [
        (
            "core2_openfoam_18um_balanced_20k",
            processed / "bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_dt010_n20000_trajectories.npz",
            0.001,
            80,
            "6,10,14",
            3.0,
            outputs
            / "bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_dt010_n20000_stride1600_outer_split_mixture_benchmark.json",
        ),
        (
            "core2_openfoam_12um_balanced_20k",
            processed
            / "bentheimer_core2_subvol1_6um_downsample2_D00225_openfoam_phys_scaled_dt010_n20000_trajectories.npz",
            0.00225,
            60,
            "9,15,21",
            4.5,
            outputs
            / "bentheimer_core2_subvol1_6um_downsample2_D00225_openfoam_phys_scaled_dt010_n20000_stride1600_outer_split_mixture_benchmark.json",
        ),
        (
            "core2_openfoam_6um_balanced_20k",
            processed / "bentheimer_core2_subvol1_6um_fullres_D009_openfoam_strict_dt010_n20000_trajectories.npz",
            0.009,
            60,
            "18,30,42",
            9.0,
            outputs
            / "bentheimer_core2_subvol1_6um_fullres_D009_openfoam_strict_dt010_n20000_stride1600_outer_split_mixture_benchmark.json",
        ),
    ]
    for name, input_path, diffusivity, match_steps, planes, scale, output in openfoam_balanced:
        jobs.append(
            BenchmarkJob(
                name=name,
                output=output,
                command=outer_split_command(
                    input_path=input_path,
                    output=output,
                    diffusivity=diffusivity,
                    dt=0.1,
                    segment_steps=160,
                    match_steps=match_steps,
                    segment_stride=1600,
                    n_segments=8,
                    planes=planes,
                    time_indices="500,1000,1500,2000",
                    bin_size=scale,
                    reaction_radius=scale,
                ),
            )
        )

    jobs.append(
        BenchmarkJob(
            name="core1_baseline_graph_objectives_20k",
            output=outputs / "bentheimer_6um_downsample3_D001_n20000_stride400_objective_weight_sensitivity.json",
            command=objective_command(
                input_path=processed / "bentheimer_6um_downsample3_D001_n20000_steps800_trajectories.npz",
                output=outputs / "bentheimer_6um_downsample3_D001_n20000_stride400_objective_weight_sensitivity.json",
                diffusivity=0.001,
                dt=0.5,
                segment_steps=36,
                match_steps=20,
                segment_stride=400,
                n_segments=32,
                planes="6,10,14",
                time_indices="100,200,300,400",
                bin_size=3.0,
                reaction_radius=3.0,
                regimes=None,
            ),
        )
    )
    jobs.append(
        BenchmarkJob(
            name="core1_baseline_graph_breakthrough_only_20k",
            output=outputs / "bentheimer_6um_downsample3_D001_n20000_stride400_breakthrough_only_failure.json",
            command=objective_command(
                input_path=processed / "bentheimer_6um_downsample3_D001_n20000_steps800_trajectories.npz",
                output=outputs / "bentheimer_6um_downsample3_D001_n20000_stride400_breakthrough_only_failure.json",
                diffusivity=0.001,
                dt=0.5,
                segment_steps=36,
                match_steps=20,
                segment_stride=400,
                n_segments=32,
                planes="6,10,14",
                time_indices="100,200,300,400",
                bin_size=3.0,
                reaction_radius=3.0,
                regimes="balanced,breakthrough_only",
            ),
        )
    )

    for name, input_path, diffusivity, match_steps, planes, scale, output in openfoam_balanced:
        sensitivity_output = Path(str(output).replace("_outer_split_mixture_benchmark.json", "_objective_weight_sensitivity.json"))
        jobs.append(
            BenchmarkJob(
                name=name.replace("_balanced_", "_objectives_"),
                output=sensitivity_output,
                command=objective_command(
                    input_path=input_path,
                    output=sensitivity_output,
                    diffusivity=diffusivity,
                    dt=0.1,
                    segment_steps=160,
                    match_steps=match_steps,
                    segment_stride=1600,
                    n_segments=8,
                    planes=planes,
                    time_indices="500,1000,1500,2000",
                    bin_size=scale,
                    reaction_radius=scale,
                    regimes=None,
                ),
            )
        )

    return jobs


def outer_split_command(
    *,
    input_path: Path,
    output: Path,
    diffusivity: float,
    dt: float,
    segment_steps: int,
    match_steps: int,
    segment_stride: int,
    n_segments: int,
    planes: str,
    time_indices: str,
    bin_size: float,
    reaction_radius: float,
) -> list[str]:
    return [
        sys.executable,
        "scripts/outer_split_mixture_benchmark.py",
        "--input",
        str(input_path),
        "--n-outer-splits",
        "4",
        "--n-repeats",
        "3",
        "--grid-step",
        "0.25",
        "--n-validation-generated",
        "60",
        "--n-test-generated",
        "120",
        "--pair-samples",
        "3000",
        "--contrastive-epochs",
        "300",
        "--contrastive-negative-ratio",
        "6",
        "--hybrid-learned-weight",
        "0.25",
        "--pair-rerank-weight",
        "0.25",
        "--segment-steps",
        str(segment_steps),
        "--match-steps",
        str(match_steps),
        "--segment-stride",
        str(segment_stride),
        "--n-segments",
        str(n_segments),
        "--dt",
        str(dt),
        "--diffusivity",
        str(diffusivity),
        "--planes",
        planes,
        "--time-indices",
        time_indices,
        "--bin-size",
        str(bin_size),
        "--reaction-radius",
        str(reaction_radius),
        "--btc-weight",
        "1",
        "--pair-weight",
        "20",
        "--dilution-weight",
        "120",
        "--reaction-weight",
        "1000",
        "--output",
        str(output),
    ]


def objective_command(
    *,
    input_path: Path,
    output: Path,
    diffusivity: float,
    dt: float,
    segment_steps: int,
    match_steps: int,
    segment_stride: int,
    n_segments: int,
    planes: str,
    time_indices: str,
    bin_size: float,
    reaction_radius: float,
    regimes: str | None,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/objective_weight_sensitivity.py",
        "--input",
        str(input_path),
        "--n-outer-splits",
        "4",
        "--n-repeats",
        "3",
        "--grid-step",
        "0.25",
        "--n-validation-generated",
        "50",
        "--n-test-generated",
        "100",
        "--pair-samples",
        "2500",
        "--contrastive-epochs",
        "300",
        "--contrastive-negative-ratio",
        "6",
        "--hybrid-learned-weight",
        "0.25",
        "--pair-rerank-weight",
        "0.25",
        "--segment-steps",
        str(segment_steps),
        "--match-steps",
        str(match_steps),
        "--segment-stride",
        str(segment_stride),
        "--n-segments",
        str(n_segments),
        "--dt",
        str(dt),
        "--diffusivity",
        str(diffusivity),
        "--planes",
        planes,
        "--time-indices",
        time_indices,
        "--bin-size",
        str(bin_size),
        "--reaction-radius",
        str(reaction_radius),
        "--output",
        str(output),
    ]
    if regimes is not None:
        command.extend(["--regimes", regimes])
    return command


if __name__ == "__main__":
    main()

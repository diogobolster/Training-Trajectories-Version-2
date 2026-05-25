from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ArchiveJob:
    name: str
    output: Path
    command: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the 20,000-particle final trajectory archives for graph-flow and OpenFOAM cases."
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="Optional job names to run. Defaults to every archive not already complete.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate archives even if a complete output exists.")
    parser.add_argument(
        "--particles",
        type=int,
        default=20000,
        help="Number of particles to track for each archive.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = build_jobs(args.particles)
    selected = set(args.only or [job.name for job in jobs])
    unknown = selected.difference(job.name for job in jobs)
    if unknown:
        raise ValueError(f"unknown job names: {', '.join(sorted(unknown))}")

    for job in jobs:
        if job.name not in selected:
            continue
        if not args.force and archive_complete(job.output, args.particles):
            print(f"[skip] {job.name}: {job.output} already has {args.particles} trajectories", flush=True)
            continue
        print(f"[run] {job.name}", flush=True)
        print("      " + " ".join(job.command), flush=True)
        subprocess.run(job.command, cwd=ROOT, check=True)
        if not archive_complete(job.output, args.particles):
            raise RuntimeError(f"{job.output} was written but does not report {args.particles} trajectories")
        print(f"[done] {job.name}: {job.output}", flush=True)


def archive_complete(path: Path, particles: int) -> bool:
    summary_path = path.with_suffix(".summary.json")
    if not path.exists() or not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return int(summary.get("n_trajectories", 0)) >= particles


def build_jobs(particles: int) -> list[ArchiveJob]:
    core1_raw = ROOT / "data" / "raw" / "Core1_Subvol1_6micron_225cube_16bit_LE.raw"
    core2_raw = ROOT / "data" / "raw" / "Core2_Subvol1_6micron_225cube_16bit_LE.raw"
    processed = ROOT / "data" / "processed"

    jobs: list[ArchiveJob] = []

    graph_cases = [
        (
            "core1_high_pe_graph_20k",
            core1_raw,
            0.0003,
            processed / "bentheimer_6um_downsample3_D0003_n20000_steps800_trajectories.npz",
            20260525,
        ),
        (
            "core1_baseline_graph_20k",
            core1_raw,
            0.001,
            processed / "bentheimer_6um_downsample3_D001_n20000_steps800_trajectories.npz",
            20260524,
        ),
        (
            "core1_low_pe_graph_20k",
            core1_raw,
            0.003,
            processed / "bentheimer_6um_downsample3_D003_n20000_steps800_trajectories.npz",
            20260526,
        ),
        (
            "core2_baseline_graph_20k",
            core2_raw,
            0.001,
            processed / "bentheimer_core2_subvol1_6um_downsample3_D001_n20000_steps800_trajectories.npz",
            20260527,
        ),
    ]
    for name, raw, diffusivity, output, seed in graph_cases:
        jobs.append(
            ArchiveJob(
                name=name,
                output=output,
                command=[
                    sys.executable,
                    "scripts/build_bentheimer_trajectories.py",
                    "--raw",
                    str(raw),
                    "--shape",
                    "225,225,225",
                    "--voxel-size",
                    "6e-6",
                    "--downsample-factor",
                    "3",
                    "--pressure-iters",
                    "1200",
                    "--particles",
                    str(particles),
                    "--steps",
                    "800",
                    "--dt",
                    "0.5",
                    "--diffusivity",
                    str(diffusivity),
                    "--seed",
                    str(seed),
                    "--output",
                    str(output),
                ],
            )
        )

    openfoam_cases = [
        (
            "core2_openfoam_18um_20k",
            ROOT / "openfoam_cases" / "bentheimer_core2_subvol1_6um_downsample3_voxel_flow",
            "103",
            "3",
            False,
            0.06,
            0.001,
            0.03333333333333333,
            0.025,
            processed / "bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_dt010_n20000_trajectories.npz",
            20260524,
        ),
        (
            "core2_openfoam_12um_20k",
            ROOT / "openfoam_cases" / "bentheimer_core2_subvol1_6um_downsample2_voxel_flow",
            "105",
            "2",
            True,
            0.09,
            0.00225,
            0.05,
            0.0375,
            processed
            / "bentheimer_core2_subvol1_6um_downsample2_D00225_openfoam_phys_scaled_dt010_n20000_trajectories.npz",
            20260525,
        ),
        (
            "core2_openfoam_6um_20k",
            ROOT / "openfoam_cases" / "bentheimer_core2_subvol1_6um_fullres_voxel_flow_strict",
            "604",
            "1",
            False,
            0.18,
            0.009,
            0.10,
            0.075,
            processed / "bentheimer_core2_subvol1_6um_fullres_D009_openfoam_strict_dt010_n20000_trajectories.npz",
            20260526,
        ),
    ]
    for (
        name,
        case_dir,
        time_dir,
        downsample_factor,
        trim_to_factor,
        target_mean_speed,
        diffusivity,
        max_advective_step,
        max_diffusive_step,
        output,
        seed,
    ) in openfoam_cases:
        command = [
            sys.executable,
            "scripts/build_openfoam_trajectories.py",
            "--raw",
            str(core2_raw),
            "--case-dir",
            str(case_dir),
            "--time",
            time_dir,
            "--shape",
            "225,225,225",
            "--voxel-size",
            "6e-6",
            "--downsample-factor",
            downsample_factor,
            "--target-mean-speed",
            str(target_mean_speed),
            "--particles",
            str(particles),
            "--steps",
            "4000",
            "--dt",
            "0.1",
            "--diffusivity",
            str(diffusivity),
            "--max-advective-step",
            str(max_advective_step),
            "--max-diffusive-step",
            str(max_diffusive_step),
            "--max-substeps",
            "128",
            "--seed",
            str(seed),
            "--output",
            str(output),
        ]
        if trim_to_factor:
            command.insert(command.index("--target-mean-speed"), "--trim-to-factor")
        jobs.append(ArchiveJob(name=name, output=output, command=command))

    return jobs


if __name__ == "__main__":
    main()

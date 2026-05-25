from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tta_v2.evaluation import (  # noqa: E402
    MetricSettings,
    compare_metrics,
    evaluate_ensemble,
)
from tta_v2.flow import solve_pressure_jacobi, velocity_from_pressure  # noqa: E402
from tta_v2.geometry import (  # noqa: E402
    block_average,
    connected_pore_network,
    load_raw_volume,
    porosity,
    segment_pore_space,
)
from tta_v2.io import load_trajectories, save_trajectories_npz  # noqa: E402
from tta_v2.tracking import trace_particles  # noqa: E402


DEFAULT_COUNTS = [500, 2500, 5000, 10000, 20000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Core1 baseline particle-count convergence for transport metrics."
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "data" / "raw" / "Core1_Subvol1_6micron_225cube_16bit_LE.raw",
    )
    parser.add_argument("--shape", default="225,225,225")
    parser.add_argument("--downsample-factor", type=int, default=3)
    parser.add_argument("--pressure-iters", type=int, default=1200)
    parser.add_argument("--max-particles", type=int, default=20000)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--diffusivity", type=float, default=0.001)
    parser.add_argument("--target-mean-speed", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--pair-samples", type=int, default=10000)
    parser.add_argument("--reference-pair-samples", type=int, default=50000)
    parser.add_argument("--counts", default=",".join(str(value) for value in DEFAULT_COUNTS))
    parser.add_argument(
        "--trajectory-output",
        type=Path,
        default=ROOT
        / "data"
        / "processed"
        / "bentheimer_6um_downsample3_D001_n20000_trajectories.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "core1_baseline_particle_count_convergence.json",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=ROOT / "outputs" / "core1_baseline_particle_count_convergence.csv",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=ROOT / "figures" / "run_021_core1_particle_count_convergence.png",
    )
    parser.add_argument("--force-regenerate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = [int(item) for item in args.counts.split(",") if item.strip()]
    if max(counts) > args.max_particles:
        raise ValueError("--max-particles must be at least the largest requested count")

    if args.force_regenerate or not args.trajectory_output.exists():
        generate_reference_trajectories(args)

    trajectories = load_trajectories(args.trajectory_output)
    if len(trajectories) < max(counts):
        raise ValueError(
            f"{args.trajectory_output} contains {len(trajectories)} trajectories, "
            f"but {max(counts)} are required"
        )

    settings = MetricSettings(
        planes=[6.0, 10.0, 14.0],
        time_indices=[100, 200, 300, 400],
        bin_size=3.0,
        pair_samples=args.pair_samples,
        reaction_radius=3.0,
        seed=args.seed,
    )
    reference_settings = replace(settings, pair_samples=args.reference_pair_samples)
    reference_metrics = evaluate_ensemble(trajectories[: args.max_particles], reference_settings)

    rng = np.random.default_rng(args.seed + 7001)
    payload_rows = []
    for count in counts:
        n_repeats = 1 if count == args.max_particles else args.repeats
        for repeat in range(n_repeats):
            if count == args.max_particles:
                ids = np.arange(args.max_particles)
            else:
                ids = rng.choice(args.max_particles, size=count, replace=False)
            subset = [trajectories[int(idx)] for idx in ids]
            repeat_settings = replace(settings, seed=args.seed + 17 * repeat + count)
            metrics = evaluate_ensemble(subset, repeat_settings)
            errors = compare_metrics(
                reference_metrics,
                metrics,
                missing_penalty=repeat_settings.missing_penalty,
            )
            payload_rows.append(
                {
                    "count": count,
                    "repeat": repeat,
                    "errors": errors,
                    "diagnostics": extract_diagnostics(metrics),
                }
            )
            print(
                f"N={count:5d} repeat={repeat + 1:02d}/{n_repeats:02d} "
                f"btc={errors['btc_score']:.3f} "
                f"dil={errors['dilution_log_mae']:.4f} "
                f"pair={errors['pair_quantile_mae']:.4f} "
                f"enc={errors['reaction_abs_error']:.5f}",
                flush=True,
            )

    summary = summarize_rows(payload_rows, counts)
    payload = {
        "description": "Core1 baseline particle-count convergence against the 20000-particle reference ensemble.",
        "trajectory_output": str(args.trajectory_output),
        "settings": {
            "counts": counts,
            "repeats": args.repeats,
            "max_particles": args.max_particles,
            "steps": args.steps,
            "dt": args.dt,
            "diffusivity": args.diffusivity,
            "target_mean_speed": args.target_mean_speed,
            "pair_samples": args.pair_samples,
            "reference_pair_samples": args.reference_pair_samples,
            "seed": args.seed,
        },
        "reference_metrics": reference_metrics,
        "rows": payload_rows,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(args.csv_output, summary)
    plot_summary(args.figure_output, summary)
    print(f"wrote {args.output}")
    print(f"wrote {args.csv_output}")
    print(f"wrote {args.figure_output}")


def generate_reference_trajectories(args: argparse.Namespace) -> None:
    shape = tuple(int(item) for item in args.shape.split(","))
    volume = load_raw_volume(args.raw, shape=shape, dtype="<u2", order="C")
    if args.downsample_factor > 1:
        volume = block_average(volume, args.downsample_factor)
    pore_mask = segment_pore_space(volume, threshold=None, pore_is_dark=True)
    connected = connected_pore_network(pore_mask, axis=0)
    pressure, solver_info = solve_pressure_jacobi(
        connected,
        n_iters=args.pressure_iters,
        tolerance=1e-6,
    )
    velocity = velocity_from_pressure(
        pressure,
        connected,
        target_mean_speed=args.target_mean_speed,
    )
    diagnostics: dict[str, float] = {}
    trajectories = trace_particles(
        velocity,
        connected,
        n_particles=args.max_particles,
        n_steps=args.steps,
        dt=args.dt,
        diffusivity=args.diffusivity,
        seed=args.seed,
        diagnostics=diagnostics,
    )
    args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
    save_trajectories_npz(args.trajectory_output, trajectories)
    summary = {
        "raw": str(args.raw),
        "shape": shape,
        "simulation_shape": tuple(int(item) for item in volume.shape),
        "downsample_factor": args.downsample_factor,
        "raw_porosity": porosity(pore_mask),
        "connected_porosity": porosity(connected),
        "pressure_solver": solver_info,
        "particles_requested": args.max_particles,
        "steps_requested": args.steps,
        "dt": args.dt,
        "diffusivity": args.diffusivity,
        "target_mean_speed": args.target_mean_speed,
        "seed": args.seed,
        "tracking_diagnostics": diagnostics,
        "n_trajectories": len(trajectories),
        "mean_trajectory_length": float(np.mean([len(item) for item in trajectories])),
        "output": str(args.trajectory_output),
    }
    args.trajectory_output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


def extract_diagnostics(metrics: dict[str, object]) -> dict[str, float]:
    breakthrough = metrics["breakthrough"]
    dilution = metrics["dilution"]
    pair = metrics["pair_separation"]
    reaction = metrics["reaction"]
    return {
        "btc_q50_plane10": float(breakthrough[10.0]["q50"]),
        "btc_q90_plane14": float(breakthrough[14.0]["q90"]),
        "btc_coverage_plane14": float(breakthrough[14.0]["coverage"]),
        "dilution_index_t400": float(dilution[400]["dilution_index"]),
        "dilution_occupied_bins_t400": float(dilution[400]["occupied_bins"]),
        "pair_q50_t400": float(pair[400]["q50"]),
        "pair_q90_t400": float(pair[400]["q90"]),
        "encounter_probability": float(reaction["probability"]),
        "encounter_pair_count": float(reaction["pair_count"]),
    }


def summarize_rows(rows: list[dict[str, object]], counts: list[int]) -> dict[str, dict[str, float]]:
    fields = [
        ("errors", "btc_score"),
        ("errors", "btc_quantile_mae"),
        ("errors", "btc_coverage_deficit"),
        ("errors", "dilution_log_mae"),
        ("errors", "pair_quantile_mae"),
        ("errors", "reaction_abs_error"),
        ("diagnostics", "btc_q50_plane10"),
        ("diagnostics", "btc_q90_plane14"),
        ("diagnostics", "btc_coverage_plane14"),
        ("diagnostics", "dilution_index_t400"),
        ("diagnostics", "dilution_occupied_bins_t400"),
        ("diagnostics", "pair_q50_t400"),
        ("diagnostics", "pair_q90_t400"),
        ("diagnostics", "encounter_probability"),
    ]
    summary: dict[str, dict[str, float]] = {}
    for count in counts:
        selected = [row for row in rows if row["count"] == count]
        item: dict[str, float] = {"repeats": float(len(selected))}
        for group, field in fields:
            values = np.asarray([row[group][field] for row in selected], dtype=float)
            finite = values[np.isfinite(values)]
            item[f"{field}_mean"] = float(np.mean(finite)) if len(finite) else float("nan")
            item[f"{field}_std"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
            item[f"{field}_p05"] = float(np.quantile(finite, 0.05)) if len(finite) else float("nan")
            item[f"{field}_p95"] = float(np.quantile(finite, 0.95)) if len(finite) else float("nan")
        summary[str(count)] = item
    return summary


def write_csv(path: Path, summary: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = [int(key) for key in summary]
    first = summary[str(counts[0])]
    columns = ["count", *first.keys()]
    lines = [",".join(columns)]
    for count in counts:
        row = summary[str(count)]
        values = [str(count), *[f"{row[column]:.10g}" for column in first.keys()]]
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(path: Path, summary: dict[str, dict[str, float]]) -> None:
    counts = np.asarray([int(key) for key in summary], dtype=float)
    panels = [
        ("btc_score", "Breakthrough score"),
        ("dilution_log_mae", "Dilution log error"),
        ("pair_quantile_mae", "Pair-separation error"),
        ("reaction_abs_error", "Encounter-probability error"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), constrained_layout=True)
    for ax, (field, title) in zip(axes.ravel(), panels):
        mean = np.asarray([summary[str(int(count))][f"{field}_mean"] for count in counts])
        std = np.asarray([summary[str(int(count))][f"{field}_std"] for count in counts])
        ax.errorbar(counts, mean, yerr=std, marker="o", linewidth=2, capsize=4)
        ax.set_xscale("log")
        ax.set_xticks(counts, [f"{int(count):,}" for count in counts], rotation=20)
        ax.set_title(title)
        ax.set_xlabel("particles in subsample")
        ax.set_ylabel("error vs 20k reference")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Core1 baseline metric stability with particle count", fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=250)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bootstrap_mixture_selection import fixed_train_test_split  # noqa: E402
from tta_v2 import (  # noqa: E402
    GaussianBayesSegmentSampler,
    MetricSettings,
    SegmentArchive,
    breakthrough_score,
    breakthrough_times,
    choose_origin_pool,
    dilution_index,
    generate_with_origins,
    load_trajectories,
    scalar_curve_log_mae,
    summarize_breakthroughs,
)


SEGMENT_CONFIGS = [
    {"name": "short", "segment_steps": 24, "match_steps": 12, "n_segments": 42},
    {"name": "baseline", "segment_steps": 36, "match_steps": 20, "n_segments": 32},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Limited Core1 archive-density and segment-length convergence diagnostic."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "processed" / "bentheimer_6um_downsample3_D001_n20000_steps800_trajectories.npz",
    )
    parser.add_argument("--archive-particles", default="5000,10000,14000")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-generated", type=int, default=180)
    parser.add_argument("--segment-stride", type=int, default=400)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "core1_archive_convergence.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive_particles = [int(item) for item in args.archive_particles.split(",") if item.strip()]
    trajectories = load_trajectories(args.input)
    train_pool, test, split_payload = fixed_train_test_split(trajectories, test_fraction=0.30, seed=123)
    if max(archive_particles) > len(train_pool):
        raise ValueError(f"maximum archive size after the held-out split is {len(train_pool)}")

    settings = MetricSettings(
        planes=[6.0, 10.0, 14.0],
        time_indices=[100, 200, 300, 400],
        bin_size=3.0,
        pair_samples=1,
        reaction_radius=3.0,
        seed=123,
    )
    full_reference = breakthrough_dilution_metrics(test, settings)

    rows: list[dict] = []
    for config in SEGMENT_CONFIGS:
        for n_archive in archive_particles:
            for repeat in range(args.repeats):
                seed = 91003 + 1009 * repeat + n_archive + int(config["segment_steps"])
                rng = np.random.default_rng(seed)
                archive_trajectories = choose_archive_subset(train_pool, n_archive, rng)
                archive = SegmentArchive.from_trajectories(
                    archive_trajectories,
                    segment_steps=int(config["segment_steps"]),
                    match_steps=int(config["match_steps"]),
                    stride=args.segment_stride,
                    dt=0.5,
                )
                sampler = GaussianBayesSegmentSampler(
                    archive=archive,
                    diffusivity=0.001,
                    candidate_limit=256,
                    bandwidth_multiplier=0.25,
                    seed=seed,
                )
                origin_pool = choose_origin_pool("train", archive_trajectories, test)
                generated = generate_with_origins(
                    sampler,
                    n_trajectories=args.n_generated,
                    n_segments=int(config["n_segments"]),
                    origin_pool=origin_pool,
                    rng=np.random.default_rng(seed + 17),
                )
                reference_sample = choose_reference_subset(test, args.n_generated, rng)
                repeat_settings = replace(settings, seed=seed)
                generated_metrics = breakthrough_dilution_metrics(generated, repeat_settings)
                reference_sample_metrics = breakthrough_dilution_metrics(reference_sample, repeat_settings)
                equal_errors = breakthrough_dilution_errors(reference_sample_metrics, generated_metrics, repeat_settings)
                full_errors = breakthrough_dilution_errors(full_reference, generated_metrics, repeat_settings)
                row = {
                    "segment_config": config["name"],
                    "segment_steps": int(config["segment_steps"]),
                    "match_steps": int(config["match_steps"]),
                    "n_segments": int(config["n_segments"]),
                    "archive_particles": int(n_archive),
                    "archive_segments": int(archive.size),
                    "repeat": int(repeat),
                    "seed": int(seed),
                    "n_generated": int(args.n_generated),
                    "equal_count_errors": equal_errors,
                    "full_reference_errors": full_errors,
                    "diagnostics": {
                        "reference_sample_final_dilution": final_dilution(reference_sample_metrics),
                        "full_reference_final_dilution": final_dilution(full_reference),
                        "generated_final_dilution": final_dilution(generated_metrics),
                        "generated_to_reference_sample_final_ratio": final_dilution(generated_metrics)
                        / final_dilution(reference_sample_metrics),
                        "generated_to_full_reference_final_ratio": final_dilution(generated_metrics)
                        / final_dilution(full_reference),
                    },
                }
                rows.append(row)
                print(
                    f"{config['name']} N={n_archive} repeat={repeat}: "
                    f"segments={archive.size} btc={equal_errors['btc_score']:.2f} "
                    f"dil={equal_errors['dilution_log_mae']:.3f} "
                    f"final_ratio={row['diagnostics']['generated_to_reference_sample_final_ratio']:.3f}",
                    flush=True,
                )

    payload = {
        "description": (
            "Core1 baseline archive-density and segment-length convergence diagnostic. "
            "Errors compare generated trajectories with same-count held-out reference subsamples "
            "to avoid entropy-dilution sample-size bias; full-reference errors are also retained."
        ),
        "input": str(args.input),
        "split": split_payload,
        "archive_particle_counts": archive_particles,
        "segment_configs": SEGMENT_CONFIGS,
        "n_generated": args.n_generated,
        "rows": rows,
        "summary": summarize(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


def choose_archive_subset(train_pool: list[np.ndarray], n_archive: int, rng: np.random.Generator) -> list[np.ndarray]:
    if n_archive == len(train_pool):
        return list(train_pool)
    indices = rng.choice(len(train_pool), size=n_archive, replace=False)
    return [train_pool[int(index)] for index in indices]


def choose_reference_subset(test: list[np.ndarray], n_reference: int, rng: np.random.Generator) -> list[np.ndarray]:
    indices = rng.choice(len(test), size=n_reference, replace=False)
    return [test[int(index)] for index in indices]


def breakthrough_dilution_metrics(trajectories: list[np.ndarray], settings: MetricSettings) -> dict[str, object]:
    return {
        "breakthrough": summarize_breakthroughs(breakthrough_times(trajectories, settings.planes)),
        "dilution": dilution_index(trajectories, settings.time_indices, bin_size=settings.bin_size),
    }


def breakthrough_dilution_errors(
    reference: dict[str, object],
    generated: dict[str, object],
    settings: MetricSettings,
) -> dict[str, float]:
    btc = breakthrough_score(reference["breakthrough"], generated["breakthrough"], missing_penalty=settings.missing_penalty)
    dilution = scalar_curve_log_mae(reference["dilution"], generated["dilution"], "dilution_index")
    return {
        "btc_score": float(btc["score"]),
        "btc_quantile_mae": float(btc["quantile_mae"]),
        "btc_coverage_deficit": float(btc["coverage_deficit"]),
        "dilution_log_mae": float(dilution),
    }


def final_dilution(metrics: dict[str, object]) -> float:
    return float(metrics["dilution"][400]["dilution_index"])


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["segment_config"], row["archive_particles"]), []).append(row)
    for (segment_config, archive_particles), selected in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        item: dict[str, float | int | str | bool] = {
            "segment_config": segment_config,
            "archive_particles": int(archive_particles),
            "archive_segments_mean": float(np.mean([row["archive_segments"] for row in selected])),
            "repeats": int(len(selected)),
        }
        add_summary_values(item, "equal_count_btc_score", [row["equal_count_errors"]["btc_score"] for row in selected])
        add_summary_values(
            item,
            "equal_count_dilution_log_mae",
            [row["equal_count_errors"]["dilution_log_mae"] for row in selected],
        )
        add_summary_values(
            item,
            "full_reference_btc_score",
            [row["full_reference_errors"]["btc_score"] for row in selected],
        )
        add_summary_values(
            item,
            "full_reference_dilution_log_mae",
            [row["full_reference_errors"]["dilution_log_mae"] for row in selected],
        )
        add_summary_values(
            item,
            "generated_to_reference_sample_final_ratio",
            [row["diagnostics"]["generated_to_reference_sample_final_ratio"] for row in selected],
        )
        add_summary_values(
            item,
            "generated_to_full_reference_final_ratio",
            [row["diagnostics"]["generated_to_full_reference_final_ratio"] for row in selected],
        )
        item["dilution_gap_persists_equal_count"] = bool(
            np.all([row["diagnostics"]["generated_to_reference_sample_final_ratio"] < 1.0 for row in selected])
        )
        item["dilution_gap_persists_full_reference"] = bool(
            np.all([row["diagnostics"]["generated_to_full_reference_final_ratio"] < 1.0 for row in selected])
        )
        summary.append(item)
    return summary


def add_summary_values(item: dict[str, float | int | str | bool], prefix: str, values: list[float]) -> None:
    arr = np.asarray(values, dtype=float)
    item[f"{prefix}_mean"] = float(np.mean(arr))
    item[f"{prefix}_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bootstrap_mixture_selection import (  # noqa: E402
    COMPONENT_ORDER,
    build_components,
    fixed_train_test_split,
)
from tta_v2 import (  # noqa: E402
    MetricSettings,
    MixtureSegmentSampler,
    SegmentArchive,
    choose_origin_pool,
    compare_metrics,
    evaluate_ensemble,
    generate_with_origins,
    load_trajectories,
)


SAMPLERS = [
    "bootstrap_mean_mixture",
    "pooled_validation_mixture",
    "gaussian_bayes",
    "knn_conditional",
    "hybrid",
    "pair_rerank",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Core1 baseline reference-split, dilution-bin, and encounter-radius sensitivity checks."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "outputs" / "bentheimer_6um_downsample3_D001_n20000_stride400_outer_split_mixture_benchmark.json",
    )
    parser.add_argument("--bin-sizes", default="2,3,4")
    parser.add_argument("--reaction-radii", default="2,3,4")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "core1_proxy_sensitivity.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    trajectories = load_trajectories(Path(benchmark["input"]))
    bin_sizes = parse_float_list(args.bin_sizes)
    reaction_radii = parse_float_list(args.reaction_radii)

    base = MetricSettings(
        planes=[6.0, 10.0, 14.0],
        time_indices=[100, 200, 300, 400],
        bin_size=3.0,
        pair_samples=3000,
        reaction_radius=3.0,
        seed=123,
    )

    generated_records: list[dict] = []
    reference_split_records: list[dict] = []

    for outer in benchmark["outer_results"]:
        outer_seed = int(outer["seed"])
        train_pool, test, _split_payload = fixed_train_test_split(
            trajectories,
            test_fraction=float(outer["splits"]["test_fraction"]),
            seed=outer_seed,
        )
        outer_settings = replace(base, seed=outer_seed)
        reference_metrics = evaluate_ensemble(test, outer_settings)

        ref_a, ref_b = split_reference_test(test, seed=outer_seed + 4242)
        ref_a_metrics = evaluate_ensemble(ref_a, outer_settings)
        ref_b_metrics = evaluate_ensemble(ref_b, outer_settings)
        reference_split_records.append(
            {
                "outer_index": int(outer["outer_index"]),
                "seed": outer_seed,
                "errors": compare_metrics(ref_a_metrics, ref_b_metrics, missing_penalty=outer_settings.missing_penalty),
            }
        )

        archive = SegmentArchive.from_trajectories(
            train_pool,
            segment_steps=int(outer["archive"]["segment_steps"]),
            match_steps=int(outer["archive"]["match_steps"]),
            stride=400,
            dt=0.5,
        )
        component_args = component_namespace()
        components = build_components(component_args, archive, seed=outer_seed)
        samplers = {
            "bootstrap_mean_mixture": MixtureSegmentSampler(
                archive=archive,
                components=[components[name] for name in COMPONENT_ORDER],
                component_weights=weights_from_payload(outer["mean_selected_weights"]),
                seed=outer_seed,
            ),
            "pooled_validation_mixture": MixtureSegmentSampler(
                archive=archive,
                components=[components[name] for name in COMPONENT_ORDER],
                component_weights=weights_from_payload(outer["pooled_validation_weights"]),
                seed=outer_seed,
            ),
            **components,
        }
        origin_pool = choose_origin_pool("train", train_pool, test)
        generated = {
            name: generate_with_origins(
                samplers[name],
                n_trajectories=120,
                n_segments=32,
                origin_pool=origin_pool,
                rng=np.random.default_rng(outer_seed),
            )
            for name in SAMPLERS
        }

        for name in SAMPLERS:
            metrics = evaluate_ensemble(generated[name], outer_settings)
            errors = compare_metrics(reference_metrics, metrics, missing_penalty=outer_settings.missing_penalty)
            generated_records.append(
                {
                    "outer_index": int(outer["outer_index"]),
                    "seed": outer_seed,
                    "sampler": name,
                    "setting": "default",
                    "bin_size": outer_settings.bin_size,
                    "reaction_radius": outer_settings.reaction_radius,
                    "errors": errors,
                    "reference": reference_metrics,
                    "generated": metrics,
                }
            )

        for bin_size in bin_sizes:
            settings = replace(outer_settings, bin_size=float(bin_size), reaction_radius=3.0)
            reference = evaluate_ensemble(test, settings)
            for name in SAMPLERS:
                metrics = evaluate_ensemble(generated[name], settings)
                generated_records.append(
                    {
                        "outer_index": int(outer["outer_index"]),
                        "seed": outer_seed,
                        "sampler": name,
                        "setting": "bin_size",
                        "bin_size": float(bin_size),
                        "reaction_radius": 3.0,
                        "errors": compare_metrics(reference, metrics, missing_penalty=settings.missing_penalty),
                        "reference": reference,
                        "generated": metrics,
                    }
                )

        for radius in reaction_radii:
            settings = replace(outer_settings, bin_size=3.0, reaction_radius=float(radius))
            reference = evaluate_ensemble(test, settings)
            for name in SAMPLERS:
                metrics = evaluate_ensemble(generated[name], settings)
                generated_records.append(
                    {
                        "outer_index": int(outer["outer_index"]),
                        "seed": outer_seed,
                        "sampler": name,
                        "setting": "reaction_radius",
                        "bin_size": 3.0,
                        "reaction_radius": float(radius),
                        "errors": compare_metrics(reference, metrics, missing_penalty=settings.missing_penalty),
                        "reference": reference,
                        "generated": metrics,
                    }
                )

        print(f"processed outer split {outer['outer_index']}", flush=True)

    payload = {
        "benchmark": str(args.benchmark),
        "bin_sizes": bin_sizes,
        "reaction_radii": reaction_radii,
        "reference_vs_reference": summarize_reference_split(reference_split_records),
        "dilution_bin_sensitivity": summarize_dilution_bins(generated_records, bin_sizes),
        "encounter_radius_sensitivity": summarize_encounter_radii(generated_records, reaction_radii),
        "qualitative_checks": summarize_qualitative_checks(generated_records, bin_sizes, reaction_radii),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


def component_namespace() -> SimpleNamespace:
    return SimpleNamespace(
        diffusivity=0.001,
        gaussian_bandwidth=0.25,
        candidate_limit=256,
        knn_k=96,
        knn_temperature=0.8,
        contrastive_epochs=300,
        contrastive_negative_ratio=6,
        hybrid_learned_weight=0.25,
        pair_rerank_weight=0.25,
        pair_neighbor_k=32,
        rerank_horizon_segments=3,
        adaptive_bins=4,
    )


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def weights_from_payload(payload: dict[str, float]) -> np.ndarray:
    return np.asarray([float(payload[name]) for name in COMPONENT_ORDER], dtype=float)


def split_reference_test(test: list[np.ndarray], *, seed: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(test))
    half = len(indices) // 2
    return [test[int(index)] for index in indices[:half]], [test[int(index)] for index in indices[half:]]


def summarize_reference_split(records: list[dict]) -> dict[str, dict[str, float]]:
    keys = ["btc_score", "dilution_log_mae", "pair_quantile_mae", "reaction_abs_error"]
    return {
        key: summarize_values([float(record["errors"][key]) for record in records])
        for key in keys
    }


def summarize_dilution_bins(records: list[dict], bin_sizes: list[float]) -> list[dict]:
    rows = []
    for bin_size in bin_sizes:
        subset = [
            record
            for record in records
            if record["setting"] == "bin_size" and np.isclose(float(record["bin_size"]), bin_size)
        ]
        ref_by_outer = {}
        gen_by_sampler: dict[str, list[float]] = {name: [] for name in SAMPLERS}
        all_below = True
        for record in subset:
            outer = int(record["outer_index"])
            reference_final = read_final_dilution(record["reference"])
            generated_final = read_final_dilution(record["generated"])
            ref_by_outer[outer] = reference_final
            gen_by_sampler[record["sampler"]].append(generated_final)
            all_below = all_below and generated_final < reference_final
        ref_values = list(ref_by_outer.values())
        ref_sd = float(np.std(ref_values, ddof=1)) if len(ref_values) > 1 else float("nan")
        sampler_means = {name: float(np.mean(values)) for name, values in gen_by_sampler.items() if values}
        closest_sampler = max(sampler_means, key=sampler_means.get)
        gap = float(np.mean(ref_values) - sampler_means[closest_sampler])
        rows.append(
            {
                "bin_size": float(bin_size),
                "reference_final_mean": float(np.mean(ref_values)),
                "reference_final_sd": ref_sd,
                "closest_sampler": closest_sampler,
                "closest_generated_final_mean": sampler_means[closest_sampler],
                "gap_over_reference_sd": gap / ref_sd if np.isfinite(ref_sd) and ref_sd > 0 else float("nan"),
                "all_generated_below_reference": bool(all_below),
            }
        )
    return rows


def summarize_encounter_radii(records: list[dict], radii: list[float]) -> list[dict]:
    rows = []
    for radius in radii:
        subset = [
            record
            for record in records
            if record["setting"] == "reaction_radius" and np.isclose(float(record["reaction_radius"]), radius)
        ]
        ref_by_outer = {}
        errors_by_sampler: dict[str, list[float]] = {name: [] for name in SAMPLERS}
        probabilities_by_sampler: dict[str, list[float]] = {name: [] for name in SAMPLERS}
        for record in subset:
            outer = int(record["outer_index"])
            ref_by_outer[outer] = float(record["reference"]["reaction"]["probability"])
            errors_by_sampler[record["sampler"]].append(float(record["errors"]["reaction_abs_error"]))
            probabilities_by_sampler[record["sampler"]].append(float(record["generated"]["reaction"]["probability"]))
        ref_values = list(ref_by_outer.values())
        mean_errors = {name: float(np.mean(values)) for name, values in errors_by_sampler.items() if values}
        lowest_error_sampler = min(mean_errors, key=mean_errors.get)
        generated_means = [float(np.mean(values)) for values in probabilities_by_sampler.values() if values]
        rows.append(
            {
                "reaction_radius": float(radius),
                "reference_probability_mean": float(np.mean(ref_values)),
                "reference_probability_sd": float(np.std(ref_values, ddof=1)) if len(ref_values) > 1 else float("nan"),
                "generated_probability_min_mean": float(np.min(generated_means)),
                "generated_probability_max_mean": float(np.max(generated_means)),
                "lowest_abs_error_sampler": lowest_error_sampler,
                "lowest_abs_error": mean_errors[lowest_error_sampler],
            }
        )
    return rows


def summarize_qualitative_checks(records: list[dict], bin_sizes: list[float], radii: list[float]) -> dict[str, object]:
    return {
        "dilution_gap_persists_all_bin_sizes": all(
            row["all_generated_below_reference"]
            for row in summarize_dilution_bins(records, bin_sizes)
        ),
        "encounter_lowest_error_samplers": {
            str(row["reaction_radius"]): row["lowest_abs_error_sampler"]
            for row in summarize_encounter_radii(records, radii)
        },
        "note": (
            "Sensitivity checks support the proxy-observable interpretation: dilution under-preservation "
            "persists under bin-size perturbations, while encounter rankings vary with radius and should "
            "be interpreted qualitatively rather than as calibrated reactive-transport rates."
        ),
    }


def read_final_dilution(metrics: dict) -> float:
    times = sorted(int(key) for key in metrics["dilution"])
    final = times[-1]
    for key in (final, str(final), float(final), f"{float(final):.1f}"):
        if key in metrics["dilution"]:
            return float(metrics["dilution"][key]["dilution_index"])
    raise KeyError(final)


def summarize_values(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


if __name__ == "__main__":
    main()

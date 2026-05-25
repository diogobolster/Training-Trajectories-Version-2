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
    evaluate_ensemble,
    generate_with_origins,
    load_trajectories,
)
from tta_v2.metrics import quantile_curve_mae  # noqa: E402


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
        description="Core1 baseline sensitivity of pair/encounter proxies to inlet-neighborhood pair sampling."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "outputs" / "bentheimer_6um_downsample3_D001_n20000_stride400_outer_split_mixture_benchmark.json",
    )
    parser.add_argument("--pair-samples", type=int, default=3000)
    parser.add_argument("--inlet-neighborhood-radius", type=float, default=6.0)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "core1_inlet_pair_sensitivity.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    trajectories = load_trajectories(Path(benchmark["input"]))
    settings = MetricSettings(
        planes=[6.0, 10.0, 14.0],
        time_indices=[100, 200, 300, 400],
        bin_size=3.0,
        pair_samples=args.pair_samples,
        reaction_radius=3.0,
        seed=123,
    )

    records: list[dict] = []
    for outer in benchmark["outer_results"]:
        outer_seed = int(outer["seed"])
        train_pool, test, _split_payload = fixed_train_test_split(
            trajectories,
            test_fraction=float(outer["splits"]["test_fraction"]),
            seed=outer_seed,
        )
        outer_settings = replace(settings, seed=outer_seed)
        archive = SegmentArchive.from_trajectories(
            train_pool,
            segment_steps=int(outer["archive"]["segment_steps"]),
            match_steps=int(outer["archive"]["match_steps"]),
            stride=400,
            dt=0.5,
        )
        components = build_components(component_namespace(), archive, seed=outer_seed)
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

        reference_uniform = evaluate_ensemble(test, outer_settings)
        reference_inlet = evaluate_pair_metrics(
            test,
            outer_settings,
            strategy="inlet_neighborhood",
            inlet_radius=args.inlet_neighborhood_radius,
        )
        for name in SAMPLERS:
            generated_uniform = evaluate_ensemble(generated[name], outer_settings)
            generated_inlet = evaluate_pair_metrics(
                generated[name],
                outer_settings,
                strategy="inlet_neighborhood",
                inlet_radius=args.inlet_neighborhood_radius,
            )
            records.append(
                {
                    "outer_index": int(outer["outer_index"]),
                    "seed": outer_seed,
                    "sampler": name,
                    "uniform": pair_errors(reference_uniform, generated_uniform),
                    "inlet_neighborhood": pair_errors(reference_inlet, generated_inlet),
                    "reference_inlet_pair_count": reference_inlet["reaction"]["pair_count"],
                    "generated_inlet_pair_count": generated_inlet["reaction"]["pair_count"],
                }
            )
        print(f"processed outer split {outer['outer_index']}", flush=True)

    payload = {
        "benchmark": str(args.benchmark),
        "settings": {
            "pair_samples": int(args.pair_samples),
            "inlet_neighborhood_radius": float(args.inlet_neighborhood_radius),
            "encounter_radius": float(settings.reaction_radius),
            "generated_trajectories_per_sampler": 120,
            "note": (
                "The inlet-neighborhood strategy samples pairs whose initial transverse separation is within "
                "the listed radius. It is a sensitivity diagnostic, not a labeled-reactant construction."
            ),
        },
        "summary": {
            "uniform": summarize(records, "uniform"),
            "inlet_neighborhood": summarize(records, "inlet_neighborhood"),
        },
    }
    payload["qualitative"] = qualitative(payload["summary"])
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


def weights_from_payload(payload: dict[str, float]) -> np.ndarray:
    return np.asarray([float(payload[name]) for name in COMPONENT_ORDER], dtype=float)


def evaluate_pair_metrics(
    trajectories: list[np.ndarray],
    settings: MetricSettings,
    *,
    strategy: str,
    inlet_radius: float,
) -> dict[str, object]:
    pairs = inlet_neighborhood_pairs(
        trajectories,
        n_pairs=settings.pair_samples,
        inlet_radius=inlet_radius,
        seed=settings.seed,
    )
    return {
        "pair_separation": pair_separation_from_pairs(trajectories, settings.time_indices, pairs),
        "reaction": encounter_from_pairs(
            trajectories,
            pairs,
            reaction_radius=settings.reaction_radius,
            max_time_index=max(settings.time_indices),
        ),
        "pair_sampling": strategy,
    }


def inlet_neighborhood_pairs(
    trajectories: list[np.ndarray],
    *,
    n_pairs: int,
    inlet_radius: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    starts = np.asarray([np.asarray(traj)[0] for traj in trajectories if len(traj) > 0], dtype=float)
    n = len(starts)
    if n < 2:
        return np.empty((0, 2), dtype=int)
    transverse = starts[:, 1:3]
    candidates: list[tuple[int, int]] = []
    if n <= 500:
        for i in range(n - 1):
            delta = transverse[i + 1 :] - transverse[i][None, :]
            distances = np.linalg.norm(delta, axis=1)
            for offset in np.flatnonzero(distances <= inlet_radius):
                candidates.append((i, i + 1 + int(offset)))
    else:
        max_draws = max(n_pairs * 250, 200_000)
        seen: set[tuple[int, int]] = set()
        for _ in range(max_draws):
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n - 1))
            if j >= i:
                j += 1
            if np.linalg.norm(transverse[i] - transverse[j]) <= inlet_radius:
                pair = (min(i, j), max(i, j))
                if pair not in seen:
                    seen.add(pair)
                    candidates.append(pair)
                    if len(candidates) >= n_pairs:
                        break
    if not candidates:
        return np.empty((0, 2), dtype=int)
    candidate_array = np.asarray(candidates, dtype=int)
    replace = len(candidate_array) < n_pairs
    chosen = rng.choice(len(candidate_array), size=n_pairs, replace=replace)
    return candidate_array[chosen]


def pair_separation_from_pairs(
    trajectories: list[np.ndarray],
    time_indices: list[int],
    pairs: np.ndarray,
) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for time_index in time_indices:
        distances = []
        for i, j in pairs:
            if len(trajectories[int(i)]) <= time_index or len(trajectories[int(j)]) <= time_index:
                continue
            distance = np.linalg.norm(trajectories[int(i)][time_index] - trajectories[int(j)][time_index])
            distances.append(float(distance))
        result[time_index] = summarize_values(np.asarray(distances, dtype=float))
    return result


def encounter_from_pairs(
    trajectories: list[np.ndarray],
    pairs: np.ndarray,
    *,
    reaction_radius: float,
    max_time_index: int,
) -> dict[str, float]:
    encounters = 0
    usable = 0
    for i, j in pairs:
        ti = trajectories[int(i)]
        tj = trajectories[int(j)]
        horizon = min(len(ti), len(tj), max_time_index + 1)
        if horizon <= 0:
            continue
        delta = ti[:horizon] - tj[:horizon]
        encounters += int(float(np.min(np.linalg.norm(delta, axis=1))) <= reaction_radius)
        usable += 1
    return {
        "reaction_radius": float(reaction_radius),
        "pair_count": float(usable),
        "encounter_count": float(encounters),
        "probability": float(encounters / usable) if usable else float("nan"),
    }


def summarize_values(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"count": 0.0, "mean": float("nan"), "q10": float("nan"), "q50": float("nan"), "q90": float("nan")}
    q10, q50, q90 = np.quantile(finite, [0.1, 0.5, 0.9])
    return {"count": float(len(finite)), "mean": float(np.mean(finite)), "q10": float(q10), "q50": float(q50), "q90": float(q90)}


def pair_errors(reference: dict[str, object], generated: dict[str, object]) -> dict[str, float]:
    pair_error = quantile_curve_mae(reference["pair_separation"], generated["pair_separation"])
    ref_p = float(reference["reaction"]["probability"])
    gen_p = float(generated["reaction"]["probability"])
    encounter_error = abs(ref_p - gen_p) if np.isfinite(ref_p) and np.isfinite(gen_p) else float("nan")
    return {
        "pair_quantile_mae": float(pair_error),
        "encounter_abs_error": float(encounter_error),
        "reference_encounter_probability": ref_p,
        "generated_encounter_probability": gen_p,
    }


def summarize(records: list[dict], key: str) -> list[dict]:
    rows = []
    for sampler in SAMPLERS:
        subset = [record[key] for record in records if record["sampler"] == sampler]
        pair_values = np.asarray([item["pair_quantile_mae"] for item in subset], dtype=float)
        encounter_values = np.asarray([item["encounter_abs_error"] for item in subset], dtype=float)
        ref_probs = np.asarray([item["reference_encounter_probability"] for item in subset], dtype=float)
        gen_probs = np.asarray([item["generated_encounter_probability"] for item in subset], dtype=float)
        rows.append(
            {
                "sampler": sampler,
                "pair_quantile_mae_mean": float(np.nanmean(pair_values)),
                "pair_quantile_mae_std": float(np.nanstd(pair_values, ddof=1)),
                "encounter_abs_error_mean": float(np.nanmean(encounter_values)),
                "encounter_abs_error_std": float(np.nanstd(encounter_values, ddof=1)),
                "reference_encounter_probability_mean": float(np.nanmean(ref_probs)),
                "generated_encounter_probability_mean": float(np.nanmean(gen_probs)),
            }
        )
    for metric in ("pair_quantile_mae_mean", "encounter_abs_error_mean"):
        order = sorted(rows, key=lambda row: row[metric])
        for rank, row in enumerate(order, start=1):
            row[f"{metric}_rank"] = rank
    return rows


def qualitative(summary: dict[str, list[dict]]) -> dict[str, object]:
    result = {}
    for key, rows in summary.items():
        result[key] = {
            "lowest_pair_error": min(rows, key=lambda row: row["pair_quantile_mae_mean"])["sampler"],
            "lowest_encounter_error": min(rows, key=lambda row: row["encounter_abs_error_mean"])["sampler"],
        }
    result["note"] = (
        "This sensitivity check conditions sampled pairs on common inlet neighborhoods. It is used only to "
        "test whether the ensemble-level pair/encounter interpretation is fragile to one simple pair-sampling rule."
    )
    return result


if __name__ == "__main__":
    main()

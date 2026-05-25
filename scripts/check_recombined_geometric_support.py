from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bootstrap_mixture_selection import fixed_train_test_split  # noqa: E402
from bootstrap_mixture_selection import build_components, COMPONENT_ORDER  # noqa: E402
from tta_v2 import (  # noqa: E402
    MixtureSegmentSampler,
    SegmentArchive,
    choose_origin_pool,
    generate_with_origins,
    load_trajectories,
)
from tta_v2.geometry import (  # noqa: E402
    block_average,
    connected_pore_network,
    load_raw_volume,
    segment_pore_space,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check pore-mask occupancy of representative recombined TTA paths."
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "data" / "raw" / "Core1_Subvol1_6micron_225cube_16bit_LE.raw",
    )
    parser.add_argument("--shape", default="225,225,225")
    parser.add_argument("--downsample-factor", type=int, default=3)
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=ROOT
        / "data"
        / "processed"
        / "bentheimer_6um_downsample3_D001_n20000_steps800_trajectories.npz",
    )
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--segment-steps", type=int, default=36)
    parser.add_argument("--match-steps", type=int, default=20)
    parser.add_argument("--segment-stride", type=int, default=400)
    parser.add_argument("--n-generated", type=int, default=1000)
    parser.add_argument("--n-reference", type=int, default=1000)
    parser.add_argument("--n-segments", type=int, default=32)
    parser.add_argument("--diffusivity", type=float, default=0.001)
    parser.add_argument("--candidate-limit", type=int, default=256)
    parser.add_argument("--gaussian-bandwidth", type=float, default=0.25)
    parser.add_argument("--knn-k", type=int, default=96)
    parser.add_argument("--knn-temperature", type=float, default=0.8)
    parser.add_argument("--contrastive-epochs", type=int, default=300)
    parser.add_argument("--contrastive-negative-ratio", type=int, default=6)
    parser.add_argument("--hybrid-learned-weight", type=float, default=0.25)
    parser.add_argument("--pair-rerank-weight", type=float, default=0.25)
    parser.add_argument("--pair-neighbor-k", type=int, default=32)
    parser.add_argument("--rerank-horizon-segments", type=int, default=3)
    parser.add_argument("--adaptive-bins", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "outputs" / "bentheimer_6um_downsample3_D001_n20000_stride400_outer_split_mixture_benchmark.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "recombined_geometric_support.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask = load_connected_mask(args)
    trajectories = load_trajectories(args.trajectories)
    train_pool, test, split_payload = fixed_train_test_split(
        trajectories,
        test_fraction=args.test_fraction,
        seed=args.split_seed,
    )
    archive = SegmentArchive.from_trajectories(
        train_pool,
        segment_steps=args.segment_steps,
        match_steps=args.match_steps,
        stride=args.segment_stride,
        dt=0.5,
    )
    components = build_components(args, archive, seed=args.seed)
    samplers = build_eval_samplers(args, archive, components)
    origin_pool = choose_origin_pool("train", train_pool, test)
    reference = sample_reference(test, args.n_reference, seed=args.seed + 31)
    benchmark_summary = benchmark_dilution_errors(args.benchmark)

    sampler_payload = {}
    for offset, (name, sampler) in enumerate(samplers.items()):
        generated = generate_with_origins(
            sampler,
            n_trajectories=args.n_generated,
            n_segments=args.n_segments,
            origin_pool=origin_pool,
            rng=np.random.default_rng(args.seed + 17),
        )
        sampler_payload[name] = {
            "occupancy": occupancy_summary(generated, mask),
            "benchmark_dilution_log_mae": benchmark_summary.get(name),
        }
        print(
            f"{name}: "
            f"viol={100.0 * sampler_payload[name]['occupancy']['violation_fraction_all_points']:.2f}% "
            f"dil={benchmark_summary.get(name, {}).get('mean', float('nan')):.3f}",
            flush=True,
        )

    payload = {
        "description": (
            "Representative pore-mask occupancy diagnostic for Core1 baseline "
            "recombined trajectories by sampler. Occupancy uses the same "
            "floor-position voxel convention as the resolved tracker. Dilution "
            "errors are taken from the manuscript-facing 20,000-particle "
            "outer-split benchmark."
        ),
        "raw": str(args.raw),
        "trajectory_input": str(args.trajectories),
        "benchmark": str(args.benchmark),
        "mask_shape": list(mask.shape),
        "split": split_payload,
        "archive": {
            "segment_steps": args.segment_steps,
            "match_steps": args.match_steps,
            "segment_stride": args.segment_stride,
            "archive_segments": int(archive.size),
            "train_pool": int(len(train_pool)),
        },
        "diagnostic_settings": {
            "diffusivity": args.diffusivity,
            "candidate_limit": args.candidate_limit,
            "gaussian_bandwidth": args.gaussian_bandwidth,
            "n_generated": args.n_generated,
            "n_segments": args.n_segments,
        },
        "samplers": sampler_payload,
        "occupancy": {
            "held_out_reference_sample": occupancy_summary(reference, mask),
        },
        "correlations": violation_dilution_correlations(sampler_payload),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_summary(payload)
    print(f"wrote {args.output}")


def load_connected_mask(args: argparse.Namespace) -> np.ndarray:
    shape = tuple(int(item) for item in args.shape.split(","))
    volume = load_raw_volume(args.raw, shape=shape, dtype="<u2", order="C")
    if args.downsample_factor > 1:
        volume = block_average(volume, args.downsample_factor)
    pore_mask = segment_pore_space(volume, threshold=None, pore_is_dark=True)
    return connected_pore_network(pore_mask, axis=0)


def sample_reference(trajectories: list[np.ndarray], n_reference: int, *, seed: int) -> list[np.ndarray]:
    n = min(n_reference, len(trajectories))
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(trajectories), size=n, replace=False)
    return [trajectories[int(idx)] for idx in ids]


def occupancy_summary(trajectories: list[np.ndarray], mask: np.ndarray) -> dict[str, float | int]:
    points = np.concatenate([np.asarray(path, dtype=float) for path in trajectories], axis=0)
    indices = np.floor(points).astype(int)
    lower_ok = np.all(indices >= 0, axis=1)
    upper_ok = np.all(indices < np.asarray(mask.shape)[None, :], axis=1)
    in_domain = lower_ok & upper_ok

    solid = np.zeros(len(points), dtype=bool)
    if np.any(in_domain):
        valid_indices = indices[in_domain]
        solid[in_domain] = ~mask[
            valid_indices[:, 0],
            valid_indices[:, 1],
            valid_indices[:, 2],
        ]
    outside_domain = ~in_domain
    violation = outside_domain | solid
    in_domain_count = int(np.sum(in_domain))
    return {
        "n_trajectories": int(len(trajectories)),
        "n_points": int(len(points)),
        "outside_domain_points": int(np.sum(outside_domain)),
        "inside_domain_solid_points": int(np.sum(solid)),
        "violation_points": int(np.sum(violation)),
        "outside_domain_fraction_all_points": fraction(np.sum(outside_domain), len(points)),
        "inside_domain_solid_fraction_all_points": fraction(np.sum(solid), len(points)),
        "violation_fraction_all_points": fraction(np.sum(violation), len(points)),
        "inside_domain_solid_fraction_in_domain_points": fraction(np.sum(solid), in_domain_count),
        "mean_path_length": float(np.mean([len(path) for path in trajectories])),
    }


def build_eval_samplers(args: argparse.Namespace, archive: SegmentArchive, components: dict[str, object]) -> dict[str, object]:
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8")) if args.benchmark.exists() else {}
    outer = benchmark.get("outer_results", [{}])[0] if benchmark.get("outer_results") else {}
    mean_weights = np.asarray(
        [
            outer.get("mean_selected_weights", {}).get(name, 1.0 / len(COMPONENT_ORDER))
            for name in COMPONENT_ORDER
        ],
        dtype=float,
    )
    pooled_weights = np.asarray(
        [
            outer.get("pooled_validation_weights", {}).get(name, 1.0 / len(COMPONENT_ORDER))
            for name in COMPONENT_ORDER
        ],
        dtype=float,
    )
    component_list = [components[name] for name in COMPONENT_ORDER]
    return {
        "bootstrap_mean_mixture": MixtureSegmentSampler(
            archive=archive,
            components=component_list,
            component_weights=mean_weights,
            seed=args.seed,
        ),
        "pooled_validation_mixture": MixtureSegmentSampler(
            archive=archive,
            components=component_list,
            component_weights=pooled_weights,
            seed=args.seed,
        ),
        **components,
    }


def benchmark_dilution_errors(path: Path) -> dict[str, dict[str, float | list[float]]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    results: dict[str, dict[str, float | list[float]]] = {}
    outer_results = payload.get("outer_results", [])
    if not outer_results:
        return results
    sampler_names = list(outer_results[0].get("test", {}).keys())
    for name in sampler_names:
        values = np.asarray(
            [
                result["test"][name]["errors"]["dilution_log_mae"]
                for result in outer_results
                if name in result.get("test", {})
            ],
            dtype=float,
        )
        if len(values) == 0:
            continue
        results[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "split_values": [float(value) for value in values],
        }
    return results


def violation_dilution_correlations(
    sampler_payload: dict[str, dict[str, object]]
) -> dict[str, dict[str, float | int] | str]:
    names = []
    violations = []
    dilutions = []
    for name, payload in sampler_payload.items():
        dilution = payload.get("benchmark_dilution_log_mae")
        if not dilution:
            continue
        names.append(name)
        violations.append(float(payload["occupancy"]["violation_fraction_all_points"]))
        dilutions.append(float(dilution["mean"]))
    if len(violations) < 3:
        return {"note": "fewer than three paired sampler values available"}
    x = np.asarray(violations, dtype=float)
    y = np.asarray(dilutions, dtype=float)
    return {
        "samplers": names,
        "pearson": correlation_payload(x, y),
        "spearman": correlation_payload(rankdata(x), rankdata(y)),
        "interpretation": (
            "Positive values indicate that samplers with more pore-mask violations "
            "also have larger held-out dilution log errors in this diagnostic."
        ),
    }


def correlation_payload(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    if len(x) != len(y) or len(x) < 2:
        return {"n": int(len(x)), "r": float("nan")}
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return {"n": int(len(x)), "r": float("nan")}
    return {"n": int(len(x)), "r": float(np.corrcoef(x, y)[0, 1])}


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def fraction(numerator: int | np.integer, denominator: int | np.integer) -> float:
    denom = int(denominator)
    return float(numerator) / denom if denom else float("nan")


def print_summary(payload: dict[str, object]) -> None:
    reference = payload["occupancy"]["held_out_reference_sample"]
    print(
        f"reference: points={reference['n_points']} "
        f"viol={100.0 * reference['violation_fraction_all_points']:.2f}% "
        f"solid={100.0 * reference['inside_domain_solid_fraction_all_points']:.2f}% "
        f"outside={100.0 * reference['outside_domain_fraction_all_points']:.2f}%"
    )
    pearson = payload["correlations"].get("pearson", {})
    spearman = payload["correlations"].get("spearman", {})
    print(
        "violation-vs-dilution correlation: "
        f"Pearson r={pearson.get('r', float('nan')):.3f}; "
        f"Spearman r={spearman.get('r', float('nan')):.3f}"
    )


if __name__ == "__main__":
    main()

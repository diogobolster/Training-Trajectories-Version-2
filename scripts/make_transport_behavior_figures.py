from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tta_v2 import MetricSettings, evaluate_ensemble, load_trajectories  # noqa: E402


FIGURES = ROOT / "figures"
PAPER_FIGURES = ROOT / "paper" / "figures"
OUTER_BENCHMARK = ROOT / "outputs" / "bentheimer_6um_downsample3_outer_split_mixture_benchmark.json"

SAMPLERS = [
    "reference",
    "pooled_validation_mixture",
    "gaussian_bayes",
    "hybrid",
    "knn_conditional",
    "pair_rerank",
]

LABELS = {
    "reference": "reference",
    "pooled_validation_mixture": "validation mixture",
    "gaussian_bayes": "physics kernel",
    "hybrid": "learned hybrid",
    "knn_conditional": "nearest neighbors",
    "pair_rerank": "pair-aware",
}

COLORS = {
    "reference": "#111827",
    "pooled_validation_mixture": "#2563eb",
    "gaussian_bayes": "#243b53",
    "hybrid": "#2f855a",
    "knn_conditional": "#d97706",
    "pair_rerank": "#b83280",
}


def main() -> None:
    setup_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)

    benchmark = load_json(OUTER_BENCHMARK)
    trajectories = load_trajectories(ROOT / benchmark["input"])
    records = collect_records(benchmark, trajectories)
    summary = summarize_records(records)
    fig = behavior_figure(summary)
    save_all("figure3_transport_behavior_comparison", fig)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 8.8,
            "legend.fontsize": 8.2,
            "figure.dpi": 170,
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_records(benchmark: dict, trajectories: list[np.ndarray]) -> dict[str, list[dict]]:
    records: dict[str, list[dict]] = {name: [] for name in SAMPLERS}
    for outer in benchmark["outer_results"]:
        split = outer["splits"]
        test = fixed_test_split(
            trajectories,
            test_fraction=float(split["test_fraction"]),
            seed=int(split["seed"]),
        )
        settings = metric_settings_from_outer(outer)
        records["reference"].append(evaluate_ensemble(test, settings))
        for sampler in SAMPLERS:
            if sampler == "reference":
                continue
            records[sampler].append(outer["test"][sampler]["metrics"])
    return records


def fixed_test_split(
    trajectories: list[np.ndarray],
    *,
    test_fraction: float,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(trajectories))
    n_test = max(1, int(round(test_fraction * len(trajectories))))
    n_test = min(n_test, len(trajectories) - 2)
    return [trajectories[int(index)] for index in indices[:n_test]]


def metric_settings_from_outer(outer: dict) -> MetricSettings:
    first_metrics = next(iter(outer["test"].values()))["metrics"]
    planes = sorted(float(key) for key in first_metrics["breakthrough"])
    time_indices = sorted(int(key) for key in first_metrics["dilution"])
    return MetricSettings(
        planes=planes,
        time_indices=time_indices,
        bin_size=3.0,
        pair_samples=1200,
        reaction_radius=3.0,
        seed=int(outer["seed"]),
    )


def summarize_records(records: dict[str, list[dict]]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name, items in records.items():
        planes = sorted(float(key) for key in items[0]["breakthrough"])
        times = sorted(int(key) for key in items[0]["dilution"])
        out[name] = {
            "planes": planes,
            "times": times,
            "breakthrough": {
                stat: np.asarray(
                    [
                        np.nanmean([float(read_metric(item["breakthrough"], plane)[stat]) for item in items])
                        for plane in planes
                    ],
                    dtype=float,
                )
                for stat in ("q10", "q50", "q90", "coverage")
            },
            "breakthrough_std": {
                stat: np.asarray(
                    [
                        np.nanstd(
                            [float(read_metric(item["breakthrough"], plane)[stat]) for item in items],
                            ddof=1,
                        )
                        for plane in planes
                    ],
                    dtype=float,
                )
                for stat in ("q10", "q50", "q90", "coverage")
            },
            "dilution_mean": np.asarray(
                [
                    np.nanmean(
                        [
                            float(read_metric(item["dilution"], time_index)["dilution_index"])
                            for item in items
                        ]
                    )
                    for time_index in times
                ],
                dtype=float,
            ),
            "dilution_std": np.asarray(
                [
                    np.nanstd(
                        [
                            float(read_metric(item["dilution"], time_index)["dilution_index"])
                            for item in items
                        ],
                        ddof=1,
                    )
                    for time_index in times
                ],
                dtype=float,
            ),
        }
    return out


def read_metric(metrics: dict, key: float | int) -> dict:
    for candidate in (key, float(key), int(key), str(key), f"{float(key):.1f}", str(int(key))):
        if candidate in metrics:
            return metrics[candidate]
    raise KeyError(key)


def behavior_figure(summary: dict[str, dict]) -> plt.Figure:
    fig = plt.figure(figsize=(7.45, 6.75), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[1.14, 0.86], hspace=0.40, wspace=0.34)
    ax_btc = fig.add_subplot(gs[0, 0])
    ax_cov = fig.add_subplot(gs[0, 1])
    ax_dil = fig.add_subplot(gs[1, :])

    fig.suptitle(
        "Held-out transport behavior behind the multi-objective scores",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=13.4,
        fontweight="bold",
        color="#102a43",
    )
    fig.text(
        0.02,
        0.945,
        "Averaged over five outer test splits; reference trajectories are shown in black.",
        ha="left",
        fontsize=9.1,
        color="#415161",
    )

    draw_breakthrough_panel(ax_btc, summary)
    draw_coverage_panel(ax_cov, summary)
    draw_dilution_panel(ax_dil, summary)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.075, top=0.88)
    return fig


def draw_breakthrough_panel(ax: plt.Axes, summary: dict[str, dict]) -> None:
    planes = np.asarray(summary["reference"]["planes"], dtype=float)
    offsets = np.linspace(-0.32, 0.32, len(SAMPLERS))
    for idx, sampler in enumerate(SAMPLERS):
        stats = summary[sampler]["breakthrough"]
        q50 = stats["q50"]
        q10 = stats["q10"]
        q90 = stats["q90"]
        color = COLORS[sampler]
        lw = 2.3 if sampler == "reference" else 1.45
        alpha = 1.0 if sampler == "reference" else 0.86
        x = planes + offsets[idx]
        ax.errorbar(
            x,
            q50,
            yerr=np.vstack([q50 - q10, q90 - q50]),
            color=color,
            marker="o",
            markersize=4.1,
            linewidth=lw,
            elinewidth=1.0,
            capsize=2.0,
            alpha=alpha,
            label=LABELS[sampler],
        )
    ax.set_title("Breakthrough timing summaries", fontweight="bold", color="#102a43")
    ax.set_xlabel("control plane, x")
    ax.set_ylabel("first-passage time")
    ax.set_xticks(planes)
    ax.grid(True, color="#d9e2ec", linewidth=0.8, alpha=0.75)
    ax.text(
        0.02,
        0.96,
        "points: q50\nbars: q10-q90",
        transform=ax.transAxes,
        va="top",
        fontsize=7.4,
        color="#52606d",
    )


def draw_coverage_panel(ax: plt.Axes, summary: dict[str, dict]) -> None:
    planes = np.asarray(summary["reference"]["planes"], dtype=float)
    for sampler in SAMPLERS:
        stats = summary[sampler]["breakthrough"]
        color = COLORS[sampler]
        lw = 2.3 if sampler == "reference" else 1.5
        ax.plot(
            planes,
            stats["coverage"],
            color=color,
            marker="o",
            markersize=4.0,
            linewidth=lw,
            alpha=1.0 if sampler == "reference" else 0.86,
        )
    ax.set_title("Crossing coverage", fontweight="bold", color="#102a43")
    ax.set_xlabel("control plane, x")
    ax.set_ylabel("fraction crossed")
    ax.set_xticks(planes)
    ax.set_ylim(0, 1.03)
    ax.grid(True, color="#d9e2ec", linewidth=0.8, alpha=0.75)


def draw_dilution_panel(ax: plt.Axes, summary: dict[str, dict]) -> None:
    for sampler in SAMPLERS:
        times = np.asarray(summary[sampler]["times"], dtype=float)
        mean = summary[sampler]["dilution_mean"]
        std = summary[sampler]["dilution_std"]
        color = COLORS[sampler]
        lw = 2.4 if sampler == "reference" else 1.65
        ax.plot(
            times,
            mean,
            color=color,
            marker="o",
            markersize=4.0,
            linewidth=lw,
            alpha=1.0 if sampler == "reference" else 0.88,
            label=LABELS[sampler],
        )
        if sampler in {"reference", "pooled_validation_mixture", "gaussian_bayes"}:
            ax.fill_between(times, mean - std, mean + std, color=color, alpha=0.08, linewidth=0)
    ax.set_title("Dilution index through time", fontweight="bold", color="#102a43")
    ax.set_xlabel("time step")
    ax.set_ylabel("dilution index")
    ax.grid(True, color="#d9e2ec", linewidth=0.8, alpha=0.75)
    ax.legend(ncol=3, loc="upper left", frameon=False, handlelength=2.2, columnspacing=1.1)


def save_all(name: str, fig: plt.Figure) -> None:
    for directory in (FIGURES, PAPER_FIGURES):
        fig.savefig(directory / f"{name}.png", bbox_inches="tight")
        fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


if __name__ == "__main__":
    main()

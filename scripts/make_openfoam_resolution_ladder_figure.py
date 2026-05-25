from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "openfoam_resolution_ladder_summary.json"
BENCHMARKS = {
    "downsample3": ROOT / "outputs" / "bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_dt010_n20000_stride1600_outer_split_mixture_benchmark.json",
    "downsample2": ROOT / "outputs" / "bentheimer_core2_subvol1_6um_downsample2_D00225_openfoam_phys_scaled_dt010_n20000_stride1600_outer_split_mixture_benchmark.json",
    "fullres": ROOT / "outputs" / "bentheimer_core2_subvol1_6um_fullres_D009_openfoam_strict_dt010_n20000_stride1600_outer_split_mixture_benchmark.json",
}
OUT = ROOT / "figures" / "run_016_openfoam_resolution_ladder.svg"
OUT_PNG = ROOT / "figures" / "run_016_openfoam_resolution_ladder.png"
OUT_PDF = ROOT / "figures" / "run_016_openfoam_resolution_ladder.pdf"
PAPER_OUT = ROOT / "paper" / "figures" / "run_016_openfoam_resolution_ladder.svg"
PAPER_OUT_PNG = ROOT / "paper" / "figures" / "run_016_openfoam_resolution_ladder.png"
PAPER_OUT_PDF = ROOT / "paper" / "figures" / "run_016_openfoam_resolution_ladder.pdf"

ORDER = ["downsample3", "downsample2", "fullres"]
XTICK = ["18 um", "12 um", "6 um"]
MEMORY_ORDER = ["gaussian_bayes", "knn_conditional", "hybrid", "pair_rerank"]
SAMPLERS = ["pooled_validation_mixture", "gaussian_bayes", "knn_conditional", "hybrid", "pair_rerank"]

MEMORY_LABELS = {
    "pooled_validation_mixture": "validation mixture",
    "gaussian_bayes": "velocity continuity",
    "knn_conditional": "archive proximity",
    "hybrid": "learned context",
    "pair_rerank": "pair organization",
}
SHORT = {
    "pooled_validation_mixture": "mixture",
    "gaussian_bayes": "velocity",
    "knn_conditional": "archive",
    "hybrid": "learned",
    "pair_rerank": "pair org.",
}
COLORS = {
    "gaussian_bayes": "#4E79A7",
    "knn_conditional": "#B07D12",
    "hybrid": "#5C8D6A",
    "pair_rerank": "#9B6A8F",
    "pooled_validation_mixture": "#6F6F6F",
    "bootstrap_mean_mixture": "#9A9A9A",
    "graph": "#A8A8A8",
    "openfoam18": "#7A7A7A",
    "openfoam12": "#4D4D4D",
    "openfoam6": "#9E6240",
    "ink": "#1A1A1A",
    "muted": "#666666",
    "grid": "#E1E1E1",
}


def main() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    benchmarks = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in BENCHMARKS.items()}
    cases = payload["cases"]
    x = np.arange(len(ORDER), dtype=float)

    setup_style()
    fig = plt.figure(figsize=(7.45, 6.25), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.12], hspace=0.62, wspace=0.34)
    ax_k = fig.add_subplot(gs[0, 0])
    ax_corr = fig.add_subplot(gs[0, 1])
    ax_scores = fig.add_subplot(gs[1, 0])
    ax_weights = fig.add_subplot(gs[1, 1])

    draw_permeability(ax_k, cases, x)
    draw_autocorrelation(ax_corr, cases)
    draw_validation_scores(ax_scores, benchmarks, x)
    draw_weight_variability(ax_weights, benchmarks, x)
    add_panel_labels([ax_k, ax_corr, ax_scores, ax_weights])
    fig.subplots_adjust(top=0.955, bottom=0.175, left=0.085, right=0.985)

    for path in (OUT, OUT_PNG, OUT_PDF, PAPER_OUT, PAPER_OUT_PNG, PAPER_OUT_PDF):
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=320, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(PAPER_OUT, bbox_inches="tight")
    fig.savefig(PAPER_OUT_PNG, dpi=320, bbox_inches="tight")
    fig.savefig(PAPER_OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"Wrote {OUT_PNG.relative_to(ROOT)}")
    print(f"Wrote {OUT_PDF.relative_to(ROOT)}")


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.9,
            "axes.titlesize": 8.3,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.1,
            "legend.fontsize": 6.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def draw_permeability(ax: plt.Axes, cases: dict, x: np.ndarray) -> None:
    values = np.asarray([cases[name]["apparent_permeability"] * 1e12 for name in ORDER], dtype=float)
    cells = np.asarray([cases[name]["n_cells"] for name in ORDER], dtype=float)
    ax.plot(x, values, color=COLORS["ink"], marker="o", lw=1.45, ms=4.3)
    ax.set_xticks(x, [f"{label}\n{cell/1e6:.2f}M cells" for label, cell in zip(XTICK, cells)])
    ax.set_ylabel(r"apparent $k$ ($10^{-12}$ m$^2$)")
    ax.set_title("hydraulic response tightens", loc="left", fontweight="normal", color=COLORS["ink"])
    ax.set_ylim(values.min() - 0.12, values.max() + 0.18)
    ax.grid(True, color=COLORS["grid"], linewidth=0.65)
    for xi, value in zip(x, values):
        ax.text(xi, value + 0.045, f"{value:.2f}", ha="center", va="bottom", fontsize=6.9, color=COLORS["ink"])
    pct01 = 100.0 * (values[1] / values[0] - 1.0)
    pct12 = 100.0 * (values[2] / values[1] - 1.0)
    pct02 = 100.0 * (values[2] / values[0] - 1.0)
    annotate_segment(ax, x[0], values[0], x[1], values[1], f"{pct01:.0f}%")
    annotate_segment(ax, x[1], values[1], x[2], values[2], f"{pct12:.0f}%")
    ax.text(
        0.97,
        0.92,
        f"total {pct02:.0f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color=COLORS["muted"],
    )


def annotate_segment(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, label: str) -> None:
    xm = 0.5 * (x0 + x1)
    ym = 0.5 * (y0 + y1)
    ax.annotate(
        label,
        xy=(xm, ym),
        xytext=(xm, ym + 0.10),
        ha="center",
        fontsize=6.4,
        color=COLORS["muted"],
        arrowprops=dict(arrowstyle="-", color=COLORS["grid"], lw=0.8),
    )


def draw_autocorrelation(ax: plt.Axes, cases: dict) -> None:
    labels = {"downsample3": "18 um", "downsample2": "12 um", "fullres": "6 um"}
    colors = {"downsample3": COLORS["openfoam18"], "downsample2": COLORS["openfoam12"], "fullres": COLORS["openfoam6"]}
    all_values = []
    for name in ORDER:
        corr = cases[name]["autocorrelation"]
        lags = np.asarray([int(key) for key in corr.keys()], dtype=float)
        vals = np.asarray([corr[str(int(lag))] for lag in lags], dtype=float)
        all_values.extend(vals.tolist())
        ax.plot(lags, vals, marker="o", lw=1.5, ms=3.3, color=colors[name], label=labels[name])
    ax.set_xlabel("lag in saved steps")
    ax.set_ylabel("axial velocity autocorrelation")
    ax.set_title("OpenFOAM velocity persistence is stable", loc="left", fontweight="normal", color=COLORS["ink"])
    ax.set_xlim(8, 82)
    ymin = max(0.0, min(all_values) - 0.015)
    ymax = max(all_values) + 0.018
    ax.set_ylim(ymin, ymax)
    ax.grid(True, color=COLORS["grid"], linewidth=0.65)
    ax.legend(frameon=False, ncols=1, loc="upper right", columnspacing=0.8, handlelength=1.5)


def draw_validation_scores(ax: plt.Axes, benchmarks: dict, x: np.ndarray) -> None:
    offsets = np.linspace(-0.18, 0.18, len(SAMPLERS))
    rng = np.random.default_rng(74)
    for offset, sampler in zip(offsets, SAMPLERS):
        means = []
        stds = []
        for i, key in enumerate(ORDER):
            summary = benchmarks[key]["summary"]["samplers"][sampler]
            means.append(float(summary["mean_objective"]))
            stds.append(float(summary["std_objective"]))
            split_values = [float(outer["test"][sampler]["objective"]) for outer in benchmarks[key]["outer_results"]]
            jitter = rng.normal(0.0, 0.011, size=len(split_values))
            ax.scatter(
                np.full(len(split_values), x[i] + offset) + jitter,
                split_values,
                s=10,
                color=COLORS[sampler],
                alpha=0.25,
                edgecolor="none",
                zorder=2,
            )
        ax.errorbar(
            x + offset,
            means,
            yerr=stds,
            color=COLORS[sampler],
            marker="o",
            lw=1.0,
            elinewidth=0.75,
            capsize=2.0,
            markersize=3.0,
            label=MEMORY_LABELS[sampler],
            zorder=4,
        )
    ax.set_xticks(x, XTICK)
    ax.set_ylabel("held-out objective")
    ax.set_title("validation changes retained state", loc="left", fontweight="normal", color=COLORS["ink"])
    ax.grid(True, color=COLORS["grid"], linewidth=0.65)
    ylo, yhi = ax.get_ylim()
    yrange = yhi - ylo
    ax.set_ylim(ylo, yhi + 0.18 * yrange)
    for i, key in enumerate(ORDER):
        summary = benchmarks[key]["summary"]["samplers"]
        best = min(SAMPLERS, key=lambda sampler: float(summary[sampler]["mean_objective"]))
        wins = int(summary[best].get("wins", 0))
        ax.text(
            x[i],
            yhi + 0.105 * yrange,
            f"lowest mean:\n{SHORT[best]}\nsplit wins: {wins}/4",
            ha="center",
            va="center",
            fontsize=5.6,
            color=COLORS["muted"],
            linespacing=0.92,
        )
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.52, -0.22), columnspacing=0.8, handlelength=1.1)


def draw_weight_variability(ax: plt.Axes, benchmarks: dict, x: np.ndarray) -> None:
    offsets = np.linspace(-0.15, 0.15, len(MEMORY_ORDER))
    rng = np.random.default_rng(18)
    for offset, memory in zip(offsets, MEMORY_ORDER):
        mean_values = []
        std_values = []
        for i, key in enumerate(ORDER):
            samples = selected_weight_samples(benchmarks[key], memory)
            mean_values.append(float(np.mean(samples)))
            std_values.append(float(np.std(samples, ddof=1)))
            jitter = rng.normal(0.0, 0.010, size=len(samples))
            ax.scatter(np.full(len(samples), x[i] + offset) + jitter, samples, s=9, color=COLORS[memory], alpha=0.24, edgecolor="none")
        ax.errorbar(
            x + offset,
            mean_values,
            yerr=std_values,
            color=COLORS[memory],
            marker="o",
            lw=1.05,
            elinewidth=0.8,
            capsize=2.1,
            markersize=3.3,
            label=MEMORY_LABELS[memory],
        )
    ax.set_xticks(x, XTICK)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("selected mixture weight")
    ax.set_title("mixture weights include split variability", loc="left", fontweight="normal", color=COLORS["ink"])
    ax.grid(True, color=COLORS["grid"], linewidth=0.65)
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.53, -0.22), columnspacing=0.8, handlelength=1.1)


def selected_weight_samples(benchmark: dict, memory: str) -> np.ndarray:
    values = []
    for outer in benchmark["outer_results"]:
        values.append(float(outer["pooled_validation_weights"][memory]))
        for repeat in outer["repeat_results"]:
            values.append(float(repeat["selected_weights"][memory]))
    return np.asarray(values, dtype=float)


def add_panel_labels(axes: list[plt.Axes]) -> None:
    for idx, ax in enumerate(axes):
        ax.text(
            -0.14,
            1.08,
            f"({chr(ord('a') + idx)})",
            transform=ax.transAxes,
            fontsize=9.2,
            fontweight="bold",
            ha="left",
            va="bottom",
            color=COLORS["ink"],
        )


if __name__ == "__main__":
    main()

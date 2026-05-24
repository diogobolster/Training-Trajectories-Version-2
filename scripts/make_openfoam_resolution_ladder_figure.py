from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "openfoam_resolution_ladder_summary.json"
OUT = ROOT / "figures" / "run_016_openfoam_resolution_ladder.svg"
OUT_PNG = ROOT / "figures" / "run_016_openfoam_resolution_ladder.png"

ORDER = ["downsample3", "downsample2", "fullres"]
XTICK = ["18 um", "12 um", "6 um"]
MEMORY_ORDER = ["gaussian_bayes", "knn_conditional", "hybrid", "pair_rerank"]
MEMORY_LABELS = {
    "gaussian_bayes": "velocity",
    "knn_conditional": "archive",
    "hybrid": "learned",
    "pair_rerank": "pairs",
}
COLORS = {
    "gaussian_bayes": "#1f4e79",
    "knn_conditional": "#d97706",
    "hybrid": "#2f855a",
    "pair_rerank": "#b83280",
    "pooled_validation_mixture": "#2563eb",
    "bootstrap_mean_mixture": "#60a5fa",
    "ink": "#102a43",
    "muted": "#64748b",
    "grid": "#e5e7eb",
}


def main() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    cases = payload["cases"]
    x = np.arange(len(ORDER))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 10.6,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.4,
        }
    )
    fig = plt.figure(figsize=(10.6, 6.1), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.58, wspace=0.44)
    ax_k = fig.add_subplot(gs[0, 0])
    ax_corr = fig.add_subplot(gs[0, 1:])
    ax_weights = fig.add_subplot(gs[1, 0:2])
    ax_obj = fig.add_subplot(gs[1, 2])

    draw_permeability(ax_k, cases, x)
    draw_autocorrelation(ax_corr, cases)
    draw_weights(ax_weights, cases, x)
    draw_objective(ax_obj, cases, x)

    fig.suptitle(
        "Full-resolution OpenFOAM sharpens, but does not simplify, the memory question",
        x=0.03,
        y=0.975,
        ha="left",
        fontsize=14.0,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.03,
        0.933,
        "Permeability tightens across resolution; tight 5000-particle validation selects different memories rather than a universal sampler.",
        ha="left",
        va="top",
        fontsize=9.6,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(top=0.84, bottom=0.16, left=0.08, right=0.985)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"Wrote {OUT_PNG.relative_to(ROOT)}")


def draw_permeability(ax, cases: dict, x: np.ndarray) -> None:
    values = [cases[name]["apparent_permeability"] * 1e12 for name in ORDER]
    cells = [cases[name]["n_cells"] for name in ORDER]
    ax.plot(x, values, color=COLORS["ink"], marker="o", lw=2.0, ms=5.5)
    ax.set_xticks(x, XTICK)
    ax.set_ylabel("apparent k (10^-12 m^2)")
    ax.set_title("Bulk flow response tightens", loc="left", fontweight="bold", color=COLORS["ink"])
    ax.set_ylim(min(values) - 0.08, max(values) + 0.14)
    ax.grid(True, color=COLORS["grid"], linewidth=0.8)
    for xi, val, n_cells in zip(x, values, cells):
        ax.text(xi, val + 0.045, f"{val:.2f}", ha="center", va="bottom", fontsize=8.0, color=COLORS["ink"])
        ax.text(
            xi,
            0.02,
            f"{n_cells/1e6:.2f}M cells",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=7.4,
            color=COLORS["muted"],
        )
    ax.spines[["top", "right"]].set_visible(False)


def draw_autocorrelation(ax, cases: dict) -> None:
    labels = {
        "downsample3": "18 um",
        "downsample2": "12 um",
        "fullres": "6 um",
    }
    colors = {
        "downsample3": "#94a3b8",
        "downsample2": "#475569",
        "fullres": "#dc2626",
    }
    for name in ORDER:
        corr = cases[name]["autocorrelation"]
        lags = np.array([int(k) for k in corr.keys()])
        vals = np.array([corr[str(lag)] for lag in lags])
        ax.plot(lags, vals, marker="o", lw=2.0, ms=4.2, color=colors[name], label=labels[name])
    ax.set_xscale("log")
    all_vals = []
    for name in ORDER:
        all_vals.extend(float(value) for value in cases[name]["autocorrelation"].values())
    lower = max(0.0, min(all_vals) - 0.04)
    upper = min(1.0, max(all_vals) + 0.04)
    ax.set_xticks([10, 20, 40, 80], ["10", "20", "40", "80"])
    ax.set_ylim(lower, upper)
    ax.set_ylabel("axial velocity autocorrelation")
    ax.set_xlabel("lag in saved steps (dt = 0.1)")
    ax.set_title("Velocity memory persists in the tight archives", loc="left", fontweight="bold", color=COLORS["ink"])
    ax.grid(True, color=COLORS["grid"], linewidth=0.8)
    ax.legend(frameon=False, ncols=3, loc="upper right", fontsize=8.4)
    ax.spines[["top", "right"]].set_visible(False)


def draw_weights(ax, cases: dict, x: np.ndarray) -> None:
    bottoms = np.zeros(len(ORDER))
    for memory in MEMORY_ORDER:
        vals = [cases[name]["balanced"]["mean_selected_weights"][memory] for name in ORDER]
        ax.bar(
            x,
            vals,
            bottom=bottoms,
            width=0.62,
            color=COLORS[memory],
            edgecolor="white",
            linewidth=0.9,
            label=MEMORY_LABELS[memory],
        )
        bottoms += np.asarray(vals)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x, XTICK)
    ax.set_ylabel("selected mixture weight")
    ax.set_title("Balanced validation keeps multiple memories alive", loc="left", fontweight="bold", color=COLORS["ink"])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.50, -0.15), fontsize=8.4)
    ax.spines[["top", "right"]].set_visible(False)


def draw_objective(ax, cases: dict, x: np.ndarray) -> None:
    vals = [cases[name]["balanced"]["best_mean_objective"] for name in ORDER]
    colors = [COLORS.get(cases[name]["balanced"]["best_sampler"], COLORS["muted"]) for name in ORDER]
    ax.bar(x, vals, width=0.62, color=colors, edgecolor="white", linewidth=0.9)
    ax.set_xticks(x, XTICK)
    ax.set_ylabel("best mean objective")
    ax.set_title("Held-out score", loc="left", fontweight="bold", color=COLORS["ink"])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    pad = 0.035 * max(vals)
    ax.set_ylim(0, max(vals) + 3 * pad)
    for xi, val in zip(x, vals):
        ax.text(xi, val + pad * 0.25, f"{val:.0f}", ha="center", va="bottom", fontsize=8.0, color=COLORS["ink"])
    for xi, name in enumerate(ORDER):
        best = cases[name]["balanced"]["best_sampler"]
        ax.text(
            xi,
            vals[xi] * 0.52,
            sampler_label(best),
            ha="center",
            va="center",
            fontsize=8.1,
            color="white",
            fontweight="bold",
        )
    ax.spines[["top", "right"]].set_visible(False)


def sampler_label(name: str) -> str:
    return {
        "pooled_validation_mixture": "mixture",
        "bootstrap_mean_mixture": "mean mix",
        "gaussian_bayes": "velocity",
        "knn_conditional": "archive",
        "hybrid": "learned",
        "pair_rerank": "pairs",
    }.get(name, name)


if __name__ == "__main__":
    main()

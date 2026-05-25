from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "core1_archive_convergence.json"
FIGURES = ROOT / "paper" / "figures"
COLORS = {"short": "#2f855a", "baseline": "#1f4e79"}
LABELS = {"short": "24-step motifs", "baseline": "36-step motifs"}


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.7,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.6,
            "legend.fontsize": 7.6,
            "figure.dpi": 170,
            "savefig.dpi": 320,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    summary = payload["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.2), facecolor="white", gridspec_kw={"wspace": 0.32})
    draw_ratio_panel(
        axes[0],
        summary,
        "generated_to_full_reference_final_ratio",
        "Against full held-out reference",
        "generated / full-reference final dilution",
        (0.0, 1.32),
    )
    draw_ratio_panel(
        axes[1],
        summary,
        "generated_to_reference_sample_final_ratio",
        "Against same-count reference subsamples",
        "generated / same-count final dilution",
        (0.0, 1.45),
    )
    fig.suptitle("Archive-density and segment-length diagnostic", x=0.02, ha="left", fontsize=12.0, fontweight="bold", color="#102a43")
    fig.text(
        0.02,
        0.900,
        "Core1 baseline Gaussian/Bayes recombination; 180 generated paths per repeat, 3 repeats per setting.",
        ha="left",
        fontsize=8.0,
        color="#52606d",
    )
    fig.subplots_adjust(left=0.080, right=0.985, top=0.78, bottom=0.20)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(FIGURES / f"figureS_archive_convergence.{suffix}", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print("wrote figureS_archive_convergence")


def draw_ratio_panel(
    ax: plt.Axes,
    rows: list[dict],
    prefix: str,
    title: str,
    ylabel: str,
    ylim: tuple[float, float],
) -> None:
    for segment in ("short", "baseline"):
        selected = [row for row in rows if row["segment_config"] == segment]
        selected.sort(key=lambda row: row["archive_particles"])
        x = np.asarray([row["archive_segments_mean"] / 1000.0 for row in selected], dtype=float)
        y = np.asarray([row[f"{prefix}_mean"] for row in selected], dtype=float)
        err = np.asarray([row[f"{prefix}_std"] for row in selected], dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=err,
            color=COLORS[segment],
            marker="o",
            lw=2.0,
            capsize=2.5,
            label=LABELS[segment],
        )
    ax.axhline(1.0, color="#64748b", lw=0.9, ls="--")
    ax.set_title(title, fontweight="bold", color="#102a43")
    ax.set_xlabel("archive segments (thousands)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_xticks([10, 20, 28])
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")
    if "same-count" in title:
        ax.text(0.03, 0.09, "sign changes after sample-count control", transform=ax.transAxes, fontsize=7.4, color="#475569")
    else:
        ax.text(0.03, 0.09, "full-reference gap persists", transform=ax.transAxes, fontsize=7.4, color="#475569")


if __name__ == "__main__":
    main()

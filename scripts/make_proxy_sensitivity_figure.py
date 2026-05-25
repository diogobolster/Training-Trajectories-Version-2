from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "core1_proxy_sensitivity.json"
FIGURES = ROOT / "paper" / "figures"


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.7,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.7,
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
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.15), facecolor="white", gridspec_kw={"wspace": 0.34})
    draw_bin_sensitivity(axes[0], payload["dilution_bin_sensitivity"])
    draw_radius_sensitivity(axes[1], payload["encounter_radius_sensitivity"])
    fig.suptitle("Proxy-observable sensitivity checks", x=0.02, ha="left", fontsize=12.2, fontweight="bold", color="#102a43")
    fig.text(
        0.02,
        0.905,
        "Core1 baseline, 20,000-particle archive; generated ensembles are re-evaluated under metric perturbations.",
        ha="left",
        fontsize=8.2,
        color="#52606d",
    )
    fig.subplots_adjust(left=0.080, right=0.985, top=0.78, bottom=0.20)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(FIGURES / f"figureS_proxy_sensitivity.{suffix}", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print("wrote figureS_proxy_sensitivity")


def draw_bin_sensitivity(ax: plt.Axes, rows: list[dict]) -> None:
    x = np.asarray([row["bin_size"] for row in rows], dtype=float)
    y = np.asarray([row["gap_over_reference_sd"] for row in rows], dtype=float)
    ax.plot(x, y, color="#2f855a", marker="o", lw=2.1)
    ax.fill_between(x, 0, y, color="#2f855a", alpha=0.12)
    ax.axhline(1.0, color="#64748b", lw=0.9, ls="--")
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 2.0, f"{yi:.1f}x", ha="center", fontsize=7.6, color="#334155", fontweight="bold")
    ax.set_title("Dilution gap survives bin-size choice", fontweight="bold", color="#102a43")
    ax.set_xlabel("dilution bin size (cells)")
    ax.set_ylabel("final dilution gap / ref. split SD")
    ax.set_xticks(x)
    ax.set_ylim(0, max(y) * 1.20)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax.text(0.02, 0.92, "all generated curves remain below reference", transform=ax.transAxes, fontsize=7.5, color="#475569")


def draw_radius_sensitivity(ax: plt.Axes, rows: list[dict]) -> None:
    radius = np.asarray([row["reaction_radius"] for row in rows], dtype=float)
    ref = np.asarray([row["reference_probability_mean"] for row in rows], dtype=float)
    ref_sd = np.asarray([row["reference_probability_sd"] for row in rows], dtype=float)
    gen_min = np.asarray([row["generated_probability_min_mean"] for row in rows], dtype=float)
    gen_max = np.asarray([row["generated_probability_max_mean"] for row in rows], dtype=float)
    ax.fill_between(radius, gen_min, gen_max, color="#b83280", alpha=0.16, label="generated range")
    ax.errorbar(radius, ref, yerr=ref_sd, color="#111827", marker="o", lw=2.0, capsize=2.5, label="reference")
    ax.plot(radius, 0.5 * (gen_min + gen_max), color="#b83280", marker="s", lw=1.6, label="generated mean range")
    ax.set_title("Encounter proxy is radius-sensitive", fontweight="bold", color="#102a43")
    ax.set_xlabel("encounter radius (cells)")
    ax.set_ylabel("encounter probability")
    ax.set_xticks(radius)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.02, 0.08, "used qualitatively, not as a rate law", transform=ax.transAxes, fontsize=7.5, color="#475569")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import make_transport_behavior_figures as behavior  # noqa: E402


FIGURES = ROOT / "figures"
PAPER_FIGURES = ROOT / "paper" / "figures"
OUTER = ROOT / "outputs" / "bentheimer_6um_downsample3_outer_split_mixture_benchmark.json"
OBJECTIVE = ROOT / "outputs" / "bentheimer_6um_downsample3_objective_weight_sensitivity.json"
HIGH_PE = ROOT / "outputs" / "bentheimer_6um_downsample3_D0003_outer_split_mixture_benchmark.json"
LOW_PE = ROOT / "outputs" / "bentheimer_6um_downsample3_D003_outer_split_mixture_benchmark.json"
CORE2_GRAPH = ROOT / "outputs" / "bentheimer_core2_subvol1_6um_downsample3_D001_outer_split_mixture_benchmark.json"
CORE2_OPENFOAM = ROOT / "outputs" / "bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_outer_split_mixture_benchmark.json"
OPENFOAM_OBJECTIVE = ROOT / "outputs" / "bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_objective_weight_sensitivity.json"
FLOW_PHYSICS = ROOT / "outputs" / "core2_graph_vs_openfoam_physics_comparison.json"
EVIDENCE = ROOT / "outputs" / "master_evidence_table.json"

COMPONENTS = ["gaussian_bayes", "knn_conditional", "hybrid", "pair_rerank"]
SAMPLERS = ["pooled_validation_mixture", "gaussian_bayes", "hybrid", "knn_conditional", "pair_rerank"]
FIXED_SAMPLERS = ["gaussian_bayes", "hybrid", "knn_conditional", "pair_rerank"]

LABELS = {
    "reference": "reference",
    "pooled_validation_mixture": "validation mixture",
    "bootstrap_mean_mixture": "mean mixture",
    "gaussian_bayes": "velocity memory",
    "knn_conditional": "archive proximity",
    "hybrid": "learned context",
    "pair_rerank": "pair organization",
    "balanced": "balanced",
    "btc_heavy": "arrival",
    "dilution_heavy": "dilution",
    "pair_heavy": "pairs",
    "reaction_heavy": "encounters",
    "reaction_light": "reaction light",
    "no_reaction": "no reaction",
}

SHORT_LABELS = {
    "pooled_validation_mixture": "mixture",
    "gaussian_bayes": "velocity",
    "knn_conditional": "archive",
    "hybrid": "learned",
    "pair_rerank": "pairs",
}

COLORS = {
    "reference": "#111827",
    "pooled_validation_mixture": "#2563eb",
    "bootstrap_mean_mixture": "#0f766e",
    "gaussian_bayes": "#1f4e79",
    "knn_conditional": "#d97706",
    "hybrid": "#2f855a",
    "pair_rerank": "#b83280",
    "arrival": "#2563eb",
    "dilution": "#2f855a",
    "pairs": "#d97706",
    "encounters": "#b83280",
    "graph": "#64748b",
    "openfoam": "#dc2626",
}

OUTCOME_ERRORS = [
    ("arrival", "btc_score"),
    ("dilution", "dilution_log_mae"),
    ("pairs", "pair_quantile_mae"),
    ("encounters", "reaction_abs_error"),
]


def main() -> None:
    setup_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)

    outer = load_json(OUTER)
    behavior_summary = load_behavior_summary(outer)
    objective = load_json(OBJECTIVE)
    peclet = {
        "high Pe": load_json(HIGH_PE),
        "baseline": outer,
        "low Pe": load_json(LOW_PE),
    }
    core2_graph = load_json(CORE2_GRAPH)
    core2_openfoam = load_json(CORE2_OPENFOAM)
    openfoam_objective = load_json(OPENFOAM_OBJECTIVE)
    flow_physics = load_json(FLOW_PHYSICS)
    evidence = load_json(EVIDENCE)["conditions"]

    save_all("figure3_behavior_memory_gap", behavior_memory_gap(outer, behavior_summary))
    save_all("figure4_outcome_memory_selection", outcome_memory_selection(objective))
    save_all("figure5_diffusion_memory_shift", diffusion_memory_shift(peclet))
    save_all(
        "figure6_flow_fidelity_memory_test",
        flow_fidelity_memory_test(core2_graph, core2_openfoam, openfoam_objective, flow_physics),
    )
    save_all("figure7_memory_map", memory_map(evidence, openfoam_objective))


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.0,
            "legend.fontsize": 8.2,
            "figure.dpi": 170,
            "savefig.dpi": 320,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_behavior_summary(outer: dict) -> dict[str, dict]:
    trajectories = behavior.load_trajectories(ROOT / outer["input"])
    records = behavior.collect_records(outer, trajectories)
    return behavior.summarize_records(records)


def save_all(name: str, fig: plt.Figure) -> None:
    for directory in (FIGURES, PAPER_FIGURES):
        fig.savefig(directory / f"{name}.png", bbox_inches="tight")
        fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


def behavior_memory_gap(outer: dict, summary: dict[str, dict]) -> plt.Figure:
    fig = plt.figure(figsize=(7.45, 7.35), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.94, 1.16], width_ratios=[1.03, 0.97], hspace=0.48, wspace=0.36)
    ax_btc = fig.add_subplot(gs[0, 0])
    ax_err = fig.add_subplot(gs[0, 1])
    ax_dil = fig.add_subplot(gs[1, :])

    title(fig, "The archive predicts arrival better than dilution")
    subtitle(fig, "Held-out tests reveal which transport memories are preserved and which still need help.")
    draw_breakthrough_story(ax_btc, summary)
    draw_error_fingerprint(ax_err, mean_errors_from_outer(outer), include_mixture=True)
    draw_dilution_story(ax_dil, summary)
    add_corner_note(
        ax_dil,
        "Scientific read",
        "The generators retain enough path memory to match arrival trends,\nbut all sit below the reference dilution trajectory.",
        loc="upper left",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.08)
    return fig


def draw_breakthrough_story(ax: plt.Axes, summary: dict[str, dict]) -> None:
    planes = np.asarray(summary["reference"]["planes"], dtype=float)
    for sampler in ["reference", "pooled_validation_mixture", "gaussian_bayes", "hybrid"]:
        stats = summary[sampler]["breakthrough"]
        q50 = stats["q50"]
        q10 = stats["q10"]
        q90 = stats["q90"]
        color = COLORS[sampler]
        ax.plot(planes, q50, marker="o", lw=2.4 if sampler == "reference" else 1.75, color=color, label=LABELS[sampler])
        ax.fill_between(planes, q10, q90, color=color, alpha=0.07 if sampler != "reference" else 0.10, linewidth=0)
    ax.set_title("Arrival memory is largely retained", fontweight="bold", color="#102a43")
    ax.set_xlabel("downstream control plane")
    ax.set_ylabel("first-passage time")
    ax.set_xticks(planes)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left", handlelength=2.0)


def draw_dilution_story(ax: plt.Axes, summary: dict[str, dict]) -> None:
    for sampler in ["reference", "pooled_validation_mixture", "gaussian_bayes", "hybrid", "knn_conditional", "pair_rerank"]:
        times = np.asarray(summary[sampler]["times"], dtype=float)
        mean = summary[sampler]["dilution_mean"]
        color = COLORS[sampler]
        ax.plot(times, mean, marker="o", markersize=4, lw=2.6 if sampler == "reference" else 1.75, color=color, label=LABELS[sampler])
    ref = summary["reference"]["dilution_mean"]
    mix = summary["pooled_validation_mixture"]["dilution_mean"]
    times = np.asarray(summary["reference"]["times"], dtype=float)
    ax.fill_between(times, mix, ref, where=ref >= mix, color="#94a3b8", alpha=0.20, linewidth=0)
    ax.annotate(
        "persistent dilution gap",
        xy=(times[-2], 0.5 * (ref[-2] + mix[-2])),
        xytext=(times[-2] - 110, ref[-2] + 120),
        arrowprops=dict(arrowstyle="-|>", color="#64748b", lw=1.2),
        fontsize=9,
        color="#334155",
    )
    ax.set_title("Spatial organization memory is still under-preserved", fontweight="bold", color="#102a43")
    ax.set_xlabel("time step")
    ax.set_ylabel("dilution index")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False, ncol=3, loc="lower right", handlelength=2.0)


def draw_error_fingerprint(ax: plt.Axes, errors: dict[str, dict[str, float]], include_mixture: bool) -> None:
    samplers = SAMPLERS if include_mixture else FIXED_SAMPLERS
    outcomes = [name for name, _ in OUTCOME_ERRORS]
    matrix = np.zeros((len(samplers), len(outcomes)))
    for j, (_, key) in enumerate(OUTCOME_ERRORS):
        best = min(errors[s][key] for s in samplers)
        for i, sampler in enumerate(samplers):
            matrix[i, j] = errors[sampler][key] / best if best > 0 else 1.0

    cmap = LinearSegmentedColormap.from_list("story_error", ["#e0f2fe", "#fef3c7", "#fecaca"])
    im = ax.imshow(np.clip(matrix, 1, 2.5), cmap=cmap, vmin=1, vmax=2.5, aspect="auto")
    ax.set_title("Each memory fails differently", fontweight="bold", color="#102a43")
    ax.set_xticks(range(len(outcomes)), [label.title() for label in outcomes], rotation=35, ha="right")
    ax.set_yticks(range(len(samplers)), [SHORT_LABELS[s] for s in samplers])
    for i in range(len(samplers)):
        for j in range(len(outcomes)):
            ax.text(j, i, f"{matrix[i, j]:.1f}x", ha="center", va="center", fontsize=8.2, color="#111827")
    ax.tick_params(length=0)
    ax.set_xlabel("held-out outcome")
    ax.set_ylabel("transition memory")
    cbar = plt.colorbar(im, ax=ax, fraction=0.050, pad=0.03)
    cbar.set_label("relative error\n(1 = best)", rotation=270, labelpad=22)


def outcome_memory_selection(objective: dict) -> plt.Figure:
    regimes = ["balanced", "btc_heavy", "dilution_heavy", "pair_heavy", "reaction_heavy"]
    fig = plt.figure(figsize=(7.45, 7.10), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.50)
    ax_weights = fig.add_subplot(gs[0, 0])
    ax_pareto = fig.add_subplot(gs[1, 0])
    title(fig, "Different outcomes choose different memories")
    subtitle(fig, "Changing the validation target changes the selected mixture and the best fixed memory.")
    draw_objective_memory_bars(ax_weights, objective, regimes)
    draw_pareto_story(ax_pareto, objective)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.85, bottom=0.09)
    return fig


def draw_objective_memory_bars(ax: plt.Axes, objective: dict, regimes: list[str]) -> None:
    y = np.arange(len(regimes))[::-1]
    left = np.zeros(len(regimes))
    height = 0.56
    for component in COMPONENTS:
        values = np.array([objective["summary"][r]["selected_weights"]["pooled_validation_mixture"]["mean"][component] for r in regimes])
        ax.barh(y, values, left=left, height=height, color=COLORS[component], edgecolor="white", linewidth=0.8, label=LABELS[component])
        for yi, lft, val in zip(y, left, values):
            if val >= 0.12:
                ax.text(lft + val / 2, yi, f"{val:.2f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        left += values

    for yi, regime in zip(y, regimes):
        best = best_sampler(objective["summary"][regime])
        ax.text(1.03, yi, f"best memory: {SHORT_LABELS.get(best, best)}", va="center", fontsize=8.6, color=COLORS.get(best, "#334155"))
    ax.set_yticks(y, [LABELS[r] for r in regimes])
    ax.set_xlim(0, 1.33)
    ax.set_xlabel("validation-selected mixture weight")
    ax.set_title("Validation target selects the memory mixture", fontweight="bold", color="#102a43")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.46, -0.16), columnspacing=1.0)


def draw_pareto_story(ax: plt.Axes, objective: dict) -> None:
    records = objective["outer_results"]
    means = {s: {key: [] for _, key in OUTCOME_ERRORS} for s in SAMPLERS}
    for outer in records:
        for sampler in SAMPLERS:
            errs = outer["regime_results"]["balanced"]["test"][sampler]["errors"]
            for _, key in OUTCOME_ERRORS:
                means[sampler][key].append(float(errs[key]))
    for sampler in SAMPLERS:
        x = np.mean(means[sampler]["btc_score"])
        y = np.mean(means[sampler]["pair_quantile_mae"])
        size = 340 * np.mean(means[sampler]["dilution_log_mae"]) / 0.25
        ax.scatter(x, y, s=size, color=COLORS[sampler], alpha=0.82, edgecolor="white", linewidth=1.4)
        if sampler == "knn_conditional":
            ax.text(x - 1.0, y + 0.020, SHORT_LABELS[sampler], fontsize=8.2, color="#334155", ha="right")
        elif sampler == "pair_rerank":
            ax.text(x + 1.2, y + 0.020, SHORT_LABELS[sampler], fontsize=8.2, color="#334155", ha="left")
        else:
            ax.text(x + 1.2, y + 0.020, SHORT_LABELS[sampler], fontsize=8.2, color="#334155")
    ax.set_xlabel("arrival error")
    ax.set_ylabel("pair-separation error")
    ax.set_title("No single memory preserves every outcome", fontweight="bold", color="#102a43")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.text(0.02, 0.06, "circle size = dilution error", transform=ax.transAxes, fontsize=8.4, color="#64748b")


def diffusion_memory_shift(peclet: dict[str, dict]) -> plt.Figure:
    fig = plt.figure(figsize=(7.45, 5.95), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 0.82], hspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    title(fig, "Diffusion erodes velocity memory and elevates learned context")
    subtitle(fig, "As diffusivity increases, validation shifts weight away from strict velocity continuity.")
    draw_peclet_weight_shift(ax, peclet)
    draw_peclet_rank_shift(ax2, peclet)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.83, bottom=0.10)
    return fig


def draw_peclet_weight_shift(ax: plt.Axes, peclet: dict[str, dict]) -> None:
    labels = list(peclet)
    x = np.arange(len(labels))
    width = 0.56
    bottom = np.zeros(len(labels))
    for component in COMPONENTS:
        vals = np.array([peclet[label]["summary"]["mean_selected_weights"][component] for label in labels])
        ax.bar(x, vals, bottom=bottom, width=width, color=COLORS[component], edgecolor="white", linewidth=0.8, label=LABELS[component])
        for xi, btm, val in zip(x, bottom, vals):
            if val >= 0.12:
                ax.text(xi, btm + val / 2, f"{val:.2f}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        bottom += vals
    ax.set_xticks(x, ["high Pe\nlow diffusion", "baseline", "low Pe\nmore diffusion"])
    ax.set_ylabel("selected mixture weight")
    ax.set_ylim(0, 1.05)
    ax.set_title("Memory selected by validation", fontweight="bold", color="#102a43")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), columnspacing=1.0)
    ax.add_patch(FancyArrowPatch((0.13, 0.93), (0.87, 0.93), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=13, lw=1.2, color="#94a3b8"))
    ax.text(0.14, 0.875, "velocity memory matters", transform=ax.transAxes, fontsize=8.4, color="#475569")
    ax.text(0.67, 0.875, "context matters more", transform=ax.transAxes, fontsize=8.4, color="#475569")


def draw_peclet_rank_shift(ax: plt.Axes, peclet: dict[str, dict]) -> None:
    labels = list(peclet)
    x = np.arange(len(labels))
    for sampler in ["gaussian_bayes", "hybrid", "knn_conditional", "pooled_validation_mixture"]:
        ranks = [peclet[label]["summary"]["samplers"][sampler]["mean_rank"] for label in labels]
        ax.plot(x, ranks, marker="o", lw=2.0, color=COLORS[sampler], label=LABELS[sampler])
        for xi, rank in zip(x, ranks):
            ax.text(xi, rank - 0.16, f"{rank:.1f}", ha="center", va="bottom", fontsize=7.8, color=COLORS[sampler])
    ax.invert_yaxis()
    ax.set_xticks(x, ["high Pe", "baseline", "low Pe"])
    ax.set_ylabel("mean held-out rank\n(lower is better)")
    ax.set_title("Rank shift confirms the regime dependence", fontweight="bold", color="#102a43")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(ncol=2, frameon=False, loc="lower left")


def flow_fidelity_memory_test(core2_graph: dict, core2_openfoam: dict, openfoam_objective: dict, flow: dict) -> plt.Figure:
    fig = plt.figure(figsize=(7.45, 7.2), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.55], width_ratios=[1.02, 0.98], hspace=0.48, wspace=0.36)
    ax_corr = fig.add_subplot(gs[0, 0])
    ax_rank = fig.add_subplot(gs[0, 1])
    ax_obj = fig.add_subplot(gs[1, :])
    title(fig, "OpenFOAM strengthens the memory used by the physics kernel")
    subtitle(fig, "Higher flow fidelity increases velocity autocorrelation and shifts the strongest fixed memory toward velocity memory.")
    draw_autocorrelation(ax_corr, flow)
    draw_flow_rank_bars(ax_rank, core2_graph, core2_openfoam)
    draw_openfoam_objective_exception(ax_obj, openfoam_objective)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.84, bottom=0.12)
    return fig


def draw_autocorrelation(ax: plt.Axes, flow: dict) -> None:
    lags = np.array(flow["velocity_autocorrelation"]["lags"])
    graph = np.array(flow["velocity_autocorrelation"]["graph"])
    openfoam = np.array(flow["velocity_autocorrelation"]["openfoam"])
    ax.plot(lags, graph, color=COLORS["graph"], lw=2.0, label="graph-flow")
    ax.plot(lags, openfoam, color=COLORS["openfoam"], lw=2.0, label="OpenFOAM")
    ax.fill_between(lags, graph, openfoam, where=openfoam >= graph, color=COLORS["openfoam"], alpha=0.12, linewidth=0)
    ax.set_xlim(0, 80)
    ax.set_xlabel("lag")
    ax.set_ylabel("axial velocity autocorrelation")
    ax.set_title("Cause: stronger velocity memory", fontweight="bold", color="#102a43")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False)


def draw_flow_rank_bars(ax: plt.Axes, core2_graph: dict, core2_openfoam: dict) -> None:
    samplers = ["gaussian_bayes", "hybrid", "pooled_validation_mixture", "knn_conditional"]
    labels = ["Core2 graph", "Core2 OpenFOAM"]
    data = [core2_graph, core2_openfoam]
    x = np.arange(len(labels))
    width = 0.17
    offsets = np.linspace(-0.26, 0.26, len(samplers))
    for off, sampler in zip(offsets, samplers):
        ranks = [d["summary"]["samplers"][sampler]["mean_rank"] for d in data]
        ax.bar(x + off, ranks, width=width, color=COLORS[sampler], label=SHORT_LABELS[sampler])
        for xi, rank in zip(x + off, ranks):
            ax.text(xi, rank + 0.10, f"{rank:.2g}", ha="center", va="bottom", fontsize=7.6, color="#334155")
    ax.invert_yaxis()
    ax.set_xticks(x, labels)
    ax.set_ylabel("mean rank\n(lower is better)")
    ax.set_title("Effect: velocity memory wins", fontweight="bold", color="#102a43")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.legend(frameon=False, ncol=2, loc="lower left")


def draw_openfoam_objective_exception(ax: plt.Axes, openfoam_objective: dict) -> None:
    regimes = ["balanced", "btc_heavy", "dilution_heavy", "pair_heavy", "reaction_heavy", "no_reaction"]
    x = np.arange(len(regimes))
    best = [best_sampler(openfoam_objective["summary"][r]) for r in regimes]
    colors = [COLORS[b] for b in best]
    ax.bar(x, [0.52] * len(regimes), bottom=[0.24] * len(regimes), color=colors, edgecolor="white", linewidth=1.0)
    for xi, b in zip(x, best):
        ax.text(xi, 0.50, SHORT_LABELS[b], ha="center", va="center", fontsize=8.4, color="white", fontweight="bold")
    ax.set_xticks(x, [LABELS[r] for r in regimes], rotation=20, ha="right")
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.set_title("But pair-heavy validation still selects learned context", fontweight="bold", color="#102a43")
    ax.text(
        0.01,
        -0.38,
        "Each tile shows the best fixed memory under an OpenFOAM objective-weight regime.",
        transform=ax.transAxes,
        fontsize=8.4,
        color="#64748b",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)


def memory_map(evidence: list[dict], openfoam_objective: dict) -> plt.Figure:
    fig = plt.figure(figsize=(7.45, 5.0), facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    title(fig, "A validation map of Lagrangian memory", y=0.965)
    subtitle(fig, "The useful memory changes with diffusion, flow fidelity, and the transport outcome.", y=0.925)

    x0, x1 = 0.045, 0.705
    y = 0.58
    xs = np.linspace(x0, x1, len(evidence))
    for idx, (x, row) in enumerate(zip(xs, evidence)):
        draw_condition_memory_card(ax, x, y, row)
        if idx < len(evidence) - 1:
            ax.add_patch(FancyArrowPatch((x + 0.062, y), (xs[idx + 1] - 0.062, y), arrowstyle="-|>", mutation_scale=10, lw=1.1, color="#cbd5e1"))

    ax.text(0.05, 0.30, "low diffusion / high Pe", fontsize=8.8, color="#64748b")
    ax.text(0.42, 0.30, "more diffusion", fontsize=8.8, color="#64748b", ha="center")
    ax.text(0.79, 0.30, "higher flow fidelity", fontsize=8.8, color="#64748b", ha="right")
    ax.add_patch(FancyArrowPatch((0.06, 0.335), (0.78, 0.335), arrowstyle="-|>", mutation_scale=12, lw=1.1, color="#94a3b8"))

    rounded_box(ax, 0.80, 0.50, 0.17, 0.30, "#fff7ed", "#fb923c", 1.0)
    ax.text(0.825, 0.75, "Takeaway", fontsize=10, fontweight="bold", color="#9a3412")
    ax.text(
        0.825,
        0.70,
        "No single memory\nwins because no\nsingle memory\ncontrols all\ntransport outcomes.",
        fontsize=8.4,
        color="#7c2d12",
        va="top",
        linespacing=1.15,
    )

    ax.text(0.815, 0.37, "Memory colors", fontsize=9.2, fontweight="bold", color="#102a43")
    for idx, comp in enumerate(COMPONENTS):
        yy = 0.32 - idx * 0.055
        ax.add_patch(Rectangle((0.815, yy - 0.016), 0.030, 0.026, color=COLORS[comp], transform=ax.transAxes))
        ax.text(0.853, yy - 0.003, LABELS[comp], fontsize=7.9, color="#334155", va="center")
    return fig


def draw_condition_memory_card(ax: plt.Axes, x: float, y: float, row: dict) -> None:
    w, h = 0.115, 0.25
    rounded_box(ax, x - w / 2, y - h / 2, w, h, "white", "#cbd5e1", 1.0)
    ax.text(x, y + 0.095, row["short"].replace(" ", "\n", 1), ha="center", va="center", fontsize=8.2, color="#102a43", fontweight="bold")
    best = row["best_sampler"]
    rounded_box(ax, x - 0.048, y + 0.015, 0.096, 0.044, COLORS[best], COLORS[best], 0)
    ax.text(x, y + 0.037, SHORT_LABELS.get(best, best), ha="center", va="center", fontsize=7.0, color="white", fontweight="bold")
    bx, by, bw, bh = x - 0.050, y - 0.045, 0.100, 0.030
    left = bx
    for comp in COMPONENTS:
        val = row["selected_weights"][comp]
        ax.add_patch(Rectangle((left, by), bw * val, bh, color=COLORS[comp], transform=ax.transAxes))
        left += bw * val
    ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, edgecolor="#334155", lw=0.6, transform=ax.transAxes))
    ax.text(x, y - 0.088, f"rank {row['best_mean_rank']:.2g}", ha="center", fontsize=7.4, color="#475569")


def mean_errors_from_outer(outer: dict) -> dict[str, dict[str, float]]:
    out = {sampler: {key: [] for _, key in OUTCOME_ERRORS} for sampler in SAMPLERS}
    for split in outer["outer_results"]:
        for sampler in SAMPLERS:
            errs = split["test"][sampler]["errors"]
            for _, key in OUTCOME_ERRORS:
                out[sampler][key].append(float(errs[key]))
    return {sampler: {key: float(np.mean(vals)) for key, vals in metrics.items()} for sampler, metrics in out.items()}


def best_sampler(summary: dict) -> str:
    return min(summary["samplers"], key=lambda name: summary["samplers"][name]["mean_objective"])


def title(fig: plt.Figure, text: str, y: float = 0.985) -> None:
    fig.text(0.02, y, text, ha="left", va="top", fontsize=14.2, fontweight="bold", color="#102a43")


def subtitle(fig: plt.Figure, text: str, y: float = 0.945) -> None:
    fig.text(0.02, y, text, ha="left", va="top", fontsize=9.3, color="#475569")


def rounded_box(ax: plt.Axes, x: float, y: float, w: float, h: float, fc: str, ec: str, lw: float) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            transform=ax.transAxes,
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
        )
    )


def add_corner_note(ax: plt.Axes, header: str, body: str, loc: str = "upper right") -> None:
    x, y = (0.03, 0.94) if loc == "upper left" else (0.62, 0.94)
    ax.text(
        x,
        y,
        header,
        transform=ax.transAxes,
        fontsize=8.8,
        fontweight="bold",
        color="#102a43",
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cbd5e1", alpha=0.92),
    )
    ax.text(x, y - 0.075, body, transform=ax.transAxes, fontsize=8.0, color="#334155", va="top")


if __name__ == "__main__":
    main()

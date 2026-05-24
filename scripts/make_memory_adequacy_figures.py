from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import make_story_figures as story  # noqa: E402
from scripts import render_3d_porous_medium as pore3d  # noqa: E402


FIGURES = ROOT / "figures"
PAPER_FIGURES = ROOT / "paper" / "figures"

OUTER = ROOT / "outputs" / "bentheimer_6um_downsample3_outer_split_mixture_benchmark.json"
OBJECTIVE = ROOT / "outputs" / "bentheimer_6um_downsample3_objective_weight_sensitivity.json"
HIGH_PE = ROOT / "outputs" / "bentheimer_6um_downsample3_D0003_outer_split_mixture_benchmark.json"
LOW_PE = ROOT / "outputs" / "bentheimer_6um_downsample3_D003_outer_split_mixture_benchmark.json"
CORE2_GRAPH = ROOT / "outputs" / "bentheimer_core2_subvol1_6um_downsample3_D001_outer_split_mixture_benchmark.json"
CORE2_OPENFOAM = ROOT / "outputs" / "bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_dt010_n5000_stride400_outer_split_mixture_benchmark.json"
OPENFOAM_OBJECTIVE = ROOT / "outputs" / "bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_dt010_n5000_stride400_objective_weight_sensitivity.json"
FLOW_PHYSICS = ROOT / "outputs" / "core2_graph_vs_openfoam_physics_comparison.json"
EVIDENCE = ROOT / "outputs" / "master_evidence_table.json"
BREAKTHROUGH_ONLY = ROOT / "outputs" / "bentheimer_6um_downsample3_breakthrough_only_failure.json"

COMPONENTS = ["gaussian_bayes", "knn_conditional", "hybrid", "pair_rerank"]
SAMPLERS = ["pooled_validation_mixture", "gaussian_bayes", "hybrid", "knn_conditional", "pair_rerank"]
FIXED = ["gaussian_bayes", "knn_conditional", "hybrid", "pair_rerank"]

LABELS = {
    "pooled_validation_mixture": "memory mixture",
    "bootstrap_mean_mixture": "mean mixture",
    "gaussian_bayes": "velocity memory",
    "knn_conditional": "archive proximity",
    "hybrid": "learned context",
    "pair_rerank": "pair organization",
    "balanced": "balanced",
    "breakthrough_only": "breakthrough only",
    "btc_heavy": "arrival",
    "dilution_heavy": "dilution",
    "pair_heavy": "pairs",
    "reaction_heavy": "encounters",
    "reaction_light": "reaction light",
    "no_reaction": "no reaction",
}

SHORT = {
    "pooled_validation_mixture": "mixture",
    "gaussian_bayes": "velocity",
    "knn_conditional": "archive",
    "hybrid": "learned",
    "pair_rerank": "pairs",
}

COLORS = {
    "reference": "#111827",
    "pooled_validation_mixture": "#2563eb",
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
    "ink": "#102a43",
    "muted": "#64748b",
    "grid": "#e5e7eb",
}

OBSERVABLES = [
    ("arrival", "arrival", "Did mass arrive?"),
    ("dilution", "dilution", "Did dilution survive?"),
    ("pairs", "pairs", "Did pairs separate?"),
    ("encounters", "encounters", "Did encounters occur?"),
]


def main() -> None:
    setup_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)

    outer = load_json(OUTER)
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
    breakthrough_only = load_json(BREAKTHROUGH_ONLY)

    volume = pore3d.load_downsampled_volume()
    pore = pore3d.inlet_outlet_pore_mask(volume)
    trajectories = pore3d.load_trajectories()
    behavior_summary = story.load_behavior_summary(outer)

    save_all("figure1_memory_landscape", figure1_memory_landscape(pore, trajectories), svg=False)
    save_all("figure2_memory_assay", figure2_memory_assay(trajectories, evidence))
    save_all("figure3_arrival_can_lie", figure3_arrival_can_lie(behavior_summary, breakthrough_only))
    save_all("figure4_observable_memory_selection", figure4_observable_memory_selection(objective))
    save_all("figure5_diffusion_memory_erosion", figure5_diffusion_memory_erosion(peclet))
    save_all(
        "figure6_flow_fidelity_velocity_memory",
        figure6_flow_fidelity_velocity_memory(core2_graph, core2_openfoam, openfoam_objective, flow_physics),
    )
    save_all("figure7_memory_adequacy_atlas", figure7_memory_adequacy_atlas(evidence))


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.1,
            "axes.titlesize": 10.4,
            "axes.labelsize": 8.9,
            "legend.fontsize": 8.0,
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


def save_all(name: str, fig: plt.Figure, svg: bool = True) -> None:
    for directory in (FIGURES, PAPER_FIGURES):
        fig.savefig(directory / f"{name}.png", bbox_inches="tight", pad_inches=0.035)
        fig.savefig(directory / f"{name}.pdf", bbox_inches="tight", pad_inches=0.035)
        if svg:
            fig.savefig(directory / f"{name}.svg", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(f"wrote {name}")


def title(fig: plt.Figure, text: str, y: float = 0.982) -> None:
    fig.text(0.02, y, text, ha="left", va="top", fontsize=14.1, fontweight="bold", color=COLORS["ink"])


def subtitle(fig: plt.Figure, text: str, y: float = 0.942) -> None:
    fig.text(0.02, y, text, ha="left", va="top", fontsize=9.15, color=COLORS["muted"])


def panel(ax: plt.Axes, letter: str, label: str | None = None, x: float = 0.0, y: float = 1.01) -> None:
    text_fn = ax.text2D if hasattr(ax, "text2D") else ax.text
    text_fn(x, y, letter, transform=ax.transAxes, fontsize=10.2, fontweight="bold", color=COLORS["ink"], va="bottom")
    if label:
        text_fn(x + 0.045, y, label, transform=ax.transAxes, fontsize=9.0, color=COLORS["ink"], va="bottom")


def rounded(ax: plt.Axes, x: float, y: float, w: float, h: float, fc: str, ec: str, lw: float = 1.0, alpha: float = 1.0) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.020",
            transform=ax.transAxes,
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            alpha=alpha,
        )
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#94a3b8", scale: int = 12) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=scale,
            lw=1.15,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def tint(color: str, amount: float) -> tuple[float, float, float]:
    rgb = np.array(to_rgb(color))
    return tuple(1 - amount * (1 - rgb))


def figure1_memory_landscape(pore: np.ndarray, trajectories: list[np.ndarray]) -> plt.Figure:
    fig = plt.figure(figsize=(7.45, 7.25), facecolor="white")
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.55, 1.0, 1.0],
        height_ratios=[1.0, 0.92],
        wspace=0.18,
        hspace=0.32,
    )
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_frag = fig.add_subplot(gs[0, 1:])
    ax_strip = fig.add_subplot(gs[1, 1:])

    title(fig, "The full memory exists before the model forgets")
    subtitle(fig, "A resolved pore-scale simulation contains geometry, velocity history, path order, and particle neighborhoods.")
    fig.subplots_adjust(left=0.035, right=0.985, top=0.86, bottom=0.055)

    draw_3d_memory_object(ax3d, pore, trajectories)
    draw_history_fragments(ax_frag, trajectories)
    draw_memory_strip(ax_strip)
    panel(ax3d, "a", "memory-rich pore world", x=0.02, y=0.98)
    panel(ax_frag, "b", "trajectory histories", x=0.00, y=1.02)
    panel(ax_strip, "c", "what reduced models discard", x=0.00, y=1.02)
    return fig


def draw_3d_memory_object(ax: plt.Axes, pore: np.ndarray, trajectories: list[np.ndarray]) -> None:
    step = 2
    pore_plot = pore[::step, ::step, ::step]
    rgba = np.zeros(pore_plot.shape + (4,), dtype=float)
    rgba[..., 0] = 0.10
    rgba[..., 1] = 0.58
    rgba[..., 2] = 0.74
    rgba[..., 3] = 0.31
    ax.voxels(pore_plot, facecolors=rgba, edgecolors=(1, 1, 1, 0.02), linewidth=0.035, shade=True)

    selected = pore3d.select_trajectories(trajectories, n=11)
    colors = plt.cm.plasma(np.linspace(0.08, 0.90, len(selected)))
    for traj, color in zip(selected, colors):
        pts = traj[:, :3] / step
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] < 2:
            continue
        stride = max(1, pts.shape[0] // 210)
        pts = pts[::stride]
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = Line3DCollection(segs, colors=[color], linewidths=2.05, alpha=0.94)
        ax.add_collection3d(lc)
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], s=30, color=color, edgecolor="white", linewidth=0.7)
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], s=38, color="white", edgecolor=color, linewidth=1.0)

    n = pore_plot.shape[0]
    ax.view_init(elev=23, azim=-51, roll=0)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_zlim(0, n)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.text2D(
        0.05,
        0.05,
        "actual 3D Bentheimer\nconnected pore volume",
        transform=ax.transAxes,
        fontsize=8.2,
        color=COLORS["ink"],
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#d9e2ec", alpha=0.88),
    )


def draw_history_fragments(ax: plt.Axes, trajectories: list[np.ndarray]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    chosen = pore3d.select_trajectories(trajectories, n=5)
    labels = [
        ("fast channel", "velocity history", COLORS["gaussian_bayes"]),
        ("slow/tortuous path", "path order", COLORS["knn_conditional"]),
        ("pair divergence", "neighborhood loss", COLORS["hybrid"]),
        ("near encounter", "reaction opportunity", COLORS["pair_rerank"]),
    ]
    y_positions = [0.80, 0.57, 0.34, 0.11]
    for idx, ((headline, body, color), y0) in enumerate(zip(labels, y_positions)):
        rounded(ax, 0.03, y0 - 0.055, 0.94, 0.15, "white", "#d9e2ec", 0.8)
        if idx < 2:
            traj = chosen[idx][:, :3]
            start = min(25 + idx * 100, max(0, len(traj) - 90))
            seg = traj[start : start + 70, [0, 1]]
            draw_normalized_path(ax, seg, x0=0.08, y0=y0 - 0.010, w=0.42, h=0.075, color=color, lw=2.0)
        elif idx == 2:
            draw_pair_paths(ax, 0.08, y0 - 0.010, 0.42, 0.075, color)
        else:
            draw_encounter_paths(ax, 0.08, y0 - 0.010, 0.42, 0.075, color)
        ax.text(0.57, y0 + 0.028, headline, fontsize=8.8, color=color, fontweight="bold", va="center")
        ax.text(0.57, y0 - 0.032, body, fontsize=7.9, color="#334155", va="center")


def draw_normalized_path(ax: plt.Axes, seg: np.ndarray, x0: float, y0: float, w: float, h: float, color: str, lw: float = 2.0) -> None:
    seg = np.asarray(seg, dtype=float)
    seg = seg[np.isfinite(seg).all(axis=1)]
    if seg.shape[0] < 2:
        return
    seg = seg - seg.min(axis=0)
    span = np.ptp(seg, axis=0)
    span[span == 0] = 1
    seg = seg / span
    pts = np.column_stack([x0 + w * seg[:, 0], y0 + h * (seg[:, 1] - 0.5)])
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, solid_capstyle="round", transform=ax.transAxes)
    ax.scatter(pts[0, 0], pts[0, 1], s=22, color=color, edgecolor="white", linewidth=0.6, transform=ax.transAxes, zorder=4)
    ax.scatter(pts[-1, 0], pts[-1, 1], s=25, color="white", edgecolor=color, linewidth=0.8, transform=ax.transAxes, zorder=4)


def draw_pair_paths(ax: plt.Axes, x0: float, y0: float, w: float, h: float, color: str) -> None:
    t = np.linspace(0, 1, 90)
    p1 = np.column_stack([x0 + w * t, y0 + h * (0.12 * np.sin(8 * t))])
    p2 = np.column_stack([x0 + w * t, y0 + h * (0.08 * np.cos(6 * t) + 1.1 * (t - 0.2))])
    ax.plot(p1[:, 0], p1[:, 1], color=color, lw=1.9, transform=ax.transAxes)
    ax.plot(p2[:, 0], p2[:, 1], color=color, lw=1.9, alpha=0.70, transform=ax.transAxes)
    ax.add_patch(Circle((x0 + 0.76 * w, y0 + 0.54 * h), 0.020, transform=ax.transAxes, facecolor="white", edgecolor=color, lw=1.0))


def draw_encounter_paths(ax: plt.Axes, x0: float, y0: float, w: float, h: float, color: str) -> None:
    t = np.linspace(0, 1, 90)
    p1 = np.column_stack([x0 + w * t, y0 + h * (0.18 * np.sin(2.4 * np.pi * t))])
    p2 = np.column_stack([x0 + w * t, y0 + h * (0.10 * np.cos(2.0 * np.pi * t) - 0.22 * (t - 0.45))])
    ax.plot(p1[:, 0], p1[:, 1], color=color, lw=1.9, transform=ax.transAxes)
    ax.plot(p2[:, 0], p2[:, 1], color=color, lw=1.9, alpha=0.72, transform=ax.transAxes)
    ax.add_patch(Circle((x0 + 0.47 * w, y0 + 0.01), 0.027, transform=ax.transAxes, facecolor=tint(color, 0.10), edgecolor=color, lw=1.1))


def draw_memory_strip(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    items = [
        ("pore\ngeometry", "#94a3b8", 0.95),
        ("velocity\nhistory", COLORS["gaussian_bayes"], 0.86),
        ("path\norder", COLORS["hybrid"], 0.70),
        ("particle\nneighbors", COLORS["pair_rerank"], 0.54),
        ("observables", COLORS["arrival"], 1.0),
    ]
    xs = np.array([0.08, 0.285, 0.49, 0.695, 0.89])
    for idx, ((label, color, alpha), x) in enumerate(zip(items, xs)):
        rounded(ax, x - 0.062, 0.49, 0.124, 0.20, tint(color, 0.16), color, 1.1, alpha=alpha)
        ax.text(x, 0.59, label, ha="center", va="center", fontsize=7.25, color=COLORS["ink"], fontweight="bold")
        if idx < len(items) - 1:
            arrow(ax, (x + 0.069, 0.59), (xs[idx + 1] - 0.069, 0.59))
    rounded(ax, 0.06, 0.15, 0.61, 0.18, "#f8fafc", "#cbd5e1", 0.8)
    ax.text(0.09, 0.24, "reduced model", fontsize=8.7, color=COLORS["ink"], fontweight="bold", va="center")
    ax.text(0.39, 0.24, "cannot keep\nthe whole history", fontsize=7.7, color="#334155", va="center")
    rounded(ax, 0.74, 0.15, 0.20, 0.18, "#fff7ed", "#fb923c", 1.0)
    ax.text(0.84, 0.24, "which forgetting\nis safe?", ha="center", va="center", fontsize=8.2, color="#9a3412", fontweight="bold")


def figure2_memory_assay(trajectories: list[np.ndarray], evidence: list[dict]) -> plt.Figure:
    fig = plt.figure(figsize=(7.45, 6.45), facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    title(fig, "Reduced transport is selective forgetting")
    subtitle(fig, "Validation is the assay that reveals which memory is adequate for the prediction being made.")
    fig.subplots_adjust(left=0.035, right=0.985, top=0.86, bottom=0.055)

    draw_motif_source(ax, trajectories)
    draw_forgetting_gate(ax)
    draw_memory_hypotheses(ax)
    draw_validation_stations(ax)
    draw_mini_atlas(ax, evidence)
    return fig


def draw_motif_source(ax: plt.Axes, trajectories: list[np.ndarray]) -> None:
    rounded(ax, 0.035, 0.58, 0.245, 0.30, "white", "#cbd5e1", 1.0)
    ax.text(0.058, 0.835, "resolved paths", fontsize=9.2, color=COLORS["ink"], fontweight="bold", transform=ax.transAxes)
    colors = [COLORS["arrival"], COLORS["knn_conditional"], COLORS["hybrid"], COLORS["pair_rerank"]]
    chosen = pore3d.select_trajectories(trajectories, n=4)
    for idx, traj in enumerate(chosen):
        start = min(30 + idx * 80, max(0, len(traj) - 75))
        seg = traj[start : start + 68, [0, 1]]
        draw_normalized_path(ax, seg, 0.065, 0.775 - 0.060 * idx, 0.15, 0.045, colors[idx], lw=1.8)
        ax.plot([0.225, 0.255], [0.775 - 0.060 * idx, 0.775 - 0.060 * idx], color=colors[idx], lw=3, transform=ax.transAxes, solid_capstyle="round")
    ax.text(
        0.058,
        0.615,
        "motifs keep observed\nlocal path shapes",
        fontsize=7.6,
        color="#334155",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="none", alpha=0.78),
    )
    arrow(ax, (0.286, 0.73), (0.365, 0.73))


def draw_forgetting_gate(ax: plt.Axes) -> None:
    rounded(ax, 0.375, 0.55, 0.135, 0.36, "#f8fafc", "#94a3b8", 1.0)
    ax.text(0.442, 0.865, "forgetting\ngate", ha="center", va="center", fontsize=9.0, color=COLORS["ink"], fontweight="bold", transform=ax.transAxes)
    gate_items = ["geometry", "velocity", "order", "neighbors"]
    alphas = [0.35, 0.85, 0.70, 0.58]
    for idx, (item, alpha) in enumerate(zip(gate_items, alphas)):
        y = 0.785 - 0.052 * idx
        ax.add_patch(Rectangle((0.407, y - 0.015), 0.070, 0.026, transform=ax.transAxes, facecolor="#cbd5e1", edgecolor="none", alpha=alpha))
        ax.text(0.442, y - 0.001, item, ha="center", va="center", fontsize=7.2, color="#334155", transform=ax.transAxes)
    ax.text(0.442, 0.585, "rules retain\ndifferent pieces", ha="center", va="center", fontsize=7.0, color="#475569", transform=ax.transAxes)
    arrow(ax, (0.516, 0.73), (0.585, 0.73))


def draw_memory_hypotheses(ax: plt.Axes) -> None:
    x0 = 0.595
    ys = [0.820, 0.745, 0.670, 0.595]
    cards = [
        ("velocity", "diffusion-scaled\ncontinuity", "gaussian_bayes"),
        ("archive", "empirical\nproximity", "knn_conditional"),
        ("learned", "transition\ncontext", "hybrid"),
        ("pairs", "short-horizon\norganization", "pair_rerank"),
    ]
    ax.text(x0, 0.875, "competing memory hypotheses", fontsize=9.2, color=COLORS["ink"], fontweight="bold", transform=ax.transAxes)
    for (head, body, key), y in zip(cards, ys):
        color = COLORS[key]
        rounded(ax, x0, y - 0.035, 0.215, 0.055, tint(color, 0.12), color, 1.1)
        ax.text(x0 + 0.022, y - 0.007, head, va="center", fontsize=8.2, color=color, fontweight="bold", transform=ax.transAxes)
        ax.text(x0 + 0.102, y - 0.007, body, va="center", fontsize=6.9, color="#334155", transform=ax.transAxes)
    arrow(ax, (0.817, 0.73), (0.885, 0.73))


def draw_validation_stations(ax: plt.Axes) -> None:
    ax.text(0.055, 0.455, "validation listening stations", fontsize=9.5, color=COLORS["ink"], fontweight="bold", transform=ax.transAxes)
    xs = [0.075, 0.305, 0.535, 0.765]
    for x, (key, short, question) in zip(xs, OBSERVABLES):
        color = COLORS[key]
        rounded(ax, x - 0.050, 0.210, 0.165, 0.190, "white", color, 1.1)
        draw_observable_icon(ax, key, x + 0.032, 0.338, color)
        ax.text(x + 0.032, 0.285, short, ha="center", fontsize=8.4, color=color, fontweight="bold", transform=ax.transAxes)
        ax.text(x + 0.032, 0.242, question, ha="center", fontsize=6.8, color="#334155", transform=ax.transAxes)
    ax.text(
        0.055,
        0.135,
        "The validation objective is not a final scorecard; it is the experiment that asks which memory the observable needs.",
        fontsize=8.25,
        color="#475569",
        transform=ax.transAxes,
    )


def draw_observable_icon(ax: plt.Axes, key: str, x: float, y: float, color: str) -> None:
    if key == "arrival":
        t = np.linspace(0, 1, 40)
        ax.plot(x - 0.038 + 0.076 * t, y - 0.020 + 0.040 * (1 - np.exp(-4 * t)), color=color, lw=1.8, transform=ax.transAxes)
        ax.add_patch(FancyArrowPatch((x + 0.025, y + 0.015), (x + 0.040, y + 0.020), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=8, lw=1.3, color=color))
    elif key == "dilution":
        for r, alpha in [(0.014, 0.85), (0.030, 0.45), (0.047, 0.22)]:
            ax.add_patch(Circle((x, y), r, transform=ax.transAxes, facecolor="none", edgecolor=color, lw=1.2, alpha=alpha))
    elif key == "pairs":
        ax.plot([x - 0.038, x + 0.037], [y - 0.020, y + 0.025], color=color, lw=1.6, transform=ax.transAxes)
        ax.plot([x - 0.038, x + 0.037], [y + 0.020, y - 0.025], color=color, lw=1.6, transform=ax.transAxes)
        ax.scatter([x - 0.038, x + 0.037], [y - 0.020, y + 0.025], s=12, color=color, transform=ax.transAxes)
    else:
        ax.plot([x - 0.035, x + 0.030], [y - 0.025, y + 0.020], color=color, lw=1.6, transform=ax.transAxes)
        ax.plot([x - 0.035, x + 0.030], [y + 0.025, y - 0.020], color=color, lw=1.6, transform=ax.transAxes)
        ax.add_patch(Circle((x, y), 0.018, transform=ax.transAxes, facecolor=tint(color, 0.14), edgecolor=color, lw=1.0))


def draw_mini_atlas(ax: plt.Axes, evidence: list[dict]) -> None:
    rounded(ax, 0.885, 0.565, 0.095, 0.315, "#fff7ed", "#fb923c", 1.0)
    ax.text(0.932, 0.850, "memory\nmap", ha="center", va="center", fontsize=8.6, color="#9a3412", fontweight="bold", transform=ax.transAxes)
    y0 = 0.792
    for idx, row in enumerate(evidence):
        y = y0 - 0.039 * idx
        key = row["best_sampler"]
        ax.add_patch(Rectangle((0.905, y - 0.011), 0.022, 0.020, transform=ax.transAxes, facecolor=COLORS[key], edgecolor="none"))
        ax.text(0.934, y - 0.001, row["short"].replace("Core2 ", ""), fontsize=6.4, color="#7c2d12", va="center", transform=ax.transAxes)
    ax.text(0.932, 0.595, "safe\nforgetting", ha="center", va="center", fontsize=7.1, color="#9a3412", transform=ax.transAxes)


def figure3_arrival_can_lie(summary: dict[str, dict], breakthrough_only: dict) -> plt.Figure:
    fig = plt.figure(figsize=(7.45, 5.25), facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 0.90],
        width_ratios=[1.02, 0.98],
        hspace=0.48,
        wspace=0.33,
    )
    ax_btc = fig.add_subplot(gs[0, 0])
    ax_dil = fig.add_subplot(gs[0, 1])
    ax_fail = fig.add_subplot(gs[1, :])
    title(fig, "A model can arrive correctly and still forget the plume")
    subtitle(fig, "Breakthrough-only validation selects arrival memory, but other observables expose unsafe forgetting.")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.82, bottom=0.13)

    draw_breakthrough(ax_btc, summary)
    draw_dilution(ax_dil, summary)
    draw_breakthrough_only_counterfactual(ax_fail, breakthrough_only)
    panel(ax_btc, "a", "arrival preserved", x=0.00, y=1.04)
    panel(ax_dil, "b", "dilution forgotten", x=0.00, y=1.04)
    panel(ax_fail, "c", None, x=0.00, y=1.04)
    return fig


def draw_breakthrough(ax: plt.Axes, summary: dict[str, dict]) -> None:
    planes = np.asarray(summary["reference"]["planes"], dtype=float)
    for sampler in ["reference", "pooled_validation_mixture", "gaussian_bayes", "hybrid"]:
        stats = summary[sampler]["breakthrough"]
        color = COLORS.get(sampler, "#334155")
        lw = 2.6 if sampler == "reference" else 1.85
        label = "reference" if sampler == "reference" else SHORT[sampler] if sampler in SHORT else LABELS[sampler]
        ax.plot(planes, stats["q50"], marker="o", lw=lw, color=color, label=label)
        ax.fill_between(planes, stats["q10"], stats["q90"], color=color, alpha=0.08, linewidth=0)
    ax.set_xlabel("downstream control plane")
    ax.set_ylabel("first-passage time")
    ax.set_xticks(planes)
    ax.grid(True, color=COLORS["grid"], linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.03, 0.06, "arrival memory transfers", transform=ax.transAxes, fontsize=8.2, color=COLORS["arrival"], fontweight="bold")


def draw_dilution(ax: plt.Axes, summary: dict[str, dict]) -> None:
    for sampler in ["reference", "pooled_validation_mixture", "gaussian_bayes", "hybrid", "knn_conditional", "pair_rerank"]:
        times = np.asarray(summary[sampler]["times"], dtype=float)
        mean = np.asarray(summary[sampler]["dilution_mean"], dtype=float)
        color = COLORS.get(sampler, "#334155")
        lw = 2.6 if sampler == "reference" else 1.75
        label = "reference" if sampler == "reference" else SHORT.get(sampler, sampler)
        ax.plot(times, mean, marker="o", markersize=3.6, lw=lw, color=color, label=label)
    ref = np.asarray(summary["reference"]["dilution_mean"], dtype=float)
    mix = np.asarray(summary["pooled_validation_mixture"]["dilution_mean"], dtype=float)
    times = np.asarray(summary["reference"]["times"], dtype=float)
    ax.fill_between(times, mix, ref, where=ref >= mix, color="#94a3b8", alpha=0.24, linewidth=0)
    ax.annotate(
        "spatial organization gap",
        xy=(times[-2], 0.5 * (ref[-2] + mix[-2])),
        xytext=(times[1], ref[-1] * 0.93),
        arrowprops=dict(arrowstyle="-|>", color="#64748b", lw=1.15),
        fontsize=8.2,
        color="#334155",
    )
    ax.set_xlabel("time step")
    ax.set_ylabel("dilution index")
    ax.grid(True, color=COLORS["grid"], linewidth=0.8)
    ax.legend(frameon=False, ncol=2, loc="lower right", handlelength=1.8, columnspacing=0.8)


def draw_breakthrough_only_counterfactual(ax: plt.Axes, result: dict) -> None:
    mean_errors = mean_test_errors(result)
    selected = min(
        result["summary"]["breakthrough_only"]["samplers"],
        key=lambda name: result["summary"]["breakthrough_only"]["samplers"][name]["mean_objective"],
    )
    selected_key = (
        f"breakthrough_only::{selected}"
        if selected in {"pooled_validation_mixture", "bootstrap_mean_mixture"}
        else selected
    )
    metrics = [
        ("arrival", "btc_score", "BTC"),
        ("dilution", "dilution_log_mae", "dilution"),
        ("pairs", "pair_quantile_mae", "pairs"),
        ("encounters", "reaction_abs_error", "encounters"),
    ]
    ratios = []
    colors = []
    for outcome, key, _label in metrics:
        finite = [errs[key] for errs in mean_errors.values() if np.isfinite(errs[key])]
        best = min(finite) if finite else np.nan
        value = mean_errors[selected_key][key]
        ratios.append(value / best if np.isfinite(best) and best > 0 else np.nan)
        colors.append(COLORS[outcome])

    x = np.arange(len(metrics))
    bars = ax.bar(x, ratios, color=colors, alpha=0.90, width=0.58)
    ax.axhline(1.0, color="#334155", lw=0.9, ls="--")
    for bar, ratio in zip(bars, ratios):
        if np.isfinite(ratio):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                ratio + 0.025,
                f"{ratio:.2f}x",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color="#334155",
                fontweight="bold",
            )
    ax.set_xticks(x, [label for _outcome, _key, label in metrics])
    ax.set_ylim(0.85, max(1.28, np.nanmax(ratios) + 0.12))
    ax.set_ylabel("error relative to\nobservable-specific best")
    ax.set_title("Breakthrough-only selection favors velocity memory, not universal adequacy", fontweight="bold", color=COLORS["ink"])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.text(
        0.015,
        0.90,
        f"selected by BTC-only validation: {SHORT.get(selected, selected)}",
        transform=ax.transAxes,
        fontsize=8.2,
        color=COLORS["gaussian_bayes"],
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.985,
        0.10,
        "1.0 = best memory for that observable",
        transform=ax.transAxes,
        fontsize=7.6,
        color="#64748b",
        ha="right",
        va="bottom",
    )


def mean_test_errors(result: dict) -> dict[str, dict[str, float]]:
    names = [
        "gaussian_bayes",
        "hybrid",
        "knn_conditional",
        "pair_rerank",
        "balanced::pooled_validation_mixture",
        "breakthrough_only::bootstrap_mean_mixture",
        "breakthrough_only::pooled_validation_mixture",
    ]
    keys = ["btc_score", "dilution_log_mae", "pair_quantile_mae", "reaction_abs_error"]
    out = {}
    for name in names:
        out[name] = {
            key: float(np.mean([outer["test_errors"][name][key] for outer in result["outer_results"]]))
            for key in keys
        }
    return out


def figure4_observable_memory_selection(objective: dict) -> plt.Figure:
    regimes = ["balanced", "btc_heavy", "dilution_heavy", "pair_heavy", "reaction_heavy", "no_reaction"]
    fig = plt.figure(figsize=(7.45, 5.45), facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    title(fig, "Different observables protect different pasts")
    subtitle(fig, "Changing the validation target changes the memory mixture and the best fixed transition rule.")
    fig.subplots_adjust(left=0.035, right=0.985, top=0.84, bottom=0.06)
    draw_objective_matrix(ax, objective, regimes)
    return fig


def draw_objective_matrix(ax: plt.Axes, objective: dict, regimes: list[str]) -> None:
    x0 = 0.225
    col_w = 0.115
    y_top = 0.75
    row_h = 0.105
    ax.text(0.055, 0.795, "validation target", fontsize=8.8, color=COLORS["ink"], fontweight="bold", transform=ax.transAxes)
    for j, comp in enumerate(COMPONENTS):
        x = x0 + j * col_w
        ax.text(x + 0.045, 0.795, SHORT[comp], ha="center", fontsize=8.1, color=COLORS[comp], fontweight="bold", transform=ax.transAxes)
    ax.text(0.725, 0.795, "best fixed", fontsize=8.8, color=COLORS["ink"], fontweight="bold", transform=ax.transAxes)
    ax.text(0.875, 0.795, "interpretation", fontsize=8.8, color=COLORS["ink"], fontweight="bold", transform=ax.transAxes)

    interpretation = {
        "balanced": "no single past is enough",
        "btc_heavy": "arrival tolerates velocity memory",
        "dilution_heavy": "context helps plume structure",
        "pair_heavy": "pairs ask for different history",
        "reaction_heavy": "encounters still favor velocity",
        "no_reaction": "velocity remains economical",
    }
    icon_map = {
        "balanced": "arrival",
        "btc_heavy": "arrival",
        "dilution_heavy": "dilution",
        "pair_heavy": "pairs",
        "reaction_heavy": "encounters",
        "no_reaction": "arrival",
    }
    for i, regime in enumerate(regimes):
        y = y_top - i * row_h
        if i % 2 == 0:
            ax.add_patch(Rectangle((0.035, y - 0.045), 0.93, 0.082, transform=ax.transAxes, facecolor="#f8fafc", edgecolor="none"))
        color = COLORS[icon_map[regime]]
        draw_observable_icon(ax, icon_map[regime], 0.073, y - 0.005, color)
        ax.text(0.112, y - 0.005, LABELS[regime], fontsize=8.6, color=COLORS["ink"], va="center", transform=ax.transAxes)
        weights = objective["summary"][regime]["selected_weights"]["pooled_validation_mixture"]["mean"]
        for j, comp in enumerate(COMPONENTS):
            val = float(weights[comp])
            x = x0 + j * col_w
            ax.add_patch(
                Rectangle(
                    (x + 0.010, y - 0.030),
                    0.070,
                    0.055,
                    transform=ax.transAxes,
                    facecolor=tint(COLORS[comp], 0.18 + 0.75 * val),
                    edgecolor=COLORS[comp],
                    linewidth=0.65,
                )
            )
            ax.text(x + 0.045, y - 0.004, f"{val:.2f}", ha="center", va="center", fontsize=7.2, color=COLORS["ink"], transform=ax.transAxes)
        best = story.best_sampler(objective["summary"][regime])
        rounded(ax, 0.720, y - 0.032, 0.100, 0.055, COLORS[best], COLORS[best], 0.0)
        ax.text(0.770, y - 0.005, SHORT[best], ha="center", va="center", fontsize=7.7, color="white", fontweight="bold", transform=ax.transAxes)
        ax.text(0.855, y - 0.005, interpretation[regime], fontsize=7.45, color="#334155", va="center", transform=ax.transAxes)
    draw_memory_legend(ax, 0.055, 0.070)


def figure5_diffusion_memory_erosion(peclet: dict[str, dict]) -> plt.Figure:
    fig = plt.figure(figsize=(7.45, 5.45), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[0.78, 1.0], hspace=0.42, wspace=0.24)
    ax_paths = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_ribbon = fig.add_subplot(gs[1, :])
    title(fig, "Diffusion erodes velocity memory")
    subtitle(fig, "As random motion grows, strict velocity continuity carries less of the future and learned context carries more.")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.80, bottom=0.12)
    draw_diffusion_paths(ax_paths)
    draw_memory_ribbon(ax_ribbon, peclet)
    return fig


def draw_diffusion_paths(axes: list[plt.Axes]) -> None:
    rng = np.random.default_rng(12)
    labels = [("high Pe", 0.03), ("baseline", 0.10), ("low Pe", 0.22)]
    for ax, (label, sigma) in zip(axes, labels):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        t = np.linspace(0, 1, 120)
        walk = np.cumsum(rng.normal(0, sigma, size=t.size))
        walk = (walk - walk.min()) / max(np.ptp(walk), 1e-9)
        y = 0.50 + (walk - 0.5) * 0.55
        x = 0.08 + 0.84 * t
        ax.plot(x, y, color=COLORS["gaussian_bayes"], lw=2.15, transform=ax.transAxes)
        if sigma > 0.08:
            for alpha, width in [(0.12, 7.0), (0.18, 4.8)]:
                ax.plot(x, y, color=COLORS["hybrid"], lw=width, alpha=alpha, transform=ax.transAxes)
        ax.add_patch(FancyArrowPatch((0.12, 0.18), (0.88, 0.18), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=11, lw=1.0, color="#94a3b8"))
        ax.text(0.50, 0.88, label, ha="center", fontsize=9.0, color=COLORS["ink"], fontweight="bold", transform=ax.transAxes)
        ax.text(0.50, 0.05, "advective history" if sigma < 0.08 else "diffusive blurring", ha="center", fontsize=7.6, color="#475569", transform=ax.transAxes)


def draw_memory_ribbon(ax: plt.Axes, peclet: dict[str, dict]) -> None:
    labels = list(peclet)
    x = np.arange(len(labels), dtype=float)
    weights = np.vstack([[peclet[label]["summary"]["mean_selected_weights"][comp] for label in labels] for comp in COMPONENTS])
    ax.stackplot(x, weights, colors=[COLORS[c] for c in COMPONENTS], alpha=0.92, edgecolor="white", linewidth=0.8)
    centers = np.cumsum(weights, axis=0) - weights / 2
    for j, label in enumerate(labels):
        x_text = x[j]
        if j == 0:
            x_text += 0.075
        elif j == len(labels) - 1:
            x_text -= 0.075
        for i, comp in enumerate(COMPONENTS):
            if weights[i, j] >= 0.10:
                ax.text(x_text, centers[i, j], f"{weights[i, j]:.2f}", ha="center", va="center", fontsize=8.0, color="white", fontweight="bold")
    ax.set_xticks(x, ["high Pe\nlow diffusion", "baseline", "low Pe\nmore diffusion"])
    ax.set_xlim(-0.18, 2.18)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("selected mixture weight")
    ax.set_title("Validation-selected memory mixture", fontweight="bold", color=COLORS["ink"])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.legend([LABELS[c] for c in COMPONENTS], ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.50, -0.20), columnspacing=1.0)
    ax.annotate(
        "learned context takes over",
        xy=(1.88, weights[0, 2] + weights[1, 2] + 0.5 * weights[2, 2]),
        xytext=(1.20, 0.82),
        arrowprops=dict(arrowstyle="-|>", lw=1.1, color=COLORS["hybrid"]),
        fontsize=8.2,
        color=COLORS["hybrid"],
    )


def figure6_flow_fidelity_velocity_memory(core2_graph: dict, core2_openfoam: dict, openfoam_objective: dict, flow: dict) -> plt.Figure:
    fig = plt.figure(figsize=(7.45, 6.35), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.78, 1.05], hspace=0.42, wspace=0.34)
    ax_graph = fig.add_subplot(gs[0, 0])
    ax_openfoam = fig.add_subplot(gs[0, 1])
    ax_corr = fig.add_subplot(gs[1, 0])
    ax_outcome = fig.add_subplot(gs[1, 1])
    title(fig, "Higher-fidelity flow gives velocity memory something real to hold")
    subtitle(fig, "OpenFOAM increases velocity autocorrelation, but tight held-out validation still decides which memory is adequate.")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.82, bottom=0.12)
    draw_flow_memory_cartoon(ax_graph, "graph-flow", coherent=False)
    draw_flow_memory_cartoon(ax_openfoam, "OpenFOAM", coherent=True)
    draw_autocorr(ax_corr, flow)
    draw_flow_outcome(ax_outcome, core2_graph, core2_openfoam, openfoam_objective)
    return fig


def draw_flow_memory_cartoon(ax: plt.Axes, label: str, coherent: bool) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rounded(ax, 0.03, 0.08, 0.94, 0.78, "white", "#cbd5e1", 1.0)
    rng = np.random.default_rng(4 if coherent else 9)
    for i in range(11):
        t = np.linspace(0, 1, 80)
        x = 0.10 + 0.80 * t
        phase = i * 0.45
        base = 0.18 + 0.060 * i
        noise = rng.normal(0, 0.015 if coherent else 0.045, t.size).cumsum()
        noise = noise - np.mean(noise)
        y = base + (0.035 if coherent else 0.075) * np.sin(2 * np.pi * (t + phase)) + 0.18 * noise / max(np.max(np.abs(noise)), 1e-6)
        color = COLORS["openfoam"] if coherent else COLORS["graph"]
        ax.plot(x, y, color=color, lw=1.25, alpha=0.62, transform=ax.transAxes)
    ax.text(0.08, 0.80, label, fontsize=9.0, color=COLORS["openfoam"] if coherent else COLORS["graph"], fontweight="bold", transform=ax.transAxes)
    ax.text(0.08, 0.15, "persistent velocity history" if coherent else "weaker velocity persistence", fontsize=7.8, color="#334155", transform=ax.transAxes)


def draw_autocorr(ax: plt.Axes, flow: dict) -> None:
    lags = np.asarray(flow["velocity_autocorrelation"]["lags"], dtype=float)
    graph = np.asarray(flow["velocity_autocorrelation"]["graph"], dtype=float)
    openfoam = np.asarray(flow["velocity_autocorrelation"]["openfoam"], dtype=float)
    ax.plot(lags, graph, color=COLORS["graph"], lw=2.1, label="graph-flow")
    ax.plot(lags, openfoam, color=COLORS["openfoam"], lw=2.1, label="OpenFOAM")
    ax.fill_between(lags, graph, openfoam, where=openfoam >= graph, color=COLORS["openfoam"], alpha=0.12, linewidth=0)
    ax.set_xlim(0, 80)
    ax.set_xlabel("lag")
    ax.set_ylabel("axial velocity autocorrelation")
    ax.set_title("cause: stronger velocity memory", fontweight="bold", color=COLORS["ink"])
    ax.grid(True, color=COLORS["grid"], linewidth=0.8)
    ax.legend(frameon=False)


def draw_flow_outcome(ax: plt.Axes, core2_graph: dict, core2_openfoam: dict, openfoam_objective: dict) -> None:
    samplers = ["gaussian_bayes", "hybrid", "pooled_validation_mixture", "knn_conditional"]
    x = np.arange(2)
    width = 0.17
    offsets = np.linspace(-0.26, 0.26, len(samplers))
    data = [core2_graph, core2_openfoam]
    for off, sampler in zip(offsets, samplers):
        ranks = [d["summary"]["samplers"][sampler]["mean_rank"] for d in data]
        ax.bar(x + off, ranks, width=width, color=COLORS[sampler], label=SHORT[sampler])
        for xi, rank in zip(x + off, ranks):
            ax.text(xi, rank + 0.08, f"{rank:.2g}", ha="center", va="bottom", fontsize=7.3, color="#334155")
    ax.invert_yaxis()
    ax.set_xticks(x, ["Core2 graph", "Core2 OpenFOAM"])
    ax.set_ylabel("mean rank\n(lower is better)")
    ax.set_title("effect: memory adequacy changes", fontweight="bold", color=COLORS["ink"])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.legend(frameon=False, ncol=2, loc="lower left", columnspacing=0.9)
    best_pair = story.best_sampler(openfoam_objective["summary"]["pair_heavy"])
    ax.text(
        0.02,
        -0.28,
        f"pair-heavy OpenFOAM winner: {SHORT[best_pair]}",
        transform=ax.transAxes,
        fontsize=8.2,
        color=COLORS[best_pair],
        fontweight="bold",
    )


def figure7_memory_adequacy_atlas(evidence: list[dict]) -> plt.Figure:
    fig = plt.figure(figsize=(7.45, 6.45), facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    title(fig, "A memory-adequacy atlas for non-Fickian transport")
    subtitle(fig, "The product is not a winning sampler; it is a map of which forgetting is safe under each condition.")
    fig.subplots_adjust(left=0.035, right=0.985, top=0.83, bottom=0.06)
    draw_atlas(ax, evidence)
    return fig


def draw_atlas(ax: plt.Axes, evidence: list[dict]) -> None:
    y_top = 0.705
    row_h = 0.086
    x_label = 0.055
    x_weights = 0.305
    x_best = 0.695
    x_read = 0.815
    ax.text(x_label, 0.780, "condition", fontsize=8.8, color=COLORS["ink"], fontweight="bold", transform=ax.transAxes)
    ax.text(x_weights, 0.780, "validation-selected memory mixture", fontsize=8.8, color=COLORS["ink"], fontweight="bold", transform=ax.transAxes)
    ax.text(x_best, 0.780, "winner", fontsize=8.8, color=COLORS["ink"], fontweight="bold", transform=ax.transAxes)
    ax.text(x_read, 0.780, "scientific read", fontsize=8.8, color=COLORS["ink"], fontweight="bold", transform=ax.transAxes)
    phrases = {
        "core1_high_pe": "velocity persists",
        "core1_baseline": "mixed memory is safest",
        "core1_low_pe": "diffusion erodes velocity",
        "core2_graph": "geometry restores velocity",
        "core2_openfoam": "archive protects coarse flow",
        "core2_openfoam_12um": "mixture wins at 12 um",
        "core2_openfoam_6um": "strict flow keeps mixtures alive",
    }
    for i, row in enumerate(evidence):
        y = y_top - i * row_h
        if i % 2 == 0:
            ax.add_patch(Rectangle((0.035, y - 0.040), 0.93, 0.074, transform=ax.transAxes, facecolor="#f8fafc", edgecolor="none"))
        ax.text(x_label, y - 0.004, row["short"], fontsize=8.5, color=COLORS["ink"], va="center", transform=ax.transAxes)
        draw_weight_strip(ax, x_weights, y - 0.021, 0.325, 0.036, row["selected_weights"])
        best = row["best_sampler"]
        rounded(ax, x_best, y - 0.027, 0.095, 0.045, COLORS[best], COLORS[best], 0.0)
        ax.text(x_best + 0.047, y - 0.006, SHORT[best], ha="center", va="center", fontsize=7.5, color="white", fontweight="bold", transform=ax.transAxes)
        ax.text(x_read, y - 0.005, phrases.get(row["id"], row["interpretation"]), fontsize=7.9, color="#334155", va="center", transform=ax.transAxes)
    draw_memory_legend(ax, 0.055, 0.038)
    rounded(ax, 0.600, 0.018, 0.350, 0.112, "#fff7ed", "#fb923c", 1.0)
    ax.text(0.622, 0.102, "main result", fontsize=8.4, color="#9a3412", fontweight="bold", transform=ax.transAxes)
    ax.text(
        0.622,
        0.045,
        "No memory wins universally because different observables\nand flow regimes protect different parts of the past.",
        fontsize=7.15,
        color="#7c2d12",
        transform=ax.transAxes,
    )


def draw_weight_strip(ax: plt.Axes, x: float, y: float, w: float, h: float, weights: dict[str, float]) -> None:
    left = x
    for comp in COMPONENTS:
        val = float(weights[comp])
        if val > 0:
            ax.add_patch(Rectangle((left, y), w * val, h, transform=ax.transAxes, facecolor=COLORS[comp], edgecolor="white", linewidth=0.7))
        left += w * val
    ax.add_patch(Rectangle((x, y), w, h, transform=ax.transAxes, fill=False, edgecolor="#334155", linewidth=0.65))


def draw_memory_legend(ax: plt.Axes, x: float, y: float) -> None:
    for i, comp in enumerate(COMPONENTS):
        xx = x + i * 0.135
        ax.add_patch(Rectangle((xx, y), 0.026, 0.022, transform=ax.transAxes, facecolor=COLORS[comp], edgecolor="none"))
        ax.text(xx + 0.032, y + 0.011, SHORT[comp], va="center", fontsize=7.45, color="#334155", transform=ax.transAxes)


if __name__ == "__main__":
    main()

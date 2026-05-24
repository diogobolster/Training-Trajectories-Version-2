from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
PAPER_FIGURES = ROOT / "paper" / "figures"

COMPONENTS = [
    ("gaussian_bayes", "Physics\ncontinuity", "#243b53"),
    ("knn_conditional", "Nearest\nneighbors", "#d97706"),
    ("hybrid", "Learned\ncontext", "#2f855a"),
    ("pair_rerank", "Pair\norganization", "#b83280"),
]

SAMPLER_LABELS = {
    "pooled_validation_mixture": "validation mixture",
    "bootstrap_mean_mixture": "mean mixture",
    "gaussian_bayes": "physics kernel",
    "knn_conditional": "nearest neighbors",
    "hybrid": "learned context",
    "pair_rerank": "pair rerank",
}


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    setup_style()

    volume = load_downsampled_volume()
    trajectories = load_trajectories(ROOT / "data" / "processed" / "bentheimer_6um_downsample3_trajectories.npz")
    baseline_weights = load_json(ROOT / "outputs" / "bentheimer_6um_downsample3_outer_split_mixture_benchmark.json")[
        "summary"
    ]["mean_selected_weights"]
    evidence = load_json(ROOT / "outputs" / "master_evidence_table.json")["conditions"]

    save_all("figure1_training_trajectory_workflow", workflow_figure(volume, trajectories, baseline_weights))
    save_all("figure2_validation_mechanism_map", mechanism_map_figure(evidence))


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9,
            "figure.dpi": 170,
            "savefig.dpi": 300,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_trajectories(path: Path) -> list[np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return [np.asarray(item, dtype=float) for item in data["trajectories"]]


def load_downsampled_volume() -> np.ndarray:
    raw_path = ROOT / "data" / "raw" / "Core1_Subvol1_6micron_225cube_16bit_LE.raw"
    arr = np.fromfile(raw_path, dtype="<u2")
    vol = arr.reshape((225, 225, 225))
    vol = vol.reshape(75, 3, 75, 3, 75, 3).mean(axis=(1, 3, 5))
    return vol


def otsu_threshold(values: np.ndarray, bins: int = 512) -> float:
    counts, edges = np.histogram(values.ravel(), bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight_b = np.cumsum(counts)
    weight_f = counts.sum() - weight_b
    sum_b = np.cumsum(counts * centers)
    total = sum_b[-1]
    mean_b = sum_b / np.maximum(weight_b, 1)
    mean_f = (total - sum_b) / np.maximum(weight_f, 1)
    variance = weight_b * weight_f * (mean_b - mean_f) ** 2
    return float(centers[np.argmax(variance)])


def save_all(name: str, fig: plt.Figure) -> None:
    for directory in (FIGURES, PAPER_FIGURES):
        fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
        fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(directory / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def workflow_figure(
    volume: np.ndarray,
    trajectories: list[np.ndarray],
    weights: dict[str, float],
) -> plt.Figure:
    fig = plt.figure(figsize=(7.45, 6.35), facecolor="white")
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.08, 1.0],
        height_ratios=[1.02, 1.0],
        wspace=0.23,
        hspace=0.40,
    )
    ax0 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        "Training trajectories make resolved paths reusable",
        x=0.02,
        y=0.965,
        ha="left",
        fontsize=13.0,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.905,
        "Keep observed path motifs, recombine them with candidate transition rules, then pick rules by held-out transport scores.",
        ha="left",
        fontsize=9.5,
        color="#415161",
    )
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.07, top=0.815)

    draw_pore_panel(ax0, volume, trajectories)
    draw_segment_panel(ax1, trajectories)
    draw_sampler_panel(ax2)
    draw_validation_panel(ax3, weights)

    add_panel_label(ax0, "a", "Resolved pore-scale paths")
    add_panel_label(ax1, "b", "Archive path motifs")
    add_panel_label(ax2, "c", "Choose next segment")
    add_panel_label(ax3, "d", "Validate what matters")
    return fig


def draw_pore_panel(ax: plt.Axes, volume: np.ndarray, trajectories: list[np.ndarray]) -> None:
    step = 3
    pore = volume < otsu_threshold(volume)
    coords = np.argwhere(pore[::step, ::step, ::step])
    rng = np.random.default_rng(17)
    if coords.shape[0] > 2600:
        coords = coords[rng.choice(coords.shape[0], 2600, replace=False)]
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        s=2.4,
        c="#1f78a8",
        alpha=0.12,
        depthshade=True,
        linewidths=0,
    )

    chosen = select_trajectories(trajectories, n=9)
    colors = plt.cm.turbo(np.linspace(0.08, 0.88, len(chosen)))
    for idx, traj in enumerate(chosen):
        pts = traj[:, :3] / step
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] < 2:
            continue
        stride = max(1, pts.shape[0] // 140)
        pts = pts[::stride]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=colors[idx], lw=1.45, alpha=0.92)
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], s=18, color=colors[idx], edgecolor="white", linewidth=0.4)

    n = volume.shape[0] / step
    ax.view_init(elev=21, azim=-49, roll=0)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_zlim(0, n)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    ax.text2D(
        0.05,
        0.05,
        "Actual 3D Bentheimer\npore network",
        transform=ax.transAxes,
        color="#102a43",
        fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.82, edgecolor="#d9e2ec"),
    )


def select_trajectories(trajectories: list[np.ndarray], n: int) -> list[np.ndarray]:
    spans = np.array([t[-1, 0] - t[0, 0] for t in trajectories])
    order = np.argsort(spans)[::-1]
    picks = []
    for idx in order:
        traj = trajectories[int(idx)]
        if len(picks) == 0 or min(np.linalg.norm(traj[0, 1:3] - p[0, 1:3]) for p in picks) > 4:
            picks.append(traj)
        if len(picks) >= n:
            break
    return picks


def draw_segment_panel(ax: plt.Axes, trajectories: list[np.ndarray]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    palette = ["#0e7490", "#d97706", "#2f855a", "#b83280"]
    base = trajectories[5]
    starts = [30, 115, 235, 350]
    for row, start in enumerate(starts):
        seg = base[start : start + 52, 1:3]
        seg = seg - seg[0]
        span = np.ptp(seg, axis=0)
        span[span == 0] = 1
        seg = seg / span
        seg[:, 0] = 0.13 + seg[:, 0] * 0.58
        y0 = 0.82 - row * 0.2
        seg[:, 1] = y0 - 0.055 + seg[:, 1] * 0.11
        ax.plot(seg[:, 0], seg[:, 1], color=palette[row], lw=2.0, solid_capstyle="round")
        ax.scatter(seg[0, 0], seg[0, 1], s=24, color=palette[row], edgecolor="white", zorder=4)
        ax.scatter(seg[-1, 0], seg[-1, 1], s=24, color="white", edgecolor=palette[row], zorder=4)
        ax.text(0.76, y0, f"segment {row + 1}", va="center", fontsize=8.6, color="#334e68")


def draw_sampler_panel(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.08, 0.72, "Physics", "velocity continuity + diffusion", "#243b53"),
        (0.08, 0.49, "Archive", "nearest observed start", "#d97706"),
        (0.08, 0.26, "Learned", "transition context", "#2f855a"),
        (0.08, 0.03, "Pairs", "short-horizon organization", "#b83280"),
    ]
    for x, y, title, body, color in boxes:
        rounded_box(ax, x, y, 0.84, 0.17, fc="#f8fafc", ec=color, lw=1.4)
        ax.text(x + 0.04, y + 0.112, title, fontsize=8.9, color=color, fontweight="bold", va="center")
        ax.text(
            x + 0.04,
            y + 0.055,
            body,
            fontsize=7.45,
            color="#334e68",
            va="center",
        )
    ax.add_patch(FancyArrowPatch((0.50, 0.94), (0.50, 0.89), arrowstyle="-|>", mutation_scale=10, lw=1.2, color="#627d98"))
    ax.text(0.50, 0.965, "candidate rules", ha="center", fontsize=9, color="#334e68")


def draw_validation_panel(ax: plt.Axes, weights: dict[str, float]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    metrics = [
        ("Breakthrough", "#2563eb"),
        ("Dilution", "#2f855a"),
        ("Pairs", "#d97706"),
        ("Encounters", "#b83280"),
    ]
    for idx, (label, color) in enumerate(metrics):
        y = 0.79 - idx * 0.17
        ax.plot([0.10, 0.34], [y, y], color=color, lw=4, solid_capstyle="round")
        ax.scatter([0.11, 0.24, 0.33], [y + 0.04, y - 0.025, y + 0.02], s=12, color=color, zorder=3)
        ax.text(0.42, y - 0.01, label, va="center", fontsize=8.7, color="#102a43")
    ax.text(0.10, 0.92, "Held-out score", fontsize=10.2, color="#102a43", fontweight="bold")
    ax.text(0.10, 0.06, "selected mixture", fontsize=8.5, color="#627d98")
    x0, y0, w, h = 0.10, 0.15, 0.78, 0.075
    left = x0
    for key, label, color in COMPONENTS:
        value = weights.get(key, 0.0)
        ax.add_patch(Rectangle((left, y0), w * value, h, facecolor=color, edgecolor="white", lw=0.8))
        if value > 0.08:
            ax.text(left + w * value / 2, y0 + h / 2, f"{value:.2f}", ha="center", va="center", color="white", fontsize=7.2)
        left += w * value
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="none", edgecolor="#334e68", lw=0.8))


def draw_story_strip(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        ("resolved\ntrajectories", "#0e7490"),
        ("archive\nsegments", "#d97706"),
        ("sample next\npath piece", "#2f855a"),
        ("score against\nheld-out metrics", "#b83280"),
        ("select rule\nor mixture", "#243b53"),
    ]
    xs = np.linspace(0.08, 0.92, len(steps))
    for idx, ((label, color), x) in enumerate(zip(steps, xs)):
        ax.scatter(x, 0.58, s=720, color=color, alpha=0.14, edgecolor=color, linewidth=1.2)
        ax.scatter(x, 0.58, s=110, color=color, edgecolor="white", linewidth=1.2, zorder=4)
        ax.text(x, 0.24, label, ha="center", va="center", fontsize=8.8, color="#102a43", linespacing=1.1)
        if idx < len(steps) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 0.055, 0.58),
                    (xs[idx + 1] - 0.055, 0.58),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    lw=1.4,
                    color="#9fb3c8",
                )
            )
    rounded_box(ax, 0.025, 0.03, 0.95, 0.90, fc="#f8fafc", ec="#d9e2ec", lw=1.0, zorder=-1)


def mechanism_map_figure(evidence: list[dict]) -> plt.Figure:
    fig = plt.figure(figsize=(9.6, 5.25), facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.02,
        0.965,
        "The selected transport mechanism changes with the physics and the question",
        fontsize=15,
        fontweight="bold",
        color="#102a43",
    )
    ax.text(
        0.02,
        0.925,
        "Each column is a held-out condition. The badge is the best fixed sampler; the bar is the validation-selected mixture.",
        fontsize=10.4,
        color="#415161",
    )

    left, right = 0.035, 0.82
    top, bottom = 0.84, 0.17
    col_w = (right - left) / len(evidence)
    for idx, row in enumerate(evidence):
        x = left + idx * col_w
        card = (x + 0.008, bottom, col_w - 0.016, top - bottom)
        draw_condition_card(ax, card, row, idx)

    legend_x = 0.845
    ax.text(legend_x, 0.82, "Mixture components", fontsize=10.4, fontweight="bold", color="#102a43")
    for idx, (_, label, color) in enumerate(COMPONENTS):
        y = 0.77 - idx * 0.066
        ax.add_patch(Rectangle((legend_x, y - 0.018), 0.035, 0.032, facecolor=color, edgecolor="none"))
        ax.text(legend_x + 0.048, y, label.replace("\n", " "), va="center", fontsize=8.7, color="#334e68")

    rounded_box(ax, 0.835, 0.18, 0.14, 0.26, fc="#fffaf0", ec="#f0b429", lw=1.1)
    ax.text(0.852, 0.395, "Takeaway", fontsize=10.2, fontweight="bold", color="#8a5a00")
    ax.text(
        0.852,
        0.352,
        "No sampler wins\neverywhere.\nValidation reveals\nwhen physics or\nlearned context\ncarries the signal.",
        fontsize=8.6,
        color="#513c06",
        linespacing=1.12,
        va="top",
    )

    ax.text(0.04, 0.08, "low diffusion / high Peclet", fontsize=8.8, color="#627d98")
    ax.text(0.40, 0.08, "more diffusion", fontsize=8.8, color="#627d98", ha="center")
    ax.text(0.79, 0.08, "higher fidelity flow", fontsize=8.8, color="#627d98", ha="right")
    ax.add_patch(FancyArrowPatch((0.05, 0.105), (0.80, 0.105), arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#9fb3c8"))
    return fig


def draw_condition_card(ax: plt.Axes, card: tuple[float, float, float, float], row: dict, idx: int) -> None:
    x, y, w, h = card
    rounded_box(ax, x, y, w, h, fc="white", ec="#d9e2ec", lw=1.1)
    ax.text(x + 0.018, y + h - 0.055, row["short"], fontsize=8.6, fontweight="bold", color="#102a43")
    badge_color = sampler_color(row["best_sampler"])
    rounded_box(ax, x + 0.018, y + h - 0.145, w - 0.036, 0.070, fc=badge_color, ec=badge_color, lw=0.0)
    badge = short_badge(row["best_sampler"])
    ax.text(
        x + w / 2,
        y + h - 0.110,
        badge,
        ha="center",
        va="center",
        fontsize=6.4,
        color="white",
        fontweight="bold",
        linespacing=0.9,
    )

    bar_x, bar_y, bar_w, bar_h = x + 0.018, y + h - 0.245, w - 0.036, 0.062
    left = bar_x
    for key, _, color in COMPONENTS:
        value = row["selected_weights"].get(key, 0.0)
        ax.add_patch(Rectangle((left, bar_y), bar_w * value, bar_h, facecolor=color, edgecolor="white", lw=0.7))
        left += bar_w * value
    ax.add_patch(Rectangle((bar_x, bar_y), bar_w, bar_h, facecolor="none", edgecolor="#334e68", lw=0.8))

    rank = row["best_mean_rank"]
    wins = row["best_wins"]
    ax.text(x + 0.018, y + h - 0.320, f"rank {rank:.2f}", fontsize=7.8, color="#334e68")
    ax.text(x + 0.018, y + h - 0.367, f"wins {wins}", fontsize=7.8, color="#334e68")

    if idx < 4:
        ax.add_patch(FancyArrowPatch((x + w + 0.001, y + h * 0.50), (x + w + 0.023, y + h * 0.50), arrowstyle="-|>", mutation_scale=9, lw=1.1, color="#bcccdc"))


def sampler_color(name: str) -> str:
    return {
        "pooled_validation_mixture": "#2563eb",
        "bootstrap_mean_mixture": "#0f766e",
        "gaussian_bayes": "#243b53",
        "knn_conditional": "#d97706",
        "hybrid": "#2f855a",
        "pair_rerank": "#b83280",
    }.get(name, "#334e68")


def short_badge(name: str) -> str:
    return {
        "pooled_validation_mixture": "validation\nmixture",
        "bootstrap_mean_mixture": "mean\nmixture",
        "gaussian_bayes": "physics\nkernel",
        "knn_conditional": "nearest\nneighbors",
        "hybrid": "learned\ncontext",
        "pair_rerank": "pair\nrerank",
    }.get(name, f"best:\n{SAMPLER_LABELS.get(name, name)}")


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fc: str,
    ec: str,
    lw: float,
    zorder: float = 1,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(patch)


def add_panel_label(ax: plt.Axes, letter: str, title: str) -> None:
    draw_text = ax.text2D if hasattr(ax, "text2D") else ax.text
    draw_text(
        0.0,
        1.015,
        f"{letter}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="circle,pad=0.20", facecolor="#102a43", edgecolor="none"),
    )
    draw_text(
        0.12,
        1.018,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#102a43",
    )


def wrap(text: str, width: int) -> str:
    words = text.split()
    lines = []
    current = []
    count = 0
    for word in words:
        add = len(word) + (1 if current else 0)
        if current and count + add > width:
            lines.append(" ".join(current))
            current = [word]
            count = len(word)
        else:
            current.append(word)
            count += add
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Core1_Subvol1_6micron_225cube_16bit_LE.raw"
TRAJ = ROOT / "data" / "processed" / "bentheimer_6um_downsample3_trajectories.npz"
FIGURES = ROOT / "figures"
PAPER_FIGURES = ROOT / "paper" / "figures"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 170,
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )

    volume = load_downsampled_volume()
    pore = inlet_outlet_pore_mask(volume)
    trajectories = load_trajectories()

    fig = render_scene(pore, trajectories)
    for directory in (FIGURES, PAPER_FIGURES):
        fig.savefig(directory / "bentheimer_3d_porous_medium.png", bbox_inches="tight", pad_inches=0.04)
        fig.savefig(directory / "bentheimer_3d_porous_medium.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote figures/bentheimer_3d_porous_medium.png")


def load_downsampled_volume() -> np.ndarray:
    raw = np.fromfile(RAW, dtype="<u2").reshape((225, 225, 225))
    return raw.reshape(75, 3, 75, 3, 75, 3).mean(axis=(1, 3, 5))


def otsu_threshold(values: np.ndarray, bins: int = 512) -> float:
    counts, edges = np.histogram(values.ravel(), bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight_b = np.cumsum(counts)
    weight_f = counts.sum() - weight_b
    sum_b = np.cumsum(counts * centers)
    sum_total = sum_b[-1]
    mean_b = sum_b / np.maximum(weight_b, 1)
    mean_f = (sum_total - sum_b) / np.maximum(weight_f, 1)
    variance = weight_b * weight_f * (mean_b - mean_f) ** 2
    return float(centers[np.argmax(variance)])


def inlet_outlet_pore_mask(volume: np.ndarray) -> np.ndarray:
    pore = volume < otsu_threshold(volume)
    labels, n_labels = ndimage.label(pore, structure=np.ones((3, 3, 3), dtype=bool))
    if n_labels == 0:
        return pore

    inlet = set(np.unique(labels[0, :, :])) - {0}
    outlet = set(np.unique(labels[-1, :, :])) - {0}
    through_labels = np.array(sorted(inlet & outlet), dtype=int)
    if through_labels.size:
        return np.isin(labels, through_labels)

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))


def load_trajectories() -> list[np.ndarray]:
    with np.load(TRAJ, allow_pickle=True) as data:
        return [np.asarray(item, dtype=float) for item in data["trajectories"]]


def render_scene(pore: np.ndarray, trajectories: list[np.ndarray]) -> plt.Figure:
    step = 2
    pore_plot = pore[::step, ::step, ::step]
    n = pore_plot.shape[0]

    fig = plt.figure(figsize=(7.6, 6.2), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    rgba = np.zeros(pore_plot.shape + (4,), dtype=float)
    rgba[..., 0] = 0.12
    rgba[..., 1] = 0.54
    rgba[..., 2] = 0.72
    rgba[..., 3] = 0.40
    ax.voxels(pore_plot, facecolors=rgba, edgecolors=(1, 1, 1, 0.015), linewidth=0.04, shade=True)

    add_sandstone_surface_context(ax, pore, step)
    add_trajectory_lines(ax, trajectories, step)
    add_bounding_box(ax, n)

    ax.view_init(elev=23, azim=-50, roll=0)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_zlim(0, n)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()

    fig.text(
        0.03,
        0.955,
        "Actual 3D Bentheimer pore space",
        fontsize=14.5,
        fontweight="bold",
        color="#102a43",
    )
    fig.text(
        0.03,
        0.915,
        "Connected void volume from the micro-CT data, with resolved particle paths overlaid",
        fontsize=9.2,
        color="#415161",
    )
    return fig


def add_sandstone_surface_context(ax: plt.Axes, pore: np.ndarray, step: int) -> None:
    solid = ~pore
    solid_plot = solid[::step, ::step, ::step]
    n = solid_plot.shape[0]
    x, y, z = np.indices(solid_plot.shape)
    cutaway = ~((x > 0.43 * n) & (y < 0.62 * n) & (z > 0.30 * n))
    solid_cut = solid_plot & cutaway
    surface = solid_cut & ~ndimage.binary_erosion(solid_cut, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    coords = np.argwhere(surface)
    if coords.size == 0:
        return
    rng = np.random.default_rng(7)
    if coords.shape[0] > 12000:
        coords = coords[rng.choice(coords.shape[0], 12000, replace=False)]
    ax.scatter(
        coords[:, 0] + 0.5,
        coords[:, 1] + 0.5,
        coords[:, 2] + 0.5,
        s=2.0,
        c="#d2b48c",
        alpha=0.10,
        depthshade=True,
        linewidths=0,
    )


def add_trajectory_lines(ax: plt.Axes, trajectories: list[np.ndarray], step: int) -> None:
    selected = select_trajectories(trajectories, n=9)
    colors = plt.cm.plasma(np.linspace(0.08, 0.88, len(selected)))
    for traj, color in zip(selected, colors):
        pts = traj[:, :3] / step
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] < 2:
            continue
        stride = max(1, pts.shape[0] // 220)
        pts = pts[::stride]
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = Line3DCollection(segs, colors=[color], linewidths=2.25, alpha=0.92)
        ax.add_collection3d(lc)
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], s=34, color=color, edgecolor="white", linewidth=0.7)
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], s=42, color="white", edgecolor=color, linewidth=1.1)


def select_trajectories(trajectories: list[np.ndarray], n: int) -> list[np.ndarray]:
    spans = np.array([traj[-1, 0] - traj[0, 0] for traj in trajectories])
    order = np.argsort(spans)[::-1]
    picks: list[np.ndarray] = []
    for idx in order:
        traj = trajectories[int(idx)]
        if len(traj) < 10:
            continue
        if not picks or min(np.linalg.norm(traj[0, 1:3] - item[0, 1:3]) for item in picks) > 5:
            picks.append(traj)
        if len(picks) >= n:
            break
    return picks


def add_bounding_box(ax: plt.Axes, n: int) -> None:
    color = "#102a43"
    alpha = 0.18
    corners = np.array(
        [
            [0, 0, 0],
            [n, 0, 0],
            [n, n, 0],
            [0, n, 0],
            [0, 0, n],
            [n, 0, n],
            [n, n, n],
            [0, n, n],
        ],
        dtype=float,
    )
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        ax.plot(*zip(corners[a], corners[b]), color=color, alpha=alpha, lw=0.8)


if __name__ == "__main__":
    main()

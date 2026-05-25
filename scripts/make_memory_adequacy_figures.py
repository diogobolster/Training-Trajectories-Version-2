from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from bootstrap_mixture_selection import build_components, fixed_train_test_split  # noqa: E402
from scripts import make_story_figures as story  # noqa: E402
from scripts import render_3d_porous_medium as pore3d  # noqa: E402
from tta_v2 import MixtureSegmentSampler, SegmentArchive, choose_origin_pool, generate_with_origins  # noqa: E402
from tta_v2.metrics import dilution_index, pair_separation_summary, velocity_autocorrelation  # noqa: E402


FIGURES = ROOT / "figures"
PAPER_FIGURES = ROOT / "paper" / "figures"

OUTER = ROOT / "outputs" / "bentheimer_6um_downsample3_D001_n20000_stride400_outer_split_mixture_benchmark.json"
OBJECTIVE = ROOT / "outputs" / "bentheimer_6um_downsample3_D001_n20000_stride400_objective_weight_sensitivity.json"
HIGH_PE = ROOT / "outputs" / "bentheimer_6um_downsample3_D0003_n20000_stride400_outer_split_mixture_benchmark.json"
LOW_PE = ROOT / "outputs" / "bentheimer_6um_downsample3_D003_n20000_stride400_outer_split_mixture_benchmark.json"
CORE2_GRAPH = ROOT / "outputs" / "bentheimer_core2_subvol1_6um_downsample3_D001_n20000_stride400_outer_split_mixture_benchmark.json"
CORE2_OPENFOAM = ROOT / "outputs" / "bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_dt010_n20000_stride1600_outer_split_mixture_benchmark.json"
OPENFOAM_OBJECTIVE = ROOT / "outputs" / "bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_dt010_n20000_stride1600_objective_weight_sensitivity.json"
FLOW_PHYSICS = ROOT / "outputs" / "core2_graph_vs_openfoam_physics_comparison.json"
OPENFOAM_LADDER = ROOT / "outputs" / "openfoam_resolution_ladder_summary.json"
OPENFOAM_LADDER_BENCHMARKS = {
    "downsample3": ROOT / "outputs" / "bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_dt010_n20000_stride1600_outer_split_mixture_benchmark.json",
    "downsample2": ROOT / "outputs" / "bentheimer_core2_subvol1_6um_downsample2_D00225_openfoam_phys_scaled_dt010_n20000_stride1600_outer_split_mixture_benchmark.json",
    "fullres": ROOT / "outputs" / "bentheimer_core2_subvol1_6um_fullres_D009_openfoam_strict_dt010_n20000_stride1600_outer_split_mixture_benchmark.json",
}
EVIDENCE = ROOT / "outputs" / "master_evidence_table.json"
BREAKTHROUGH_ONLY = ROOT / "outputs" / "bentheimer_6um_downsample3_D001_n20000_stride400_breakthrough_only_failure.json"
CORE1_BASELINE_TRAJ = ROOT / "data" / "processed" / "bentheimer_6um_downsample3_D001_n20000_steps800_trajectories.npz"
HIGH_PE_TRAJ = ROOT / "data" / "processed" / "bentheimer_6um_downsample3_D0003_n20000_steps800_trajectories.npz"
LOW_PE_TRAJ = ROOT / "data" / "processed" / "bentheimer_6um_downsample3_D003_n20000_steps800_trajectories.npz"
FIGURE3_VISUAL_CACHE = ROOT / "outputs" / "figure3_core1_baseline_visual_diagnostics.npz"
FIGURE5_DIAGNOSTICS_CACHE = ROOT / "outputs" / "figure5_peclet_reference_diagnostics.npz"
FIGURE6_DIAGNOSTICS_CACHE = ROOT / "outputs" / "figure6_openfoam_physical_diagnostics.npz"
VOXEL_UM = 18.0
FIGURE3_VISUAL_VERSION = "figure3_visual_v2"
FIGURE5_DIAGNOSTICS_VERSION = "figure5_peclet_reference_v1"
FIGURE6_DIAGNOSTICS_VERSION = "figure6_openfoam_physical_v1"

COMPONENTS = ["gaussian_bayes", "knn_conditional", "hybrid", "pair_rerank"]
SAMPLERS = ["pooled_validation_mixture", "gaussian_bayes", "hybrid", "knn_conditional", "pair_rerank"]
FIXED = ["gaussian_bayes", "knn_conditional", "hybrid", "pair_rerank"]

LABELS = {
    "pooled_validation_mixture": "pooled-validation mixture",
    "bootstrap_mean_mixture": "bootstrap-mean mixture",
    "gaussian_bayes": "velocity continuity",
    "knn_conditional": "archive proximity",
    "hybrid": "learned context",
    "pair_rerank": "pair organization",
    "balanced": "balanced",
    "breakthrough_only": "breakthrough-only",
    "btc_heavy": "BTC-heavy",
    "pair_heavy": "pair-heavy",
    "dilution_heavy": "occupied-volume-heavy",
    "reaction_light": "encounter-light",
    "reaction_heavy": "encounter-heavy",
    "no_reaction": "no-encounter",
}

SHORT = {
    "pooled_validation_mixture": "pooled-validation",
    "bootstrap_mean_mixture": "bootstrap-mean mixture",
    "gaussian_bayes": "vel. cont.",
    "knn_conditional": "archive prox.",
    "hybrid": "learned ctx.",
    "pair_rerank": "pair org.",
}

LEGEND_LABELS = {
    "pooled_validation_mixture": "pooled-validation mixture",
    "bootstrap_mean_mixture": "bootstrap-mean mixture",
    "gaussian_bayes": "velocity continuity",
    "knn_conditional": "archive proximity",
    "hybrid": "learned context",
    "pair_rerank": "pair organization",
}

COLORS = {
    "reference": "#111111",
    "pooled_validation_mixture": "#6F6F6F",
    "bootstrap_mean_mixture": "#9A9A9A",
    "gaussian_bayes": "#4E79A7",
    "knn_conditional": "#B07D12",
    "hybrid": "#5C8D6A",
    "pair_rerank": "#9B6A8F",
    "arrival": "#4E79A7",
    "dilution": "#5C8D6A",
    "pairs": "#B07D12",
    "encounters": "#9B6A8F",
    "graph": "#7A7A7A",
    "openfoam": "#9E6240",
    "ink": "#1A1A1A",
    "muted": "#666666",
    "grid": "#E1E1E1",
}

OBSERVABLES = [
    ("arrival", "arrival", "Did mass arrive?"),
    ("dilution", "occupied volume", "Did plume organization survive?"),
    ("pairs", "pairs", "Did pairs separate?"),
    ("encounters", "encounter proxy", "Did pairs come close?"),
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
    openfoam_ladder = load_json(OPENFOAM_LADDER)
    openfoam_ladder_benchmarks = {key: load_json(path) for key, path in OPENFOAM_LADDER_BENCHMARKS.items()}
    evidence = load_json(EVIDENCE)["conditions"]
    breakthrough_only = load_json(BREAKTHROUGH_ONLY)

    volume = pore3d.load_downsampled_volume()
    pore = pore3d.inlet_outlet_pore_mask(volume)
    trajectories = load_core1_baseline_trajectories()
    behavior_summary = story.load_behavior_summary(outer)
    figure3_visual = load_or_create_figure3_visual_data(trajectories, outer)
    peclet_reference = load_or_create_figure5_reference_diagnostics()
    openfoam_physical = load_or_create_figure6_openfoam_diagnostics(flow_physics)

    save_all("figure1_memory_landscape", figure1_memory_landscape(pore, trajectories), svg=False)
    save_all("figure2_memory_assay", figure2_memory_assay(trajectories, evidence))
    save_all("figure3_arrival_can_lie", figure3_arrival_can_lie(behavior_summary, breakthrough_only, figure3_visual, outer))
    save_all("figure4_observable_memory_selection", figure4_observable_memory_selection(objective))
    save_all("figure5_diffusion_memory_erosion", figure5_diffusion_memory_erosion(peclet, peclet_reference))
    save_all(
        "figure6_openfoam_fidelity_resolution",
        figure6_openfoam_fidelity_resolution(flow_physics, openfoam_physical, openfoam_ladder, openfoam_ladder_benchmarks),
    )
    save_all("figure7_memory_adequacy_atlas", figure7_memory_adequacy_atlas(evidence))


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.9,
            "axes.titlesize": 8.3,
            "axes.labelsize": 7.8,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.1,
            "figure.dpi": 180,
            "savefig.dpi": 320,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_core1_baseline_trajectories() -> list[np.ndarray]:
    path = CORE1_BASELINE_TRAJ if CORE1_BASELINE_TRAJ.exists() else pore3d.TRAJ
    with np.load(path, allow_pickle=True) as data:
        return [np.asarray(item, dtype=float) for item in data["trajectories"]]


def load_trajectory_file(path: Path) -> list[np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return [np.asarray(item, dtype=float) for item in data["trajectories"]]


def load_or_create_figure5_reference_diagnostics() -> dict[str, np.ndarray]:
    if FIGURE5_DIAGNOSTICS_CACHE.exists():
        with np.load(FIGURE5_DIAGNOSTICS_CACHE, allow_pickle=False) as data:
            version = str(np.asarray(data["version"]).item())
            if version == FIGURE5_DIAGNOSTICS_VERSION:
                return {key: data[key].copy() for key in data.files}

    diagnostics = create_figure5_reference_diagnostics()
    FIGURE5_DIAGNOSTICS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIGURE5_DIAGNOSTICS_CACHE, **diagnostics)
    return diagnostics


def create_figure5_reference_diagnostics() -> dict[str, np.ndarray]:
    paths = [HIGH_PE_TRAJ, CORE1_BASELINE_TRAJ, LOW_PE_TRAJ]
    pe_values = np.asarray([200.0, 60.0, 20.0])
    max_lag = 50
    autocorr = []
    for path in paths:
        trajectories = load_trajectory_file(path)
        autocorr.append(velocity_autocorrelation(trajectories, max_lag=max_lag))
    return {
        "version": np.asarray(FIGURE5_DIAGNOSTICS_VERSION),
        "pe_values": pe_values,
        "lags": np.arange(max_lag + 1, dtype=float),
        "autocorr": np.asarray(autocorr, dtype=float),
    }


def load_or_create_figure6_openfoam_diagnostics(flow: dict) -> dict[str, np.ndarray]:
    if FIGURE6_DIAGNOSTICS_CACHE.exists():
        with np.load(FIGURE6_DIAGNOSTICS_CACHE, allow_pickle=False) as data:
            version = str(np.asarray(data["version"]).item())
            if version == FIGURE6_DIAGNOSTICS_VERSION:
                return {key: data[key].copy() for key in data.files}

    diagnostics = create_figure6_openfoam_diagnostics(flow)
    FIGURE6_DIAGNOSTICS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIGURE6_DIAGNOSTICS_CACHE, **diagnostics)
    return diagnostics


def create_figure6_openfoam_diagnostics(flow: dict) -> dict[str, np.ndarray]:
    graph = load_trajectory_file(Path(flow["inputs"]["graph"]))
    openfoam = load_trajectory_file(Path(flow["inputs"]["openfoam"]))
    graph_speeds = step_displacements(graph)
    openfoam_speeds = step_displacements(openfoam)
    upper = float(max(np.quantile(graph_speeds, 0.997), np.quantile(openfoam_speeds, 0.997)))
    bins = np.linspace(0.0, upper, 90)
    graph_ccdf = ccdf_on_bins(graph_speeds, bins)
    openfoam_ccdf = ccdf_on_bins(openfoam_speeds, bins)
    graph_values, graph_splits = transport_diagnostic_values(graph, graph_speeds)
    openfoam_values, openfoam_splits = transport_diagnostic_values(openfoam, openfoam_speeds)
    return {
        "version": np.asarray(FIGURE6_DIAGNOSTICS_VERSION),
        "speed_bins": bins,
        "graph_ccdf": graph_ccdf,
        "openfoam_ccdf": openfoam_ccdf,
        "transport_values": np.vstack([graph_values, openfoam_values]),
        "transport_splits": np.stack([graph_splits, openfoam_splits]),
    }


def step_displacements(trajectories: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.linalg.norm(np.diff(np.asarray(traj), axis=0), axis=1) for traj in trajectories if len(traj) > 1])


def ccdf_on_bins(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.asarray([np.mean(values >= threshold) for threshold in bins], dtype=float)


def transport_diagnostic_values(trajectories: list[np.ndarray], speeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    full = transport_diagnostic_vector(trajectories, speeds)
    rng = np.random.default_rng(601)
    indices = rng.permutation(len(trajectories))
    split_vectors = []
    for chunk in np.array_split(indices, 5):
        subset = [trajectories[int(i)] for i in chunk]
        split_vectors.append(transport_diagnostic_vector(subset, step_displacements(subset)))
    return full, np.asarray(split_vectors, dtype=float)


def transport_diagnostic_vector(trajectories: list[np.ndarray], speeds: np.ndarray) -> np.ndarray:
    final_index = min(400, min(len(traj) for traj in trajectories) - 1)
    positions = np.asarray([np.asarray(traj)[final_index] for traj in trajectories if len(traj) > final_index], dtype=float)
    final_x = positions[:, 0]
    dilution = dilution_index(trajectories, [final_index], bin_size=3.0)[final_index]["dilution_index"]
    pair = pair_separation_summary(trajectories, [final_index], n_pairs=min(5000, max(1000, len(trajectories) * 8)), seed=123)[final_index]["q50"]
    return np.asarray(
        [
            float(np.quantile(speeds, 0.95)),
            float(np.quantile(final_x, 0.90) - np.quantile(final_x, 0.10)),
            float(dilution),
            float(pair),
        ],
        dtype=float,
    )


def load_or_create_figure3_visual_data(trajectories: list[np.ndarray], outer: dict) -> dict[str, object]:
    if FIGURE3_VISUAL_CACHE.exists():
        with np.load(FIGURE3_VISUAL_CACHE, allow_pickle=False) as data:
            version = str(np.asarray(data["version"]).item())
            if version == FIGURE3_VISUAL_VERSION:
                return {key: data[key].copy() for key in data.files}

    visual = create_figure3_visual_data(trajectories, outer)
    FIGURE3_VISUAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIGURE3_VISUAL_CACHE, **visual)
    return visual


def create_figure3_visual_data(trajectories: list[np.ndarray], outer: dict) -> dict[str, object]:
    outer_result = outer["outer_results"][0]
    seed = int(outer_result["seed"])
    train_pool, test, _split = fixed_train_test_split(trajectories, test_fraction=0.30, seed=seed)
    archive = SegmentArchive.from_trajectories(
        train_pool,
        segment_steps=int(outer_result["archive"]["segment_steps"]),
        match_steps=int(outer_result["archive"]["match_steps"]),
        stride=400,
        dt=1.0,
    )
    args = SimpleNamespace(
        diffusivity=0.001,
        candidate_limit=256,
        gaussian_bandwidth=0.25,
        knn_k=96,
        knn_temperature=0.8,
        contrastive_negative_ratio=6,
        contrastive_epochs=400,
        hybrid_learned_weight=0.25,
        pair_rerank_weight=0.25,
        pair_neighbor_k=32,
        rerank_horizon_segments=3,
        adaptive_bins=4,
    )
    components = build_components(args, archive, seed=seed)
    pooled_weights = np.asarray([outer_result["pooled_validation_weights"][key] for key in COMPONENTS], dtype=float)
    mixture = MixtureSegmentSampler(
        archive=archive,
        components=[components[key] for key in COMPONENTS],
        component_weights=pooled_weights,
        seed=seed,
    )
    samplers = {"pooled_validation_mixture": mixture, **components}
    origin_pool = choose_origin_pool("train", train_pool, test)
    n_visual = 700
    n_segments = 32
    visual_names = ["pooled_validation_mixture", "gaussian_bayes", "knn_conditional", "hybrid", "pair_rerank"]
    rng_ref = np.random.default_rng(seed + 3101)
    ref_indices = rng_ref.choice(len(test), size=min(n_visual, len(test)), replace=False)
    ref_subset = [test[int(index)] for index in ref_indices]

    payload: dict[str, object] = {
        "version": np.asarray(FIGURE3_VISUAL_VERSION),
        "sampler_names": np.asarray(visual_names),
        "times_reference": first_passage_times_at_plane(test, 14.0),
        "map_reference": positions_at_time(ref_subset, 400),
    }
    for offset, name in enumerate(visual_names):
        generated = generate_with_origins(
            samplers[name],
            n_trajectories=n_visual,
            n_segments=n_segments,
            origin_pool=origin_pool,
            rng=np.random.default_rng(seed + 4100 + offset),
        )
        payload[f"times_{name}"] = first_passage_times_at_plane(generated, 14.0)
        payload[f"map_{name}"] = positions_at_time(generated, 400)
    return payload


def first_passage_times_at_plane(trajectories: list[np.ndarray], plane: float) -> np.ndarray:
    times: list[float] = []
    for trajectory in trajectories:
        x = np.asarray(trajectory, dtype=float)[:, 0]
        crossed = np.flatnonzero(x >= plane)
        times.append(float(crossed[0]) if len(crossed) else np.nan)
    return np.asarray(times, dtype=float)


def positions_at_time(trajectories: list[np.ndarray], time_index: int) -> np.ndarray:
    positions: list[np.ndarray] = []
    for trajectory in trajectories:
        arr = np.asarray(trajectory, dtype=float)
        if len(arr) > time_index:
            positions.append(arr[time_index, :3])
    if not positions:
        return np.empty((0, 3), dtype=float)
    return np.asarray(positions, dtype=float)


def save_all(name: str, fig: plt.Figure, svg: bool = True) -> None:
    for directory in (FIGURES, PAPER_FIGURES):
        fig.savefig(directory / f"{name}.png", bbox_inches="tight", pad_inches=0.035)
        fig.savefig(directory / f"{name}.pdf", bbox_inches="tight", pad_inches=0.035)
        if svg:
            fig.savefig(directory / f"{name}.svg", bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(f"wrote {name}")


def title(fig: plt.Figure, text: str, y: float = 0.982) -> None:
    # Main captions carry interpretation; figures themselves stay data-first.
    return None


def subtitle(fig: plt.Figure, text: str, y: float = 0.942) -> None:
    return None


def panel(ax: plt.Axes, letter: str, label: str | None = None, x: float = 0.0, y: float = 1.01) -> None:
    text_fn = ax.text2D if hasattr(ax, "text2D") else ax.text
    text_fn(x, y, f"({letter})", transform=ax.transAxes, fontsize=8.7, fontweight="bold", color=COLORS["ink"], va="bottom")
    if label:
        text_fn(x + 0.078, y, label, transform=ax.transAxes, fontsize=8.1, color=COLORS["ink"], va="bottom")


def rounded(ax: plt.Axes, x: float, y: float, w: float, h: float, fc: str, ec: str, lw: float = 1.0, alpha: float = 1.0) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
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
    fig = plt.figure(figsize=(7.45, 7.15), facecolor="white")
    gs = fig.add_gridspec(
        3,
        4,
        width_ratios=[1.22, 1.05, 1.05, 1.05],
        height_ratios=[1.0, 1.0, 0.72],
        wspace=0.34,
        hspace=0.44,
    )
    ax3d = fig.add_subplot(gs[0:2, 0:2], projection="3d")
    motif_gs = gs[0:2, 2:4].subgridspec(3, 1, hspace=0.34)
    motif_axes = [fig.add_subplot(motif_gs[i, 0]) for i in range(3)]
    ax_pair = fig.add_subplot(gs[2, 0:2])
    ax_map = fig.add_subplot(gs[2, 2:4])
    fig.subplots_adjust(left=0.055, right=0.985, top=0.975, bottom=0.070)

    motifs = select_empirical_motifs(trajectories)
    pairs = select_pair_histories(trajectories)
    speed_norm = speed_color_norm(trajectories, motifs)
    cmap = mpl.colormaps["turbo"]

    mappable = draw_empirical_pore_paths(ax3d, pore, trajectories, speed_norm, cmap)
    for ax, motif in zip(motif_axes, motifs):
        draw_empirical_motif(ax, motif, speed_norm, cmap)
    draw_pair_distance_histories(ax_pair, pairs)
    draw_state_observable_map(ax_map)

    cbar = fig.colorbar(mappable, ax=ax3d, orientation="horizontal", fraction=0.044, pad=0.010, shrink=0.54)
    cbar.set_label("local speed (cells step$^{-1}$)", fontsize=6.6)
    cbar.ax.tick_params(labelsize=6.2, length=2)

    panel(ax3d, "a", "connected pore volume and reference pathlines", x=0.00, y=1.00)
    panel(motif_axes[0], "b", "archive motifs", x=0.00, y=1.14)
    panel(ax_pair, "c", "pair histories", x=0.00, y=1.05)
    panel(ax_map, "d", "retained state and observable", x=0.00, y=1.05)
    return fig


def speed_color_norm(trajectories: list[np.ndarray], motifs: list[dict]) -> mpl.colors.Normalize:
    samples: list[np.ndarray] = []
    for traj in select_reference_trajectories(trajectories, n=36):
        pts = np.asarray(traj, dtype=float)[:, :3]
        samples.append(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    for motif in motifs:
        samples.append(motif["speed"])
    speeds = np.concatenate([item[np.isfinite(item)] for item in samples if item.size])
    vmax = float(np.nanquantile(speeds, 0.985)) if speeds.size else 0.12
    return mpl.colors.Normalize(vmin=0.0, vmax=max(vmax, 0.08))


def select_reference_trajectories(trajectories: list[np.ndarray], n: int = 36) -> list[np.ndarray]:
    spans = np.array([np.asarray(traj)[-1, 0] - np.asarray(traj)[0, 0] for traj in trajectories], dtype=float)
    order = np.argsort(spans)[::-1]
    picks: list[np.ndarray] = []
    for idx in order:
        traj = np.asarray(trajectories[int(idx)], dtype=float)
        if len(traj) < 50 or not np.isfinite(traj).all():
            continue
        if not picks:
            picks.append(traj)
        else:
            start = traj[0, 1:3]
            min_sep = min(float(np.linalg.norm(start - np.asarray(item)[0, 1:3])) for item in picks)
            if min_sep > 3.5 or len(picks) < 12:
                picks.append(traj)
        if len(picks) >= n:
            break
    return picks


def draw_empirical_pore_paths(
    ax: plt.Axes,
    pore: np.ndarray,
    trajectories: list[np.ndarray],
    norm: mpl.colors.Normalize,
    cmap: mpl.colors.Colormap,
) -> mpl.cm.ScalarMappable:
    step = 2
    pore_plot = pore[::step, ::step, ::step]
    rgba = np.zeros(pore_plot.shape + (4,), dtype=float)
    rgba[..., 0] = 0.47
    rgba[..., 1] = 0.52
    rgba[..., 2] = 0.54
    rgba[..., 3] = 0.070
    ax.voxels(pore_plot, facecolors=rgba, edgecolors=(1, 1, 1, 0.012), linewidth=0.015, shade=True)

    selected = select_reference_trajectories(trajectories, n=42)
    for traj in selected:
        pts_full = np.asarray(traj, dtype=float)[:, :3]
        pts_full = pts_full[np.isfinite(pts_full).all(axis=1)]
        if pts_full.shape[0] < 3:
            continue
        stride = max(1, pts_full.shape[0] // 180)
        pts = pts_full[::stride] / step
        if pts.shape[0] < 2:
            continue
        segment_speed = np.linalg.norm(np.diff(pts_full, axis=0), axis=1)
        segment_speed = segment_speed[::stride][: pts.shape[0] - 1]
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        halo = Line3DCollection(segs, colors=[(0.02, 0.02, 0.02, 0.46)], linewidths=1.55, alpha=0.85)
        ax.add_collection3d(halo)
        lc = Line3DCollection(segs, colors=cmap(norm(segment_speed)), linewidths=1.08, alpha=0.98)
        ax.add_collection3d(lc)
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], s=10, color="#111111", alpha=0.58, linewidth=0)
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], s=12, color="white", edgecolor="#111111", alpha=0.72, linewidth=0.45)

    n = pore_plot.shape[0]
    ax.view_init(elev=23, azim=-51, roll=0)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_zlim(0, n)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    add_empirical_3d_annotations(ax, n)
    return mpl.cm.ScalarMappable(norm=norm, cmap=cmap)


def add_empirical_3d_annotations(ax: plt.Axes, n: int) -> None:
    ax.text(0, -1.5, 3, "inlet", fontsize=6.2, color=COLORS["ink"])
    ax.text(n * 0.98, -1.5, 3, "outlet", fontsize=6.2, color=COLORS["ink"], ha="right")
    ax.plot([4, 14], [2, 2], [1, 1], color=COLORS["ink"], lw=1.2)
    ax.text(9, 2.4, 1.2, f"{int(10 * VOXEL_UM)} um", fontsize=5.9, color=COLORS["ink"], ha="center")
    ax.quiver(2, 4, 2, 8, 0, 0, color=COLORS["ink"], arrow_length_ratio=0.22, linewidth=0.8, alpha=0.65)
    ax.text(10.5, 4, 2, "x", fontsize=6.2, color=COLORS["ink"])


def select_empirical_motifs(trajectories: list[np.ndarray], window: int = 120) -> list[dict]:
    candidates: list[dict] = []
    max_traj = min(len(trajectories), 3200)
    indices = np.linspace(0, len(trajectories) - 1, max_traj, dtype=int)
    for ti in indices:
        traj = np.asarray(trajectories[int(ti)], dtype=float)[:, :3]
        if traj.shape[0] <= window or not np.isfinite(traj).all():
            continue
        for start in range(0, traj.shape[0] - window, 45):
            pts = traj[start : start + window]
            d = np.diff(pts, axis=0)
            speed = np.linalg.norm(d, axis=1)
            path_length = float(np.sum(speed))
            net = float(np.linalg.norm(pts[-1] - pts[0]))
            mean_speed = float(np.mean(speed))
            cv = float(np.std(speed) / (mean_speed + 1e-9))
            dx = float(pts[-1, 0] - pts[0, 0])
            third = max(5, len(speed) // 3)
            early = float(np.mean(speed[:third]))
            late = float(np.mean(speed[-third:]))
            candidates.append(
                {
                    "traj_index": int(ti),
                    "start": int(start),
                    "pts": pts,
                    "speed": speed,
                    "path_length": path_length,
                    "net": net,
                    "mean_speed": mean_speed,
                    "cv": cv,
                    "dx": dx,
                    "tortuosity": path_length / max(net, 1e-6),
                    "change": abs(late - early),
                    "range": float(np.nanquantile(speed, 0.92) - np.nanquantile(speed, 0.08)),
                    "early": early,
                    "late": late,
                }
            )
    if not candidates:
        fallback = np.asarray(trajectories[0], dtype=float)[:window, :3]
        speed = np.linalg.norm(np.diff(fallback, axis=0), axis=1)
        return [{"label": "motif", "pts": fallback, "speed": speed, "mean_speed": float(np.mean(speed)), "dx": 0.0, "tortuosity": 1.0, "change": 0.0}]

    used: set[tuple[int, int]] = set()

    def choose(label: str, score_fn) -> dict:
        ranked = sorted(candidates, key=score_fn, reverse=True)
        for item in ranked:
            key = (item["traj_index"], item["start"] // 90)
            if key not in used:
                used.add(key)
                out = dict(item)
                out["label"] = label
                return out
        out = dict(ranked[0])
        out["label"] = label
        return out

    fast = choose("fast persistent channel", lambda c: c["mean_speed"] / (1.0 + 0.65 * c["cv"]) + 0.003 * max(c["dx"], 0))
    slow = choose(
        "slow/tortuous motif",
        lambda c: (
            min(c["tortuosity"], 12.0) + 0.075 * c["path_length"] - 2.2 * abs(c["mean_speed"] - 0.050)
            if c["path_length"] > 4.0 and c["net"] > 1.0 and c["mean_speed"] < 0.090
            else -np.inf
        ),
    )
    changing = choose(
        "velocity-class change",
        lambda c: (
            1.2 * c["change"] + 0.55 * c["range"] + 0.2 * abs(c["late"] - c["early"])
            if c["path_length"] > 4.0 and c["mean_speed"] > 0.025 and c["range"] > 0.045
            else -np.inf
        ),
    )
    return [fast, slow, changing]


def draw_empirical_motif(
    ax: plt.Axes,
    motif: dict,
    norm: mpl.colors.Normalize,
    cmap: mpl.colors.Colormap,
) -> None:
    speed = np.asarray(motif["speed"], dtype=float)
    t = np.arange(speed.size)
    if speed.size > 1:
        pts = np.column_stack([t, speed])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, colors=cmap(norm(speed[:-1])), linewidths=1.55)
        ax.add_collection(lc)
    ax.plot(t, speed, color="#222222", lw=0.35, alpha=0.45)
    ax.axhline(float(np.mean(speed)), color="#777777", lw=0.65, ls=":")
    ax.set_xlim(0, max(1, speed.size - 1))
    ax.set_ylim(0, max(norm.vmax * 1.08, float(np.nanmax(speed)) * 1.10))
    ax.grid(True, color=COLORS["grid"], lw=0.55)
    ax.set_ylabel("speed", fontsize=6.4)
    ax.text(
        0.02,
        0.86,
        str(motif["label"]),
        transform=ax.transAxes,
        fontsize=6.9,
        color=COLORS["ink"],
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.98,
        0.84,
        f"mean={motif['mean_speed']:.3f}\nnet dx={motif['dx']:.1f}\ntort={motif['tortuosity']:.1f}",
        transform=ax.transAxes,
        fontsize=5.8,
        color=COLORS["muted"],
        ha="right",
        va="top",
    )
    if ax.get_subplotspec().is_last_row():
        ax.set_xlabel("motif step", fontsize=6.4)
    else:
        ax.set_xticklabels([])


def select_pair_histories(trajectories: list[np.ndarray], encounter_radius: float = 3.0) -> dict:
    rng = np.random.default_rng(13)
    max_n = min(len(trajectories), 2600)
    indices = rng.choice(len(trajectories), size=max_n, replace=False)
    arr = np.stack([np.asarray(trajectories[int(i)], dtype=float)[:, :3] for i in indices], axis=0)
    tmax = arr.shape[1]
    initial = arr[:, 0, :]

    divergence = None
    start_tree = cKDTree(initial[:, 1:3])
    start_pairs = list(start_tree.query_pairs(r=7.0))
    rng.shuffle(start_pairs)
    best_score = -np.inf
    for i, j in start_pairs[:45000]:
        curve = np.linalg.norm(arr[i] - arr[j], axis=1)
        if curve[0] <= encounter_radius or np.nanmin(curve[20:]) <= encounter_radius:
            continue
        score = curve[-1] - curve[0]
        if score > best_score:
            best_score = float(score)
            divergence = curve

    encounter = None
    best_time = tmax
    for time_idx in [80, 140, 220, 320, 460, 620, 780]:
        if time_idx >= tmax:
            continue
        tree = cKDTree(arr[:, time_idx, :])
        pairs = list(tree.query_pairs(r=encounter_radius))
        rng.shuffle(pairs)
        for i, j in pairs[:30000]:
            curve = np.linalg.norm(arr[i] - arr[j], axis=1)
            min_idx = int(np.nanargmin(curve[20:])) + 20
            if curve[0] > 2.0 * encounter_radius and curve[min_idx] <= encounter_radius:
                if min_idx < best_time:
                    best_time = min_idx
                    encounter = curve
        if encounter is not None and best_time <= time_idx:
            break

    if divergence is None:
        divergence = np.linalg.norm(arr[0] - arr[1], axis=1)
    if encounter is None:
        encounter = np.linalg.norm(arr[2] - arr[3], axis=1)
    return {"divergence": divergence, "encounter": encounter, "radius": encounter_radius}


def draw_pair_distance_histories(ax: plt.Axes, pairs: dict) -> None:
    radius = float(pairs["radius"])
    t = np.arange(len(pairs["divergence"]))
    ax.plot(t, pairs["divergence"], color=COLORS["hybrid"], lw=1.35, label="diverging pair")
    ax.plot(t, pairs["encounter"], color=COLORS["pair_rerank"], lw=1.35, label="near-encounter pair")
    ax.axhline(radius, color="#333333", lw=0.75, ls="--", label=f"encounter radius = {radius:g} cells")
    ax.set_xlabel("time step")
    ax.set_ylabel("pair distance (cells)")
    ax.grid(True, color=COLORS["grid"], lw=0.55)
    ax.legend(frameon=False, loc="upper right", fontsize=6.2)
    enc = pairs["encounter"]
    div = pairs["divergence"]
    ax.text(
        0.02,
        0.94,
        f"divergence: {div[0]:.1f} -> {div[-1]:.1f} cells\nencounter: min {np.nanmin(enc):.1f} cells",
        transform=ax.transAxes,
        fontsize=5.9,
        color=COLORS["muted"],
        va="top",
    )


def draw_state_observable_map(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rows = [
        ("velocity history", "arrival / BTC", COLORS["gaussian_bayes"]),
        ("spatial organization", "entropy occupied-volume proxy", COLORS["hybrid"]),
        ("pair-distance history", "separation and encounter proxy", COLORS["pair_rerank"]),
    ]
    ax.text(0.02, 0.86, "retained state", fontsize=6.8, fontweight="bold", color=COLORS["ink"], transform=ax.transAxes)
    ax.text(0.55, 0.86, "observable exposed", fontsize=6.8, fontweight="bold", color=COLORS["ink"], transform=ax.transAxes)
    for i, (state, obs, color) in enumerate(rows):
        y = 0.66 - i * 0.22
        ax.add_patch(Rectangle((0.02, y - 0.035), 0.040, 0.070, transform=ax.transAxes, facecolor=color, edgecolor="none", alpha=0.92))
        ax.text(0.075, y, state, fontsize=6.6, color=COLORS["ink"], va="center", transform=ax.transAxes)
        ax.annotate(
            "",
            xy=(0.50, y),
            xytext=(0.39, y),
            xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", lw=0.70, color="#777777"),
        )
        ax.text(0.55, y, obs, fontsize=6.6, color=COLORS["ink"], va="center", transform=ax.transAxes)


def draw_3d_memory_object(ax: plt.Axes, pore: np.ndarray, trajectories: list[np.ndarray]) -> None:
    step = 2
    pore_plot = pore[::step, ::step, ::step]
    rgba = np.zeros(pore_plot.shape + (4,), dtype=float)
    rgba[..., 0] = 0.53
    rgba[..., 1] = 0.61
    rgba[..., 2] = 0.65
    rgba[..., 3] = 0.22
    ax.voxels(pore_plot, facecolors=rgba, edgecolors=(1, 1, 1, 0.015), linewidth=0.025, shade=True)

    selected = pore3d.select_trajectories(trajectories, n=11)
    path_colors = [COLORS["gaussian_bayes"], COLORS["knn_conditional"], COLORS["hybrid"], COLORS["pair_rerank"]]
    colors = [path_colors[i % len(path_colors)] for i in range(len(selected))]
    for traj, color in zip(selected, colors):
        pts = traj[:, :3] / step
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] < 2:
            continue
        stride = max(1, pts.shape[0] // 210)
        pts = pts[::stride]
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = Line3DCollection(segs, colors=[color], linewidths=1.45, alpha=0.82)
        ax.add_collection3d(lc)
        ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], s=22, color=color, edgecolor="white", linewidth=0.5)
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], s=26, color="white", edgecolor=color, linewidth=0.7)

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
        "3D Bentheimer\nconnected pore volume",
        transform=ax.transAxes,
        fontsize=8.2,
        color=COLORS["ink"],
        bbox=dict(boxstyle="square,pad=0.25", facecolor="white", edgecolor=COLORS["grid"], alpha=0.86),
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
        ("near encounter", "encounter-opportunity proxy", COLORS["pair_rerank"]),
    ]
    y_positions = [0.80, 0.57, 0.34, 0.11]
    for idx, ((headline, body, color), y0) in enumerate(zip(labels, y_positions)):
        rounded(ax, 0.03, y0 - 0.055, 0.94, 0.15, "white", COLORS["grid"], 0.55)
        if idx < 2:
            traj = chosen[idx][:, :3]
            start = min(25 + idx * 100, max(0, len(traj) - 90))
            seg = traj[start : start + 70, [0, 1]]
            draw_normalized_path(ax, seg, x0=0.08, y0=y0 - 0.010, w=0.42, h=0.075, color=color, lw=2.0)
        elif idx == 2:
            draw_pair_paths(ax, 0.08, y0 - 0.010, 0.42, 0.075, color)
        else:
            draw_encounter_paths(ax, 0.08, y0 - 0.010, 0.42, 0.075, color)
        ax.text(0.57, y0 + 0.028, headline, fontsize=7.8, color=COLORS["ink"], va="center")
        ax.text(0.57, y0 - 0.032, body, fontsize=7.2, color=COLORS["muted"], va="center")


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
        ("pore\ngeometry", "#999999", 0.95),
        ("velocity\nhistory", COLORS["gaussian_bayes"], 0.86),
        ("path\norder", COLORS["hybrid"], 0.70),
        ("particle\nneighbors", COLORS["pair_rerank"], 0.54),
        ("observables", COLORS["arrival"], 1.0),
    ]
    xs = np.array([0.08, 0.285, 0.49, 0.695, 0.89])
    for idx, ((label, color, alpha), x) in enumerate(zip(items, xs)):
        rounded(ax, x - 0.062, 0.49, 0.124, 0.20, tint(color, 0.16), color, 1.1, alpha=alpha)
        ax.text(x, 0.59, label, ha="center", va="center", fontsize=6.85, color=COLORS["ink"], fontweight="bold")
        if idx < len(items) - 1:
            arrow(ax, (x + 0.069, 0.59), (xs[idx + 1] - 0.069, 0.59))


def figure2_memory_assay(trajectories: list[np.ndarray], evidence: list[dict]) -> plt.Figure:
    fig = plt.figure(figsize=(7.45, 5.65), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.08], height_ratios=[0.95, 1.08], wspace=0.26, hspace=0.36)
    ax_split = fig.add_subplot(gs[0, 0])
    ax_join = fig.add_subplot(gs[0, 1])
    ax_states = fig.add_subplot(gs[1, 0])
    ax_objective = fig.add_subplot(gs[1, 1])
    fig.subplots_adjust(left=0.055, right=0.985, top=0.960, bottom=0.075)

    draw_split_design(ax_split)
    draw_motif_recombination_panel(ax_join, trajectories)
    draw_candidate_state_panel(ax_states)
    draw_validation_objective_panel(ax_objective, evidence)
    panel(ax_split, "a", "trajectory-level split design", x=0.00, y=1.04)
    panel(ax_join, "b", "motif recombination", x=0.00, y=1.04)
    panel(ax_states, "c", "candidate retained states", x=0.00, y=1.04)
    panel(ax_objective, "d", "inner validation objective", x=0.00, y=1.04)
    return fig


def draw_split_design(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.88, "20,000 complete particle trajectories", fontsize=7.1, fontweight="bold", color=COLORS["ink"], transform=ax.transAxes)
    ax.text(0.02, 0.805, "splits are by particle trajectory, not by segment", fontsize=6.2, color=COLORS["muted"], transform=ax.transAxes)
    draw_split_bar(ax, 0.05, 0.66, 0.90, 0.090, [0.70, 0.30], ["train pool\n14,000", "outer test\n6,000"], [tint("#777777", 0.10), tint(COLORS["pair_rerank"], 0.17)])
    ax.annotate(
        "",
        xy=(0.36, 0.57),
        xytext=(0.36, 0.65),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", lw=0.75, color="#777777"),
    )
    draw_split_bar(
        ax,
        0.05,
        0.40,
        0.63,
        0.085,
        [0.70, 0.30],
        ["fit/archive\n9,800", "inner validation\n4,200"],
        [tint(COLORS["knn_conditional"], 0.12), tint(COLORS["hybrid"], 0.14)],
    )
    rounded(ax, 0.73, 0.395, 0.22, 0.105, "white", "#CFCFCF", 0.75)
    ax.text(0.84, 0.456, "outer held-out\nevaluation", ha="center", va="center", fontsize=6.1, color=COLORS["ink"], transform=ax.transAxes)
    ax.annotate(
        "",
        xy=(0.73, 0.448),
        xytext=(0.68, 0.448),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", lw=0.75, color="#777777"),
    )
    ax.text(0.05, 0.235, "4 outer splits", fontsize=6.4, color=COLORS["ink"], transform=ax.transAxes)
    ax.text(0.05, 0.170, "3 inner validation repeats per outer split", fontsize=6.4, color=COLORS["ink"], transform=ax.transAxes)
    ax.text(0.05, 0.105, "archive motifs are cut only from fit trajectories", fontsize=6.4, color=COLORS["ink"], transform=ax.transAxes)


def draw_split_bar(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    fractions: list[float],
    labels: list[str],
    colors: list[tuple[float, float, float]],
) -> None:
    xpos = x
    total = sum(fractions)
    for frac, label, color in zip(fractions, labels, colors):
        ww = w * frac / total
        ax.add_patch(Rectangle((xpos, y), ww, h, transform=ax.transAxes, facecolor=color, edgecolor="#9A9A9A", linewidth=0.65))
        ax.text(xpos + ww / 2, y + h / 2, label, ha="center", va="center", fontsize=5.7, color=COLORS["ink"], linespacing=0.95, transform=ax.transAxes)
        xpos += ww


def draw_motif_recombination_panel(ax: plt.Axes, trajectories: list[np.ndarray]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    motifs = select_empirical_motifs(trajectories, window=92)
    current = motifs[0]["pts"][:, [0, 1]]
    candidates = [item["pts"][:, [0, 1]] for item in motifs]
    current_pts = normalized_polyline(current, 0.07, 0.55, 0.34, 0.17)
    ax.plot(current_pts[:, 0], current_pts[:, 1], color="#202020", lw=1.55, transform=ax.transAxes)
    ax.scatter(current_pts[-1, 0], current_pts[-1, 1], s=18, color="#202020", transform=ax.transAxes, zorder=4)
    ax.text(0.075, 0.76, "current motif", fontsize=6.5, color=COLORS["ink"], transform=ax.transAxes)
    ax.annotate(
        r"$v_{end}$",
        xy=tuple(current_pts[-1]),
        xytext=(current_pts[-1, 0] - 0.04, current_pts[-1, 1] + 0.105),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        fontsize=6.4,
        color=COLORS["gaussian_bayes"],
        arrowprops=dict(arrowstyle="->", lw=0.75, color=COLORS["gaussian_bayes"]),
    )
    ax.text(0.56, 0.82, "candidate next motifs", fontsize=6.5, color=COLORS["ink"], transform=ax.transAxes)
    colors = [COLORS["gaussian_bayes"], COLORS["knn_conditional"], COLORS["hybrid"]]
    candidate_starts: list[tuple[float, float]] = []
    for i, (seg, color) in enumerate(zip(candidates, colors)):
        pts = normalized_polyline(seg, 0.57, 0.68 - i * 0.145, 0.28, 0.070)
        candidate_starts.append(tuple(pts[0]))
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.25, transform=ax.transAxes)
        ax.scatter(pts[0, 0], pts[0, 1], s=13, color="white", edgecolor=color, linewidth=0.65, transform=ax.transAxes, zorder=4)
        ax.text(0.87, pts[0, 1], r"$v_{start}$", fontsize=5.6, color=color, va="center", transform=ax.transAxes)
    ax.annotate(
        "",
        xy=(0.54, 0.565),
        xytext=(0.44, 0.565),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", lw=0.85, color="#777777"),
    )
    joined = normalized_polyline(candidates[1], current_pts[-1, 0], current_pts[-1, 1], 0.26, 0.12)
    joined[:, 0] = current_pts[-1, 0] + (joined[:, 0] - joined[0, 0])
    joined[:, 1] = current_pts[-1, 1] + (joined[:, 1] - joined[0, 1])
    ax.plot(joined[:, 0], joined[:, 1], color=COLORS["knn_conditional"], lw=1.35, transform=ax.transAxes)
    ax.scatter(joined[0, 0], joined[0, 1], s=18, color="white", edgecolor=COLORS["knn_conditional"], linewidth=0.75, transform=ax.transAxes, zorder=4)
    rounded(ax, 0.055, 0.080, 0.885, 0.135, tint("#777777", 0.045), "#C9C9C9", 0.65)
    ax.text(
        0.075,
        0.160,
        "motif displacement recombination: observed relative displacements\nare translated to the current endpoint",
        fontsize=5.85,
        color=COLORS["ink"],
        transform=ax.transAxes,
        linespacing=0.95,
    )
    ax.text(
        0.075,
        0.093,
        "not pore-mask-constrained tracking through the original geometry",
        fontsize=5.85,
        color=COLORS["muted"],
        transform=ax.transAxes,
    )


def normalized_polyline(seg: np.ndarray, x0: float, y0: float, w: float, h: float) -> np.ndarray:
    arr = np.asarray(seg, dtype=float)
    arr = arr[np.isfinite(arr).all(axis=1)]
    if arr.shape[0] < 2:
        return np.array([[x0, y0], [x0 + w, y0]], dtype=float)
    arr = arr - arr[0]
    span_x = np.ptp(arr[:, 0])
    span_y = np.ptp(arr[:, 1])
    if span_x <= 1e-12:
        arr[:, 0] = np.linspace(0.0, 1.0, arr.shape[0])
    else:
        arr[:, 0] = (arr[:, 0] - arr[:, 0].min()) / span_x
    if span_y <= 1e-12:
        arr[:, 1] = 0.5
    else:
        arr[:, 1] = (arr[:, 1] - arr[:, 1].min()) / span_y
    return np.column_stack([x0 + w * arr[:, 0], y0 + h * (arr[:, 1] - 0.5)])


def draw_candidate_state_panel(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rows = [
        ("velocity continuity", r"$\|v_{end}-v_{start}\|$ after diffusion scaling", "gaussian_bayes"),
        ("archive proximity", "nearest archived local transition state", "knn_conditional"),
        ("learned context", r"contrastive score $f(s_j,s_{j+1})$", "hybrid"),
        ("pair organization", "short-horizon separation descriptors", "pair_rerank"),
    ]
    for i, (name, desc, key) in enumerate(rows):
        y = 0.81 - i * 0.205
        rounded(ax, 0.025, y - 0.070, 0.94, 0.128, "white", "#D7D7D7", 0.65)
        ax.add_patch(Rectangle((0.050, y - 0.035), 0.035, 0.070, transform=ax.transAxes, facecolor=COLORS[key], edgecolor="none"))
        ax.text(0.105, y + 0.022, name, fontsize=6.9, color=COLORS["ink"], fontweight="bold", va="center", transform=ax.transAxes)
        ax.text(0.105, y - 0.032, desc, fontsize=6.0, color=COLORS["muted"], va="center", transform=ax.transAxes)
    ax.text(0.025, 0.055, "Each row defines what the next motif is allowed to know about the previous motif.", fontsize=5.9, color=COLORS["muted"], transform=ax.transAxes)


def draw_validation_objective_panel(ax: plt.Axes, evidence: list[dict]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    components = [
        ("BTC", r"$E_{BTC}$", "arrival"),
        ("occupied vol.", r"$E_D$", "dilution"),
        ("pairs", r"$E_{pair}$", "pairs"),
        ("enc. proxy", r"$E_{enc}$", "encounters"),
    ]
    ax.text(0.035, 0.875, r"$J = a_{BTC}E_{BTC}+a_DE_D+a_{pair}E_{pair}+a_{enc}E_{enc}$", fontsize=7.0, color=COLORS["ink"], transform=ax.transAxes)
    for i, (label, symbol, key) in enumerate(components):
        x = 0.04 + i * 0.235
        rounded(ax, x, 0.660, 0.195, 0.115, "white", "#D2D2D2", 0.65)
        ax.text(x + 0.097, 0.728, symbol, fontsize=7.1, color=COLORS[key], fontweight="bold", ha="center", transform=ax.transAxes)
        ax.text(x + 0.097, 0.684, label, fontsize=5.9, color=COLORS["muted"], ha="center", transform=ax.transAxes)
    ax.annotate(
        "inner validation selects weights",
        xy=(0.50, 0.565),
        xytext=(0.50, 0.615),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        fontsize=6.1,
        ha="center",
        color=COLORS["muted"],
        arrowprops=dict(arrowstyle="->", lw=0.70, color="#777777"),
    )
    weights = core1_baseline_weights(evidence)
    draw_weight_bar(ax, 0.085, 0.420, 0.80, 0.085, weights)
    ax.text(0.085, 0.535, "example selected mixture, Core1 baseline", fontsize=6.3, color=COLORS["ink"], transform=ax.transAxes)
    ax.annotate(
        "",
        xy=(0.50, 0.275),
        xytext=(0.50, 0.415),
        xycoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", lw=0.70, color="#777777"),
    )
    rounded(ax, 0.18, 0.145, 0.64, 0.125, "white", "#CFCFCF", 0.70)
    ax.text(0.50, 0.218, "outer held-out test reports BTC, occupied-volume,\npair-separation, and encounter-proxy errors", ha="center", va="center", fontsize=6.25, color=COLORS["ink"], transform=ax.transAxes)
    ax.text(0.04, 0.045, "Selection and final evaluation use disjoint complete trajectories.", fontsize=5.9, color=COLORS["muted"], transform=ax.transAxes)


def core1_baseline_weights(evidence: list[dict]) -> dict[str, float]:
    for row in evidence:
        if row.get("id") == "core1_baseline":
            return {key: float(row.get("selected_weights", {}).get(key, 0.0)) for key in COMPONENTS}
    return {"gaussian_bayes": 0.25, "knn_conditional": 0.23, "hybrid": 0.27, "pair_rerank": 0.25}


def draw_weight_bar(ax: plt.Axes, x: float, y: float, w: float, h: float, weights: dict[str, float]) -> None:
    xpos = x
    total = sum(max(0.0, float(weights.get(key, 0.0))) for key in COMPONENTS) or 1.0
    for key in COMPONENTS:
        frac = max(0.0, float(weights.get(key, 0.0))) / total
        ww = w * frac
        ax.add_patch(Rectangle((xpos, y), ww, h, transform=ax.transAxes, facecolor=COLORS[key], edgecolor="white", linewidth=0.65))
        if ww > 0.055:
            ax.text(xpos + ww / 2, y + h / 2, f"{frac:.2f}", ha="center", va="center", fontsize=5.7, color="white", transform=ax.transAxes)
        xpos += ww
    legend = [("velocity", "gaussian_bayes"), ("archive", "knn_conditional"), ("learned", "hybrid"), ("pair", "pair_rerank")]
    for i, (label, key) in enumerate(legend):
        lx = x + i * 0.198
        ax.add_patch(Rectangle((lx, y - 0.078), 0.022, 0.028, transform=ax.transAxes, facecolor=COLORS[key], edgecolor="none"))
        ax.text(lx + 0.030, y - 0.064, label, fontsize=5.55, color=COLORS["muted"], va="center", transform=ax.transAxes)


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
        "observed local\npath motifs",
        fontsize=7.6,
        color=COLORS["muted"],
        transform=ax.transAxes,
        bbox=dict(boxstyle="square,pad=0.18", facecolor="white", edgecolor="none", alpha=0.78),
    )
    arrow(ax, (0.286, 0.73), (0.365, 0.73))


def draw_forgetting_gate(ax: plt.Axes) -> None:
    rounded(ax, 0.375, 0.55, 0.135, 0.36, "#f8fafc", "#94a3b8", 1.0)
    ax.text(0.442, 0.865, "state\nreduction", ha="center", va="center", fontsize=9.0, color=COLORS["ink"], fontweight="bold", transform=ax.transAxes)
    gate_items = ["geometry", "velocity", "order", "neighbors"]
    alphas = [0.35, 0.85, 0.70, 0.58]
    for idx, (item, alpha) in enumerate(zip(gate_items, alphas)):
        y = 0.785 - 0.052 * idx
        ax.add_patch(Rectangle((0.407, y - 0.015), 0.070, 0.026, transform=ax.transAxes, facecolor="#cbd5e1", edgecolor="none", alpha=alpha))
        ax.text(0.442, y - 0.001, item, ha="center", va="center", fontsize=7.2, color="#334155", transform=ax.transAxes)
    ax.text(0.442, 0.585, "transition rules\nuse subsets", ha="center", va="center", fontsize=7.0, color=COLORS["muted"], transform=ax.transAxes)
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
    ax.text(x0, 0.875, "competing retained states", fontsize=9.2, color=COLORS["ink"], fontweight="bold", transform=ax.transAxes)
    for (head, body, key), y in zip(cards, ys):
        color = COLORS[key]
        rounded(ax, x0, y - 0.035, 0.215, 0.055, tint(color, 0.12), color, 1.1)
        ax.text(x0 + 0.022, y - 0.007, head, va="center", fontsize=8.2, color=color, fontweight="bold", transform=ax.transAxes)
        ax.text(x0 + 0.102, y - 0.007, body, va="center", fontsize=6.9, color="#334155", transform=ax.transAxes)


def draw_validation_stations(ax: plt.Axes) -> None:
    ax.text(0.055, 0.455, "held-out validation observables", fontsize=8.8, color=COLORS["ink"], fontweight="bold", transform=ax.transAxes)
    xs = [0.090, 0.310, 0.530, 0.750]
    for x, (key, short, question) in zip(xs, OBSERVABLES):
        color = COLORS[key]
        draw_observable_icon(ax, key, x, 0.335, color)
        ax.plot([x - 0.060, x + 0.060], [0.288, 0.288], color=color, lw=1.0, transform=ax.transAxes)
        ax.text(x, 0.252, short, ha="center", fontsize=8.0, color=COLORS["ink"], fontweight="bold", transform=ax.transAxes)
        ax.text(x, 0.218, question, ha="center", fontsize=6.7, color=COLORS["muted"], transform=ax.transAxes)
    ax.text(
        0.055,
        0.135,
        "Each observable is evaluated separately before scalar objectives are formed.",
        fontsize=8.25,
        color=COLORS["muted"],
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
    rounded(ax, 0.885, 0.565, 0.095, 0.315, "white", COLORS["grid"], 0.8)
    ax.text(0.932, 0.850, "relative\nselection", ha="center", va="center", fontsize=7.6, color=COLORS["ink"], fontweight="bold", transform=ax.transAxes)
    y0 = 0.792
    for idx, row in enumerate(evidence):
        y = y0 - 0.039 * idx
        key = row["best_sampler"]
        ax.add_patch(Rectangle((0.905, y - 0.011), 0.022, 0.020, transform=ax.transAxes, facecolor=COLORS[key], edgecolor="none"))
        ax.text(0.934, y - 0.001, row["short"].replace("Core2 ", ""), fontsize=6.4, color=COLORS["muted"], va="center", transform=ax.transAxes)
    ax.text(0.932, 0.595, "lowest\nmean", ha="center", va="center", fontsize=6.9, color=COLORS["muted"], transform=ax.transAxes)


def figure3_arrival_can_lie(summary: dict[str, dict], breakthrough_only: dict, visual: dict[str, object], outer: dict) -> plt.Figure:
    fig = plt.figure(figsize=(7.45, 7.05), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.98, 1.02], width_ratios=[1.02, 0.98], hspace=0.48, wspace=0.32)
    ax_btc = fig.add_subplot(gs[0, 0])
    map_gs = gs[0, 1].subgridspec(1, 3, wspace=0.08)
    ax_maps = [fig.add_subplot(map_gs[0, i]) for i in range(3)]
    ax_dil = fig.add_subplot(gs[1, 0])
    ax_fail = fig.add_subplot(gs[1, 1])
    fig.subplots_adjust(left=0.075, right=0.985, top=0.960, bottom=0.090)

    draw_breakthrough_cdf(ax_btc, visual, outer)
    draw_plume_occupancy_maps(ax_maps, visual)
    draw_dilution_time_series(ax_dil, summary)
    draw_breakthrough_only_counterfactual(ax_fail, breakthrough_only)
    panel(ax_btc, "a", "arrival at downstream plane x = 14", x=0.00, y=1.04)
    panel(ax_maps[0], "b", None, x=0.00, y=1.24)
    ax_maps[0].text(0.32, 1.24, "final-time occupancy, t = 400", transform=ax_maps[0].transAxes, fontsize=8.1, color=COLORS["ink"], va="bottom")
    panel(ax_dil, "c", "entropy-based occupied-volume proxy", x=0.00, y=1.04)
    panel(ax_fail, "d", "breakthrough-only selection tested on all observables", x=0.00, y=1.04)
    return fig


def draw_breakthrough_cdf(ax: plt.Axes, visual: dict[str, object], outer: dict) -> None:
    samplers = ["pooled_validation_mixture", "gaussian_bayes", "knn_conditional", "hybrid", "pair_rerank", "reference"]
    for sampler in samplers:
        key = "times_reference" if sampler == "reference" else f"times_{sampler}"
        times = np.asarray(visual[key], dtype=float)
        x, y = empirical_cdf(times)
        color = COLORS["reference"] if sampler == "reference" else COLORS[sampler]
        lw = 2.0 if sampler == "reference" else 1.2
        label = "reference" if sampler == "reference" else LEGEND_LABELS.get(sampler, sampler)
        ax.step(x, y, where="post", color=color, lw=lw, alpha=0.96 if sampler == "reference" else 0.82, label=label, zorder=5 if sampler == "reference" else 3)
    draw_split_arrival_markers(ax, outer)
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("first-passage time to x = 14")
    ax.set_ylabel("fraction arrived")
    ax.grid(True, color=COLORS["grid"], linewidth=0.65)
    ax.legend(frameon=False, loc="upper left", fontsize=5.9, handlelength=1.8)


def draw_split_arrival_markers(ax: plt.Axes, outer: dict) -> None:
    samplers = ["pooled_validation_mixture", "gaussian_bayes", "knn_conditional", "hybrid", "pair_rerank"]
    y0 = 0.815
    for idx, sampler in enumerate(samplers):
        q50s = []
        for result in outer["outer_results"]:
            stats = result["test"][sampler]["metrics"]["breakthrough"]["14.0"]
            q50 = float(stats["q50"])
            if np.isfinite(q50):
                q50s.append(q50)
        if q50s:
            ax.scatter(q50s, np.full(len(q50s), y0 + idx * 0.024), marker="|", s=42, linewidth=1.1, color=COLORS[sampler], alpha=0.95, zorder=6)
    ax.text(
        0.985,
        0.955,
        "ticks: conditional median arrival\namong crossed particles",
        transform=ax.transAxes,
        fontsize=5.8,
        color=COLORS["muted"],
        ha="right",
        va="top",
    )


def empirical_cdf(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float)
    total = max(1, len(times))
    finite = np.sort(times[np.isfinite(times)])
    if finite.size == 0:
        return np.asarray([0.0, 500.0]), np.asarray([0.0, 0.0])
    x = np.concatenate([[0.0], finite, [max(500.0, float(finite[-1]))]])
    y = np.concatenate([[0.0], np.arange(1, finite.size + 1) / total, [finite.size / total]])
    return x, y


def draw_plume_occupancy_maps(axes: list[plt.Axes], visual: dict[str, object]) -> None:
    ref_prob = occupancy_probability(np.asarray(visual["map_reference"], dtype=float))
    mix_prob = occupancy_probability(np.asarray(visual["map_pooled_validation_mixture"], dtype=float))
    maps = [
        ("reference", log_occupancy_display(ref_prob), "reference", "cividis", None),
        ("validation mixture", log_occupancy_display(mix_prob), "pooled_validation_mixture", "cividis", None),
        ("absolute difference", np.abs(ref_prob - mix_prob), "pair_rerank", "magma", None),
    ]
    density_vmax = max(float(np.max(item[1])) for item in maps[:2]) or 1.0
    diff_vmax = float(np.max(maps[2][1])) or 1.0
    maps[0] = (*maps[0][:-1], density_vmax)
    maps[1] = (*maps[1][:-1], density_vmax)
    maps[2] = (*maps[2][:-1], diff_vmax)
    for idx, (ax, (label, image, color_key, cmap, vmax)) in enumerate(zip(axes, maps)):
        ax.imshow(image.T, origin="lower", extent=[0, 75, 0, 75], cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")
        ax.set_title(label, fontsize=6.2, color=COLORS["ink"], pad=1.0)
        ax.set_xlim(0, 75)
        ax.set_ylim(0, 75)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=5.3, length=2)
        if idx == 0:
            ax.set_ylabel("z (cells)", fontsize=5.7)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("y", fontsize=5.7)
        ax.add_patch(Rectangle((2, 67), 8, 5, facecolor=COLORS["reference"] if color_key == "reference" else COLORS[color_key], edgecolor="none", alpha=0.95))


def occupancy_probability(positions: np.ndarray, bin_size: float = 3.0) -> np.ndarray:
    positions = positions[np.isfinite(positions).all(axis=1)]
    if positions.size == 0:
        return np.zeros((25, 25), dtype=float)
    y = np.clip(positions[:, 1], 0, 74.999)
    z = np.clip(positions[:, 2], 0, 74.999)
    bins = np.arange(0, 75 + bin_size, bin_size)
    hist, _, _ = np.histogram2d(y, z, bins=[bins, bins])
    return hist / max(1.0, float(np.sum(hist)))


def log_occupancy_display(probability: np.ndarray) -> np.ndarray:
    return np.log10(probability + 1e-4) - np.log10(1e-4)


def draw_dilution_time_series(ax: plt.Axes, summary: dict[str, dict]) -> None:
    samplers = ["reference", "pooled_validation_mixture", "gaussian_bayes", "knn_conditional", "hybrid", "pair_rerank"]
    for sampler in samplers:
        times = np.asarray(summary[sampler]["times"], dtype=float)
        mean = np.asarray(summary[sampler]["dilution_mean"], dtype=float)
        std = np.asarray(summary[sampler].get("dilution_std", np.zeros_like(mean)), dtype=float)
        color = COLORS["reference"] if sampler == "reference" else COLORS[sampler]
        lw = 2.0 if sampler == "reference" else 1.15
        label = "reference" if sampler == "reference" else LEGEND_LABELS.get(sampler, sampler)
        if np.any(np.isfinite(std) & (std > 0)):
            ax.fill_between(times, mean - std, mean + std, color=color, alpha=0.13 if sampler == "reference" else 0.08, linewidth=0)
        ax.plot(times, mean, marker="o", markersize=3.0, lw=lw, color=color, label=label)
    ref_final = float(summary["reference"]["dilution_mean"][-1])
    ref_sd = float(summary["reference"].get("dilution_std", [np.nan])[-1])
    if np.isfinite(ref_sd) and ref_sd > 0:
        gaps = []
        for sampler in samplers[1:]:
            gaps.append(max(0.0, (ref_final - float(summary[sampler]["dilution_mean"][-1])) / ref_sd))
        gap_label = f"{float(np.mean(gaps)):.0f}x" if np.ptp(gaps) < 0.5 else f"{min(gaps):.0f}-{max(gaps):.0f}x"
        ax.set_title(f"final generated gap / reference split SD = {gap_label}", fontsize=7.1, color=COLORS["ink"], pad=3.0)
    ax.text(
        0.985,
        0.235,
        "full-reference comparison;\nsame-count sensitivity in SI",
        transform=ax.transAxes,
        fontsize=5.9,
        color=COLORS["muted"],
        ha="right",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
    )
    ax.set_xlabel("time step")
    ax.set_ylabel("entropy occupied-volume proxy (cells$^3$)")
    ax.grid(True, color=COLORS["grid"], linewidth=0.65)
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=5.8, handlelength=1.8, columnspacing=0.7)


def draw_breakthrough_only_counterfactual(ax: plt.Axes, result: dict) -> None:
    selected = min(
        result["summary"]["breakthrough_only"]["samplers"],
        key=lambda name: result["summary"]["breakthrough_only"]["samplers"][name]["mean_objective"],
    )
    selected_key = f"breakthrough_only::{selected}" if selected in {"pooled_validation_mixture", "bootstrap_mean_mixture"} else selected
    metrics = [
        ("arrival", "btc_score", "BTC"),
        ("dilution", "dilution_log_mae", "occ. vol."),
        ("pairs", "pair_quantile_mae", "pairs"),
        ("encounters", "reaction_abs_error", "enc. proxy"),
    ]
    ratio_by_metric = splitwise_counterfactual_ratios(result, selected_key, [key for _outcome, key, _label in metrics])
    x = np.arange(len(metrics))
    means = np.asarray([np.nanmean(ratio_by_metric[key]) for _outcome, key, _label in metrics], dtype=float)
    stds = np.asarray([np.nanstd(ratio_by_metric[key], ddof=1) for _outcome, key, _label in metrics], dtype=float)
    colors = [COLORS[outcome] for outcome, _key, _label in metrics]
    ax.bar(x, means, yerr=stds, capsize=2.4, color=colors, alpha=0.78, width=0.58, edgecolor="white", linewidth=0.45)
    rng = np.random.default_rng(4)
    for xi, (_outcome, key, _label) in enumerate(metrics):
        vals = ratio_by_metric[key]
        jitter = rng.uniform(-0.085, 0.085, size=len(vals))
        ax.scatter(np.full(len(vals), xi) + jitter, vals, s=13, color="#222222", alpha=0.55, linewidth=0, zorder=4)
    ax.axhline(1.0, color="#334155", lw=0.85, ls="--")
    ax.set_xticks(x, [label for _outcome, _key, label in metrics])
    ax.set_ylim(0.82, max(1.35, float(np.nanmax(means + stds)) + 0.10))
    ax.set_ylabel("error / split-specific lowest error")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.65)
    ax.text(
        0.02,
        0.92,
        f"BTC-only selected: {LEGEND_LABELS.get(selected, selected)}",
        transform=ax.transAxes,
        fontsize=6.6,
        color=COLORS["muted"],
        va="top",
    )


def splitwise_counterfactual_ratios(result: dict, selected_key: str, metric_keys: list[str]) -> dict[str, np.ndarray]:
    candidate_names = [
        "gaussian_bayes",
        "knn_conditional",
        "hybrid",
        "pair_rerank",
        "balanced::pooled_validation_mixture",
        "breakthrough_only::pooled_validation_mixture",
        "breakthrough_only::bootstrap_mean_mixture",
    ]
    ratios: dict[str, list[float]] = {key: [] for key in metric_keys}
    for outer in result["outer_results"]:
        errors = outer["test_errors"]
        for key in metric_keys:
            finite = [errors[name][key] for name in candidate_names if name in errors and np.isfinite(errors[name][key])]
            best = min(finite) if finite else np.nan
            value = errors[selected_key][key]
            ratios[key].append(float(value / best) if np.isfinite(best) and best > 0 else np.nan)
    return {key: np.asarray(values, dtype=float) for key, values in ratios.items()}


def figure4_observable_memory_selection(objective: dict) -> plt.Figure:
    regimes = [
        "balanced",
        "breakthrough_only",
        "btc_heavy",
        "pair_heavy",
        "dilution_heavy",
        "reaction_light",
        "reaction_heavy",
        "no_reaction",
    ]
    fig = plt.figure(figsize=(7.45, 6.25), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.92, 1.18], width_ratios=[0.98, 1.28], hspace=0.45, wspace=0.34)
    ax_weights = fig.add_subplot(gs[0, 0])
    ax_mix = fig.add_subplot(gs[0, 1])
    ax_consequence = fig.add_subplot(gs[1, :])
    fig.subplots_adjust(left=0.115, right=0.985, top=0.955, bottom=0.130)
    draw_objective_weight_design(ax_weights, objective, regimes)
    draw_selected_mixture_bars(ax_mix, objective, regimes)
    draw_objective_consequence(ax_consequence, objective, regimes)
    panel(ax_weights, "a", None, x=0.00, y=1.11)
    panel(ax_mix, "b", None, x=0.00, y=1.11)
    panel(ax_consequence, "c", None, x=0.00, y=1.11)
    return fig


def draw_objective_weight_design(ax: plt.Axes, objective: dict, regimes: list[str]) -> None:
    components = [("btc", "BTC"), ("pair", "pairs"), ("dilution", "occ. volume"), ("reaction", "enc. proxy")]
    base = objective["regime_weights"]["balanced"]
    matrix = np.asarray(
        [[float(objective["regime_weights"][regime][key]) / max(float(base[key]), 1e-12) for key, _label in components] for regime in regimes],
        dtype=float,
    )
    image = ax.imshow(matrix, cmap="Greys", vmin=0.0, vmax=3.0, aspect="auto")
    ax.set_xticks(np.arange(len(components)), [label for _key, label in components], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(regimes)), [LABELS[regime] for regime in regimes])
    ax.tick_params(length=0)
    ax.set_title("objective weights relative to balanced", loc="left", fontweight="normal", color=COLORS["ink"], pad=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            label = "0" if value == 0 else f"{value:.1f}"
            color = "white" if value > 1.55 else COLORS["ink"]
            ax.text(j, i, label, ha="center", va="center", fontsize=6.6, color=color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = plt.colorbar(image, ax=ax, fraction=0.052, pad=0.025)
    cbar.ax.tick_params(labelsize=6.2, length=2.0)
    cbar.outline.set_linewidth(0.45)


def draw_selected_mixture_bars(ax: plt.Axes, objective: dict, regimes: list[str]) -> None:
    y = np.arange(len(regimes), dtype=float)
    left = np.zeros(len(regimes), dtype=float)
    for comp in COMPONENTS:
        values = np.asarray(
            [objective["summary"][regime]["selected_weights"]["pooled_validation_mixture"]["mean"][comp] for regime in regimes],
            dtype=float,
        )
        ax.barh(y, values, left=left, height=0.64, color=COLORS[comp], edgecolor="white", linewidth=0.55, label=LEGEND_LABELS[comp])
        for yi, start, value in zip(y, left, values):
            if value >= 0.12:
                ax.text(start + value / 2, yi, f"{value:.2f}", ha="center", va="center", fontsize=5.9, color="white")
        left += values
    ax.set_xlim(0, 1)
    ax.set_ylim(len(regimes) - 0.45, -0.55)
    ax.set_yticks(y, [""] * len(regimes))
    ax.set_xlabel("mixture weight")
    ax.set_title("selected pooled-validation mixture", loc="left", fontweight="normal", color=COLORS["ink"], pad=8)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.65)
    ax.legend(ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.50, -0.37), columnspacing=1.0, handlelength=1.1)


def draw_objective_consequence(ax: plt.Axes, objective: dict, regimes: list[str]) -> None:
    samplers = ["gaussian_bayes", "knn_conditional", "hybrid", "pair_rerank", "pooled_validation_mixture", "bootstrap_mean_mixture"]
    offsets = np.linspace(-0.23, 0.23, len(samplers))
    cap = 3.35
    for row, regime in enumerate(regimes):
        summary = objective["summary"][regime]["samplers"]
        best = min(samplers, key=lambda key: float(summary[key]["mean_objective"]))
        best_mean = float(summary[best]["mean_objective"])
        best_sd = max(float(summary[best].get("std_objective", np.nan)), 1e-9)
        for offset, sampler in zip(offsets, samplers):
            delta = (float(summary[sampler]["mean_objective"]) - best_mean) / best_sd
            x = min(delta, cap)
            marker = ">" if delta > cap else "o"
            size = 23 if sampler != best else 42
            ax.scatter(
                x,
                row + offset,
                s=size,
                marker=marker,
                color=COLORS[sampler],
                edgecolor="white",
                linewidth=0.45,
                zorder=3 if sampler != best else 5,
            )
        wins = int(summary[best].get("wins", 0))
        rank = float(summary[best].get("mean_rank", np.nan))
        ax.text(cap + 0.23, row, f"{SHORT[best]}; {wins}/4 wins; rank {rank:.1f}", fontsize=6.25, color=COLORS["muted"], va="center")
    ax.axvline(1.0, color="#9ca3af", lw=0.75, ls="--")
    ax.text(1.02, -0.58, "one split SD", fontsize=6.1, color=COLORS["muted"], va="bottom")
    ax.set_xlim(-0.05, cap + 1.30)
    ax.set_ylim(len(regimes) - 0.55, -0.55)
    ax.set_yticks(np.arange(len(regimes)), [LABELS[regime] for regime in regimes])
    ax.set_xlabel(r"$\Delta J$ from lowest mean / SD(lowest-mean mechanism)")
    ax.set_title("relative held-out score; lowest mean at zero", loc="left", fontweight="normal", color=COLORS["ink"], pad=8)
    ax.text(cap + 0.05, len(regimes) - 0.18, "triangles: beyond axis", fontsize=5.9, color=COLORS["muted"], va="bottom")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.65)
    handles = [
        mpl.lines.Line2D([0], [0], marker="o", lw=0, markersize=5.2, markerfacecolor=COLORS[sampler], markeredgecolor="white", label=LEGEND_LABELS[sampler])
        for sampler in samplers
    ]
    ax.legend(handles=handles, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.50, -0.22), columnspacing=1.1, handletextpad=0.35)


def figure5_diffusion_memory_erosion(peclet: dict[str, dict], reference_diagnostics: dict[str, np.ndarray]) -> plt.Figure:
    fig = plt.figure(figsize=(7.45, 5.90), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.90, 1.08], width_ratios=[1.02, 1.10], hspace=0.74, wspace=0.34)
    ax_ref = fig.add_subplot(gs[0, 0])
    ax_ribbon = fig.add_subplot(gs[0, 1])
    ax_scores = fig.add_subplot(gs[1, 0])
    ax_delta = fig.add_subplot(gs[1, 1])
    fig.subplots_adjust(left=0.095, right=0.985, top=0.955, bottom=0.165)
    draw_peclet_reference_diagnostic(ax_ref, reference_diagnostics)
    draw_memory_ribbon(ax_ribbon, peclet)
    draw_peclet_objectives(ax_scores, peclet)
    draw_peclet_relative_evidence(ax_delta, peclet)
    panel(ax_ref, "a", None, x=0.00, y=1.18)
    panel(ax_ribbon, "b", None, x=0.00, y=1.18)
    panel(ax_scores, "c", None, x=0.00, y=1.18)
    panel(ax_delta, "d", None, x=0.00, y=1.18)
    return fig


def draw_peclet_reference_diagnostic(ax: plt.Axes, diagnostics: dict[str, np.ndarray]) -> None:
    lags = np.asarray(diagnostics["lags"], dtype=float)
    autocorr = np.asarray(diagnostics["autocorr"], dtype=float)
    colors = ["#111111", "#666666", "#A6A6A6"]
    labels = ["Pe = 200", "Pe = 60", "Pe = 20"]
    for idx, (label, color) in enumerate(zip(labels, colors)):
        ax.plot(lags, autocorr[idx], color=color, lw=1.55, label=label)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("lag (saved steps)")
    ax.set_ylabel("axial step autocorrelation")
    ax.set_title("reference regime diagnostic", loc="left", fontweight="normal", color=COLORS["ink"], pad=8)
    ax.grid(True, color=COLORS["grid"], linewidth=0.65)
    ax.legend(frameon=False, loc="upper right", handlelength=1.6)


def draw_diffusion_paths(axes: list[plt.Axes]) -> None:
    rng = np.random.default_rng(12)
    labels = [("Pe = 200", "weak diffusion", 0.03), ("Pe = 60", "baseline", 0.10), ("Pe = 20", "stronger diffusion", 0.22)]
    for ax, (label, note, sigma) in zip(axes, labels):
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
        ax.text(0.50, 0.05, note, ha="center", fontsize=7.6, color="#475569", transform=ax.transAxes)


def draw_memory_ribbon(ax: plt.Axes, peclet: dict[str, dict]) -> None:
    labels = list(peclet)
    x = np.arange(len(labels), dtype=float)
    weights = np.vstack([[peclet[label]["summary"]["mean_selected_weights"][comp] for label in labels] for comp in COMPONENTS])
    bottoms = np.zeros(len(labels))
    for i, comp in enumerate(COMPONENTS):
        ax.bar(x, weights[i], bottom=bottoms, width=0.62, color=COLORS[comp], alpha=0.76, edgecolor="white", linewidth=0.55, label=LEGEND_LABELS[comp])
        for j, value in enumerate(weights[i]):
            if value >= 0.12:
                ax.text(x[j], bottoms[j] + value / 2, f"{value:.2f}", ha="center", va="center", fontsize=6.3, color="white")
        bottoms += weights[i]
    ax.set_xticks(x, ["Pe = 200", "Pe = 60", "Pe = 20"])
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("selected mixture weight")
    ax.set_title("selected pooled-validation mixture", loc="left", fontweight="normal", color=COLORS["ink"], pad=8)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.50, -0.18), columnspacing=0.85, handlelength=1.1)


def draw_peclet_objectives(ax: plt.Axes, peclet: dict[str, dict]) -> None:
    labels = list(peclet)
    x = np.arange(len(labels), dtype=float)
    samplers = ["pooled_validation_mixture", "gaussian_bayes", "knn_conditional", "hybrid", "pair_rerank"]
    offsets = np.linspace(-0.10, 0.10, len(samplers))
    rng = np.random.default_rng(52)
    for offset, sampler in zip(offsets, samplers):
        for i, label_key in enumerate(labels):
            split_values = [float(outer["test"][sampler]["objective"]) for outer in peclet[label_key]["outer_results"]]
            jitter = rng.normal(0.0, 0.010, size=len(split_values))
            ax.scatter(
                np.full(len(split_values), x[i] + offset) + jitter,
                split_values,
                s=10,
                color=COLORS[sampler],
                alpha=0.28,
                edgecolor="none",
                zorder=2,
            )
        means = [peclet[label]["summary"]["samplers"][sampler]["mean_objective"] for label in labels]
        stds = [peclet[label]["summary"]["samplers"][sampler]["std_objective"] for label in labels]
        label = LEGEND_LABELS.get(sampler, sampler)
        ax.errorbar(
            x + offset,
            means,
            yerr=stds,
            color=COLORS[sampler],
            marker="o",
            markersize=3.5,
            lw=1.15,
            capsize=2.2,
            elinewidth=0.75,
            label=label,
            alpha=0.94,
            zorder=4,
        )
    ax.set_xticks(x, ["Pe = 200", "Pe = 60", "Pe = 20"])
    ax.set_xlim(-0.42, 2.42)
    ax.set_ylabel("held-out objective")
    ax.set_title("held-out objective with split-level points", loc="left", fontweight="normal", color=COLORS["ink"], pad=8)
    ax.grid(True, color=COLORS["grid"], linewidth=0.75)
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.52, -0.23), columnspacing=0.8, handlelength=1.2)


def draw_peclet_relative_evidence(ax: plt.Axes, peclet: dict[str, dict]) -> None:
    labels = list(peclet)
    samplers = ["gaussian_bayes", "knn_conditional", "hybrid", "pair_rerank", "pooled_validation_mixture"]
    matrix = np.zeros((len(labels), len(samplers)), dtype=float)
    best_names = []
    for i, label_key in enumerate(labels):
        summary = peclet[label_key]["summary"]["samplers"]
        best = min(samplers, key=lambda key: float(summary[key]["mean_objective"]))
        best_names.append(best)
        best_mean = float(summary[best]["mean_objective"])
        best_sd = max(float(summary[best].get("std_objective", np.nan)), 1e-9)
        for j, sampler in enumerate(samplers):
            matrix[i, j] = (float(summary[sampler]["mean_objective"]) - best_mean) / best_sd
    image = ax.imshow(np.clip(matrix, 0, 2.5), cmap="Greys", vmin=0.0, vmax=2.5, aspect="auto")
    ax.set_xticks(np.arange(len(samplers)), [SHORT[sampler] for sampler in samplers], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), ["Pe = 200", "Pe = 60", "Pe = 20"])
    ax.tick_params(length=0)
    ax.set_title(r"relative evidence, $\Delta J / SD_{best}$", loc="left", fontweight="normal", color=COLORS["ink"], pad=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            label = ">2.5" if value > 2.5 else f"{value:.1f}"
            color = "white" if min(value, 2.5) > 1.35 else COLORS["ink"]
            weight = "bold" if samplers[j] == best_names[i] else "normal"
            ax.text(j, i, label, ha="center", va="center", fontsize=6.3, color=color, fontweight=weight)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = plt.colorbar(image, ax=ax, fraction=0.052, pad=0.025)
    cbar.set_label(r"$\Delta J / SD_{best}$", fontsize=6.4)
    cbar.ax.tick_params(labelsize=6.1, length=2.0)
    cbar.outline.set_linewidth(0.45)


def figure6_openfoam_fidelity_resolution(
    flow: dict,
    physical: dict[str, np.ndarray],
    ladder: dict,
    benchmarks: dict[str, dict],
) -> plt.Figure:
    fig = plt.figure(figsize=(7.45, 3.05), facecolor="white")
    gs = fig.add_gridspec(1, 3, wspace=0.44)
    ax_corr = fig.add_subplot(gs[0, 0])
    ax_dist = fig.add_subplot(gs[0, 1])
    ax_transport = fig.add_subplot(gs[0, 2])
    fig.subplots_adjust(left=0.085, right=0.985, top=0.900, bottom=0.245)
    draw_graph_openfoam_autocorr(ax_corr, flow)
    draw_step_distribution(ax_dist, physical)
    draw_openfoam_transport_diagnostics(ax_transport, physical)
    for i, ax in enumerate([ax_corr, ax_dist, ax_transport]):
        panel(ax, chr(ord("a") + i), None, x=0.00, y=1.15)
    return fig


def draw_graph_openfoam_autocorr(ax: plt.Axes, flow: dict) -> None:
    lags = np.asarray(flow["velocity_autocorrelation"]["lags"], dtype=float)
    graph = np.asarray(flow["velocity_autocorrelation"]["graph"], dtype=float)
    openfoam = np.asarray(flow["velocity_autocorrelation"]["openfoam"], dtype=float)
    ax.plot(lags, graph, color=COLORS["graph"], lw=1.75, label="graph-flow")
    ax.plot(lags, openfoam, color=COLORS["openfoam"], lw=1.75, label="18 um OpenFOAM")
    ax.fill_between(lags, graph, openfoam, where=openfoam >= graph, color=COLORS["openfoam"], alpha=0.12, linewidth=0)
    ax.set_xlim(0, 80)
    ax.set_xlabel("lag")
    ax.set_ylabel("axial velocity autocorrelation")
    ax.set_title("velocity persistence before validation", loc="left", fontweight="normal", color=COLORS["ink"], pad=8)
    ax.grid(True, color=COLORS["grid"], linewidth=0.65)
    ax.legend(frameon=False)
    inset = ax.inset_axes([0.48, 0.42, 0.48, 0.40])
    auc = [np.trapezoid(graph, lags) / max(lags[-1] - lags[0], 1.0), np.trapezoid(openfoam, lags) / max(lags[-1] - lags[0], 1.0)]
    inset.bar([0, 1], auc, color=[COLORS["graph"], COLORS["openfoam"]], width=0.62)
    inset.set_xticks([0, 1], ["graph", "OF"], fontsize=5.7)
    inset.set_ylabel("mean corr.", fontsize=5.7)
    inset.tick_params(axis="y", labelsize=5.5, length=2)
    inset.set_ylim(0, max(auc) * 1.35)
    for xi, value in enumerate(auc):
        inset.text(xi, value + 0.006, f"{value:.2f}", ha="center", va="bottom", fontsize=5.6, color=COLORS["ink"])
    inset.spines[["top", "right"]].set_visible(False)


def draw_step_distribution(ax: plt.Axes, physical: dict[str, np.ndarray]) -> None:
    bins = np.asarray(physical["speed_bins"], dtype=float)
    ax.semilogy(bins, physical["graph_ccdf"], color=COLORS["graph"], lw=1.65, label="graph-flow")
    ax.semilogy(bins, physical["openfoam_ccdf"], color=COLORS["openfoam"], lw=1.65, label="18 um OpenFOAM")
    ax.set_xlabel("step displacement")
    ax.set_ylabel("exceedance probability")
    ax.set_title("step-displacement tail", loc="left", fontweight="normal", color=COLORS["ink"], pad=8)
    ax.grid(True, color=COLORS["grid"], linewidth=0.65)
    ax.legend(frameon=False, loc="upper right")


def draw_openfoam_transport_diagnostics(ax: plt.Axes, physical: dict[str, np.ndarray]) -> None:
    labels = ["q95\nstep", "x-spread\n$t=400$", "occ. volume\n$t=400$", "pair q50\n$t=400$"]
    values = np.asarray(physical["transport_values"], dtype=float)
    splits = np.asarray(physical["transport_splits"], dtype=float)
    norm = np.maximum(values[0], 1e-12)
    x = np.arange(len(labels), dtype=float)
    for dataset, color, label, offset in [(0, COLORS["graph"], "graph-flow", -0.10), (1, COLORS["openfoam"], "18 um OpenFOAM", 0.10)]:
        split_values = splits[dataset] / norm[None, :]
        means = values[dataset] / norm
        stds = np.nanstd(split_values, axis=0, ddof=1)
        for j in range(split_values.shape[1]):
            ax.scatter(np.full(split_values.shape[0], x[j] + offset), split_values[:, j], s=9, color=color, alpha=0.26, edgecolor="none")
        ax.errorbar(x + offset, means, yerr=stds, color=color, marker="o", lw=0, elinewidth=0.85, capsize=2.3, markersize=3.6, label=label, zorder=4)
    ax.axhline(1.0, color="#9ca3af", lw=0.75, ls="--")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.78, 1.28)
    ax.set_ylabel("normalized to graph-flow")
    ax.set_title("transport summaries", loc="left", fontweight="normal", color=COLORS["ink"], pad=8)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.65)
    ax.legend(frameon=False, loc="upper left", fontsize=6.8)


def draw_openfoam_resolution_physics(ax: plt.Axes, ladder: dict) -> None:
    cases = ladder["cases"]
    order = ["downsample3", "downsample2", "fullres"]
    labels = ["18 um", "12 um", "6 um"]
    x = np.arange(len(order), dtype=float)
    permeability = np.asarray([cases[key]["apparent_permeability"] * 1e12 for key in order], dtype=float)
    corr80 = np.asarray([cases[key]["autocorrelation"]["80"] for key in order], dtype=float)
    ax.plot(x, permeability / permeability[0], color=COLORS["ink"], marker="o", lw=1.35, label=r"$k/k_{18}$")
    ax.plot(x, corr80 / corr80[0], color=COLORS["openfoam"], marker="s", lw=1.25, label=r"$C_{80}/C_{80,18}$")
    ax.set_xticks(x, labels)
    ax.set_ylabel("relative to 18 um")
    ax.set_title("resolution-ladder physics", loc="left", fontweight="normal", color=COLORS["ink"], pad=8)
    ax.set_ylim(0.78, 1.22)
    ax.grid(True, color=COLORS["grid"], linewidth=0.65)
    ax.legend(frameon=False, loc="upper right", fontsize=6.8)
    for xi, k_value, corr_value in zip(x, permeability, corr80):
        ax.text(xi, k_value / permeability[0] - 0.035, f"k={k_value:.2f}", ha="center", va="top", fontsize=5.8, color=COLORS["ink"])
        ax.text(xi, corr_value / corr80[0] + 0.030, f"C80={corr_value:.3f}", ha="center", va="bottom", fontsize=5.6, color=COLORS["openfoam"])


def draw_openfoam_resolution_weights(ax: plt.Axes, ladder: dict) -> None:
    cases = ladder["cases"]
    order = ["downsample3", "downsample2", "fullres"]
    labels = ["18 um", "12 um", "6 um"]
    x = np.arange(len(order), dtype=float)
    bottoms = np.zeros(len(order), dtype=float)
    for comp in COMPONENTS:
        values = np.asarray([cases[key]["balanced"]["mean_selected_weights"][comp] for key in order], dtype=float)
        ax.bar(x, values, bottom=bottoms, width=0.64, color=COLORS[comp], edgecolor="white", linewidth=0.6, label=LEGEND_LABELS[comp])
        for xi, bottom, value in zip(x, bottoms, values):
            if value >= 0.12:
                ax.text(xi, bottom + value / 2, f"{value:.2f}", ha="center", va="center", fontsize=5.6, color="white")
        bottoms += values
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("selected mixture weight")
    ax.set_title("balanced selected mixtures", loc="left", fontweight="normal", color=COLORS["ink"], pad=8)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.65)
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.50, -0.22), fontsize=6.6, columnspacing=0.8, handlelength=1.0)


def draw_openfoam_resolution_objectives(ax: plt.Axes, benchmarks: dict[str, dict]) -> None:
    order = ["downsample3", "downsample2", "fullres"]
    labels = ["18 um", "12 um", "6 um"]
    x = np.arange(len(order), dtype=float)
    samplers = ["pooled_validation_mixture", "gaussian_bayes", "knn_conditional", "hybrid", "pair_rerank"]
    offsets = np.linspace(-0.16, 0.16, len(samplers))
    rng = np.random.default_rng(74)
    for offset, sampler in zip(offsets, samplers):
        means = []
        stds = []
        for i, key in enumerate(order):
            summary = benchmarks[key]["summary"]["samplers"][sampler]
            means.append(summary["mean_objective"])
            stds.append(summary["std_objective"])
            split_values = [float(outer["test"][sampler]["objective"]) for outer in benchmarks[key]["outer_results"]]
            jitter = rng.normal(0.0, 0.010, size=len(split_values))
            ax.scatter(np.full(len(split_values), x[i] + offset) + jitter, split_values, s=9, color=COLORS[sampler], alpha=0.26, edgecolor="none")
        ax.errorbar(x + offset, means, yerr=stds, color=COLORS[sampler], marker="o", lw=0.95, elinewidth=0.75, capsize=2.0, markersize=3.0, label=LEGEND_LABELS[sampler])
    ylo, yhi = ax.get_ylim()
    yrange = yhi - ylo
    ax.set_ylim(ylo, yhi + 0.09 * yrange)
    for i, key in enumerate(order):
        summary = benchmarks[key]["summary"]["samplers"]
        best = min(samplers, key=lambda name: float(summary[name]["mean_objective"]))
        wins = int(summary[best].get("wins", 0))
        best_label = {
            "gaussian_bayes": "velocity",
            "knn_conditional": "archive",
            "hybrid": "learned",
            "pair_rerank": "pair org.",
            "pooled_validation_mixture": "mixture",
        }.get(best, SHORT[best])
        ax.text(x[i], yhi + 0.055 * yrange, f"{best_label}\n{wins}/4", ha="center", va="center", fontsize=5.6, color=COLORS["muted"], linespacing=0.88)
    ax.set_xticks(x, labels)
    ax.set_ylabel("held-out objective")
    ax.set_title("balanced validation scores", loc="left", fontweight="normal", color=COLORS["ink"], pad=8)
    ax.grid(True, color=COLORS["grid"], linewidth=0.65)


def figure7_memory_adequacy_atlas(evidence: list[dict]) -> plt.Figure:
    fig = plt.figure(figsize=(7.45, 4.85), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 0.72, 0.055], wspace=0.24)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_weights = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])
    fig.subplots_adjust(left=0.170, right=0.965, top=0.895, bottom=0.250)
    matrix, close, best_indices, labels, weights = build_relative_evidence_matrix(evidence)
    draw_relative_evidence_heatmap(ax_heat, cax, matrix, close, best_indices, labels)
    draw_relative_evidence_weights(ax_weights, weights, labels)
    panel(ax_heat, "a", None, x=0.00, y=1.13)
    panel(ax_weights, "b", None, x=0.00, y=1.13)
    return fig


def build_relative_evidence_matrix(evidence: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[dict[str, float]]]:
    samplers = ["gaussian_bayes", "knn_conditional", "hybrid", "pair_rerank", "pooled_validation_mixture", "bootstrap_mean_mixture"]
    matrix = np.zeros((len(evidence), len(samplers)), dtype=float)
    close = np.zeros_like(matrix, dtype=bool)
    best_indices = np.zeros(len(evidence), dtype=int)
    labels: list[str] = []
    weights: list[dict[str, float]] = []
    for i, row in enumerate(evidence):
        payload = load_json(Path(row["path"]))
        summary = payload["summary"]["samplers"]
        means = np.asarray([float(summary[sampler]["mean_objective"]) for sampler in samplers], dtype=float)
        best = int(np.argmin(means))
        denom = max(float(row.get("split_variability", np.nan)), 1e-9)
        values = (means - means[best]) / denom
        matrix[i] = values
        close[i] = values <= 1.0
        best_indices[i] = best
        labels.append(row["short"])
        weights.append(row["selected_weights"])
    return matrix, close, best_indices, labels, weights


def draw_relative_evidence_heatmap(
    ax: plt.Axes,
    cax: plt.Axes,
    matrix: np.ndarray,
    close: np.ndarray,
    best_indices: np.ndarray,
    labels: list[str],
) -> None:
    samplers = ["gaussian_bayes", "knn_conditional", "hybrid", "pair_rerank", "pooled_validation_mixture", "bootstrap_mean_mixture"]
    cap = 1.5
    image = ax.imshow(np.clip(matrix, 0.0, cap), cmap="Greys_r", vmin=0.0, vmax=cap, aspect="auto")
    ax.set_xticks(np.arange(len(samplers)), ["velocity", "archive", "learned", "pair", "pooled\nmix", "bootstrap\nmix"], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.tick_params(length=0)
    ax.set_title(r"relative held-out objective, $\Delta J/\sigma_{split}$", loc="left", fontweight="normal", color=COLORS["ink"], pad=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = float(matrix[i, j])
            shown = min(value, cap)
            label = f"{value:.1f}" if value < cap else f">{cap:.1f}"
            text_color = "white" if shown < 0.45 else COLORS["ink"]
            weight = "bold" if j == best_indices[i] else "normal"
            ax.text(j, i, label, ha="center", va="center", fontsize=6.3, color=text_color, fontweight=weight)
            if close[i, j] and j != best_indices[i]:
                ax.scatter(j + 0.31, i - 0.29, s=13, facecolor="none", edgecolor=text_color, linewidth=0.7)
        ax.add_patch(Rectangle((best_indices[i] - 0.5, i - 0.5), 1.0, 1.0, fill=False, edgecolor=COLORS["ink"], linewidth=1.0))
    ax.set_xlim(-0.5, len(samplers) - 0.5)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = plt.colorbar(image, cax=cax)
    cbar.set_label(r"$\Delta J/\sigma_{split}$", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6.1, length=2.0)
    cbar.outline.set_linewidth(0.45)
    best_handle = Rectangle((0, 0), 1, 1, fill=False, edgecolor=COLORS["ink"], linewidth=1.0, label="lowest mean")
    close_handle = mpl.lines.Line2D([0], [0], marker="o", lw=0, markersize=4.0, markerfacecolor="none", markeredgecolor=COLORS["ink"], label="within 1 split SD")
    ax.legend(handles=[best_handle, close_handle], frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.54, -0.24), columnspacing=1.2)


def draw_relative_evidence_weights(ax: plt.Axes, weights: list[dict[str, float]], labels: list[str]) -> None:
    y = np.arange(len(labels), dtype=float)
    left = np.zeros(len(labels), dtype=float)
    for comp in COMPONENTS:
        values = np.asarray([float(row[comp]) for row in weights], dtype=float)
        ax.barh(y, values, left=left, height=0.58, color=COLORS[comp], edgecolor="white", linewidth=0.55, label=LEGEND_LABELS[comp])
        for yi, start, value in zip(y, left, values):
            if value >= 0.15:
                ax.text(start + value / 2, yi, f"{value:.2f}", ha="center", va="center", fontsize=5.6, color="white")
        left += values
    ax.set_xlim(0, 1)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_yticks(y, [""] * len(labels))
    ax.set_xlabel("weight")
    ax.set_title("pooled-validation mixture", loc="left", fontweight="normal", color=COLORS["ink"], pad=8)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.65)
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.50, -0.24), columnspacing=0.8, handlelength=1.0)


def draw_weight_strip(ax: plt.Axes, x: float, y: float, w: float, h: float, weights: dict[str, float]) -> None:
    left = x
    for comp in COMPONENTS:
        val = float(weights[comp])
        if val > 0:
            ax.add_patch(Rectangle((left, y), w * val, h, transform=ax.transAxes, facecolor=COLORS[comp], edgecolor="white", linewidth=0.7))
        left += w * val
    ax.add_patch(Rectangle((x, y), w, h, transform=ax.transAxes, fill=False, edgecolor="#334155", linewidth=0.65))


def draw_memory_legend(ax: plt.Axes, x: float, y: float) -> None:
    legend_text = {
        "gaussian_bayes": "velocity continuity",
        "knn_conditional": "archive proximity",
        "hybrid": "learned context",
        "pair_rerank": "pair organization",
    }
    for i, comp in enumerate(COMPONENTS):
        xx = x + i * 0.185
        ax.add_patch(Rectangle((xx, y), 0.026, 0.022, transform=ax.transAxes, facecolor=COLORS[comp], edgecolor="none"))
        ax.text(xx + 0.032, y + 0.011, legend_text[comp], va="center", fontsize=6.75, color="#334155", transform=ax.transAxes)


if __name__ == "__main__":
    main()

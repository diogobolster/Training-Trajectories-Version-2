from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tta_v2 import (  # noqa: E402
    MetricSettings,
    compare_metrics,
    evaluate_ensemble,
    load_trajectories,
    velocity_autocorrelation,
)


LABELS = {
    "graph": "Core2 graph flow",
    "openfoam": "Core2 OpenFOAM",
}
COLORS = {
    "graph": "#2563eb",
    "openfoam": "#dc2626",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Core2 graph-flow and OpenFOAM-derived trajectory physics."
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=ROOT
        / "data"
        / "processed"
        / "bentheimer_core2_subvol1_6um_downsample3_D001_trajectories.npz",
    )
    parser.add_argument(
        "--openfoam",
        type=Path,
        default=ROOT
        / "data"
        / "processed"
        / "bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_trajectories.npz",
    )
    parser.add_argument("--planes", default="6,10,14")
    parser.add_argument("--time-indices", default="100,200,300,400")
    parser.add_argument("--bin-size", type=float, default=3.0)
    parser.add_argument("--pair-samples", type=int, default=5000)
    parser.add_argument("--reaction-radius", type=float, default=3.0)
    parser.add_argument("--max-lag", type=int, default=80)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "core2_graph_vs_openfoam_physics_comparison.json",
    )
    parser.add_argument("--figure-prefix", default="run_012_core2_graph_vs_openfoam")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    planes = parse_float_list(args.planes)
    time_indices = parse_int_list(args.time_indices)
    settings = MetricSettings(
        planes=planes,
        time_indices=time_indices,
        bin_size=args.bin_size,
        pair_samples=args.pair_samples,
        reaction_radius=args.reaction_radius,
        seed=args.seed,
    )
    graph = load_trajectories(args.graph)
    openfoam = load_trajectories(args.openfoam)
    graph_metrics = evaluate_ensemble(graph, settings)
    openfoam_metrics = evaluate_ensemble(openfoam, settings)
    errors_graph_reference = compare_metrics(
        graph_metrics,
        openfoam_metrics,
        missing_penalty=settings.missing_penalty,
    )
    errors_openfoam_reference = compare_metrics(
        openfoam_metrics,
        graph_metrics,
        missing_penalty=settings.missing_penalty,
    )

    graph_speeds = step_speeds(graph)
    openfoam_speeds = step_speeds(openfoam)
    graph_axial = step_axial_increments(graph)
    openfoam_axial = step_axial_increments(openfoam)
    graph_corr = velocity_autocorrelation(graph, max_lag=args.max_lag)
    openfoam_corr = velocity_autocorrelation(openfoam, max_lag=args.max_lag)

    payload = {
        "inputs": {
            "graph": str(args.graph),
            "openfoam": str(args.openfoam),
        },
        "settings": {
            "planes": planes,
            "time_indices": time_indices,
            "bin_size": args.bin_size,
            "pair_samples": args.pair_samples,
            "reaction_radius": args.reaction_radius,
            "max_lag": args.max_lag,
            "seed": args.seed,
        },
        "trajectory_summary": {
            "graph": trajectory_summary(graph),
            "openfoam": trajectory_summary(openfoam),
        },
        "step_displacement_summary": {
            "graph": distribution_summary(graph_speeds),
            "openfoam": distribution_summary(openfoam_speeds),
        },
        "axial_increment_summary": {
            "graph": signed_distribution_summary(graph_axial),
            "openfoam": signed_distribution_summary(openfoam_axial),
        },
        "metrics": {
            "graph": graph_metrics,
            "openfoam": openfoam_metrics,
        },
        "metric_difference": {
            "graph_as_reference": errors_graph_reference,
            "openfoam_as_reference": errors_openfoam_reference,
        },
        "velocity_autocorrelation": {
            "lags": list(range(args.max_lag + 1)),
            "graph": float_list(graph_corr),
            "openfoam": float_list(openfoam_corr),
        },
        "derived_ratios": derived_ratios(graph_speeds, openfoam_speeds, graph_metrics, openfoam_metrics),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        f"{args.figure_prefix}_btc_speed.svg": btc_speed_figure(payload),
        f"{args.figure_prefix}_mixing_memory.svg": mixing_memory_figure(payload),
    }
    for name, svg in figures.items():
        path = args.figure_dir / name
        path.write_text(svg, encoding="utf-8")
        print(path)
    print(args.output)


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def step_speeds(trajectories: list[np.ndarray]) -> np.ndarray:
    increments = np.concatenate([np.diff(np.asarray(t), axis=0) for t in trajectories], axis=0)
    return np.linalg.norm(increments, axis=1)


def step_axial_increments(trajectories: list[np.ndarray]) -> np.ndarray:
    increments = np.concatenate([np.diff(np.asarray(t), axis=0) for t in trajectories], axis=0)
    return increments[:, 0]


def trajectory_summary(trajectories: list[np.ndarray]) -> dict[str, float]:
    lengths = np.asarray([len(t) for t in trajectories], dtype=float)
    final_positions = np.stack([np.asarray(t)[-1] for t in trajectories])
    return {
        "n_trajectories": int(len(trajectories)),
        "mean_length": float(np.mean(lengths)),
        "min_length": float(np.min(lengths)),
        "max_length": float(np.max(lengths)),
        "final_x_mean": float(np.mean(final_positions[:, 0])),
        "final_x_median": float(np.median(final_positions[:, 0])),
        "final_x_q10": float(np.quantile(final_positions[:, 0], 0.1)),
        "final_x_q90": float(np.quantile(final_positions[:, 0], 0.9)),
    }


def distribution_summary(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    finite = finite[finite >= 0]
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "q10": float(np.quantile(finite, 0.1)),
        "q50": float(np.quantile(finite, 0.5)),
        "q90": float(np.quantile(finite, 0.9)),
        "q95": float(np.quantile(finite, 0.95)),
        "q99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
    }


def signed_distribution_summary(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "q10": float(np.quantile(finite, 0.1)),
        "q50": float(np.quantile(finite, 0.5)),
        "q90": float(np.quantile(finite, 0.9)),
        "positive_fraction": float(np.mean(finite > 0.0)),
        "negative_fraction": float(np.mean(finite < 0.0)),
    }


def derived_ratios(
    graph_speeds: np.ndarray,
    openfoam_speeds: np.ndarray,
    graph_metrics: dict[str, object],
    openfoam_metrics: dict[str, object],
) -> dict[str, float]:
    graph_speed_stats = distribution_summary(graph_speeds)
    openfoam_speed_stats = distribution_summary(openfoam_speeds)
    graph_reaction = graph_metrics["reaction"]["probability"]
    openfoam_reaction = openfoam_metrics["reaction"]["probability"]
    return {
        "mean_step_speed_openfoam_over_graph": safe_ratio(
            openfoam_speed_stats["mean"],
            graph_speed_stats["mean"],
        ),
        "q95_step_speed_openfoam_over_graph": safe_ratio(
            openfoam_speed_stats["q95"],
            graph_speed_stats["q95"],
        ),
        "reaction_probability_openfoam_minus_graph": float(openfoam_reaction - graph_reaction),
        "reaction_probability_openfoam_over_graph": safe_ratio(openfoam_reaction, graph_reaction),
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0 or not np.isfinite(denominator):
        return float("nan")
    return float(numerator / denominator)


def float_list(values: np.ndarray) -> list[float | None]:
    out: list[float | None] = []
    for value in values:
        out.append(float(value) if np.isfinite(value) else None)
    return out


def btc_speed_figure(payload: dict[str, object]) -> str:
    width, height = 1060, 560
    body = svg_header(width, height, "OpenFOAM changes displacement tails and breakthrough")
    body += text(78, 50, "Particle-step displacement distributions and breakthrough quantiles from the two reference trajectory sets.", "subtitle")
    body += speed_cdf_panel(payload, x0=72, y0=92, w=440, h=360)
    body += btc_panel(payload, x0=590, y0=92, w=390, h=360)
    body += legend(796, 70)
    return body + "</svg>\n"


def mixing_memory_figure(payload: dict[str, object]) -> str:
    width, height = 1120, 760
    body = svg_header(width, height, "OpenFOAM preserves stronger velocity memory")
    body += text(78, 50, "Dilution, particle-pair separation, velocity memory, and reaction encounter probability.", "subtitle")
    body += dilution_panel(payload, x0=72, y0=92, w=430, h=260)
    body += pair_panel(payload, x0=610, y0=92, w=430, h=260)
    body += autocorr_panel(payload, x0=72, y0=430, w=430, h=250)
    body += reaction_panel(payload, x0=610, y0=430, w=430, h=250)
    body += legend(842, 70)
    return body + "</svg>\n"


def speed_cdf_panel(payload: dict[str, object], *, x0: int, y0: int, w: int, h: int) -> str:
    graph = synthetic_quantiles(payload["step_displacement_summary"]["graph"])
    openfoam = synthetic_quantiles(payload["step_displacement_summary"]["openfoam"])
    all_x = [value for value, _ in graph + openfoam if value > 0]
    lo = math.log10(max(min(all_x), 1e-9))
    hi = math.log10(max(all_x))

    def sx(value: float) -> float:
        return x0 + (math.log10(max(value, 1e-9)) - lo) / (hi - lo) * w

    def sy(value: float) -> float:
        return y0 + (1.0 - value) * h

    body = panel_frame(x0, y0, w, h, "step displacement quantiles")
    for frac in (0.25, 0.5, 0.75):
        y = sy(frac)
        body += grid_line(x0, y, x0 + w, y)
        body += tick_text(x0 - 8, y + 4, f"{frac:.2f}", anchor="end")
    for key, points in (("graph", graph), ("openfoam", openfoam)):
        path = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        body += f'<polyline fill="none" stroke="{COLORS[key]}" stroke-width="2.6" points="{path}" />\n'
    for tick in nice_log_ticks(lo, hi):
        x = sx(tick)
        body += grid_line(x, y0, x, y0 + h)
        body += tick_text(x, y0 + h + 20, f"{tick:.2f}", anchor="middle")
    body += axis_labels(x0, y0, w, h, "step displacement (voxels)", "probability")
    return body


def btc_panel(payload: dict[str, object], *, x0: int, y0: int, w: int, h: int) -> str:
    planes = [float(item) for item in payload["settings"]["planes"]]
    graph = payload["metrics"]["graph"]["breakthrough"]
    openfoam = payload["metrics"]["openfoam"]["breakthrough"]
    values: list[float] = []
    for metrics in (graph, openfoam):
        for plane in planes:
            stats = metrics[str(plane)] if str(plane) in metrics else metrics[plane]
            values.extend([stats["q10"], stats["q50"], stats["q90"]])
    values = [value for value in values if value is not None and np.isfinite(value)]
    lo, hi = 0.0, max(values) * 1.08

    def sx(idx: int, offset: float) -> float:
        return x0 + (idx + 0.5 + offset) * w / len(planes)

    def sy(value: float) -> float:
        return y0 + (1.0 - (value - lo) / (hi - lo)) * h

    body = panel_frame(x0, y0, w, h, "breakthrough quantiles")
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = lo + frac * (hi - lo)
        y = sy(value)
        body += grid_line(x0, y, x0 + w, y)
        body += tick_text(x0 - 8, y + 4, f"{value:.0f}", anchor="end")
    for idx, plane in enumerate(planes):
        for key, offset in (("graph", -0.13), ("openfoam", 0.13)):
            metrics = payload["metrics"][key]["breakthrough"]
            stats = metrics[str(plane)] if str(plane) in metrics else metrics[plane]
            x = sx(idx, offset)
            body += f'<line x1="{x:.1f}" y1="{sy(stats["q10"]):.1f}" x2="{x:.1f}" y2="{sy(stats["q90"]):.1f}" stroke="{COLORS[key]}" stroke-width="2.2" />\n'
            body += f'<circle cx="{x:.1f}" cy="{sy(stats["q50"]):.1f}" r="5" fill="{COLORS[key]}" stroke="#ffffff" stroke-width="1.2" />\n'
        body += tick_text(sx(idx, 0.0), y0 + h + 20, f"x={plane:g}", anchor="middle")
    body += axis_labels(x0, y0, w, h, "control plane", "time index")
    return body


def dilution_panel(payload: dict[str, object], *, x0: int, y0: int, w: int, h: int) -> str:
    times = [int(item) for item in payload["settings"]["time_indices"]]
    values = []
    for key in ("graph", "openfoam"):
        for time in times:
            values.append(get_time_stats(payload, key, "dilution", time)["dilution_index"])
    lo = min(values) * 0.95
    hi = max(values) * 1.05
    return line_panel(
        payload,
        x0=x0,
        y0=y0,
        w=w,
        h=h,
        title="dilution index",
        y_key=("dilution", "dilution_index"),
        times=times,
        lo=lo,
        hi=hi,
        y_label="index",
    )


def pair_panel(payload: dict[str, object], *, x0: int, y0: int, w: int, h: int) -> str:
    times = [int(item) for item in payload["settings"]["time_indices"]]
    values = []
    for key in ("graph", "openfoam"):
        for time in times:
            values.append(get_time_stats(payload, key, "pair_separation", time)["q50"])
    lo = min(values) * 0.95
    hi = max(values) * 1.05
    return line_panel(
        payload,
        x0=x0,
        y0=y0,
        w=w,
        h=h,
        title="pair separation median",
        y_key=("pair_separation", "q50"),
        times=times,
        lo=lo,
        hi=hi,
        y_label="distance",
    )


def autocorr_panel(payload: dict[str, object], *, x0: int, y0: int, w: int, h: int) -> str:
    lags = payload["velocity_autocorrelation"]["lags"]
    values = [
        value
        for key in ("graph", "openfoam")
        for value in payload["velocity_autocorrelation"][key]
        if value is not None
    ]
    lo = min(-0.05, min(values))
    hi = max(1.0, max(values))

    def sx(lag: float) -> float:
        return x0 + lag / max(lags) * w

    def sy(value: float) -> float:
        return y0 + (1.0 - (value - lo) / (hi - lo)) * h

    body = panel_frame(x0, y0, w, h, "axial velocity autocorrelation")
    for frac in (0.0, 0.5, 1.0):
        value = lo + frac * (hi - lo)
        y = sy(value)
        body += grid_line(x0, y, x0 + w, y)
        body += tick_text(x0 - 8, y + 4, f"{value:.2f}", anchor="end")
    for key in ("graph", "openfoam"):
        path_points = []
        for lag, value in zip(lags, payload["velocity_autocorrelation"][key]):
            if value is not None:
                path_points.append(f"{sx(lag):.1f},{sy(value):.1f}")
        body += f'<polyline fill="none" stroke="{COLORS[key]}" stroke-width="2.4" points="{" ".join(path_points)}" />\n'
    for lag in (0, 20, 40, 60, 80):
        if lag <= max(lags):
            body += tick_text(sx(lag), y0 + h + 20, str(lag), anchor="middle")
    body += axis_labels(x0, y0, w, h, "lag", "corr.")
    return body


def reaction_panel(payload: dict[str, object], *, x0: int, y0: int, w: int, h: int) -> str:
    values = {
        key: payload["metrics"][key]["reaction"]["probability"]
        for key in ("graph", "openfoam")
    }
    hi = max(values.values()) * 1.25 if max(values.values()) > 0 else 1.0

    def sy(value: float) -> float:
        return y0 + (1.0 - value / hi) * h

    body = panel_frame(x0, y0, w, h, "reaction encounter probability")
    for frac in (0.0, 0.5, 1.0):
        value = frac * hi
        y = sy(value)
        body += grid_line(x0, y, x0 + w, y)
        body += tick_text(x0 - 8, y + 4, f"{value:.3f}", anchor="end")
    bar_w = w * 0.18
    centers = {
        "graph": x0 + w * 0.38,
        "openfoam": x0 + w * 0.62,
    }
    for key, x in centers.items():
        y = sy(values[key])
        body += f'<rect x="{x - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{y0 + h - y:.1f}" fill="{COLORS[key]}" rx="2" />\n'
        body += tick_text(x, y - 8, f"{values[key]:.3f}", anchor="middle")
        body += tick_text(x, y0 + h + 20, key, anchor="middle")
    body += axis_labels(x0, y0, w, h, "", "prob.")
    return body


def line_panel(
    payload: dict[str, object],
    *,
    x0: int,
    y0: int,
    w: int,
    h: int,
    title: str,
    y_key: tuple[str, str],
    times: list[int],
    lo: float,
    hi: float,
    y_label: str,
) -> str:
    def sx(time: int) -> float:
        return x0 + (time - min(times)) / (max(times) - min(times)) * w

    def sy(value: float) -> float:
        return y0 + (1.0 - (value - lo) / (hi - lo)) * h

    body = panel_frame(x0, y0, w, h, title)
    for frac in (0.0, 0.5, 1.0):
        value = lo + frac * (hi - lo)
        y = sy(value)
        body += grid_line(x0, y, x0 + w, y)
        body += tick_text(x0 - 8, y + 4, f"{value:.1f}", anchor="end")
    for key in ("graph", "openfoam"):
        points = []
        for time in times:
            value = get_time_stats(payload, key, y_key[0], time)[y_key[1]]
            points.append(f"{sx(time):.1f},{sy(value):.1f}")
        body += f'<polyline fill="none" stroke="{COLORS[key]}" stroke-width="2.4" points="{" ".join(points)}" />\n'
        for time in times:
            value = get_time_stats(payload, key, y_key[0], time)[y_key[1]]
            body += f'<circle cx="{sx(time):.1f}" cy="{sy(value):.1f}" r="3.8" fill="{COLORS[key]}" stroke="#ffffff" stroke-width="1" />\n'
    for time in times:
        body += tick_text(sx(time), y0 + h + 20, str(time), anchor="middle")
    body += axis_labels(x0, y0, w, h, "time index", y_label)
    return body


def synthetic_quantiles(summary: dict[str, float]) -> list[tuple[float, float]]:
    anchors = [
        (summary["q10"], 0.10),
        (summary["q50"], 0.50),
        (summary["q90"], 0.90),
        (summary["q95"], 0.95),
        (summary["q99"], 0.99),
    ]
    return anchors


def get_time_stats(payload: dict[str, object], key: str, section: str, time: int) -> dict[str, float]:
    items = payload["metrics"][key][section]
    return items.get(str(time), items[time])


def nice_log_ticks(lo: float, hi: float) -> list[float]:
    values = []
    start = math.floor(lo)
    end = math.ceil(hi)
    for exp in range(start, end + 1):
        for mantissa in (1, 2, 5):
            value = mantissa * 10**exp
            if lo <= math.log10(value) <= hi:
                values.append(value)
    return values[:8]


def svg_header(width: int, height: int, title: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  .title {{ font: 700 20px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }}
  .subtitle {{ font: 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #4b5563; }}
  .panel-title {{ font: 700 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }}
  .axis {{ stroke: #374151; stroke-width: 1.1; }}
  .grid {{ stroke: #e5e7eb; stroke-width: 1; }}
  .tick {{ font: 10.5px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #4b5563; }}
  .label {{ font: 700 11.5px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #374151; }}
  .legend {{ font: 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #374151; }}
</style>
<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" stroke="#fdfdfd" stroke-width="1" />
<rect x="{width - 1}" y="{height - 1}" width="1" height="1" fill="#f8f8f8" />
<text x="{width / 2:.1f}" y="30" text-anchor="middle" class="title">{esc(title)}</text>
'''


def panel_frame(x0: int, y0: int, w: int, h: int, title: str) -> str:
    return (
        f'<text x="{x0}" y="{y0 - 14}" class="panel-title">{esc(title)}</text>\n'
        f'<line x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}" class="axis" />\n'
        f'<line x1="{x0}" y1="{y0 + h}" x2="{x0}" y2="{y0}" class="axis" />\n'
    )


def grid_line(x1: float, y1: float, x2: float, y2: float) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="grid" />\n'


def tick_text(x: float, y: float, value: str, *, anchor: str) -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" class="tick" text-anchor="{anchor}">{esc(value)}</text>\n'


def axis_labels(x0: int, y0: int, w: int, h: int, x_label: str, y_label: str) -> str:
    body = ""
    if x_label:
        body += f'<text x="{x0 + w / 2:.1f}" y="{y0 + h + 42:.1f}" class="label" text-anchor="middle">{esc(x_label)}</text>\n'
    if y_label:
        cy = y0 + h / 2
        body += f'<text x="{x0 - 52:.1f}" y="{cy:.1f}" class="label" text-anchor="middle" transform="rotate(-90 {x0 - 52:.1f},{cy:.1f})">{esc(y_label)}</text>\n'
    return body


def legend(x: int, y: int) -> str:
    body = ""
    for idx, key in enumerate(("graph", "openfoam")):
        yy = y + idx * 24
        body += f'<line x1="{x}" y1="{yy}" x2="{x + 28}" y2="{yy}" stroke="{COLORS[key]}" stroke-width="3" />\n'
        body += f'<text x="{x + 38}" y="{yy + 4}" class="legend">{esc(LABELS[key])}</text>\n'
    return body


def text(x: float, y: float, value: str, class_name: str) -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{class_name}">{esc(value)}</text>\n'


def esc(value: object) -> str:
    return html.escape(str(value))


if __name__ == "__main__":
    main()

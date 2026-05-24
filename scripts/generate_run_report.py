from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

COLORS = {
    "reference": "#111827",
    "unconditional": "#9ca3af",
    "knn_conditional": "#2563eb",
    "gaussian_bayes": "#dc2626",
    "adaptive_gaussian": "#f59e0b",
    "contrastive": "#7c3aed",
    "hybrid": "#059669",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Run 001 Markdown report and SVG figures.")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=ROOT / "outputs" / "bentheimer_6um_downsample3_full_metrics_adaptive.json",
    )
    parser.add_argument("--title", default="Run 001: Bentheimer TTA-v2 Baselines")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "run_001_report.md")
    parser.add_argument("--asset-dir", type=Path, default=ROOT / "outputs" / "run_001_assets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.metrics.read_text(encoding="utf-8"))
    args.asset_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "btc": args.asset_dir / "btc_median.svg",
        "dilution": args.asset_dir / "dilution_index.svg",
        "pair": args.asset_dir / "pair_separation_median.svg",
        "reaction": args.asset_dir / "reaction_probability.svg",
        "errors": args.asset_dir / "metric_errors.svg",
    }

    write_text(
        figures["btc"],
        line_chart(
            title="Breakthrough Median By Control Plane",
            x_label="x control plane (voxels)",
            y_label="median first-passage time (steps)",
            series=breakthrough_series(data),
        ),
    )
    write_text(
        figures["dilution"],
        line_chart(
            title="Dilution Index",
            x_label="time step",
            y_label="occupancy entropy volume",
            series=dilution_series(data),
        ),
    )
    write_text(
        figures["pair"],
        line_chart(
            title="Particle-Pair Median Separation",
            x_label="time step",
            y_label="median separation (voxels)",
            series=pair_series(data),
        ),
    )
    write_text(
        figures["reaction"],
        bar_chart(
            title="Reaction Encounter Probability",
            y_label="probability",
            values=reaction_values(data),
        ),
    )
    write_text(
        figures["errors"],
        grouped_bar_chart(
            title="Sampler Errors",
            values=error_values(data),
        ),
    )

    report = build_report(data, args.metrics, figures, args.output, args.title)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote report: {args.output}")
    for name, path in figures.items():
        print(f"Wrote {name}: {path}")


def breakthrough_series(data: dict) -> dict[str, list[tuple[float, float]]]:
    series = {"reference": metric_curve(data["reference"], "breakthrough", "q50")}
    for name, payload in data["generated"].items():
        series[name] = metric_curve(payload["metrics"], "breakthrough", "q50")
    return series


def dilution_series(data: dict) -> dict[str, list[tuple[float, float]]]:
    series = {"reference": metric_curve(data["reference"], "dilution", "dilution_index")}
    for name, payload in data["generated"].items():
        series[name] = metric_curve(payload["metrics"], "dilution", "dilution_index")
    return series


def pair_series(data: dict) -> dict[str, list[tuple[float, float]]]:
    series = {"reference": metric_curve(data["reference"], "pair_separation", "q50")}
    for name, payload in data["generated"].items():
        series[name] = metric_curve(payload["metrics"], "pair_separation", "q50")
    return series


def reaction_values(data: dict) -> dict[str, float]:
    values = {"reference": data["reference"]["reaction"]["probability"]}
    for name, payload in data["generated"].items():
        values[name] = payload["metrics"]["reaction"]["probability"]
    return values


def error_values(data: dict) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    for name, payload in data["generated"].items():
        errors = payload["errors"]
        values[name] = {
            "BTC": errors["btc_score"],
            "pair": errors["pair_quantile_mae"] * 20.0,
            "dilution": errors["dilution_log_mae"] * 300.0,
            "reaction": errors["reaction_abs_error"] * 2000.0,
        }
    return values


def metric_curve(metrics: dict, metric_name: str, value_key: str) -> list[tuple[float, float]]:
    curve = []
    for key, stats in sorted(metrics[metric_name].items(), key=lambda item: float(item[0])):
        value = stats[value_key]
        if value == value:
            curve.append((float(key), float(value)))
    return curve


def build_report(
    data: dict,
    metrics_path: Path,
    figures: dict[str, Path],
    output_path: Path,
    title: str,
) -> str:
    rel_figures = {name: relative_path(path, output_path.parent) for name, path in figures.items()}
    archive = data["archive"]
    settings = data["settings"]
    rows = []
    for name, payload in sorted(data["generated"].items(), key=lambda item: item[1]["errors"]["btc_score"]):
        errors = payload["errors"]
        rows.append(
            "| "
            + " | ".join(
                [
                    name,
                    f"{errors['btc_score']:.2f}",
                    f"{errors['dilution_log_mae']:.3f}",
                    f"{errors['pair_quantile_mae']:.2f}",
                    f"{errors['reaction_abs_error']:.3f}",
                ]
            )
            + " |"
        )

    best_btc = min(data["generated"], key=lambda name: data["generated"][name]["errors"]["btc_score"])
    best_pair = min(data["generated"], key=lambda name: data["generated"][name]["errors"]["pair_quantile_mae"])
    best_dilution = min(data["generated"], key=lambda name: data["generated"][name]["errors"]["dilution_log_mae"])

    return "\n".join(
        [
            f"# {title}",
            "",
            "## Summary",
            "",
            "This run uses the 6 micrometer Bentheimer sandstone CT volume, block-averaged to a 75^3 bootstrap simulation, then evaluates trajectory samplers against held-out particle trajectories.",
            "",
            f"- metrics file: `{relative_path(metrics_path, output_path.parent)}`",
            f"- train trajectories: `{data['n_train']}`",
            f"- reference trajectories: `{data['n_reference']}`",
            f"- archive segments: `{archive['size']}`",
            f"- segment steps: `{archive['segment_steps']}`",
            f"- match steps: `{archive['match_steps']}`",
            f"- planes: `{settings['planes']}`",
            f"- time indices: `{settings['time_indices']}`",
            "",
            "## Main Result",
            "",
            f"- Best BTC score: `{best_btc}`",
            f"- Best pair-separation score: `{best_pair}`",
            f"- Best dilution score: `{best_dilution}`",
            "",
            "The tuned Gaussian/Bayes kernel remains the strongest overall baseline. The learned and adaptive variants are informative, but they do not yet beat the physics-informed seam kernel on the most transport-specific metrics.",
            "",
            "## Metric Table",
            "",
            "| sampler | BTC score | dilution log MAE | pair MAE | reaction abs error |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            "## Figures",
            "",
            f"![BTC median]({rel_figures['btc']})",
            "",
            f"![Dilution index]({rel_figures['dilution']})",
            "",
            f"![Pair separation]({rel_figures['pair']})",
            "",
            f"![Reaction probability]({rel_figures['reaction']})",
            "",
            f"![Metric errors]({rel_figures['errors']})",
            "",
            "## Interpretation",
            "",
            "The old TTA Gaussian/Bayes transition rule is not just a historical baseline; it is a strong inductive bias. The first learned transition models improve isolated metrics in places, but they are too local and can damage pair dynamics. The next useful model should optimize short-horizon or multi-step transport consequences rather than one-step continuation plausibility.",
            "",
        ]
    )


def line_chart(
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: dict[str, list[tuple[float, float]]],
    width: int = 860,
    height: int = 480,
) -> str:
    margin = {"left": 78, "right": 180, "top": 48, "bottom": 72}
    points = [point for values in series.values() for point in values]
    x_min, x_max = bounds([x for x, _ in points])
    y_min, y_max = bounds([y for _, y in points], pad_fraction=0.08)

    def sx(x: float) -> float:
        return margin["left"] + (x - x_min) / (x_max - x_min) * (width - margin["left"] - margin["right"])

    def sy(y: float) -> float:
        return height - margin["bottom"] - (y - y_min) / (y_max - y_min) * (height - margin["top"] - margin["bottom"])

    body = svg_header(width, height, title)
    body += axes(width, height, margin, x_label, y_label)
    body += ticks(x_min, x_max, y_min, y_max, sx, sy, width, height, margin)
    legend_y = margin["top"]
    for idx, (name, values) in enumerate(series.items()):
        color = COLORS.get(name, "#374151")
        if len(values) >= 2:
            path = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in values)
            body += f'<polyline fill="none" stroke="{color}" stroke-width="2.6" points="{path}" />\n'
        for x, y in values:
            body += f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.2" fill="{color}" />\n'
        ly = legend_y + idx * 24
        body += f'<line x1="{width - 158}" y1="{ly}" x2="{width - 132}" y2="{ly}" stroke="{color}" stroke-width="3" />\n'
        body += f'<text x="{width - 124}" y="{ly + 4}" class="legend">{escape(name)}</text>\n'
    return body + "</svg>\n"


def bar_chart(
    *,
    title: str,
    y_label: str,
    values: dict[str, float],
    width: int = 860,
    height: int = 430,
) -> str:
    margin = {"left": 78, "right": 28, "top": 48, "bottom": 104}
    labels = list(values)
    max_value = max(values.values()) * 1.15
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    bar_w = plot_w / max(len(labels), 1) * 0.64

    def sy(y: float) -> float:
        return height - margin["bottom"] - y / max_value * plot_h

    body = svg_header(width, height, title)
    body += axes(width, height, margin, "", y_label)
    for idx, label in enumerate(labels):
        x = margin["left"] + (idx + 0.5) * plot_w / len(labels)
        y = sy(values[label])
        color = COLORS.get(label, "#374151")
        body += f'<rect x="{x - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{height - margin["bottom"] - y:.1f}" fill="{color}" />\n'
        body += f'<text x="{x:.1f}" y="{height - margin["bottom"] + 18}" class="tick" text-anchor="middle" transform="rotate(35 {x:.1f},{height - margin["bottom"] + 18})">{escape(label)}</text>\n'
        body += f'<text x="{x:.1f}" y="{y - 7:.1f}" class="tick" text-anchor="middle">{values[label]:.3f}</text>\n'
    return body + "</svg>\n"


def grouped_bar_chart(
    *,
    title: str,
    values: dict[str, dict[str, float]],
    width: int = 900,
    height: int = 480,
) -> str:
    margin = {"left": 78, "right": 180, "top": 48, "bottom": 104}
    samplers = list(values)
    metrics = list(next(iter(values.values())))
    max_value = max(v for sampler in values.values() for v in sampler.values()) * 1.15
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    group_w = plot_w / len(samplers)
    bar_w = group_w / (len(metrics) + 1)
    metric_colors = {"BTC": "#dc2626", "pair": "#2563eb", "dilution": "#7c3aed", "reaction": "#059669"}

    def sy(y: float) -> float:
        return height - margin["bottom"] - y / max_value * plot_h

    body = svg_header(width, height, title)
    body += axes(width, height, margin, "sampler", "scaled error")
    for s_idx, sampler in enumerate(samplers):
        center = margin["left"] + (s_idx + 0.5) * group_w
        for m_idx, metric in enumerate(metrics):
            x = center - (len(metrics) / 2 - m_idx) * bar_w
            y = sy(values[sampler][metric])
            body += f'<rect x="{x - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w * 0.82:.1f}" height="{height - margin["bottom"] - y:.1f}" fill="{metric_colors[metric]}" />\n'
        body += f'<text x="{center:.1f}" y="{height - margin["bottom"] + 18}" class="tick" text-anchor="middle" transform="rotate(35 {center:.1f},{height - margin["bottom"] + 18})">{escape(sampler)}</text>\n'
    for idx, metric in enumerate(metrics):
        y = margin["top"] + idx * 24
        color = metric_colors[metric]
        body += f'<rect x="{width - 158}" y="{y - 10}" width="18" height="12" fill="{color}" />\n'
        body += f'<text x="{width - 132}" y="{y}" class="legend">{escape(metric)}</text>\n'
    return body + "</svg>\n"


def svg_header(width: int, height: int, title: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  .title {{ font: 700 18px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }}
  .axis {{ stroke: #374151; stroke-width: 1.2; }}
  .grid {{ stroke: #e5e7eb; stroke-width: 1; }}
  .tick {{ font: 11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #4b5563; }}
  .label {{ font: 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #374151; }}
  .legend {{ font: 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #374151; }}
</style>
<rect width="100%" height="100%" fill="#ffffff" />
<text x="{width / 2:.1f}" y="26" text-anchor="middle" class="title">{escape(title)}</text>
'''


def axes(width: int, height: int, margin: dict[str, int], x_label: str, y_label: str) -> str:
    x0 = margin["left"]
    x1 = width - margin["right"]
    y0 = height - margin["bottom"]
    y1 = margin["top"]
    return f'''<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" class="axis" />
<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" class="axis" />
<text x="{(x0 + x1) / 2:.1f}" y="{height - 18}" text-anchor="middle" class="label">{escape(x_label)}</text>
<text x="18" y="{(y0 + y1) / 2:.1f}" text-anchor="middle" class="label" transform="rotate(-90 18,{(y0 + y1) / 2:.1f})">{escape(y_label)}</text>
'''


def ticks(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    sx,
    sy,
    width: int,
    height: int,
    margin: dict[str, int],
) -> str:
    body = ""
    for idx in range(5):
        x = x_min + (x_max - x_min) * idx / 4
        y = y_min + (y_max - y_min) * idx / 4
        body += f'<line x1="{sx(x):.1f}" y1="{margin["top"]}" x2="{sx(x):.1f}" y2="{height - margin["bottom"]}" class="grid" />\n'
        body += f'<text x="{sx(x):.1f}" y="{height - margin["bottom"] + 18}" text-anchor="middle" class="tick">{x:.0f}</text>\n'
        body += f'<line x1="{margin["left"]}" y1="{sy(y):.1f}" x2="{width - margin["right"]}" y2="{sy(y):.1f}" class="grid" />\n'
        body += f'<text x="{margin["left"] - 8}" y="{sy(y) + 4:.1f}" text-anchor="end" class="tick">{y:.1f}</text>\n'
    return body


def bounds(values: list[float], pad_fraction: float = 0.0) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if low == high:
        low -= 1.0
        high += 1.0
    span = high - low
    return low - span * pad_fraction, high + span * pad_fraction


def relative_path(path: Path, start: Path) -> str:
    return os.path.relpath(path.resolve(), start.resolve())


def escape(value: object) -> str:
    return html.escape(str(value))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

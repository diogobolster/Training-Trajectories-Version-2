from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np


SAMPLER_ORDER = [
    "pooled_validation_mixture",
    "gaussian_bayes",
    "hybrid",
    "bootstrap_mean_mixture",
    "knn_conditional",
    "pair_rerank",
]
COMPONENT_ORDER = ["gaussian_bayes", "knn_conditional", "hybrid", "pair_rerank"]
REGIME_ORDER = [
    "balanced",
    "btc_heavy",
    "pair_heavy",
    "dilution_heavy",
    "reaction_light",
    "reaction_heavy",
    "no_reaction",
]
DISPLAY = {
    "pooled_validation_mixture": "pooled mix",
    "bootstrap_mean_mixture": "mean mix",
    "gaussian_bayes": "Gaussian/Bayes",
    "knn_conditional": "kNN",
    "hybrid": "hybrid",
    "pair_rerank": "pair rerank",
    "balanced": "balanced",
    "btc_heavy": "BTC-heavy",
    "pair_heavy": "pair-heavy",
    "dilution_heavy": "dilution-heavy",
    "reaction_light": "reaction-light",
    "reaction_heavy": "reaction-heavy",
    "no_reaction": "no reaction",
}
COLORS = {
    "pooled_validation_mixture": "#2563eb",
    "bootstrap_mean_mixture": "#0f766e",
    "gaussian_bayes": "#111827",
    "knn_conditional": "#d97706",
    "hybrid": "#7c3aed",
    "pair_rerank": "#be123c",
    "btc_score": "#dc2626",
    "pair_quantile_mae": "#2563eb",
    "dilution_log_mae": "#059669",
    "reaction_abs_error": "#7c3aed",
}
COMPONENT_COLORS = {
    "gaussian_bayes": "#111827",
    "knn_conditional": "#d97706",
    "hybrid": "#7c3aed",
    "pair_rerank": "#be123c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate manuscript-style selection figures.")
    parser.add_argument(
        "--run005",
        type=Path,
        default=Path("outputs/bentheimer_6um_downsample3_outer_split_mixture_benchmark.json"),
    )
    parser.add_argument(
        "--run006",
        type=Path,
        default=Path("outputs/bentheimer_6um_downsample3_objective_weight_sensitivity.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--prefix",
        default="run_006",
        help="Filename prefix for objective-sensitivity figures.",
    )
    parser.add_argument("--skip-outer-summary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run005 = load_json(args.run005)
    run006 = load_json(args.run006)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        f"{args.prefix}_weight_sensitivity_heatmap.svg": weight_sensitivity_heatmap(run006),
        f"{args.prefix}_selected_weights.svg": selected_weights_figure(run006),
        f"{args.prefix}_pareto_tradeoff.svg": pareto_tradeoff(run006),
    }
    if not args.skip_outer_summary:
        figures["run_005_outer_split_summary.svg"] = outer_split_summary(run005)
    for filename, svg in figures.items():
        path = args.output_dir / filename
        path.write_text(svg, encoding="utf-8")
        print(path)


def outer_split_summary(data: dict[str, object]) -> str:
    summary = data["summary"]["samplers"]
    rows = [
        (
            sampler,
            summary[sampler]["mean_objective"],
            summary[sampler]["std_objective"],
            summary[sampler]["mean_rank"],
            summary[sampler]["wins"],
            summary[sampler]["beats_gaussian_bayes"],
            summary[sampler]["beats_hybrid"],
        )
        for sampler in SAMPLER_ORDER
    ]
    width, height = 1040, 590
    margin = {"left": 84, "right": 230, "top": 74, "bottom": 124}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    max_value = max(value + std for _, value, std, *_ in rows) * 1.08
    group_w = plot_w / len(rows)
    bar_w = group_w * 0.52

    def sy(value: float) -> float:
        return margin["top"] + (1.0 - value / max_value) * plot_h

    body = svg_header(width, height, "Validation-selected mixtures are competitive")
    body += subtitle(84, 48, "Mean held-out objective over five independent test splits; lower values indicate better multi-objective prediction.")
    body += axes(width, height, margin, "", "mean objective")
    body += y_ticks(margin, width - margin["right"], height - margin["bottom"], 0, max_value, sy)
    for idx, (sampler, mean, std, rank, wins, beats_g, beats_h) in enumerate(rows):
        center = margin["left"] + (idx + 0.5) * group_w
        y = sy(mean)
        color = COLORS[sampler]
        body += (
            f'<rect x="{center - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
            f'height="{height - margin["bottom"] - y:.1f}" fill="{color}" rx="2" />\n'
        )
        err_top = sy(mean + std)
        err_bottom = sy(max(mean - std, 0.0))
        body += f'<line x1="{center:.1f}" y1="{err_top:.1f}" x2="{center:.1f}" y2="{err_bottom:.1f}" stroke="#374151" stroke-width="1.3" />\n'
        body += f'<line x1="{center - 9:.1f}" y1="{err_top:.1f}" x2="{center + 9:.1f}" y2="{err_top:.1f}" stroke="#374151" stroke-width="1.3" />\n'
        body += f'<line x1="{center - 9:.1f}" y1="{err_bottom:.1f}" x2="{center + 9:.1f}" y2="{err_bottom:.1f}" stroke="#374151" stroke-width="1.3" />\n'
        body += f'<text x="{center:.1f}" y="{y - 18:.1f}" class="small" text-anchor="middle">rank {rank:.1f}</text>\n'
        body += f'<text x="{center:.1f}" y="{y - 4:.1f}" class="small" text-anchor="middle">wins {wins}</text>\n'
        label_y = height - margin["bottom"] + 20
        body += (
            f'<text x="{center:.1f}" y="{label_y}" class="tick" text-anchor="end" '
            f'transform="rotate(-33 {center:.1f},{label_y})">{esc(DISPLAY[sampler])}</text>\n'
        )
        body += f'<text x="{center:.1f}" y="{height - 36}" class="small" text-anchor="middle">G {beats_g}/5</text>\n'
        body += f'<text x="{center:.1f}" y="{height - 21}" class="small" text-anchor="middle">H {beats_h}/5</text>\n'

    legend_x = width - 204
    body += legend_title(legend_x, 86, "annotation")
    body += note(legend_x, 112, "G: beats Gaussian/Bayes")
    body += note(legend_x, 134, "H: beats hybrid")
    body += note(legend_x, 166, "Error bars: split std.")
    body += note(legend_x, 188, "Lower objective is better.")
    return body + "</svg>\n"


def weight_sensitivity_heatmap(data: dict[str, object]) -> str:
    summary = data["summary"]
    width, height = 1060, 560
    margin = {"left": 150, "right": 230, "top": 86, "bottom": 82}
    regimes = [name for name in REGIME_ORDER if name in summary]
    samplers = SAMPLER_ORDER
    cell_w = (width - margin["left"] - margin["right"]) / len(samplers)
    cell_h = (height - margin["top"] - margin["bottom"]) / len(regimes)
    ranks = [
        summary[regime]["samplers"][sampler]["mean_rank"]
        for regime in regimes
        for sampler in samplers
    ]
    low, high = min(ranks), max(ranks)

    body = svg_header(width, height, "The preferred mechanism depends on the transport objective")
    body += subtitle(150, 52, "Heatmap values are mean held-out rank across four outer splits; lower ranks are better.")
    for s_idx, sampler in enumerate(samplers):
        x = margin["left"] + (s_idx + 0.5) * cell_w
        body += (
            f'<text x="{x:.1f}" y="{margin["top"] - 18}" class="tick" text-anchor="start" '
            f'transform="rotate(-32 {x:.1f},{margin["top"] - 18})">{esc(DISPLAY[sampler])}</text>\n'
        )
    for r_idx, regime in enumerate(regimes):
        y = margin["top"] + (r_idx + 0.5) * cell_h
        body += f'<text x="{margin["left"] - 14}" y="{y + 4:.1f}" class="label" text-anchor="end">{esc(DISPLAY[regime])}</text>\n'
        for s_idx, sampler in enumerate(samplers):
            rank = summary[regime]["samplers"][sampler]["mean_rank"]
            x0 = margin["left"] + s_idx * cell_w
            y0 = margin["top"] + r_idx * cell_h
            fill = heat_color(rank, low, high)
            body += f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{fill}" stroke="#ffffff" stroke-width="1.5" />\n'
            body += f'<text x="{x0 + cell_w / 2:.1f}" y="{y0 + cell_h / 2 + 5:.1f}" class="cell">{rank:.2f}</text>\n'

    legend_x = width - 184
    legend_y = 128
    body += legend_title(legend_x, 104, "mean rank")
    for idx in range(6):
        value = low + (high - low) * idx / 5
        fill = heat_color(value, low, high)
        y = legend_y + idx * 30
        body += f'<rect x="{legend_x}" y="{y - 16}" width="28" height="22" fill="{fill}" stroke="#ffffff" />\n'
        body += f'<text x="{legend_x + 38}" y="{y}" class="legend">{value:.1f}</text>\n'
    body += note(legend_x, legend_y + 202, "The preferred sampler")
    body += note(legend_x, legend_y + 222, "moves with the metric")
    body += note(legend_x, legend_y + 242, "priority.")
    return body + "</svg>\n"


def selected_weights_figure(data: dict[str, object]) -> str:
    summary = data["summary"]
    regimes = [name for name in REGIME_ORDER if name in summary]
    width, height = 1040, 560
    margin = {"left": 80, "right": 220, "top": 76, "bottom": 136}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    group_w = plot_w / len(regimes)
    bar_w = group_w * 0.62

    body = svg_header(width, height, "Validation changes the mixture weights by objective")
    body += subtitle(80, 50, "Stacked bars show pooled-validation mixture weights selected under each objective regime.")
    body += axes(width, height, margin, "", "weight")
    body += y_ticks(margin, width - margin["right"], height - margin["bottom"], 0.0, 1.0, lambda v: margin["top"] + (1.0 - v) * plot_h)

    for r_idx, regime in enumerate(regimes):
        center = margin["left"] + (r_idx + 0.5) * group_w
        top = height - margin["bottom"]
        weights = summary[regime]["selected_weights"]["pooled_validation_mixture"]["mean"]
        for component in COMPONENT_ORDER:
            value = weights[component]
            segment_h = value * plot_h
            top -= segment_h
            body += (
                f'<rect x="{center - bar_w / 2:.1f}" y="{top:.1f}" width="{bar_w:.1f}" '
                f'height="{segment_h:.1f}" fill="{COMPONENT_COLORS[component]}" />\n'
            )
            if value >= 0.12:
                body += f'<text x="{center:.1f}" y="{top + segment_h / 2 + 4:.1f}" class="stacklabel" text-anchor="middle">{value:.2f}</text>\n'
        label_y = height - margin["bottom"] + 22
        body += (
            f'<text x="{center:.1f}" y="{label_y}" class="tick" text-anchor="end" '
            f'transform="rotate(-33 {center:.1f},{label_y})">{esc(DISPLAY[regime])}</text>\n'
        )

    legend_x = width - 184
    body += legend_title(legend_x, 94, "components")
    for idx, component in enumerate(COMPONENT_ORDER):
        y = 122 + idx * 28
        body += f'<rect x="{legend_x}" y="{y - 14}" width="22" height="16" fill="{COMPONENT_COLORS[component]}" />\n'
        body += f'<text x="{legend_x + 32}" y="{y}" class="legend">{esc(DISPLAY.get(component, component))}</text>\n'
    return body + "</svg>\n"


def pareto_tradeoff(data: dict[str, object]) -> str:
    regime = "balanced"
    outer_results = data["outer_results"]
    samplers = SAMPLER_ORDER
    means = {}
    for sampler in samplers:
        errors = [
            result["regime_results"][regime]["test"][sampler]["errors"]
            for result in outer_results
        ]
        means[sampler] = {
            key: float(np.mean([item[key] for item in errors]))
            for key in [
                "btc_score",
                "pair_quantile_mae",
                "dilution_log_mae",
                "reaction_abs_error",
            ]
        }
        means[sampler]["mean_rank"] = float(data["summary"][regime]["samplers"][sampler]["mean_rank"])

    width, height = 960, 580
    margin = {"left": 86, "right": 210, "top": 76, "bottom": 80}
    x_values = [means[s]["btc_score"] for s in samplers]
    y_values = [means[s]["pair_quantile_mae"] for s in samplers]
    dilution_values = [means[s]["dilution_log_mae"] for s in samplers]
    x_min, x_max = padded_bounds(x_values, 0.12)
    y_min, y_max = padded_bounds(y_values, 0.16)
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    d_min, d_max = min(dilution_values), max(dilution_values)

    def sx(value: float) -> float:
        return margin["left"] + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return height - margin["bottom"] - (value - y_min) / (y_max - y_min) * plot_h

    body = svg_header(width, height, "No sampler dominates all transport metrics")
    body += subtitle(86, 50, "Each point is a sampler averaged over four outer splits; smaller breakthrough and pair errors are better.")
    body += axes(width, height, margin, "BTC error", "pair-separation error")
    body += xy_ticks(margin, width, height, x_min, x_max, y_min, y_max, sx, sy)
    for sampler in samplers:
        item = means[sampler]
        radius = 7 + 16 * (item["dilution_log_mae"] - d_min) / (d_max - d_min + 1e-12)
        x = sx(item["btc_score"])
        y = sy(item["pair_quantile_mae"])
        color = COLORS[sampler]
        body += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="0.82" stroke="#ffffff" stroke-width="1.8" />\n'
        body += f'<text x="{x + radius + 4:.1f}" y="{y + 4:.1f}" class="small">{esc(DISPLAY[sampler])}</text>\n'
    legend_x = width - 178
    body += legend_title(legend_x, 100, "encoding")
    body += note(legend_x, 126, "circle size: dilution")
    body += note(legend_x, 146, "log-MAE")
    body += note(legend_x, 180, "lower-left is better")
    body += note(legend_x, 214, "balanced objective")
    return body + "</svg>\n"


def svg_header(width: int, height: int, title: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  .title {{ font: 700 20px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }}
  .subtitle {{ font: 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #4b5563; }}
  .axis {{ stroke: #374151; stroke-width: 1.2; }}
  .grid {{ stroke: #e5e7eb; stroke-width: 1; }}
  .tick {{ font: 11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #4b5563; }}
  .label {{ font: 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #374151; }}
  .legend {{ font: 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #374151; }}
  .small {{ font: 10.5px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #374151; }}
  .cell {{ font: 700 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; text-anchor: middle; }}
  .stacklabel {{ font: 700 11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #ffffff; }}
</style>
<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" stroke="#fdfdfd" stroke-width="1" />
<rect x="{width - 1}" y="{height - 1}" width="1" height="1" fill="#f8f8f8" />
<text x="{width / 2:.1f}" y="28" text-anchor="middle" class="title">{esc(title)}</text>
'''


def subtitle(x: float, y: float, text: str) -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" class="subtitle">{esc(text)}</text>\n'


def axes(width: int, height: int, margin: dict[str, int], x_label: str, y_label: str) -> str:
    x0 = margin["left"]
    x1 = width - margin["right"]
    y0 = height - margin["bottom"]
    y1 = margin["top"]
    body = f'''<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" class="axis" />
<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" class="axis" />
<text x="20" y="{(y0 + y1) / 2:.1f}" text-anchor="middle" class="label" transform="rotate(-90 20,{(y0 + y1) / 2:.1f})">{esc(y_label)}</text>
'''
    if x_label:
        body += f'<text x="{(x0 + x1) / 2:.1f}" y="{height - 16}" text-anchor="middle" class="label">{esc(x_label)}</text>\n'
    return body


def y_ticks(
    margin: dict[str, int],
    x1: float,
    y0: float,
    y_min: float,
    y_max: float,
    sy,
) -> str:
    body = ""
    fmt = "{:.1f}" if abs(y_max - y_min) <= 1.5 else "{:.0f}"
    for idx in range(6):
        value = y_min + (y_max - y_min) * idx / 5
        y = sy(value)
        body += f'<line x1="{margin["left"]}" y1="{y:.1f}" x2="{x1:.1f}" y2="{y:.1f}" class="grid" />\n'
        body += f'<text x="{margin["left"] - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{fmt.format(value)}</text>\n'
    return body


def xy_ticks(
    margin: dict[str, int],
    width: int,
    height: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    sx,
    sy,
) -> str:
    body = ""
    for idx in range(5):
        x_value = x_min + (x_max - x_min) * idx / 4
        y_value = y_min + (y_max - y_min) * idx / 4
        x = sx(x_value)
        y = sy(y_value)
        body += f'<line x1="{x:.1f}" y1="{margin["top"]}" x2="{x:.1f}" y2="{height - margin["bottom"]}" class="grid" />\n'
        body += f'<text x="{x:.1f}" y="{height - margin["bottom"] + 18}" class="tick" text-anchor="middle">{x_value:.0f}</text>\n'
        body += f'<line x1="{margin["left"]}" y1="{y:.1f}" x2="{width - margin["right"]}" y2="{y:.1f}" class="grid" />\n'
        body += f'<text x="{margin["left"] - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{y_value:.2f}</text>\n'
    return body


def legend_title(x: float, y: float, text: str) -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" class="label" font-weight="700">{esc(text)}</text>\n'


def note(x: float, y: float, text: str) -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" class="legend">{esc(text)}</text>\n'


def heat_color(value: float, low: float, high: float) -> str:
    ratio = (value - low) / (high - low + 1e-12)
    ratio = min(max(ratio, 0.0), 1.0)
    start = np.array([220, 242, 236], dtype=float)
    mid = np.array([254, 243, 199], dtype=float)
    end = np.array([254, 202, 202], dtype=float)
    if ratio < 0.5:
        color = start + (mid - start) * (ratio / 0.5)
    else:
        color = mid + (end - mid) * ((ratio - 0.5) / 0.5)
    return "#%02x%02x%02x" % tuple(np.round(color).astype(int))


def padded_bounds(values: list[float], pad_fraction: float) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if low == high:
        return low - 1.0, high + 1.0
    span = high - low
    return low - pad_fraction * span, high + pad_fraction * span


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def esc(value: object) -> str:
    return html.escape(str(value))


if __name__ == "__main__":
    main()

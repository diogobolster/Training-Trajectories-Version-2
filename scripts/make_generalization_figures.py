from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


SAMPLERS = [
    "pooled_validation_mixture",
    "gaussian_bayes",
    "hybrid",
    "bootstrap_mean_mixture",
    "knn_conditional",
    "pair_rerank",
]
COMPONENTS = ["gaussian_bayes", "knn_conditional", "hybrid", "pair_rerank"]
DISPLAY = {
    "core1_graph": "Core1\ngraph flow",
    "core2_graph": "Core2\ngraph flow",
    "core2_openfoam": "Core2\nOpenFOAM",
    "pooled_validation_mixture": "pooled mix",
    "bootstrap_mean_mixture": "mean mix",
    "gaussian_bayes": "Gaussian/Bayes",
    "knn_conditional": "kNN",
    "hybrid": "hybrid",
    "pair_rerank": "pair rerank",
}
COLORS = {
    "pooled_validation_mixture": "#2563eb",
    "bootstrap_mean_mixture": "#0f766e",
    "gaussian_bayes": "#111827",
    "knn_conditional": "#d97706",
    "hybrid": "#7c3aed",
    "pair_rerank": "#be123c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate geometry/flow generalization figures.")
    parser.add_argument(
        "--core1-graph",
        type=Path,
        default=Path("outputs/bentheimer_6um_downsample3_outer_split_mixture_benchmark.json"),
    )
    parser.add_argument(
        "--core2-graph",
        type=Path,
        default=Path("outputs/bentheimer_core2_subvol1_6um_downsample3_D001_outer_split_mixture_benchmark.json"),
    )
    parser.add_argument(
        "--core2-openfoam",
        type=Path,
        default=Path("outputs/bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_outer_split_mixture_benchmark.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = {
        "core1_graph": load_json(args.core1_graph),
        "core2_graph": load_json(args.core2_graph),
        "core2_openfoam": load_json(args.core2_openfoam),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "run_010_generalization_sampler_ranks.svg": sampler_rank_lines(datasets),
        "run_010_generalization_selected_weights.svg": selected_weight_bars(datasets),
    }
    for name, svg in figures.items():
        path = args.output_dir / name
        path.write_text(svg, encoding="utf-8")
        print(path)


def sampler_rank_lines(datasets: dict[str, dict[str, object]]) -> str:
    keys = list(datasets)
    width, height = 1040, 590
    margin = {"left": 82, "right": 220, "top": 78, "bottom": 104}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    x_positions = {
        key: margin["left"] + (idx + 0.5) * plot_w / len(keys)
        for idx, key in enumerate(keys)
    }

    def sy(rank: float) -> float:
        return margin["top"] + (rank - 1.0) / 5.0 * plot_h

    body = svg_header(width, height, "Flow fidelity strengthens velocity-continuity selection")
    body += text(82, 52, "Mean held-out sampler rank; lower is better. Geometry and solver change the preferred mechanism.", "subtitle")
    body += axes(width, height, margin, "mean rank")
    for idx in range(1, 7):
        y = sy(float(idx))
        body += f'<line x1="{margin["left"]}" y1="{y:.1f}" x2="{width - margin["right"]}" y2="{y:.1f}" class="grid" />\n'
        body += f'<text x="{margin["left"] - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{idx}</text>\n'
    for key in keys:
        x = x_positions[key]
        for line_idx, label in enumerate(DISPLAY[key].split("\n")):
            body += f'<text x="{x:.1f}" y="{height - margin["bottom"] + 24 + 16 * line_idx}" class="tick" text-anchor="middle">{esc(label)}</text>\n'
    for sampler in SAMPLERS:
        points = []
        for key in keys:
            rank = datasets[key]["summary"]["samplers"][sampler]["mean_rank"]
            points.append((x_positions[key], sy(rank), rank))
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
        color = COLORS[sampler]
        body += f'<polyline fill="none" stroke="{color}" stroke-width="2.7" points="{path}" />\n'
        for x, y, rank in points:
            body += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.3" fill="{color}" stroke="#ffffff" stroke-width="1.4" />\n'
            body += f'<text x="{x:.1f}" y="{y - 8:.1f}" class="small" text-anchor="middle">{rank:.1f}</text>\n'
    body += legend(width - 188, 98)
    return body + "</svg>\n"


def selected_weight_bars(datasets: dict[str, dict[str, object]]) -> str:
    keys = list(datasets)
    width, height = 940, 560
    margin = {"left": 84, "right": 200, "top": 76, "bottom": 116}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    group_w = plot_w / len(keys)
    bar_w = group_w * 0.46

    body = svg_header(width, height, "Mixture weights respond to geometry and flow fidelity")
    body += text(84, 50, "Mean validation-selected weights across outer splits.", "subtitle")
    body += axes(width, height, margin, "weight")
    for idx in range(6):
        value = idx / 5
        y = margin["top"] + (1.0 - value) * plot_h
        body += f'<line x1="{margin["left"]}" y1="{y:.1f}" x2="{width - margin["right"]}" y2="{y:.1f}" class="grid" />\n'
        body += f'<text x="{margin["left"] - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{value:.1f}</text>\n'
    for idx, key in enumerate(keys):
        center = margin["left"] + (idx + 0.5) * group_w
        top = height - margin["bottom"]
        weights = datasets[key]["summary"]["mean_selected_weights"]
        for component in COMPONENTS:
            value = weights[component]
            h = value * plot_h
            top -= h
            body += f'<rect x="{center - bar_w / 2:.1f}" y="{top:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{COLORS[component]}" />\n'
            if value >= 0.10:
                body += f'<text x="{center:.1f}" y="{top + h / 2 + 4:.1f}" class="stacklabel" text-anchor="middle">{value:.2f}</text>\n'
        for line_idx, label in enumerate(DISPLAY[key].split("\n")):
            body += f'<text x="{center:.1f}" y="{height - margin["bottom"] + 24 + 16 * line_idx}" class="tick" text-anchor="middle">{esc(label)}</text>\n'
    body += legend(width - 174, 98, components_only=True)
    return body + "</svg>\n"


def svg_header(width: int, height: int, title: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  .title {{ font: 700 20px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }}
  .subtitle {{ font: 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #4b5563; }}
  .axis {{ stroke: #374151; stroke-width: 1.2; }}
  .grid {{ stroke: #e5e7eb; stroke-width: 1; }}
  .tick {{ font: 11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #4b5563; }}
  .label {{ font: 700 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #374151; }}
  .legend {{ font: 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #374151; }}
  .small {{ font: 10.5px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #374151; }}
  .stacklabel {{ font: 700 11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #ffffff; }}
</style>
<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" stroke="#fdfdfd" stroke-width="1" />
<rect x="{width - 1}" y="{height - 1}" width="1" height="1" fill="#f8f8f8" />
<text x="{width / 2:.1f}" y="30" text-anchor="middle" class="title">{esc(title)}</text>
'''


def axes(width: int, height: int, margin: dict[str, int], y_label: str) -> str:
    x0 = margin["left"]
    x1 = width - margin["right"]
    y0 = height - margin["bottom"]
    y1 = margin["top"]
    return f'''<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" class="axis" />
<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" class="axis" />
<text x="20" y="{(y0 + y1) / 2:.1f}" text-anchor="middle" class="label" transform="rotate(-90 20,{(y0 + y1) / 2:.1f})">{esc(y_label)}</text>
'''


def legend(x: int, y0: int, *, components_only: bool = False) -> str:
    keys = COMPONENTS if components_only else SAMPLERS
    title = "components" if components_only else "samplers"
    body = text(x, y0 - 18, title, "label")
    for idx, key in enumerate(keys):
        y = y0 + idx * 28
        if components_only:
            body += f'<rect x="{x}" y="{y - 14}" width="22" height="16" fill="{COLORS[key]}" />\n'
        else:
            body += f'<line x1="{x}" y1="{y - 5}" x2="{x + 24}" y2="{y - 5}" stroke="{COLORS[key]}" stroke-width="3" />\n'
        body += f'<text x="{x + 34}" y="{y}" class="legend">{esc(DISPLAY[key])}</text>\n'
    return body


def text(x: float, y: float, value: str, class_name: str) -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{class_name}">{esc(value)}</text>\n'


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def esc(value: object) -> str:
    return html.escape(str(value))


if __name__ == "__main__":
    main()

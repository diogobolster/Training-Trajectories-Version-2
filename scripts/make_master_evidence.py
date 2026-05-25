from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

COMPONENTS = ["gaussian_bayes", "knn_conditional", "hybrid", "pair_rerank"]
SAMPLER_DISPLAY = {
    "pooled_validation_mixture": "pooled mix",
    "bootstrap_mean_mixture": "mean mix",
    "gaussian_bayes": "Gaussian/Bayes",
    "knn_conditional": "kNN",
    "hybrid": "hybrid",
    "pair_rerank": "pair rerank",
}
COMPONENT_DISPLAY = {
    "gaussian_bayes": "Gaussian",
    "knn_conditional": "kNN",
    "hybrid": "hybrid",
    "pair_rerank": "pair",
}
COLORS = {
    "gaussian_bayes": "#111827",
    "knn_conditional": "#d97706",
    "hybrid": "#7c3aed",
    "pair_rerank": "#be123c",
    "pooled_validation_mixture": "#2563eb",
    "bootstrap_mean_mixture": "#0f766e",
}


CONDITIONS = [
    {
        "id": "core1_high_pe",
        "label": "Core1 high Pe",
        "short": "Core1 high Pe",
        "path": ROOT / "outputs" / "bentheimer_6um_downsample3_D0003_n20000_stride400_outer_split_mixture_benchmark.json",
        "interpretation": "high Pe has a hybrid lowest-mean mechanism with broad mixture support",
    },
    {
        "id": "core1_baseline",
        "label": "Core1 baseline",
        "short": "Core1 baseline",
        "path": ROOT / "outputs" / "bentheimer_6um_downsample3_D001_n20000_stride400_outer_split_mixture_benchmark.json",
        "interpretation": "Gaussian/Bayes has the lowest mean objective but all memories remain active",
    },
    {
        "id": "core1_low_pe",
        "label": "Core1 low Pe",
        "short": "Core1 low Pe",
        "path": ROOT / "outputs" / "bentheimer_6um_downsample3_D003_n20000_stride400_outer_split_mixture_benchmark.json",
        "interpretation": "higher diffusion gives the lowest mean objective to a validation mixture with substantial velocity support",
    },
    {
        "id": "core2_graph",
        "label": "Core2 graph flow",
        "short": "Core2 graph",
        "path": ROOT
        / "outputs"
        / "bentheimer_core2_subvol1_6um_downsample3_D001_n20000_stride400_outer_split_mixture_benchmark.json",
        "interpretation": "second geometry gives the lowest mean objective to a mixture with the largest learned-context share",
    },
    {
        "id": "core2_openfoam",
        "label": "Core2 OpenFOAM 18 um",
        "short": "Core2 OF 18 um",
        "path": ROOT
        / "outputs"
        / "bentheimer_core2_subvol1_6um_downsample3_D001_openfoam_dt010_n20000_stride1600_outer_split_mixture_benchmark.json",
        "interpretation": "tight coarse OpenFOAM gives the lowest mean objective to archive proximity",
    },
    {
        "id": "core2_openfoam_12um",
        "label": "Core2 OpenFOAM 12 um",
        "short": "Core2 OF 12 um",
        "path": ROOT
        / "outputs"
        / "bentheimer_core2_subvol1_6um_downsample2_D00225_openfoam_phys_scaled_dt010_n20000_stride1600_outer_split_mixture_benchmark.json",
        "interpretation": "intermediate OpenFOAM resolution gives the lowest mean objective to velocity continuity",
    },
    {
        "id": "core2_openfoam_6um",
        "label": "Core2 OpenFOAM 6 um",
        "short": "Core2 OF 6 um",
        "path": ROOT
        / "outputs"
        / "bentheimer_core2_subvol1_6um_fullres_D009_openfoam_strict_dt010_n20000_stride1600_outer_split_mixture_benchmark.json",
        "interpretation": "strict full-resolution flow gives the lowest mean objective to pair organization while mixtures retain multiple memories",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the master evidence table and summary figure.")
    parser.add_argument(
        "--openfoam-sensitivity",
        type=Path,
        default=ROOT
        / "outputs"
        / "bentheimer_core2_subvol1_6um_fullres_D009_openfoam_strict_dt010_n20000_stride1600_objective_weight_sensitivity.json",
    )
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "master_evidence_table.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "outputs" / "master_evidence_table.csv")
    parser.add_argument("--figure", type=Path, default=ROOT / "figures" / "run_013_master_evidence_matrix.svg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [condition_row(condition) for condition in CONDITIONS]
    sensitivity = sensitivity_summary(args.openfoam_sensitivity)
    payload = {
        "conditions": rows,
        "openfoam_objective_sensitivity": sensitivity,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(args.output_csv, rows)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.figure.write_text(master_svg(rows, sensitivity), encoding="utf-8")
    print(args.output_json)
    print(args.output_csv)
    print(args.figure)


def condition_row(condition: dict[str, object]) -> dict[str, object]:
    data = load_json(condition["path"])
    sampler_summary = data["summary"]["samplers"]
    ordered = sorted(
        sampler_summary.items(),
        key=lambda item: item[1]["mean_objective"],
    )
    best_sampler = ordered[0][0]
    best = sampler_summary[best_sampler]
    if len(ordered) > 1:
        second = ordered[1][1]
        mean_gap_to_second = second["mean_objective"] - best["mean_objective"]
        split_variability = max(
            best.get("std_objective", 0.0),
            second.get("std_objective", 0.0),
        )
    else:
        mean_gap_to_second = 0.0
        split_variability = best.get("std_objective", 0.0)
    uncertainty_marker = "robust" if mean_gap_to_second > split_variability else "overlapping"
    weights = data["summary"]["mean_selected_weights"]
    return {
        "id": condition["id"],
        "label": condition["label"],
        "short": condition["short"],
        "path": str(condition["path"]),
        "best_sampler": best_sampler,
        "best_sampler_display": SAMPLER_DISPLAY[best_sampler],
        "best_mean_objective": best["mean_objective"],
        "best_mean_rank": best["mean_rank"],
        "best_wins": best["wins"],
        "mean_gap_to_second": mean_gap_to_second,
        "split_variability": split_variability,
        "uncertainty_marker": uncertainty_marker,
        "beats_gaussian_bayes": best["beats_gaussian_bayes"],
        "beats_hybrid": best["beats_hybrid"],
        "selected_weights": weights,
        "dominant_selected_component": max(weights, key=weights.get),
        "interpretation": condition["interpretation"],
    }


def sensitivity_summary(path: Path) -> dict[str, object]:
    data = load_json(path)
    regimes = []
    gaussian_best_count = 0
    pair_heavy_best = None
    for regime, summary in data["summary"].items():
        sampler_summary = summary["samplers"]
        best_sampler = min(
            sampler_summary,
            key=lambda name: sampler_summary[name]["mean_objective"],
        )
        if best_sampler == "gaussian_bayes":
            gaussian_best_count += 1
        if regime == "pair_heavy":
            pair_heavy_best = best_sampler
        regimes.append(
            {
                "regime": regime,
                "best_sampler": best_sampler,
                "best_sampler_display": SAMPLER_DISPLAY[best_sampler],
                "best_mean_objective": sampler_summary[best_sampler]["mean_objective"],
                "best_mean_rank": sampler_summary[best_sampler]["mean_rank"],
                "bootstrap_mean_weights": summary["selected_weights"]["bootstrap_mean_mixture"]["mean"],
                "pooled_validation_weights": summary["selected_weights"]["pooled_validation_mixture"]["mean"],
            }
        )
    return {
        "path": str(path),
        "gaussian_best_count": gaussian_best_count,
        "n_regimes": len(regimes),
        "pair_heavy_best_sampler": pair_heavy_best,
        "regimes": regimes,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "label",
            "best_sampler",
            "best_mean_objective",
            "best_mean_rank",
            "best_wins",
            "uncertainty_marker",
            "mean_gap_to_second",
            "split_variability",
            "gaussian_weight",
            "knn_weight",
            "hybrid_weight",
            "pair_weight",
            "interpretation",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            weights = row["selected_weights"]
            writer.writerow(
                {
                    "label": row["label"],
                    "best_sampler": row["best_sampler"],
                    "best_mean_objective": row["best_mean_objective"],
                    "best_mean_rank": row["best_mean_rank"],
                    "best_wins": row["best_wins"],
                    "uncertainty_marker": row["uncertainty_marker"],
                    "mean_gap_to_second": row["mean_gap_to_second"],
                    "split_variability": row["split_variability"],
                    "gaussian_weight": weights["gaussian_bayes"],
                    "knn_weight": weights["knn_conditional"],
                    "hybrid_weight": weights["hybrid"],
                    "pair_weight": weights["pair_rerank"],
                    "interpretation": row["interpretation"],
                }
            )


def master_svg(rows: list[dict[str, object]], sensitivity: dict[str, object]) -> str:
    width = 1320
    row_h = 84
    height = 104 + len(rows) * row_h + 150
    x = {
        "condition": 56,
        "best": 250,
        "rank": 420,
        "weights": 535,
        "interpretation": 790,
    }
    y0 = 104
    body = svg_header(width, height, "Master evidence table: validation selects by regime")
    body += text(56, 52, "Across Peclet, geometry, and flow fidelity, no sampler wins universally; lowest-mean mechanisms shift with the retained-state requirement.", "subtitle")
    body += header_text(x["condition"], 84, "condition")
    body += header_text(x["best"], 84, "lowest mean sampler")
    body += header_text(x["rank"], 84, "rank / split wins")
    body += header_text(x["weights"], 84, "mean selected weights")
    body += header_text(x["interpretation"], 84, "interpretation")
    for idx, row in enumerate(rows):
        y = y0 + idx * row_h
        fill = "#f9fafb" if idx % 2 == 0 else "#ffffff"
        body += f'<rect x="40" y="{y - 30}" width="{width - 80}" height="{row_h - 8}" fill="{fill}" />\n'
        body += text(x["condition"], y, row["short"], "rowlabel")
        best_sampler = row["best_sampler"]
        body += f'<circle cx="{x["best"] + 8}" cy="{y - 4}" r="6" fill="{COLORS[best_sampler]}" />\n'
        body += text(x["best"] + 24, y, row["best_sampler_display"], "cell")
        body += text(
            x["rank"],
            y,
            f'{row["best_mean_rank"]:.2f} / {row["best_wins"]}',
            "cell",
        )
        body += stacked_weight_bar(
            x["weights"],
            y - 18,
            210,
            24,
            row["selected_weights"],
        )
        body += text(x["interpretation"], y - 8, row["interpretation"], "smallcell")
        body += text(
            x["interpretation"],
            y + 12,
            weight_sentence(row["selected_weights"]),
            "muted",
        )
    note_y = y0 + len(rows) * row_h + 32
    body += f'<rect x="40" y="{note_y - 24}" width="{width - 80}" height="92" fill="#f8fafc" stroke="#e5e7eb" />\n'
    body += header_text(56, note_y, "OpenFOAM objective sensitivity")
    body += text(
        56,
        note_y + 26,
        f'Gaussian/Bayes has the lowest mean objective in {sensitivity["gaussian_best_count"]}/{sensitivity["n_regimes"]} objective regimes; pair-heavy selects {SAMPLER_DISPLAY[sensitivity["pair_heavy_best_sampler"]]}.',
        "cell",
    )
    body += text(
        56,
        note_y + 50,
        "Interpretation: high-fidelity flow reinforces the 2019 physics kernel, while pair/mixing priorities preserve a role for learned context.",
        "muted",
    )
    body += legend(960, note_y + 22)
    return body + "</svg>\n"


def stacked_weight_bar(x: int, y: int, w: int, h: int, weights: dict[str, float]) -> str:
    body = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#ffffff" stroke="#d1d5db" />\n'
    cursor = x
    for component in COMPONENTS:
        value = float(weights[component])
        segment_w = value * w
        if segment_w <= 0:
            continue
        body += f'<rect x="{cursor:.1f}" y="{y}" width="{segment_w:.1f}" height="{h}" fill="{COLORS[component]}" />\n'
        if segment_w > 32:
            body += f'<text x="{cursor + segment_w / 2:.1f}" y="{y + 16}" class="barlabel" text-anchor="middle">{value:.2f}</text>\n'
        cursor += segment_w
    return body


def weight_sentence(weights: dict[str, float]) -> str:
    dominant = max(COMPONENTS, key=lambda key: weights[key])
    return f'dominant selected weight: {COMPONENT_DISPLAY[dominant]} ({weights[dominant]:.2f})'


def legend(x: int, y: int) -> str:
    body = ""
    for idx, component in enumerate(COMPONENTS):
        yy = y + idx * 20
        body += f'<rect x="{x}" y="{yy - 12}" width="18" height="12" fill="{COLORS[component]}" />\n'
        body += text(x + 26, yy - 2, COMPONENT_DISPLAY[component], "legend")
    return body


def svg_header(width: int, height: int, title: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  .title {{ font: 700 22px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }}
  .subtitle {{ font: 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #4b5563; }}
  .header {{ font: 700 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #374151; text-transform: uppercase; }}
  .rowlabel {{ font: 700 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }}
  .cell {{ font: 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }}
  .smallcell {{ font: 12px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #111827; }}
  .muted {{ font: 11.5px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #6b7280; }}
  .legend {{ font: 11.5px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #374151; }}
  .barlabel {{ font: 700 10.5px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #ffffff; }}
</style>
<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" stroke="#fdfdfd" stroke-width="1" />
<rect x="{width - 1}" y="{height - 1}" width="1" height="1" fill="#f8f8f8" />
<text x="{width / 2:.1f}" y="30" text-anchor="middle" class="title">{esc(title)}</text>
'''


def header_text(x: float, y: float, value: str) -> str:
    return text(x, y, value, "header")


def text(x: float, y: float, value: str, class_name: str) -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{class_name}">{esc(value)}</text>\n'


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def esc(value: object) -> str:
    return html.escape(str(value))


if __name__ == "__main__":
    main()

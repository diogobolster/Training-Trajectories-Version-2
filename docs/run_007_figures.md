# Run 007: Selection Figures

## Summary

Run 007 turns the Run 005 and Run 006 JSON outputs into manuscript-facing SVG figures. The point is to move from result tables to the visual paper claim:

```text
TTA-v2 is not "ML beats physics"; it is validation-driven navigation of
transport-metric tradeoffs among physics kernels, learned rules, and mixtures.
```

## Figures

### Outer-Split Robustness

![Outer-split robustness](../figures/run_005_outer_split_summary.svg)

This figure summarizes Run 005. It shows that the pooled validation mixture has the best mean held-out objective, while Gaussian/Bayes remains highly competitive and ties it on mean rank.

### Objective-Weight Sensitivity

![Objective-weight sensitivity](../figures/run_006_weight_sensitivity_heatmap.svg)

This heatmap summarizes Run 006. Rows are objective-weight regimes, columns are samplers, and cell values are mean held-out rank. The key message is that the preferred sampler moves with the metric priority.

### Selected Mixture Weights

![Selected mixture weights](../figures/run_006_selected_weights.svg)

This stacked-bar figure shows how pooled-validation mixture weights shift under different objective regimes. Pair-, dilution-, and reaction-sensitive objectives generally move weight toward the learned hybrid component, while balanced/no-reaction objectives keep substantial Gaussian/Bayes mass.

### Pareto-Style Tradeoff

![Pareto tradeoff](../figures/run_006_pareto_tradeoff.svg)

This scatter uses the balanced Run 006 regime. The x-axis is BTC error, the y-axis is pair-separation error, and marker size indicates dilution error. It shows why a single scalar score is not enough: samplers occupy different regions of metric space.

## Generated Files

```text
figures/run_005_outer_split_summary.svg
figures/run_006_weight_sensitivity_heatmap.svg
figures/run_006_selected_weights.svg
figures/run_006_pareto_tradeoff.svg
```

## Script

```text
scripts/make_selection_figures.py
```

Run with:

```bash
python3 scripts/make_selection_figures.py
```


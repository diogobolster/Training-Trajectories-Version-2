# Figure Captions

## Figure 1. Outer-Split Robustness of Validation-Selected Samplers

Mean held-out multi-objective transport error for six trajectory samplers over five independent outer train/test splits. The pooled validation mixture selects the simplex-grid mixture with the best mean validation score across repeated inner validation splits. The bootstrap mean mixture averages the best weight vectors selected by the inner validation splits. Error bars show the standard deviation across outer splits. Labels above each bar show mean rank and number of split wins. Bottom labels report how often each sampler beats Gaussian/Bayes (`G`) and hybrid (`H`). The pooled validation mixture has the best mean objective and beats both Gaussian/Bayes and hybrid on three of five splits, while Gaussian/Bayes remains highly competitive and ties the pooled mixture on mean rank.

File: `figures/run_005_outer_split_summary.svg`

## Figure 2. Objective-Weight Sensitivity

Mean held-out rank of each sampler under seven multi-objective scoring regimes. Rows correspond to objective-weight choices and columns correspond to candidate samplers. Green cells indicate lower mean rank, red cells indicate higher mean rank. The preferred sampler changes with the scientific priority: Gaussian/Bayes is most stable on mean score, hybrid is favored for dilution-heavy objectives, kNN is competitive under pair-heavy scoring, and validation-selected mixtures win selected splits and regimes. This figure supports the central claim that TTA-v2 should be interpreted as a validation-driven framework for navigating metric tradeoffs rather than as a universal learned replacement for the physics kernel.

File: `figures/run_006_weight_sensitivity_heatmap.svg`

## Figure 3. Objective-Dependent Mixture Weights

Pooled-validation mixture weights selected under each objective-weight regime. Each stacked bar decomposes the selected transition distribution into Gaussian/Bayes, kNN conditional, hybrid learned, and pair-aware reranking components. Pair-, dilution-, and reaction-sensitive objectives shift weight toward the learned hybrid sampler, while balanced and no-reaction objectives retain substantial Gaussian/Bayes mass. The selected mixtures therefore provide a diagnostic of which transport mechanism is being emphasized by the validation score.

File: `figures/run_006_selected_weights.svg`

## Figure 4. Balanced-Objective Pareto Tradeoff

Pareto-style view of sampler errors under the balanced Run 006 objective. Each point is averaged over four outer held-out splits. The x-axis is breakthrough-curve error, the y-axis is pair-separation error, and marker size indicates dilution log-error. Lower-left is better. No sampler simultaneously minimizes all metrics: Gaussian/Bayes has strong breakthrough performance, hybrid improves some pair/dilution behavior, and selected mixtures occupy intermediate tradeoff regions. This motivates multi-objective validation rather than one-metric model selection.

File: `figures/run_006_pareto_tradeoff.svg`

## Figure 5. Peclet-Regime Sampler Ranks

Mean held-out sampler rank across high-Peclet (`D = 0.0003`), baseline (`D = 0.001`), and low-Peclet (`D = 0.003`) trajectory ensembles. Lower rank is better. The preferred sampler changes with diffusivity: kNN velocity matching becomes more competitive in the high-Peclet case, while the hybrid learned transition rule is favored in the low-Peclet case.

File: `figures/run_009_peclet_sampler_ranks.svg`

## Figure 6. Peclet-Regime Selected Mixture Weights

Mean validation-selected mixture weights across outer splits for the three diffusivity regimes. Increasing diffusivity shifts selected weight away from local velocity matching and toward the hybrid learned transition rule. This provides a physically interpretable regime-sensitivity result: validation does not choose a fixed universal transition model, but responds to the transport physics.

File: `figures/run_009_peclet_selected_weights.svg`

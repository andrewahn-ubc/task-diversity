# Post-hoc corrected analysis

This analysis uses the already completed 25 runs; no trajectory was retrained. The original overall learning curve pooled completed evaluation episodes from depths 1 through 6. Figure 6 instead conditions on depth 6 only.

## Corrected temporal diagnostic

Only start and midpoint gradient measurements are predictors. Start cosine predicts start-to-midpoint success-rate change, and midpoint cosine predicts midpoint-to-end change. Endpoint predictors are excluded. Outcomes use the main 1,024-episode evaluations rather than the smaller diagnostic evaluations.

Each pooled model adjusts for log2 diversity, distribution phase, and half-phase interval and uses CR1 standard errors clustered by the 25 training runs. The fixed-effects model additionally removes every run's mean, testing whether within-run changes in cosine covary with within-run changes in subsequent learning.

## Gradient-group results

| Parameter group | Pooled adjusted cosine slope (95% clustered CI) | Within-run cosine slope (95% clustered CI) |
|---|---:|---:|
| Shared conv + pre-GRU + GRU | 0.017 [-0.035, 0.069] | 0.024 [-0.041, 0.090] |
| Policy head | 0.030 [-0.020, 0.080] | 0.044 [-0.017, 0.104] |
| Value head | 0.013 [-0.027, 0.053] | 0.015 [-0.035, 0.066] |

## Conflict frequency by diversity

| Group | n | Mean cosine | Fraction cosine < 0 | Mean signed dot product |
|---|---:|---:|---:|---:|
| Shared conv + pre-GRU + GRU | 1 | 0.087 | 0.433 | 0.000671 |
| Shared conv + pre-GRU + GRU | 4 | 0.123 | 0.333 | 0.000821 |
| Shared conv + pre-GRU + GRU | 16 | 0.096 | 0.400 | 0.001063 |
| Shared conv + pre-GRU + GRU | 64 | 0.204 | 0.233 | 0.000649 |
| Shared conv + pre-GRU + GRU | 256 | 0.378 | 0.133 | 0.000768 |
| Policy head | 1 | 0.168 | 0.300 | 0.001847 |
| Policy head | 4 | 0.165 | 0.367 | 0.001006 |
| Policy head | 16 | 0.140 | 0.400 | 0.003758 |
| Policy head | 64 | 0.145 | 0.300 | 0.000576 |
| Policy head | 256 | 0.237 | 0.267 | 0.001292 |
| Value head | 1 | 0.287 | 0.400 | 0.003134 |
| Value head | 4 | 0.358 | 0.233 | 0.000158 |
| Value head | 16 | 0.637 | 0.100 | 0.000253 |
| Value head | 64 | 0.816 | 0.000 | 0.000033 |
| Value head | 256 | 0.874 | 0.000 | 0.000044 |

A positive slope means more aligned current-versus-previous gradients predict greater subsequent learning. A negative slope means greater alignment predicts less subsequent learning. Confidence intervals spanning zero do not provide a stable directional association.

## Generated artifacts

- `figure6_learning_curves_depth6`: depth-6-only mean learning curves.
- `figure7_transition_metrics`: every switch and following within-distribution gain separately.
- `figure8_individual_seed_learning_curves`: all 25 trajectories without seed averaging.
- `figure9_gradient_cosine_by_group`: shared, policy-head, and value-head cosine by diversity using valid predictor checkpoints only.
- `figure10_group_conflict_interval_gain`: run-fixed-effects, temporally aligned association plots.

This is a post-hoc reanalysis. It improves temporal alignment and dependence handling but remains observational and cannot establish causality.

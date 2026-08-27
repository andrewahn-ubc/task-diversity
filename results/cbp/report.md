# Reduced Banyan topology-diversity pilot report

## Result

Under the preregistered qualitative criteria, the reduced topology-only tradeoff **replicated**. The gradient-interference association was **not supported**. These are diagnostic associations, not causal claims.

## Exact setup

A recurrent PPO + Continual Backprop (CBP) agent trained sequentially on four disjoint topology distributions at n = 1, 4, 16, 64, and 256, with five matched seeds. Each phase used 50,331,648 environment steps. The layout, deterministic object grounding procedure, architecture, optimizer, CBP configuration, evaluation budget, and diagnostic schedule were held fixed. Success was measured on independent evaluation episodes.

The official Banyan code was not public when this repository was created. The CBP generate-and-test rule and element-wise Adam-state reset are adapted from Dohare et al.'s official implementation. This repository applies them to every learned feature layer and adds a documented GRU extension. A plateau that persists under this intervention is less consistent with conventional loss of plasticity, although CBP cannot logically eliminate every plasticity-related explanation.

## Effect sizes (mean across seeds, 95% CI)

| n | Transfer gap d2-d4 | Specialization gain d2-d4 | Gradient cosine |
|---:|---:|---:|---:|
| 1 | 0.002 +/- 0.226 | 0.105 +/- 0.125 | 0.109 +/- 0.137 |
| 4 | 0.106 +/- 0.014 | 0.003 +/- 0.013 | 0.152 +/- 0.048 |
| 16 | 0.007 +/- 0.019 | 0.009 +/- 0.051 | 0.091 +/- 0.252 |
| 64 | -0.017 +/- 0.014 | -0.003 +/- 0.004 | 0.178 +/- 0.171 |
| 256 | -0.010 +/- 0.005 | -0.007 +/- 0.014 | 0.296 +/- 0.114 |

## Interference diagnostic

The pooled correlation between gradient cosine and subsequent specialization gain was 0.068. In the simple regression controlling for log2 diversity, the cosine coefficient was 0.027 (n = 225 checkpoints). No inferential p-value is reported because checkpoints within a run are not independent.

## Backward performance

Figure 5 reports B(i,1) for every condition. Positive values mean later training improved performance on d1; negative values mean forgetting. This comparison distinguishes a current-distribution specialization plateau from a total halt in learning.

## Limitations

- The environment is a documented reduced reconstruction because the authors' implementation and full hyperparameters were unavailable.
- Only topology diversity is varied, the sequence has four rather than ten distributions, and the phase budget is approximately half the paper's 100M steps.
- The official CBP code supports feed-forward networks; the GRU feature-block extension here is necessary for the recurrent policy but has not been validated by the Banyan or CBP authors.
- CBP actively mitigates loss of plasticity but cannot prove that every residual plateau is caused only by gradient interference.
- Gradient conflict is observational; it cannot establish that interference causes stalled specialization.

## Recommendation

If the tradeoff is present and stable across seeds, next run one targeted causal intervention that reduces measured conflict without changing task diversity. If it is absent, first test a longer sequence or jointly vary layouts and topologies; do not tune the current pilot after seeing this result.

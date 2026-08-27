# Reduced Banyan topology-diversity pilot

This repository implements the four-distribution pilot in
`banyan_reduced_diversity_interference_pilot.md`: topology diversity
`n in {1, 4, 16, 64, 256}`, five matched seeds, recurrent PPO + Continual Backprop
(CBP), independent success evaluation, backward evaluation on `d1`, and
current-versus-previous PPO-gradient cosines.

The public Banyan project page still says **"Code (soon)"** (checked
2026-08-26), so this is a documented clean-room reduced implementation, not the
authors' unreleased code. It preserves the paper's task factorization and
interaction semantics while holding one layout and one deterministic object
grounding per topology fixed. See [docs/EXPERIMENT.md](docs/EXPERIMENT.md) for
the exact correspondences and deviations, and [docs/NARVAL.md](docs/NARVAL.md)
for the dependency audit.

## One command on Narval

Clone the repository on Narval, enter it, and run:

```bash
bash scripts/narval_submit.sh
```

That command does all of the following without contacting PyPI:

1. loads `StdEnv/2023` and `python/3.11`;
2. asks Narval's `avail_wheels` to verify every pinned requirement;
3. creates/validates `.venv-narval` with a repository-local `virtualenv` seed
   cache;
4. resolves the full transitive closure offline, rejecting source archives and
   anything outside the Alliance CVMFS wheelhouse, then installs the three
   top-level packages with both `PIP_NO_INDEX=1` and `--no-index`;
5. verifies exact package versions, the Alliance CUDA-enabled Torch build, and
   the resolved dependency graph;
6. trains the full-budget first phase for the five seed-0 conditions;
7. verifies nontrivial low/intermediate-diversity learning and actual CBP
   replacements before resuming those checkpoints in the full 25-run array; and
8. aggregates the complete sweep into CSV/JSON, five figures, and a Markdown
   report.

The A100 training arrays charge `rrg-mijungp_gpu`; the CPU-only smoke gate and
aggregation charge `def-mijungp_cpu`. Before submitting anything, the launcher
requires both exact associations to appear in `sshare -l -U "$USER"`. The
submission command prints every job ID and the exact `squeue` command needed to
monitor the pipeline.

Useful variants:

```bash
bash scripts/narval_submit.sh --preflight-only
bash scripts/narval_submit.sh --smoke-only
bash scripts/narval_submit.sh --resume-main
```

All training jobs are idempotent: completed runs exit immediately, and
interrupted runs resume from `checkpoints/latest.pt`. Training is submitted as
finite chains of distinct, dependent SLURM arrays, each capped at one hour. The
smoke chain has 8 chunks and the main chain has 21. Fifteen minutes before each
limit, an unfinished run saves its complete training state; the next array job
then resumes it. Corresponding array elements are chained independently, so a
policy does not wait for the slowest policy between chunks. The final chunk
fails explicitly if the finite allocation budget is exhausted. Allowing for
every checkpoint signal, the chains provide 6 hours for the smoke phase and 15
hours 45 minutes for the main run. This exceeds the respective 5-hour-30-minute
and 15-hour targets obtained by applying a 1.5x buffer to the
3-hour-40-minute and 10-hour planning estimates, while leaving ample time for
an in-progress evaluation or gradient diagnostic to finish and checkpoint.

## Outputs

After the final aggregation job succeeds:

```text
results/
  raw-cbp/n{1,4,16,64,256}/seed_{0..4}/
    metrics.jsonl
    diagnostics.jsonl
    checkpoints/latest.pt
    completed.json
  cbp/
    aggregated/
      phase_metrics.csv
      interference_metrics.csv
      summary.json
    figures/
      figure1_learning_curves.png
      figure2_transfer_specialization.png
      figure3_gradient_interference.png
      figure4_interference_specialization.png
      figure5_backward_d1.png
    report.md
  environment/
    offline-resolution.json
    pip-inspect.json
    pip-config.txt
    pip-freeze.txt
    preflight-login.json
```

## Post-hoc corrected analysis

After copying a complete `results/raw-cbp` directory, the existing trajectories
can be reanalysed without retraining:

```bash
python -m banyan_pilot.reanalyze
```

This writes `results/cbp/reanalysis/`, including a depth-6-only learning curve,
per-transition and per-seed plots, shared/policy/value gradient diagnostics,
temporally aligned half-phase outcomes, and run-clustered/fixed-effects
summaries. The reanalysis is explicitly post hoc and does not replace the
preregistered aggregate report.

## Focused local checks

The core checks use the standard library's `unittest` runner:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m banyan_pilot.preflight --config configs/smoke.toml
PYTHONPATH=src python3 -m banyan_pilot.train \
  --config configs/smoke.toml --diversity 1 --seed 0 \
  --output-root /tmp/banyan-pilot-check --override-steps 4096
```

## Scientific guardrails

- Diversity is topology count only; optimization steps, evaluation episodes,
  architecture, task catalog seed, and object/layout generators are matched.
- The phase budget is fixed before the sweep at 50,331,648 environment steps
  (384 complete recurrent PPO rollouts). The smoke gate does not tune or
  increase it.
- PPO is paired with CBP at every optimizer minibatch. Contribution utility,
  decay `0.99`, replacement rate `1e-4`, and maturity threshold `100` are fixed
  across conditions. Replacement counts are logged and the smoke gate rejects
  a run unless every controlled feature layer has actually been exercised.
- Reported success uses evaluation episodes, never training rollout returns.
- Gradient cosines use fresh, balanced current/previous rollouts collected at
  the same checkpoint without model updates.
- The diagnostic is correlational and must not be described as causal.
- The feed-forward generate-and-test rule and per-element Adam-state reset are
  adapted from the official
  [`loss-of-plasticity`](https://github.com/shibhansh/loss-of-plasticity)
  implementation. Because that code does not support recurrent policies, the
  GRU extension is local and explicitly documented in
  [docs/EXPERIMENT.md](docs/EXPERIMENT.md).

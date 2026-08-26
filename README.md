# Reduced Banyan topology-diversity pilot

This repository implements the four-distribution pilot in
`banyan_reduced_diversity_interference_pilot.md`: topology diversity
`n in {1, 16, 256}`, five matched seeds, recurrent PPO, independent success
evaluation, backward evaluation on `d1`, and current-versus-previous PPO-gradient
cosines.

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
6. submits a three-condition smoke array;
7. gates the full 15-run array on both successful smoke jobs and a nontrivial
   learning check; and
8. aggregates the complete sweep into CSV/JSON, five figures, and a Markdown
   report.

The jobs charge `def-mijungp`. The submission command prints every job ID and
the exact `squeue` command needed to monitor the pipeline.

Useful variants:

```bash
bash scripts/narval_submit.sh --preflight-only
bash scripts/narval_submit.sh --smoke-only
bash scripts/narval_submit.sh --resume-main
```

All training jobs are idempotent: completed runs exit immediately, and
interrupted runs resume from `checkpoints/latest.pt`. A three-minute SLURM
signal saves a checkpoint and requeues the task.

## Outputs

After the final aggregation job succeeds:

```text
results/
  raw/n{1,16,256}/seed_{0..4}/
    metrics.jsonl
    diagnostics.jsonl
    checkpoints/latest.pt
    completed.json
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
- The phase budget is fixed before the sweep at 25,165,824 environment steps
  (192 complete recurrent PPO rollouts). The smoke gate does not tune or
  increase it.
- Reported success uses evaluation episodes, never training rollout returns.
- Gradient cosines use fresh, balanced current/previous rollouts collected at
  the same checkpoint without model updates.
- The diagnostic is correlational and must not be described as causal.
- This version uses plain PPO. Continual Backprop is intentionally omitted
  because the official implementation is unavailable and reimplementing it
  would dominate this reduced pilot.

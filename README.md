# Banyan-inspired depth-6 learnability diagnostic

The default experiment is a **single-distribution validation study**, not a
continual-learning sweep. It tests whether the reconstructed environment can
support depth-6 learning before any more expensive continual experiment is
considered.

The study crosses:

- algorithms: plain PPO and PPO + Continual Backprop (CBP);
- diversity: `n in {1, 4}` layouts and `n` topologies, sampled as the full
  `n x n` layout-topology cross product;
- two independently generated topology/layout catalogs; and
- two policy-training seeds.

This gives 16 independent 100,663,296-transition runs. The public Banyan
project page still says **"Code (soon)"** (checked 2026-08-26), so the
environment remains a documented clean-room reconstruction rather than the
authors' implementation. See [docs/EXPERIMENT.md](docs/EXPERIMENT.md).

## One command on Narval

```bash
cd /home/taegyoem/scratch/task-diversity
git pull --ff-only origin main
bash scripts/narval_submit.sh
```

The command performs the complete offline dependency audit, verifies the RRG
GPU and DEF CPU associations, submits the 16-run A100 array, and then submits a
CPU analysis job. It does **not** submit a smoke-to-main pipeline or any
continual jobs.

Useful variants:

```bash
bash scripts/narval_submit.sh --preflight-only
bash scripts/narval_submit.sh --resume
```

Every training allocation is capped at one hour. A checkpoint signal arrives
at 45 minutes; a second corresponding array job resumes any unfinished run.
Completed runs exit without changing their results.

## What changed relative to the earlier pilot

- One distribution rather than four; nothing continual is launched.
- `n` layouts are crossed with `n` topologies instead of holding one layout
  fixed.
- 100,663,296 transitions rather than 50,331,648.
- 256 recurrent steps and 256 environments per rollout, producing 1,536 PPO
  updates per run instead of 384 per old phase.
- 1,536 evaluation episodes per checkpoint, exactly 256 per depth.
- Overall success is the arithmetic mean of depths 1 through 6, matching the
  paper's reported aggregate instead of pooling whichever episodes terminate
  first.
- Evaluation logs include timeout, no-op, manipulation-attempt, effective
  pickup/drop/merge/toggle, and depth-specific rates.
- Plain PPO isolates whether the custom recurrent CBP extension prevents
  first-distribution learning, where loss of plasticity is not yet relevant.

## Outputs

```text
results/learnability/
  raw/
    {ppo,ppo_cbp}/catalog_{260600880,260600881}/n{1,4}/seed_{0,1}/
      run.json
      metrics.jsonl
      checkpoints/latest.pt
      completed.json
  analysis/
    learning_curves.csv
    learning_curves_overall.png
    learning_curves_depth6.png
    learning_curves_timeout.png
    learning_curves_effective_manipulation.png
    learnability_gate.json
    report.md
```

The gate passes only if PPO+CBP reaches mean endpoint depth-6 success of at
least 0.10 at both diversity levels and at least three of four run clusters per
level reach 0.05. It is diagnostic only and controls no downstream submission.

## Focused local checks

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m banyan_pilot.preflight --config configs/learnability.toml
PYTHONPATH=src python3 -m banyan_pilot.train \
  --config configs/learnability.toml --diversity 1 --seed 0 \
  --catalog-seed 260600880 --algorithm ppo --vary-layout \
  --output-root /tmp/banyan-learnability-check --override-steps 65536
```

The completed four-distribution pilot and its post-hoc analysis remain under
`results/raw-cbp` and `results/cbp`; the new launcher never resumes or extends
those trajectories.

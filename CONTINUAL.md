# Seven-distribution continual PPO experiment

This implements the Figure 6 protocol from *Task diversity produces systematic transfer but inhibits continual reinforcement learning* with the requested changes: `n ∈ {1, 4, 16, 64, 256}` and seven sequential distributions. The paper does not publish its PPO configuration or training code. This is an independent recurrent PPO implementation following the PureJaxRL pattern, with a pilot to choose among explicit hyperparameter candidates. It is an approximation of the published run, not a claim of bit-for-bit replication.

## Run on Narval

Copy this repository to Narval storage, `cd` into it, then run:

```bash
bash scripts/setup_narval.sh
bash scripts/submit_narval.sh pilot
```

`setup_narval.sh` loads Narval's Python 3.13.2 module and installs the exact Linux/CUDA dependencies from `uv.lock` using binary wheels. It requires Narval's `$SCRATCH` directory and stores the pinned `uv` executable, wheel cache, virtual environment, datasets, checkpoints, and Slurm logs there. The repository's `.venv`, `outputs`, and `logs` paths are symlinks to that storage. [Alliance scratch storage is purgeable](https://docs.alliancecan.ca/mediawiki/images/9/99/File_Systems.pdf), so move completed results to persistent storage. If any of those paths already exists as a directory or points elsewhere, the setup script stops and tells you where to move the data before retrying. Alliance configures `pip` to search its CVMFS wheelhouse without PyPI, and that wheelhouse may lack `uv` and these exact dependency versions. The setup script bypasses that `pip` configuration only while downloading the `uv` wheel from PyPI, without using your home-directory pip cache. It also clears Alliance's custom `PYTHONPATH`, which can mask external manylinux wheels and virtual-environment packages. It checks installed dependency requirements before submitting anything. The submission script first runs a 10-minute GPU preflight that checks Python, glibc, locked package versions, the NVIDIA driver, GPU discovery, and a compiled JAX calculation. Preparation and training jobs start only if that check succeeds; inspect `logs/preflight-<job-id>.out` if it fails. It then creates two CPU preparation jobs and 16 independent GPU trajectories (four variants × two diversity settings × two seeds). Their training chunks automatically submit the next chunk until complete. All preflight, preparation, training, and automatically resumed jobs use Slurm account `rrg-mijungp_gpu`. The scripts use the partition selected by your Narval environment. Monitor with `squeue -u "$USER"` and `logs/`.

### Dependency audit

The September 2026 `uv.lock` contains 75 third-party packages. Every one has a prebuilt wheel whose tags match CPython 3.13 on Linux x86-64 with glibc 2.34; the local project itself is installed editable. In particular, `jax`, `jaxlib`, `jax-cuda12-plugin`, and `jax-cuda12-pjrt` are all locked to 0.11.0. The `jaxlib` and CUDA plugin CPython 3.13 wheels are published on [PyPI](https://pypi.org/project/jaxlib/0.11.0/) and [PyPI](https://pypi.org/project/jax-cuda12-plugin/0.11.0/) with glibc 2.27 tags, and their SHA256 hashes match `uv.lock`. Among the matching x86-64 wheels, the highest minimum glibc version is 2.28, from packages including `wandb`.

[Alliance lists Python 3.13.2](https://docs.alliancecan.ca/wiki/Available_software) and [reports Narval's upgrade to AlmaLinux 9.6](https://status.alliancecan.ca/view_incident?incident=1460), whose glibc is 2.34. The locked CUDA runtime is 12.9; [NVIDIA requires driver 575.51.03 or newer](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html), and [Alliance reports driver 580 across Narval GPU nodes](https://status.alliancecan.ca/view_incident?incident=1611). These are compatibility checks from published files and documentation, not a claim that a wheel has already run on a Narval GPU. The automatic preflight performs that final check on an allocated node.

Run from a shell without a separately loaded CUDA/cuDNN module. [JAX notes](https://docs.jax.dev/en/latest/installation.html) that `LD_LIBRARY_PATH` can override the CUDA libraries supplied by its pip wheels.

When all pilot trajectories finish, choose the winner and view the pilot curves:

```bash
./.venv/bin/python scripts/select_pilot.py
./.venv/bin/python scripts/plot_continual.py --mode pilot
```

`select_pilot.py` writes `outputs/continual/pilot_best.json`. It requires all 16 runs to be complete. The full submission command also runs selection, so the explicit selection step is optional:

```bash
bash scripts/submit_narval.sh full
```

After all full trajectories finish:

```bash
./.venv/bin/python scripts/plot_continual.py --mode full
```

The last command writes `outputs/continual/full_figure6.png` and `outputs/continual/full_figure6.svg`, with all-depth and depth-6 panels, phase boundaries, and mean ± one standard deviation across the three seeds. Plots use actual logged success rates; none are fabricated in advance.

## Design

| Component | Pilot | Full run |
| --- | --- | --- |
| Diversity values | 4, 64 | 1, 4, 16, 64, 256 |
| Distributions | 3 | 7 |
| Steps per distribution | 10,010,624 | 100,007,936 |
| Seeds | 0, 1 | 0, 1, 2 |
| Variants | base; lower learning rate; higher entropy; higher CBP replacement rate | pilot winner |
| Evaluation | 64 fixed-seed episodes per depth, about every 2M steps | 64 fixed-seed episodes per depth, about every 5M steps |

The step budgets are rounded upward to a whole 128-environment × 128-step rollout. Every distribution contains `n` distinct depth-6 topologies and `n` distinct connected wall layouts; the task generator also supplies `n` strict-subtree tasks at each depth 1–5. Layouts and depth-6 topologies are disjoint across distributions. Each `(task, layout)` pair has fixed item positions; episode resets randomize the agent start. That gives `n²` depth-6 task/layout combinations per distribution. The object vocabulary can overlap across distributions, as in a shared-object shift.

The agent sees the grid, inventory, and goal, but not the task rules. Success pays `+1`; pickup shaping and step cost are disabled. The environment is configured to pay `-1` for distractor dead ends, but this repository's round-disjoint dataset generator does not produce a distractor table, so that branch is inactive in this experiment. The paper's exact layout-generation settings are also unavailable; this implementation uses fixed, connected 8×8 random wall masks. These are material differences when interpreting the result.

The network has two 128-unit ReLU encoder layers and a 128-unit GRU, with separate categorical policy and value heads. PPO uses clipped policy/value losses, GAE, four epochs, four environment-sequence minibatches, Adam, and gradient clipping. Continual Backprop tracks activation-weight utility in both encoder layers and replaces mature, low-utility units; the GRU itself is retained across distributions and is not reset by CBP. The optimizer and CBP traces persist across all seven distributions.

The pilot compares:

| Variant | Learning rate | Entropy coefficient | CBP fraction per PPO update |
| --- | ---: | ---: | ---: |
| base | 3e-4 | .01 | 2e-4 |
| lower_lr | 1e-4 | .01 | 2e-4 |
| more_entropy | 3e-4 | .03 | 2e-4 |
| more_cbp | 3e-4 | .01 | 1e-3 |

Selection averages a score over both diversity values and both seeds: `0.5 × final all-depth success + 0.3 × final depth-6 success + 0.2 × phase-average area under the all-depth learning curve`. This rewards both terminal performance and learning speed. The full run uses only the winning variant; inspect `pilot_best.json` before submission if you want to override that choice.

## Job limits, checkpoints, and recovery

Each Slurm job has a 59-minute wall limit. The Python runner stops after 40 minutes or on Slurm's 20-minute warning, checkpoints after the current PPO update, and the job script submits its successor with an `afterok` dependency. A single phase can span many jobs. Checkpoints store policy, value head, GRU, Adam moments, CBP traces, RNG, and live environments; the successor resumes without restarting the phase. Metrics are append-only JSONL files under `outputs/continual/runs/`.

If a job exits with an error, inspect `logs/train-<job-id>.out`, fix the issue, and submit that trajectory again:

```bash
bash scripts/resume_narval.sh pilot 4 0 base
# or: bash scripts/resume_narval.sh full 256 0
```

The measured GPU throughput is hardware and allocation dependent. The 59-minute limit is enforced by Slurm, but a batch interrupted during its first compilation can lose its uncheckpointed work; the last completed checkpoint remains valid. The full experiment contains roughly 10.5 billion environment steps across 15 trajectories and may require substantial aggregate GPU time even though every individual job stays under one hour.

## Outputs and provenance

`outputs/continual/datasets/` contains the generated task files, their metadata, and layout masks. `outputs/continual/runs/` contains checkpoints, progress, and success-rate records. The dataset generator's metadata records topology and producer-signature disjointness. The local code smoke test covers reset, rollout, PPO, CBP, evaluation, and checkpoint resume; performance and scientific agreement must be assessed from the Narval pilot before the full run.

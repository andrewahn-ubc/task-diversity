# Narval dependency and scheduling audit

Checked on 2026-08-26 against the Digital Research Alliance of Canada
documentation.

## Cluster facts used by the scripts

- Narval's compute nodes do not have internet access. The submission and jobs
  therefore never use PyPI; every install command includes `--no-index`.
- Narval's default supported stack is `StdEnv/2023`; older standard
  environments are blocked.
- GPU nodes expose four 40GB NVIDIA A100 SXM4 GPUs. A full device is requested
  with `--gpus=a100:1`.
- Narval permits jobs up to 168 hours. The first-phase pilot jobs request 4
  hours and the resumable primary runs request 10 hours.

Source: [Alliance Narval documentation](https://docs.alliancecan.ca/wiki/Narval/en).

## Python installation path

The Alliance recommends `virtualenv --no-download` and `pip install
--no-index` so packages resolve from its CVMFS wheelhouse. Its PyTorch page
specifically recommends installing the `torch` wheel this way and does not
require a separate CUDA module in the batch example. The repository follows
that pattern exactly: `StdEnv/2023` plus `python/3.11`, then a persistent
project-local virtual environment. The launcher also gives `virtualenv` a
repository-local app-data directory, so a stale or corrupt seed cache under
`~/.local/share/virtualenv` cannot affect environment creation. It validates
both Python 3.11 and `pip` before reusing `.venv-narval`; a partial environment
left by an interrupted setup is rebuilt automatically. A schema-and-requirement
stamp is written only after the entire audit passes, so environments made by an
older launcher or interrupted at any point are rebuilt cleanly. Pip's index,
version check, and cache are also disabled explicitly, and Matplotlib uses a
project-local writable cache during login-node preflight. Inherited
`PYTHONPATH`, `PYTHONHOME`, and user-site packages are disabled in both the
launcher and every batch job.

Sources: [Alliance Python documentation](https://docs.alliancecan.ca/wiki/Python/en),
[Alliance PyTorch documentation](https://docs.alliancecan.ca/wiki/PyTorch/en).

## Pins and compatibility

Only three top-level packages are installed:

| Package | Pin | Reason |
|---|---:|---|
| PyTorch | 2.6.0 | PPO, the vector environment, GPU execution, and autograd diagnostics |
| NumPy | 1.26.4 | aggregation and deterministic catalog helpers |
| Matplotlib | 3.9.2 | the five required figures |

The Alliance Python 3.11 wheel catalog lists all three exact versions. PyTorch
2.6 officially supports Python 3.13 in addition to earlier supported versions,
so Python 3.11 is within the release's supported range. Matplotlib 3.9 requires
NumPy >=1.23, which is satisfied by NumPy 1.26.4. The code also accounts for
PyTorch 2.6's `torch.load(weights_only=True)` default change by explicitly
loading its own trusted checkpoints with `weights_only=False`.

Narval's Torch wheel is the Alliance's CUDA-enabled `+computecanada` build, not
the upstream PyPI binary. The launcher therefore does not assume PyPI's NVIDIA
package closure. It asks Narval's own pip configuration to resolve all
transitive dependencies, records the resolution, and rejects any artifact that
is not a wheel under the Alliance CVMFS wheelhouse. The public Alliance wheel
builder recipe builds Torch against the cluster CUDA/cuDNN/NCCL stack and tests
the completed wheel after unloading its build-only modules.

Sources: [Alliance Python 3.11 wheel catalog](https://docs.alliancecan.ca/wiki/Available_Python_wheels),
[PyTorch 2.6 release notes](https://pytorch.org/blog/pytorch2-6/),
[Matplotlib 3.9 dependency floor](https://matplotlib.org/stable/api/prev_api_changes/api_changes_3.9.0.html#increase-to-minimum-supported-versions-of-dependencies).

## Runtime verification layers

The one-command launcher fails before submitting any GPU work unless all of
these pass:

1. Narval CVMFS Python 3.11 is the actual base interpreter;
2. `avail_wheels -r requirements-narval.txt` finds all direct pins in the
   active standard environment;
3. pip performs a clean offline dry-run of the complete dependency closure,
   using wheels only, and every reported artifact is checked to be under the
   Alliance CVMFS wheelhouse;
4. an actual wheels-only `pip install --no-index` of the exact pins;
5. `pip check`, exact direct-version checks, and a check that Torch is a CUDA
   build;
6. CPU imports and a deterministic 6,144-task catalog/environment probe; and
7. on every allocated GPU, CUDA availability, A100 `sm_80` kernel support,
   catalog disjointness, an environment step, a real convolution/GRU
   forward-and-backward pass, one custom Adam step, forced CBP replacements in
   every controlled layer, and CUDA synchronization.

The resolution report, installed metadata, freeze, pip configuration, and
login-node audit are saved under `results/environment/`; the CUDA audit appears
in each SLURM log. This final live check is intentional because the public
wheel catalog warns that a wheel shown across environments may not be present
in a particular active `StdEnv`.

Alliance build source: [ComputeCanada wheels_builder](https://github.com/ComputeCanada/wheels_builder),
[Torch recipe used in the 2.6-era stack](https://github.com/ComputeCanada/wheels_builder/blob/6a4c628/config/torch.sh).

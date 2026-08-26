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
- Narval permits jobs up to 168 hours. The pilot requests 1 hour for smoke
  jobs and 10 hours for primary runs.

Source: [Alliance Narval documentation](https://docs.alliancecan.ca/wiki/Narval/en).

## Python installation path

The Alliance recommends `virtualenv --no-download` and `pip install
--no-index` so packages resolve from its CVMFS wheelhouse. Its PyTorch page
specifically recommends installing the `torch` wheel this way and does not
require a separate CUDA module in the batch example. The repository follows
that pattern exactly: `StdEnv/2023` plus `python/3.11`, then a persistent
project-local virtual environment.

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

Sources: [Alliance Python 3.11 wheel catalog](https://docs.alliancecan.ca/wiki/Available_Python_wheels),
[PyTorch 2.6 release notes](https://pytorch.org/blog/pytorch2-6/),
[Matplotlib 3.9 dependency floor](https://matplotlib.org/stable/api/prev_api_changes/api_changes_3.9.0.html#increase-to-minimum-supported-versions-of-dependencies).

## Runtime verification layers

The one-command launcher fails before submitting any GPU work unless all of
these pass:

1. `avail_wheels -r requirements-narval.txt` in the active Narval standard
   environment;
2. an actual `pip install --no-index` of the exact pins;
3. `pip check` for the resolved environment;
4. CPU imports and a deterministic 6,144-task catalog/environment probe; and
5. on every allocated GPU, CUDA availability, Ampere-or-newer compute
   capability, imports, catalog disjointness, and an environment step.

The resolved transitive environment is saved to
`results/environment/pip-freeze.txt`; the login-node audit is saved to
`results/environment/preflight-login.json`; and the CUDA audit appears in each
SLURM log. This final live check is intentional because the public wheel
catalog warns that a wheel shown across environments may not be present in a
particular active `StdEnv`.

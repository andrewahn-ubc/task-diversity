<h1 align="center">Banyan</h1>

[**Installation**](#installation) | [**Quick Start**](#quick-start) | [**Dataset Generation**](DATASETS.md) | [**Training**](#training)

---

Banyan is a single-agent grid world for compositional-task reinforcement learning in JAX. An agent collects, transforms, and combines colored objects to produce a goal object; tasks are specified by rulesets of varying depth and tree topology. The repository also includes task-dataset generators and a baseline-agnostic multi-round training and evaluation harness.

## Installation

```bash
uv sync --frozen
```

Creates `.venv` from `uv.lock` (Python 3.13+); run commands with `uv run --frozen`. On Linux this installs CUDA 12 JAX — see the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for other hardware.

## Quick Start

Gymnax-style functional API:

```python
import jax
from banyan_grid import make
from banyan_grid.tasks.ruleset_factory import build_ruleset
from banyan_grid.utils.banyan import get_map

key_rules, key_map, key_reset, key_action, key_step = jax.random.split(
    jax.random.PRNGKey(0), 5
)
ruleset = build_ruleset(key_rules, depth=3, base_seed=0)
map_array, color_map = get_map(key_map, 5, ruleset)

env = make(
    "Banyan",
    grid_size=5,
    map_array=map_array,
    color_map=color_map,
    ruleset=ruleset,
)
obs, state = env.reset(key_reset)
action = env.action_space().sample(key_action)
obs, state, reward, done, info = jax.jit(env.step)(key_step, state, action)
```

Actions and rewards are scalars; observations are flat arrays; `done` is a scalar bool. `step` does not auto-reset. `reset` and `step` support `jit`/`vmap`.

## Training

[`run_tournament`](banyan_grid/rounds/tournament.py) takes a config plus user-supplied `make_train` / `make_eval` factories, and provides round scheduling, checkpointing, replay, and cross-round evaluation. Baselines are not bundled.

## Development

```bash
uv run --frozen --extra dev python -m pytest -q
```

## License

[Apache 2.0](LICENSE).

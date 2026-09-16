# Dataset Generation

[**Generate**](#generate) | [**Load a Task**](#load-a-task) | [**Round-Disjoint Datasets**](#round-disjoint-datasets) | [**Tools**](#tools)

Datasets store rulesets; sample maps separately with `get_map`. See the [README](README.md#installation) for installation.

## Generate

```bash
uv run --frozen python -m banyan_grid.tasks.make_ruleset_dataset \
  --n 16 --max-depth 3 --pool-size 8 \
  --tree-topology balanced --base-seed 0 --bench-seed 0 \
  --out-dir outputs/datasets --name example
```

Writes **52 tasks** (8 depth-1, 28 depth-2, 16 depth-3). The balanced default enforces subtree closure: lower-depth subtrees also appear as standalone tasks.

| Option | Meaning |
| --- | --- |
| `--n` | Highest-depth task count in the default schedule (max depth ≥ 3). |
| `--max-depth` | Highest depth bucket; supported values depend on topology below. |
| `--pool-size` | Number of distinct leaf `(item, color)` pairs. |
| `--base-seed` | Leaf pool and output mapping; use `0` for compact encodings. |
| `--bench-seed` | Task sampling and distractor selection. |
| `--distractor-density` | Probability of selecting each unordered pair of pool tokens (default `0`). |

Default counts: `pool_size` at depth 1, `pool_size * (pool_size - 1) // 2` at depth 2, and `n * 2**(max_depth - d)` at depths `3 <= d <= max_depth`. `--n` is not the total size.

`--tree-topology`:

| Mode | Supported `--max-depth` | Task structure |
| --- | --- | --- |
| `balanced` | 1–6 | Binary trees with strict subtree closure. |
| `asym_d2_d3_6l` | 3 | Six-leaf trees in bucket 3 (actual tree depth 4, counting leaves as depth 1). |
| `mixed_u1_b2` | 1–6 | Unary-transform and binary-combine trees. |
| `mixed_u1_b2_t3` | 1–6 | Also permits ternary combines. |

Mixed modes don't guarantee subtree closure, and changing only `--bench-seed` doesn't guarantee disjoint tasks — use the pool/exclusion options in `--help` for train/test splits. Generators overwrite matching filenames, so use a fresh `--out-dir` or `--name` per configuration.

### Files

The example writes:

- `example_d1-2-3_n52_ps8_bs0_rs0.uint32.npy.bz2`: packed `uint32` array of shape `(tasks, padded_rules)`.
- `example_d1-2-3_n52_ps8_bs0_rs0.uint32_meta.json`: depth counts, seeds, pool IDs, generation settings.
- With nonzero distractors, `*.uint32_distractor_table.npy`: a **global** lookup table, not per-task.

Metadata `n` is the total row count; `structure` gives contiguous ascending-depth row groups.

## Load a Task

```python
import jax
import jax.numpy as jnp
from banyan_grid import make
from banyan_grid.tasks.ruleset_codec import unpack_rules_uint32_np
from banyan_grid.tasks.ruleset_dataset_compact import (
    load_distractor_table,
    load_packed_u32_bz2,
)
from banyan_grid.utils.banyan import get_map

folder = "outputs/datasets"
filename = "example_d1-2-3_n52_ps8_bs0_rs0.uint32.npy.bz2"
packed = load_packed_u32_bz2(folder, filename)
ruleset = jnp.asarray(unpack_rules_uint32_np(packed[0]), dtype=jnp.int32)
key_map, key_reset = jax.random.split(jax.random.PRNGKey(0))
map_array, color_map = get_map(key_map, 5, ruleset)

env = make(
    "Banyan",
    grid_size=5,
    map_array=map_array,
    color_map=color_map,
    ruleset=ruleset,
    max_rules=packed.shape[1],
    distractor_table=load_distractor_table(folder, filename),
)
obs, state = env.reset(key_reset)
```

Keep `max_rules` at the dataset width so observation sizes stay fixed across tasks; larger tasks need enough map cells for their leaves.

## Round-Disjoint Datasets

```bash
uv run --frozen python -m banyan_grid.tasks.make_depth6_round_disjoint_dataset \
  --n 4 --rounds 2 --pool-size 120 \
  --base-seed 0 --master-seed 7 \
  --out-dir outputs/round_datasets --name example
```

Writes **24 tasks per round** to `outputs/round_datasets/n0004/` (`example_n4_r00`, `example_n4_r01`, plus summaries). Each depth-6 task contributes one subtree per lower depth — not exhaustive closure. For each `--n`, depth-6 topologies and producer input signatures are disjoint across rounds generated together; item pools may overlap.

## Tools

- [`merge_ruleset_datasets.py`](scripts/merge_ruleset_datasets.py): merge datasets by depth; global distractor tables must agree.
- [`prune_ruleset_dataset_by_depth.py`](scripts/prune_ruleset_dataset_by_depth.py): keep selected depth groups.
- [`validate_ruleset_disjoint_sampling.py`](scripts/validate_ruleset_disjoint_sampling.py): sampled task uniqueness and pool overlap; unknown pool provenance is inconclusive.
- [`validate_dataset_map_solvability.py`](scripts/validate_dataset_map_solvability.py): sampled leaf placement and symbolic solvability; check `num_failures` in the report, not just exit status.

All take `--help`. Merge and prune require `--allow-overwrite` to replace existing outputs.

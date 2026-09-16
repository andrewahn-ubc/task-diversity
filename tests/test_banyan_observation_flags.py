import jax
import jax.numpy as jnp
import numpy as np

from banyan_grid.environment import Banyan
from banyan_grid.environment.constants import (
    Colors,
    TileType,
)


def _make_env() -> Banyan:
    map_array = jnp.full((5, 5), TileType.OPEN_FAST, dtype=jnp.int32)
    color_map = jnp.full((5, 5), Colors.BLACK, dtype=jnp.int32)
    return Banyan(
        grid_size=5,
        max_steps=10,
        map_array=map_array,
        color_map=color_map,
        include_rules_in_obs=False,
    )


def _with_inventory_item(
    state: Banyan.State,
    item_idx: int,
    color: int,
) -> Banyan.State:
    inventories = state.inventories.at[item_idx].set(True)
    inventory_colors = state.inventory_colors.at[item_idx].set(color)
    return state.replace(  # ty:ignore[unresolved-attribute]
        inventories=inventories, inventory_colors=inventory_colors
    )


def test_obs_matches_declared_observation_space():
    env = _make_env()
    key = jax.random.PRNGKey(0)
    obs, _state = env.reset(key)
    assert obs.shape == env.observation_space().shape


def test_own_inventory_changes_obs():
    key = jax.random.PRNGKey(0)

    env = _make_env()
    _, state = env.reset(key)
    obs_base = np.array(env.get_obs(state))

    state_changed = _with_inventory_item(state, item_idx=0, color=Colors.GREEN)
    obs_changed = np.array(env.get_obs(state_changed))
    assert not np.array_equal(obs_base, obs_changed)

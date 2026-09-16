import jax
import jax.numpy as jnp
import numpy as np

from banyan_grid.environment import Banyan
from banyan_grid.environment.constants import (
    Colors,
    NUM_ACTIONS,
    NUM_COLORS,
    NUM_ITEMS,
    TileType,
    WALKABLE_MASK,
)


def _make_env() -> Banyan:
    map_array = jnp.full((5, 5), TileType.OPEN_FAST, dtype=jnp.int32)
    color_map = jnp.full((5, 5), Colors.BLACK, dtype=jnp.int32)
    return Banyan(
        grid_size=5,
        max_steps=10,
        map_array=map_array,
        color_map=color_map,
    )


def test_state_is_single_agent():
    env = _make_env()
    _, state = env.reset(jax.random.PRNGKey(0))

    assert state.positions.shape == (2,)
    assert state.directions.shape == ()
    assert state.inventories.shape == (NUM_ITEMS,)
    assert state.inventory_colors.shape == (NUM_ITEMS,)
    assert state.items_ever_picked.shape == (NUM_ITEMS, NUM_COLORS)
    assert not hasattr(state, "num_agents")
    assert not hasattr(env, "num_agents")
    assert not hasattr(env, "agents")


def test_step_returns_scalars():
    env = _make_env()
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    obs, _state, reward, done, info = env.step(
        key, state, jnp.asarray(0, dtype=jnp.int32)
    )

    assert reward.shape == ()
    assert reward.dtype == jnp.float32
    assert done.shape == ()
    assert done.dtype == jnp.bool_
    for k, v in info.items():
        assert jnp.shape(v) == (), f"info[{k!r}] has shape {jnp.shape(v)}"
    assert obs.shape == env.observation_space().shape


def test_vmap_reset_and_step():
    env = _make_env()
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    obs, states = jax.vmap(env.reset)(keys)
    assert states.positions.shape == (4, 2)

    step_keys = jax.random.split(jax.random.PRNGKey(1), 4)
    actions = jnp.arange(4, dtype=jnp.int32) % NUM_ACTIONS
    obs, states, rewards, dones, info = jax.vmap(env.step)(
        step_keys, states, actions
    )
    assert states.positions.shape == (4, 2)
    assert rewards.shape == (4,)
    assert dones.shape == (4,)
    assert obs.shape == (4,) + env.observation_space().shape


def test_spawn_on_walkable_tile():
    env = _make_env()
    for i in range(20):
        _, state = env.reset(jax.random.PRNGKey(i))
        pos = np.asarray(state.positions)
        tile = np.asarray(state.map_array)[pos[0], pos[1]]
        assert bool(np.asarray(WALKABLE_MASK)[tile])

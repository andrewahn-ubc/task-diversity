import jax
import jax.numpy as jnp
import numpy as np

from banyan_grid.continual_ppo import (
    adam_init, cbp_init, cbp_step, gae, init_params, step_model,
)


def test_gae_stops_at_episode_boundary():
    advantage, target = gae(
        jnp.array([[1.0], [1.0]]), jnp.array([[1.0], [0.0]]),
        jnp.zeros((2, 1)), jnp.zeros((1,)), 1.0, 1.0,
    )
    np.testing.assert_allclose(advantage[:, 0], [1.0, 1.0])
    np.testing.assert_allclose(target[:, 0], [1.0, 1.0])


def test_gru_hidden_state_clears_at_reset():
    params = init_params(jax.random.key(1), 3, 2, 8)
    obs = jnp.ones((2, 3))
    hidden = jnp.ones((2, 8))
    reset_output = step_model(params, obs, hidden, jnp.ones((2,), dtype=bool))
    zero_output = step_model(params, obs, jnp.zeros_like(hidden), jnp.zeros((2,), dtype=bool))
    for left, right in zip(reset_output, zero_output):
        np.testing.assert_allclose(left, right, atol=1e-6)


def test_cbp_replaces_mature_units_and_clears_optimizer_moments():
    params = init_params(jax.random.key(2), 3, 2, 8)
    m, v, t = adam_init(params)
    m = jax.tree.map(jnp.ones_like, m)
    v = jax.tree.map(jnp.ones_like, v)
    state = cbp_init(8)
    state["age1"] = jnp.full((8,), 100)
    state["age2"] = jnp.full((8,), 100)
    cfg = {"cbp_rate": 1 / 8, "cbp_decay": 0.0, "cbp_maturity": 100}
    run = jax.jit(lambda p, o, s: cbp_step(
        p, o, s, jax.random.key(3), jnp.ones((8,)), jnp.ones((8,)), cfg))
    new_params, (new_m, new_v, _), new_state, _, counts = run(params, (m, v, t), state)
    assert tuple(map(int, counts)) == (1, 1)
    first = np.flatnonzero(np.asarray(new_state["age1"]) == 0)
    second = np.flatnonzero(np.asarray(new_state["age2"]) == 0)
    assert len(first) == len(second) == 1
    unaffected_columns = [i for i in range(8) if i not in second]
    np.testing.assert_allclose(new_params["enc2"]["w"][np.ix_(first, unaffected_columns)], 0)
    np.testing.assert_allclose(new_m["enc1"]["w"][:, first], 0)
    np.testing.assert_allclose(new_v["enc1"]["w"][:, first], 0)
    for gate in ("z", "r", "n"):
        np.testing.assert_allclose(new_params["w" + gate][second, :], 0)
        np.testing.assert_allclose(new_m["w" + gate][second, :], 0)

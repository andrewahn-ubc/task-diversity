"""Recurrent PPO with Continual Backprop on the feedforward encoder.

Pure JAX implementation: rollout, recurrent PPO updates, and CBP are jit/vmap/scan
compatible. CBP replaces mature, low-utility neurons in both encoder layers while
limiting their immediate influence by zeroing their outgoing weights.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def _dense(key: jax.Array, ins: int, outs: int, scale: float = 1.0) -> dict:
    return {"w": jax.nn.initializers.orthogonal(scale)(key, (ins, outs)),
            "b": jnp.zeros((outs,), dtype=jnp.float32)}


def init_params(key: jax.Array, obs_dim: int, actions: int, width: int) -> dict:
    keys = jax.random.split(key, 13)
    p = {"enc1": _dense(keys[0], obs_dim, width, jnp.sqrt(2.0)),
         "enc2": _dense(keys[1], width, width, jnp.sqrt(2.0)),
         "actor": _dense(keys[2], width, actions, 0.01),
         "critic": _dense(keys[3], width, 1, 1.0)}
    for i, gate in enumerate(("z", "r", "n")):
        p["w" + gate] = jax.nn.initializers.orthogonal(1.0)(keys[4 + i], (width, width))
        p["u" + gate] = jax.nn.initializers.orthogonal(1.0)(keys[7 + i], (width, width))
        p["b" + gate] = jnp.zeros((width,), dtype=jnp.float32)
    return p


def step_model(p: dict, obs: jax.Array, hidden: jax.Array, start: jax.Array):
    hidden = jnp.where(start[:, None], 0.0, hidden)
    h1 = jax.nn.relu(obs @ p["enc1"]["w"] + p["enc1"]["b"])
    h2 = jax.nn.relu(h1 @ p["enc2"]["w"] + p["enc2"]["b"])
    z = jax.nn.sigmoid(h2 @ p["wz"] + hidden @ p["uz"] + p["bz"])
    r = jax.nn.sigmoid(h2 @ p["wr"] + hidden @ p["ur"] + p["br"])
    candidate = jnp.tanh(h2 @ p["wn"] + (r * hidden) @ p["un"] + p["bn"])
    new_hidden = (1.0 - z) * candidate + z * hidden
    logits = new_hidden @ p["actor"]["w"] + p["actor"]["b"]
    value = (new_hidden @ p["critic"]["w"] + p["critic"]["b"]).squeeze(-1)
    return new_hidden, logits, value, h1, h2


def sequence_model(p: dict, obs: jax.Array, hidden: jax.Array, starts: jax.Array):
    def one(carry, data):
        h, logits, value, a1, a2 = step_model(p, data[0], carry, data[1])
        return h, (logits, value, a1, a2)
    return jax.lax.scan(one, hidden, (obs, starts))


def log_prob(logits: jax.Array, actions: jax.Array) -> jax.Array:
    return jnp.take_along_axis(jax.nn.log_softmax(logits), actions[..., None], -1).squeeze(-1)


def gae(reward: jax.Array, done: jax.Array, value: jax.Array,
        bootstrap: jax.Array, gamma: float, lam: float):
    next_values = jnp.concatenate((value[1:], bootstrap[None]), axis=0)
    def one(carry, data):
        r, d, v, nv = data
        delta = r + gamma * (1.0 - d) * nv - v
        advantage = delta + gamma * lam * (1.0 - d) * carry
        return advantage, advantage
    _, reverse = jax.lax.scan(one, jnp.zeros_like(bootstrap),
                              (reward[::-1], done[::-1], value[::-1], next_values[::-1]))
    advantages = reverse[::-1]
    return advantages, advantages + value


def adam_init(p: dict) -> tuple[Any, Any, jax.Array]:
    return (jax.tree.map(jnp.zeros_like, p), jax.tree.map(jnp.zeros_like, p),
            jnp.asarray(0, dtype=jnp.int32))


def adam_step(p, opt, grads, lr: float, max_grad_norm: float):
    m, v, t = opt
    norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads)))
    grads = jax.tree.map(lambda g: g * jnp.minimum(1.0, max_grad_norm / (norm + 1e-8)), grads)
    t = t + 1
    m = jax.tree.map(lambda old, g: .9 * old + .1 * g, m, grads)
    v = jax.tree.map(lambda old, g: .999 * old + .001 * g * g, v, grads)
    p = jax.tree.map(lambda x, mm, vv: x - lr * (mm / (1.0 - .9 ** t)) /
                     (jnp.sqrt(vv / (1.0 - .999 ** t)) + 1e-5), p, m, v)
    return p, (m, v, t), norm


def ppo_update(p, opt, key, batch, cfg):
    """Shuffle complete environment sequences, preserving recurrent state."""
    envs = batch["obs"].shape[1]
    per_minibatch = envs // cfg["minibatches"]
    if envs % cfg["minibatches"]:
        raise ValueError("num_envs must be divisible by minibatches")
    advantage = batch["advantage"]
    batch = {**batch, "advantage": (advantage - advantage.mean()) /
             (advantage.std() + 1e-8)}

    def loss_fn(params, indices):
        obs = batch["obs"][:, indices]
        starts = batch["start"][:, indices]
        _, (logits, value, _, _) = sequence_model(params, obs, batch["hidden0"][indices], starts)
        actions = batch["action"][:, indices]
        old_logp = batch["logp"][:, indices]
        old_value = batch["value"][:, indices]
        adv = batch["advantage"][:, indices]
        target = batch["target"][:, indices]
        ratio = jnp.exp(log_prob(logits, actions) - old_logp)
        clipped = jnp.clip(ratio, 1.0 - cfg["clip_eps"], 1.0 + cfg["clip_eps"])
        policy_loss = -jnp.minimum(ratio * adv, clipped * adv).mean()
        value_clipped = old_value + jnp.clip(value - old_value,
                                              -cfg["clip_eps"], cfg["clip_eps"])
        value_loss = .5 * jnp.maximum((value - target) ** 2,
                                       (value_clipped - target) ** 2).mean()
        probs = jax.nn.softmax(logits)
        entropy = -(probs * jax.nn.log_softmax(logits)).sum(-1).mean()
        total = policy_loss + cfg["value_coef"] * value_loss - cfg["entropy_coef"] * entropy
        return total, jnp.array((policy_loss, value_loss, entropy))

    def epoch(carry, _):
        params, optimizer, rng = carry
        rng, perm_key = jax.random.split(rng)
        indices = jax.random.permutation(perm_key, envs).reshape(cfg["minibatches"], per_minibatch)
        def minibatch(inner, idx):
            pp, oo = inner
            (loss, pieces), grad = jax.value_and_grad(loss_fn, has_aux=True)(pp, idx)
            pp, oo, grad_norm = adam_step(pp, oo, grad, cfg["learning_rate"], cfg["max_grad_norm"])
            return (pp, oo), jnp.concatenate((jnp.array([loss, grad_norm]), pieces))
        (params, optimizer), metrics = jax.lax.scan(minibatch, (params, optimizer), indices)
        return (params, optimizer, rng), metrics.mean(0)
    (p, opt, key), metrics = jax.lax.scan(epoch, (p, opt, key), None,
                                          length=cfg["update_epochs"])
    return p, opt, key, metrics.mean(0)


def cbp_init(width: int):
    return {"u1": jnp.zeros((width,)), "u2": jnp.zeros((width,)),
            "age1": jnp.zeros((width,), dtype=jnp.int32),
            "age2": jnp.zeros((width,), dtype=jnp.int32),
            "credit1": jnp.asarray(0.0), "credit2": jnp.asarray(0.0)}


def cbp_step(p, opt, state, key, activation1, activation2, cfg):
    """CBP utility trace and replacement for both encoder layers.

    The gate input matrices are the outgoing connections of encoder layer 2.
    Adam moments at reset coordinates are cleared with the weights.
    """
    m, v, t = opt
    rate = cfg["cbp_rate"]
    decay = cfg["cbp_decay"]
    min_age = cfg["cbp_maturity"]
    width = activation1.shape[0]
    out1 = jnp.abs(p["enc2"]["w"]).sum(1)
    out2 = sum(jnp.abs(p["w" + gate]).sum(1) for gate in ("z", "r", "n"))
    utility1 = decay * state["u1"] + (1.0 - decay) * activation1 * out1
    utility2 = decay * state["u2"] + (1.0 - decay) * activation2 * out2
    age1 = state["age1"] + 1
    age2 = state["age2"] + 1
    credit1 = state["credit1"] + rate * width
    credit2 = state["credit2"] + rate * width
    def choose(utility, age, credit):
        eligible = age >= min_age
        count = jnp.minimum(jnp.floor(credit).astype(jnp.int32), eligible.sum())
        order = jnp.argsort(jnp.where(eligible, utility, jnp.inf))
        return jnp.zeros((width,), dtype=bool).at[order].set(jnp.arange(width) < count), count
    replace1, count1 = choose(utility1, age1, credit1)
    replace2, count2 = choose(utility2, age2, credit2)
    key1, key2, key = jax.random.split(key, 3)
    fresh1 = jax.random.normal(key1, p["enc1"]["w"].shape) * jnp.sqrt(2.0 / p["enc1"]["w"].shape[0])
    fresh2 = jax.random.normal(key2, p["enc2"]["w"].shape) * jnp.sqrt(2.0 / width)
    # Incoming resets use fresh samples; outgoing resets use zeros.
    for tree in (p, m, v):
        tree["enc1"]["w"] = jnp.where(replace1[None, :],
            fresh1 if tree is p else 0.0, tree["enc1"]["w"])
        tree["enc1"]["b"] = jnp.where(replace1, 0.0, tree["enc1"]["b"])
        tree["enc2"]["w"] = jnp.where(replace1[:, None], 0.0, tree["enc2"]["w"])
        tree["enc2"]["w"] = jnp.where(replace2[None, :],
            fresh2 if tree is p else 0.0, tree["enc2"]["w"])
        tree["enc2"]["b"] = jnp.where(replace2, 0.0, tree["enc2"]["b"])
        for gate in ("z", "r", "n"):
            tree["w" + gate] = jnp.where(replace2[:, None], 0.0, tree["w" + gate])
    state = {"u1": jnp.where(replace1, 0.0, utility1),
             "u2": jnp.where(replace2, 0.0, utility2),
             "age1": jnp.where(replace1, 0, age1),
             "age2": jnp.where(replace2, 0, age2),
             "credit1": credit1 - count1, "credit2": credit2 - count2}
    return p, (m, v, t), state, key, (count1, count2)

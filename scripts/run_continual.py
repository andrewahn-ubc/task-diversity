"""Run one resumable pilot or full continual PPO + CBP trajectory."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import signal
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from banyan_grid.continual_ppo import (  # noqa: E402
    adam_init, cbp_init, cbp_step, gae, init_params, log_prob,
    ppo_update, step_model,
)
from continual_config import DEFAULTS, VARIANTS, N_VALUES  # noqa: E402
from banyan_grid.environment.banyan import Banyan  # noqa: E402
from banyan_grid.environment.constants import TileType  # noqa: E402
from banyan_grid.tasks.ruleset_codec import unpack_rules_uint32_np  # noqa: E402
from banyan_grid.tasks.ruleset_dataset_compact import load_packed_u32_bz2  # noqa: E402
from banyan_grid.utils.banyan import get_map  # noqa: E402


STOP = False


def _stop(_signum, _frame):
    global STOP
    STOP = True


def config_for(mode: str, variant: str, selected: Path | None) -> dict:
    cfg = dict(DEFAULTS)
    if mode == "pilot":
        cfg.update(VARIANTS[variant])
        cfg.update(rounds=3, steps_per_phase=10_000_000, eval_every_steps=2_000_000)
    elif mode == "full":
        if variant == "selected":
            if selected is None or not selected.exists():
                raise FileNotFoundError("Run the pilot and select_pilot.py before full reproduction")
            selection = json.loads(selected.read_text())
            cfg.update(selection["hyperparameters"])
        else:
            cfg.update(VARIANTS[variant])
        cfg.update(rounds=7, steps_per_phase=100_000_000)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return cfg


def dataset_path(root: Path, mode: str, n: int) -> Path:
    return root / "datasets" / mode / f"n{n:04d}"


def load_phase(root: Path, mode: str, n: int, phase: int, cfg: dict):
    folder = dataset_path(root, mode, n)
    file_name = f"continual_n{n}_r{phase:02d}.uint32.npy.bz2"
    packed = load_packed_u32_bz2(str(folder), file_name)
    if packed.shape[0] != 6 * n:
        raise ValueError(f"Expected {6*n} depth-stratified tasks, got {packed.shape[0]}")
    rules = np.stack([unpack_rules_uint32_np(row) for row in packed])
    if rules.shape[1] != packed.shape[1]:
        raise ValueError("Unexpected packed-rule width")
    layouts = np.load(folder / "layouts.npz")["obstacle_mask"]
    if layouts.shape != (cfg["rounds"], n, 8, 8):
        raise ValueError(f"Unexpected layout bank shape: {layouts.shape}")
    return jnp.asarray(rules, dtype=jnp.int32), jnp.asarray(layouts[phase]), packed.shape[1]


def make_phase_ops(env: Banyan, rules: jax.Array, layouts: jax.Array, n: int, cfg: dict):
    """JIT functions specific to one round's task and layout arrays."""
    compile_batch = jax.jit(jax.vmap(lambda codes: Banyan.compile_task_reset_metadata(
        codes, include_rules_in_obs=False, depth_weighted_pickup_shaping=True,
        pickup_shaping_leaf_reward=0.0, pickup_shaping_root_reward=0.0,
    )))
    chunks = [compile_batch(rules[start:start + 64])
              for start in range(0, rules.shape[0], 64)]
    metadata = jax.tree.map(lambda *parts: jnp.concatenate(parts, axis=0), *chunks)

    def reset_one(key, index, layout_index):
        # Fix leaf placement for every (task, layout) pair. A fresh map key on
        # each episode would silently make the layout diversity unbounded.
        k_map = jax.random.fold_in(jax.random.key(2026), layout_index)
        k_reset = key
        codes = rules[index]
        mask = layouts[layout_index]
        map_array, color_map = get_map(k_map, 8, codes, reserved_mask=mask)
        map_array = jnp.where(mask, jnp.asarray(TileType.BLOCK, dtype=jnp.int32), map_array)
        task_meta = jax.tree.map(lambda x: x[index], metadata)
        return env.reset(k_reset, {"map_array": map_array, "color_map": color_map,
                                   "ruleset": codes, **task_meta})

    def sample_indices(key, count):
        kd, kt, kl = jax.random.split(key, 3)
        depths = jax.random.randint(kd, (count,), 0, 6)
        task = depths * n + jax.random.randint(kt, (count,), 0, n)
        layout = jax.random.randint(kl, (count,), 0, n)
        return task, layout

    @jax.jit
    def init_batch(key):
        key, ks, ki = jax.random.split(key, 3)
        task, layout = sample_indices(ki, cfg["num_envs"])
        obs, states = jax.vmap(reset_one)(jax.random.split(ks, cfg["num_envs"]), task, layout)
        return key, obs, states

    @jax.jit
    def collect(params, key, obs, states, hidden, starts):
        hidden0 = hidden
        def one(carry, _):
            key, obs, states, hidden, starts = carry
            key, action_key, step_key, reset_key, index_key = jax.random.split(key, 5)
            hidden_new, logits, value, a1, a2 = step_model(params, obs, hidden, starts)
            actions = jax.random.categorical(action_key, logits)
            logp = log_prob(logits, actions)
            next_obs, next_states, rewards, done, _ = jax.vmap(env.step)(
                jax.random.split(step_key, cfg["num_envs"]), states, actions)
            task, layout = sample_indices(index_key, cfg["num_envs"])
            keys = jax.random.split(reset_key, cfg["num_envs"])
            def maybe_reset(d, k, ti, li, o, s):
                return jax.lax.cond(d, lambda _: reset_one(k, ti, li),
                                    lambda _: (o, s), operand=None)
            next_obs, next_states = jax.vmap(maybe_reset)(
                done, keys, task, layout, next_obs, next_states)
            transition = (obs, starts, actions, logp, value, rewards,
                          done.astype(jnp.float32), a1.mean(0), a2.mean(0))
            return (key, next_obs, next_states, hidden_new, done), transition
        (key, obs, states, hidden, starts), data = jax.lax.scan(
            one, (key, obs, states, hidden, starts), None,
            length=cfg["rollout_steps"])
        _, _, bootstrap, _, _ = step_model(params, obs, hidden, starts)
        (old_obs, old_starts, action, logp, value, reward, done, a1, a2) = data
        advantage, target = gae(reward, done, value, bootstrap, cfg["gamma"], cfg["gae_lambda"])
        batch = {"obs": old_obs, "start": old_starts, "action": action, "logp": logp,
                 "value": value, "advantage": advantage, "target": target,
                 "hidden0": hidden0}
        stats = (a1.mean(0), a2.mean(0), reward.mean(), done.mean())
        return (key, obs, states, hidden, starts), batch, stats

    @jax.jit
    def evaluate(params, key, depth):
        count = cfg["eval_episodes_per_depth"]
        key, kt, kl, kr = jax.random.split(key, 4)
        tasks = depth * n + jax.random.randint(kt, (count,), 0, n)
        layout_ids = jax.random.randint(kl, (count,), 0, n)
        obs, states = jax.vmap(reset_one)(jax.random.split(kr, count), tasks, layout_ids)
        hidden = jnp.zeros((count, cfg["width"]), dtype=jnp.float32)
        done = jnp.zeros((count,), dtype=bool)
        success = jnp.zeros((count,), dtype=bool)
        def one(carry, _):
            key, obs, states, hidden, done, success = carry
            key, step_key = jax.random.split(key)
            hidden, logits, _, _, _ = step_model(params, obs, hidden, jnp.zeros_like(done))
            actions = jnp.argmax(logits, axis=-1)
            def maybe_step(k, s, o, d, a):
                def step(_):
                    oo, ss, _, dd, info = env.step(k, s, a)
                    return oo, ss, dd, info["goal_achieved"] > 0
                return jax.lax.cond(d, lambda _: (o, s, d, jnp.asarray(False)),
                                    step, operand=None)
            obs, states, ended, achieved = jax.vmap(maybe_step)(
                jax.random.split(step_key, count), states, obs, done, actions)
            done = done | ended
            success = success | achieved
            return (key, obs, states, hidden, done, success), None
        final, _ = jax.lax.scan(one, (key, obs, states, hidden, done, success), None,
                                length=cfg["max_steps"])
        return final[-1].mean()

    return init_batch, collect, evaluate


def save_checkpoint(path: Path, state: dict, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        pickle.dump({"state": jax.device_get(state), "config": cfg}, stream,
                    protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def log_eval(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=("selected", *VARIANTS), default=None)
    parser.add_argument("--root", type=Path, default=Path("outputs/continual"))
    parser.add_argument("--selected", type=Path, default=Path("outputs/continual/pilot_best.json"))
    parser.add_argument("--max-minutes", type=float, default=48.0)
    args = parser.parse_args()
    if args.n not in N_VALUES:
        parser.error(f"--n must be one of {N_VALUES}")
    if args.mode == "pilot" and args.n not in (4, 64):
        parser.error("Pilot uses n=4 and n=64")
    variant = args.variant or ("base" if args.mode == "pilot" else "selected")
    if args.mode == "pilot" and variant == "selected":
        parser.error("The pilot requires a named hyperparameter variant")
    cfg = config_for(args.mode, variant, args.selected)
    root = args.root.resolve()
    run_dir = root / "runs" / args.mode / f"n{args.n:04d}" / variant / f"seed{args.seed}"
    checkpoint = run_dir / "checkpoint.pkl"
    metrics_file = run_dir / "metrics.jsonl"
    progress_file = run_dir / "progress.json"
    batch_steps = cfg["num_envs"] * cfg["rollout_steps"]
    phase_updates = math.ceil(cfg["steps_per_phase"] / batch_steps)
    phase_steps = phase_updates * batch_steps
    signal.signal(signal.SIGUSR1, _stop)
    signal.signal(signal.SIGTERM, _stop)
    deadline = time.monotonic() + args.max_minutes * 60
    start_time = time.monotonic()

    if checkpoint.exists():
        payload = pickle.loads(checkpoint.read_bytes())
        if payload["config"] != cfg:
            raise ValueError("Checkpoint config differs from requested run")
        state = jax.device_put(payload["state"])
    else:
        state = {"phase": 0, "phase_updates": 0, "last_eval_step": -1,
                 "last_eval_phase": -1,
                 "key": jax.random.key(args.seed), "params": None, "optimizer": None,
                 "cbp": None, "obs": None, "env_state": None,
                 "hidden": None, "starts": None}

    while state["phase"] < cfg["rounds"]:
        phase = int(state["phase"])
        rules, layouts, max_rules = load_phase(root, args.mode, args.n, phase, cfg)
        env = Banyan(grid_size=8, max_steps=cfg["max_steps"], max_rules=max_rules,
                     include_rules_in_obs=False, distractor_combine_penalty=-1.0,
                     timeout_penalty=0.0, depth_weighted_pickup_shaping=True,
                     pickup_shaping_leaf_reward=0.0, pickup_shaping_root_reward=0.0,
                     step_penalty=0.0)
        init_batch, collect, evaluate = make_phase_ops(env, rules, layouts, args.n, cfg)
        update = jax.jit(lambda p, opt, key, batch: ppo_update(p, opt, key, batch, cfg))
        replace = jax.jit(lambda p, opt, cbp, key, a1, a2:
                          cbp_step(p, opt, cbp, key, a1, a2, cfg))
        if state["params"] is None:
            state["key"], init_key, net_key = jax.random.split(state["key"], 3)
            state["params"] = init_params(net_key, env.observation_space().shape[0],
                                           env.action_space().n, cfg["width"])
            state["optimizer"] = adam_init(state["params"])
            state["cbp"] = cbp_init(cfg["width"])
            state["key"], state["obs"], state["env_state"] = init_batch(init_key)
            state["hidden"] = jnp.zeros((cfg["num_envs"], cfg["width"]), dtype=jnp.float32)
            state["starts"] = jnp.ones((cfg["num_envs"],), dtype=bool)
        elif state["obs"] is None:
            state["key"], reset_key = jax.random.split(state["key"])
            state["key"], state["obs"], state["env_state"] = init_batch(reset_key)
            state["hidden"] = jnp.zeros((cfg["num_envs"], cfg["width"]), dtype=jnp.float32)
            state["starts"] = jnp.ones((cfg["num_envs"],), dtype=bool)

        def current_step():
            return phase * phase_steps + int(state["phase_updates"]) * batch_steps

        def record_eval():
            step = current_step()
            if step == int(state["last_eval_step"]) and phase == int(state["last_eval_phase"]):
                return
            values = []
            for depth in range(6):
                eval_key = jax.random.fold_in(jax.random.key(args.seed + 13_000), phase * 6 + depth)
                values.append(float(evaluate(state["params"], eval_key, jnp.asarray(depth)).block_until_ready()))
            record = {"mode": args.mode, "n": args.n, "seed": args.seed,
                      "variant": variant, "phase": phase + 1,
                      "env_steps": step, "phase_steps": int(state["phase_updates"]) * batch_steps,
                      "all_depths": float(np.mean(values)),
                      **{f"depth{depth+1}": value for depth, value in enumerate(values)}}
            log_eval(metrics_file, record)
            state["last_eval_step"] = step
            state["last_eval_phase"] = phase
            print(json.dumps(record), flush=True)

        record_eval()
        save_checkpoint(checkpoint, state, cfg)
        while int(state["phase_updates"]) < phase_updates:
            if STOP or time.monotonic() >= deadline:
                break
            state["key"], rollout_key, update_key, cbp_key = jax.random.split(state["key"], 4)
            (rollout_key, state["obs"], state["env_state"], state["hidden"],
             state["starts"]), batch, stats = collect(
                 state["params"], rollout_key, state["obs"], state["env_state"],
                 state["hidden"], state["starts"])
            state["params"], state["optimizer"], update_key, losses = update(
                state["params"], state["optimizer"], update_key, batch)
            state["params"], state["optimizer"], state["cbp"], cbp_key, replacements = replace(
                state["params"], state["optimizer"], state["cbp"], cbp_key,
                stats[0], stats[1])
            # JAX dispatch is asynchronous. Synchronize so the wall clock
            # deadline tracks completed updates rather than queued work.
            replacements[0].block_until_ready()
            state["key"] = jax.random.fold_in(state["key"], int(state["phase_updates"]))
            state["phase_updates"] += 1
            step = current_step()
            if (step // cfg["eval_every_steps"] > int(state["last_eval_step"]) // cfg["eval_every_steps"]
                    or state["phase_updates"] == phase_updates):
                record_eval()
            if (state["phase_updates"] % max(1, cfg["checkpoint_every_steps"] // batch_steps) == 0
                    or state["phase_updates"] == phase_updates or STOP or time.monotonic() >= deadline):
                save_checkpoint(checkpoint, state, cfg)
            if state["phase_updates"] % 20 == 0:
                print(f"phase={phase+1} steps={step:,} loss={np.asarray(losses)[0]:.4f} "
                      f"CBP={tuple(map(int, replacements))}", flush=True)
        if int(state["phase_updates"]) < phase_updates:
            break
        state["phase"] += 1
        state["phase_updates"] = 0
        state["obs"] = None
        state["env_state"] = None
        state["hidden"] = None
        state["starts"] = None
        save_checkpoint(checkpoint, state, cfg)
        if STOP or time.monotonic() >= deadline:
            break

    progress = {"complete": bool(state["phase"] >= cfg["rounds"]),
                "phase": int(state["phase"]), "phase_updates": int(state["phase_updates"]),
                "target_phase_updates": phase_updates,
                "elapsed_this_job_seconds": time.monotonic() - start_time}
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    progress_file.write_text(json.dumps(progress, indent=2) + "\n")
    print(json.dumps(progress), flush=True)


if __name__ == "__main__":
    main()

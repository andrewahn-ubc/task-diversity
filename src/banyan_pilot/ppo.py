from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import torch

from .config import PPOConfig
from .env import ACTION_COUNT, CompactObs, VectorBanyan, stack_observations
from .model import RecurrentActorCritic

if TYPE_CHECKING:
    from .continual_backprop import ContinualBackprop


@dataclasses.dataclass
class Rollout:
    obs: CompactObs
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    episode_starts: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    initial_hidden: torch.Tensor
    next_obs: CompactObs
    next_episode_start: torch.Tensor
    next_hidden: torch.Tensor


@torch.no_grad()
def collect_rollout(
    model: RecurrentActorCritic,
    env: VectorBanyan,
    obs: CompactObs,
    hidden: torch.Tensor,
    episode_start: torch.Tensor,
    *,
    rollout_steps: int,
    gamma: float,
    gae_lambda: float,
) -> Rollout:
    observations: list[CompactObs] = []
    actions: list[torch.Tensor] = []
    logprobs: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    dones: list[torch.Tensor] = []
    starts: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    initial_hidden = hidden.clone()
    for _ in range(rollout_steps):
        observations.append(obs.clone())
        starts.append(episode_start.clone())
        action, logprob, _, value, next_hidden = model.act(obs, hidden, episode_start)
        next_obs, reward, done, _ = env.step(action)
        actions.append(action)
        logprobs.append(logprob)
        rewards.append(reward)
        dones.append(done)
        values.append(value)
        obs, hidden, episode_start = next_obs, next_hidden, done
    _, next_value, _ = model.forward_step(obs, hidden, episode_start)
    reward_tensor = torch.stack(rewards)
    done_tensor = torch.stack(dones)
    value_tensor = torch.stack(values)
    advantages = torch.zeros_like(reward_tensor)
    last_gae = torch.zeros(env.num_envs, device=reward_tensor.device)
    for step in reversed(range(rollout_steps)):
        if step == rollout_steps - 1:
            next_nonterminal = (~done_tensor[step]).float()
            following_value = next_value
        else:
            next_nonterminal = (~done_tensor[step]).float()
            following_value = value_tensor[step + 1]
        delta = reward_tensor[step] + gamma * following_value * next_nonterminal - value_tensor[step]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[step] = last_gae
    returns = advantages + value_tensor
    return Rollout(
        obs=stack_observations(observations),
        actions=torch.stack(actions),
        logprobs=torch.stack(logprobs),
        rewards=reward_tensor,
        dones=done_tensor,
        episode_starts=torch.stack(starts),
        values=value_tensor,
        advantages=advantages,
        returns=returns,
        initial_hidden=initial_hidden,
        next_obs=obs,
        next_episode_start=episode_start,
        next_hidden=hidden,
    )


def _slice_envs(obs: CompactObs, indices: torch.Tensor) -> CompactObs:
    return CompactObs(*(value[:, indices] for value in obs.as_tuple()))


def ppo_losses(
    model: RecurrentActorCritic,
    rollout: Rollout,
    config: PPOConfig,
    env_indices: torch.Tensor | None = None,
    *,
    capture_cbp: bool = False,
) -> dict[str, torch.Tensor]:
    if env_indices is None:
        env_indices = torch.arange(rollout.actions.shape[1], device=rollout.actions.device)
    new_logprobs, entropy, new_values = model.evaluate_sequence(
        _slice_envs(rollout.obs, env_indices),
        rollout.initial_hidden[env_indices],
        rollout.episode_starts[:, env_indices],
        rollout.actions[:, env_indices],
        capture_cbp=capture_cbp,
    )
    old_logprobs = rollout.logprobs[:, env_indices]
    advantages = rollout.advantages[:, env_indices]
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    ratio = (new_logprobs - old_logprobs).exp()
    policy_unclipped = -advantages * ratio
    policy_clipped = -advantages * torch.clamp(
        ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef
    )
    policy_loss = torch.maximum(policy_unclipped, policy_clipped).mean()
    value_loss = 0.5 * (new_values - rollout.returns[:, env_indices]).square().mean()
    entropy_loss = entropy.mean()
    total = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_loss
    approx_kl = ((ratio - 1.0) - (new_logprobs - old_logprobs)).mean()
    clip_fraction = ((ratio - 1.0).abs() > config.clip_coef).float().mean()
    return {
        "total": total,
        "policy": policy_loss,
        "value": value_loss,
        "entropy": entropy_loss,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
    }


def update_ppo(
    model: RecurrentActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    config: PPOConfig,
    generator: torch.Generator,
    continual_backprop: ContinualBackprop | None = None,
) -> dict[str, float]:
    num_envs = rollout.actions.shape[1]
    accumulated: dict[str, list[float]] = {}
    for _ in range(config.update_epochs):
        permutation = torch.randperm(num_envs, generator=generator, device=rollout.actions.device)
        for start in range(0, num_envs, config.minibatch_envs):
            indices = permutation[start : start + config.minibatch_envs]
            losses = ppo_losses(
                model,
                rollout,
                config,
                indices,
                capture_cbp=continual_backprop is not None,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            if continual_backprop is not None:
                continual_backprop.step(model.cbp_feature_stats())
                recurrent = continual_backprop.last_gru_indices
                if recurrent.numel():
                    rollout.initial_hidden[:, recurrent] = 0.0
                    rollout.next_hidden[:, recurrent] = 0.0
            values = {name: value.detach().item() for name, value in losses.items()}
            values["grad_norm"] = float(grad_norm)
            for name, value in values.items():
                accumulated.setdefault(name, []).append(value)
    result = {name: float(sum(values) / len(values)) for name, values in accumulated.items()}
    if continual_backprop is not None:
        for name, count in continual_backprop.totals().items():
            result[f"cbp_total_replacements_{name}"] = float(count)
    return result


@torch.no_grad()
def _evaluate_subset(
    model: RecurrentActorCritic,
    catalog_task_ids: torch.Tensor,
    env_factory: Any,
    *,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    device = next(model.parameters()).device
    successes = torch.zeros((), dtype=torch.int64, device=device)
    dead_ends = torch.zeros((), dtype=torch.int64, device=device)
    timeouts = torch.zeros((), dtype=torch.int64, device=device)
    completed = 0
    depth_success = torch.zeros(7, dtype=torch.float64, device=device)
    depth_count = torch.zeros(7, dtype=torch.float64, device=device)
    action_counts = torch.zeros(ACTION_COUNT, dtype=torch.int64, device=device)
    effective_counts = {
        name: torch.zeros((), dtype=torch.int64, device=device)
        for name in (
            "movement",
            "pickup",
            "drop",
            "merge",
            "toggle",
        )
    }
    evaluation_env_transitions = torch.zeros((), dtype=torch.int64, device=device)
    while completed < episodes:
        round_seed = seed + 10_000_019 * completed
        env: VectorBanyan = env_factory(catalog_task_ids, round_seed)
        obs = env.observe()
        hidden = model.initial_hidden(env.num_envs, env.device)
        episode_start = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        take = min(episodes - completed, env.num_envs)
        active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        active[:take] = True
        action_generator = torch.Generator(device=env.device)
        action_generator.manual_seed(round_seed + 99173)
        for _ in range(env.max_episode_steps):
            action, _, _, _, hidden = model.act(
                obs,
                hidden,
                episode_start,
                deterministic=False,
                generator=action_generator,
            )
            active_before = active.clone()
            obs, _, done, info = env.step(action)
            active_actions = action[active_before]
            action_counts += torch.bincount(
                active_actions, minlength=ACTION_COUNT
            )
            evaluation_env_transitions += active_before.sum()
            for name in effective_counts:
                effective_counts[name] += info[f"{name}_effective"][active_before].sum()
            finished = done & active_before
            successes += (info["success"] & finished).sum()
            dead_ends += (info["dead_end"] & finished).sum()
            timeouts += (info["timeout"] & finished).sum()
            for depth in range(1, 7):
                depth_finished = finished & (info["depth"] == depth)
                depth_count[depth] += depth_finished.sum()
                depth_success[depth] += (info["success"] & depth_finished).sum()
            active &= ~done
            episode_start = done
        if active.any().item():
            raise RuntimeError("Evaluation episode exceeded the configured horizon")
        completed += take
    result = {
        "success_rate": int(successes.item()) / completed,
        "dead_end_rate": int(dead_ends.item()) / completed,
        "timeout_rate": int(timeouts.item()) / completed,
        "episodes": completed,
        "evaluation_env_transitions": int(evaluation_env_transitions.item()),
    }
    for depth in range(1, 7):
        denominator = max(1.0, float(depth_count[depth].item()))
        result[f"success_depth_{depth}"] = float(depth_success[depth].item()) / denominator
    action_names = ("up", "down", "left", "right", "stay", "pickup", "drop", "toggle")
    denominator = max(1, int(evaluation_env_transitions.item()))
    for index, name in enumerate(action_names):
        result[f"action_{name}_rate"] = float(action_counts[index].item()) / denominator
    integer_effective_counts = {
        name: int(count.item()) for name, count in effective_counts.items()
    }
    for name, count in integer_effective_counts.items():
        result[f"effective_{name}_rate"] = count / denominator
    manipulation_attempts = int(action_counts[5:].sum().item())
    effective_manipulations = (
        integer_effective_counts["pickup"]
        + integer_effective_counts["drop"]
        + integer_effective_counts["toggle"]
    )
    result["manipulation_attempt_rate"] = manipulation_attempts / denominator
    result["effective_manipulation_rate"] = effective_manipulations / denominator
    result["noop_action_rate"] = 1.0 - (
        integer_effective_counts["movement"] + effective_manipulations
    ) / denominator
    return result


@torch.no_grad()
def evaluate_policy(
    model: RecurrentActorCritic,
    catalog_task_ids: torch.Tensor,
    env_factory: Any,
    *,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    """Evaluate with equal depth weight, matching the paper's aggregate metric.

    Episodes are allocated as evenly as possible across the available depths.
    The headline success, timeout, dead-end, and action rates are arithmetic
    means of the per-depth rates, so short shallow episodes cannot dominate the
    result merely because they finish and reset faster.
    """
    if episodes < 1:
        raise ValueError("Evaluation requires at least one episode")
    probe = env_factory(catalog_task_ids, seed)
    task_ids = catalog_task_ids.to(probe.device)
    task_depths = probe.catalog_depths[task_ids]
    depths = sorted(set(int(value) for value in task_depths.tolist()))
    if episodes < len(depths):
        raise ValueError(
            f"Evaluation episodes ({episodes}) must cover all {len(depths)} depths"
        )
    quotient, remainder = divmod(episodes, len(depths))
    per_depth: dict[int, dict[str, float]] = {}
    for offset, depth in enumerate(depths):
        depth_ids = task_ids[task_depths == depth].cpu()
        per_depth[depth] = _evaluate_subset(
            model,
            depth_ids,
            env_factory,
            episodes=quotient + (1 if offset < remainder else 0),
            seed=seed + 1_000_003 * depth,
        )
    averaged_keys = (
        "success_rate",
        "dead_end_rate",
        "timeout_rate",
        "action_up_rate",
        "action_down_rate",
        "action_left_rate",
        "action_right_rate",
        "action_stay_rate",
        "action_pickup_rate",
        "action_drop_rate",
        "action_toggle_rate",
        "effective_movement_rate",
        "effective_pickup_rate",
        "effective_drop_rate",
        "effective_merge_rate",
        "effective_toggle_rate",
        "manipulation_attempt_rate",
        "effective_manipulation_rate",
        "noop_action_rate",
    )
    result = {
        key: sum(per_depth[depth][key] for depth in depths) / len(depths)
        for key in averaged_keys
    }
    result["episodes"] = sum(int(per_depth[depth]["episodes"]) for depth in depths)
    result["evaluation_env_transitions"] = sum(
        int(per_depth[depth]["evaluation_env_transitions"]) for depth in depths
    )
    for depth in range(1, 7):
        if depth not in per_depth:
            result[f"episodes_depth_{depth}"] = 0
            result[f"success_depth_{depth}"] = 0.0
            result[f"timeout_depth_{depth}"] = 0.0
            result[f"dead_end_depth_{depth}"] = 0.0
            result[f"effective_manipulation_depth_{depth}"] = 0.0
            continue
        row = per_depth[depth]
        result[f"episodes_depth_{depth}"] = int(row["episodes"])
        result[f"success_depth_{depth}"] = row[f"success_depth_{depth}"]
        result[f"timeout_depth_{depth}"] = row["timeout_rate"]
        result[f"dead_end_depth_{depth}"] = row["dead_end_rate"]
        result[f"effective_manipulation_depth_{depth}"] = row[
            "effective_manipulation_rate"
        ]
    return result


def flattened_gradients(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    pieces = [
        torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ]
    return torch.cat(pieces)


def gradient_cosines(
    model: RecurrentActorCritic,
    current: Rollout,
    previous: Rollout,
    config: PPOConfig,
) -> dict[str, float]:
    current_loss = ppo_losses(model, current, config)["total"]
    previous_loss = ppo_losses(model, previous, config)["total"]
    groups = model.parameter_groups()
    current_vectors: dict[str, torch.Tensor] = {}
    previous_vectors: dict[str, torch.Tensor] = {}
    names = list(groups)
    for index, name in enumerate(names):
        current_vectors[name] = flattened_gradients(
            current_loss, groups[name], retain_graph=True
        )
        previous_vectors[name] = flattened_gradients(
            previous_loss, groups[name], retain_graph=index < len(names) - 1
        )
    result: dict[str, float] = {}
    for name in names:
        left, right = current_vectors[name], previous_vectors[name]
        denominator = left.norm() * right.norm()
        cosine = torch.tensor(float("nan"), device=left.device)
        if denominator.item() > 0:
            cosine = torch.dot(left, right) / denominator
        result[f"cosine_{name}"] = float(cosine.detach().cpu())
        result[f"norm_current_{name}"] = float(left.norm().detach().cpu())
        result[f"norm_previous_{name}"] = float(right.norm().detach().cpu())
    return result

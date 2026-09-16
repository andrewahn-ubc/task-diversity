"""metrics.py

Tournament-level metrics helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import wandb


def update_transfer_tensor(
    transfer_tensor: dict[tuple[int, int], dict[str, dict[int, dict[str, float]]]],
    eval_results: dict[tuple[int, int, str], dict[str, float]],
    round_idx: int,
) -> dict[tuple[int, int], dict[str, dict[int, dict[str, float]]]]:
    """Merge eval results into the transfer tensor."""
    for (i, j, task_set), metrics in eval_results.items():
        transfer_tensor.setdefault((i, j), {}).setdefault(task_set, {})[round_idx] = (
            metrics
        )
    return transfer_tensor


def tensor_to_wandb_table(
    transfer_tensor: dict[tuple[int, int], dict[str, dict[int, dict[str, float]]]],
) -> wandb.Table:
    """Convert transfer tensor to a WandB Table for logging."""
    # Determine max number of probes across all metrics
    max_probes = 0
    for task_dict in transfer_tensor.values():
        for round_dict in task_dict.values():
            for metrics in round_dict.values():
                num_probes = metrics.get("num_probes", 0)
                if isinstance(num_probes, (int, float)):
                    max_probes = max(max_probes, int(num_probes))

    columns = [
        "agent_i",
        "agent_j",
        "task_set",
        "round",
        "success_rate",
        "success_rate_d1",
        "success_rate_d2",
        "success_rate_d3",
        "success_rate_d4",
        "success_rate_d5",
        "success_rate_d6",
        "num_episodes",
        "avg_return",
        "avg_ep_length",
        "avg_ep_length_d1",
        "avg_ep_length_d2",
        "avg_ep_length_d3",
        "avg_ep_length_d4",
        "avg_ep_length_d5",
        "avg_ep_length_d6",
    ]

    # Add probe columns dynamically
    for p in range(max_probes):
        columns.append(f"probe_{p}_success_rate")
        columns.append(f"probe_{p}_episodes")

    data = []

    for (i, j), task_dict in transfer_tensor.items():
        for task_set, round_dict in task_dict.items():
            for round_idx, metrics in round_dict.items():
                row = [
                    i,
                    j,
                    task_set,
                    round_idx,
                    metrics.get("success_rate", 0.0),
                    metrics.get("success_rate_d1", 0.0),
                    metrics.get("success_rate_d2", 0.0),
                    metrics.get("success_rate_d3", 0.0),
                    metrics.get("success_rate_d4", 0.0),
                    metrics.get("success_rate_d5", 0.0),
                    metrics.get("success_rate_d6", 0.0),
                    metrics.get("num_episodes", 0.0),
                    metrics.get("avg_return", 0.0),
                    metrics.get("avg_ep_length", 0.0),
                    metrics.get("avg_ep_length_d1", 0.0),
                    metrics.get("avg_ep_length_d2", 0.0),
                    metrics.get("avg_ep_length_d3", 0.0),
                    metrics.get("avg_ep_length_d4", 0.0),
                    metrics.get("avg_ep_length_d5", 0.0),
                    metrics.get("avg_ep_length_d6", 0.0),
                ]

                # Add probe metrics
                probe_success_rates = metrics.get("probe_success_rates", [])
                probe_episodes = metrics.get("probe_episodes", [])
                for p in range(max_probes):
                    if p < len(probe_success_rates):
                        row.append(probe_success_rates[p])
                    else:
                        row.append(0.0)
                    if p < len(probe_episodes):
                        row.append(probe_episodes[p])
                    else:
                        row.append(0.0)

                data.append(row)

    import wandb

    return wandb.Table(columns=columns, data=data)

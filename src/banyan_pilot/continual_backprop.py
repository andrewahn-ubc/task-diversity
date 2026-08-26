from __future__ import annotations

import dataclasses
from typing import Any

import torch
from torch import nn

from .config import CBPConfig
from .continual_adam import ContinualAdam
from .model import RecurrentActorCritic


@dataclasses.dataclass
class _LayerState:
    utility: torch.Tensor
    mean_activation: torch.Tensor
    ages: torch.Tensor
    replacement_accumulator: float = 0.0
    total_replacements: int = 0


class ContinualBackprop:
    """Generate-and-test CBP adapted to this shared convolutional GRU policy.

    The feed-forward replacement rule follows Dohare et al.'s official
    implementation.  A GRU hidden unit is treated as the three corresponding
    gate rows plus its recurrent/output columns, so no recurrent feature block
    is left outside the plasticity intervention.
    """

    FORMAT_VERSION = 1
    LAYER_NAMES = ("conv1", "conv2", "pre_gru", "gru")

    def __init__(
        self,
        model: RecurrentActorCritic,
        optimizer: ContinualAdam,
        config: CBPConfig,
    ) -> None:
        if config.utility != "contribution":
            raise ValueError("Only the published contribution utility is supported")
        self.model = model
        self.optimizer = optimizer
        self.config = config
        device = next(model.parameters()).device
        sizes = {
            "conv1": model.encoder[0].out_channels,
            "conv2": model.encoder[2].out_channels,
            "pre_gru": model.pre_gru[0].out_features,
            "gru": model.hidden_size,
        }
        self.layers = {
            name: _LayerState(
                utility=torch.zeros(size, device=device),
                mean_activation=torch.zeros(
                    (size, model.grid_size * model.grid_size)
                    if name == "conv2"
                    else size,
                    device=device,
                ),
                ages=torch.zeros(size, device=device),
            )
            for name, size in sizes.items()
        }
        self._initial_row_norms = {
            "conv1": self._row_norms(model.encoder[0].weight),
            "conv2": self._row_norms(model.encoder[2].weight),
            "pre_gru": self._row_norms(model.pre_gru[0].weight),
            "gru_ih": self._row_norms(model.gru.weight_ih),
            "gru_hh": self._row_norms(model.gru.weight_hh),
        }
        self.last_gru_indices = torch.empty(0, dtype=torch.long, device=device)

    @staticmethod
    def _row_norms(weight: torch.Tensor) -> torch.Tensor:
        return weight.detach().flatten(1).norm(dim=1).clone()

    @staticmethod
    def _bias_corrected(value: torch.Tensor, decay: float, ages: torch.Tensor) -> torch.Tensor:
        correction = 1.0 - torch.pow(
            torch.as_tensor(decay, device=value.device, dtype=value.dtype), ages
        )
        while correction.ndim < value.ndim:
            correction = correction.unsqueeze(-1)
        return value / correction.clamp_min(torch.finfo(value.dtype).eps)

    def _outgoing_magnitude(self, name: str) -> torch.Tensor:
        model = self.model
        if name == "conv1":
            return model.encoder[2].weight.detach().abs().mean(dim=(0, 2, 3))
        if name == "conv2":
            weight = model.pre_gru[0].weight.detach().abs()
            pixels = model.grid_size * model.grid_size
            channels = model.encoder[2].out_channels
            return weight[:, : channels * pixels].reshape(
                weight.shape[0], channels, pixels
            ).mean(dim=0)
        if name == "pre_gru":
            return model.gru.weight_ih.detach().abs().mean(dim=0)
        if name == "gru":
            outgoing = torch.cat(
                (
                    model.gru.weight_hh.detach(),
                    model.actor.weight.detach(),
                    model.critic.weight.detach(),
                ),
                dim=0,
            )
            return outgoing.abs().mean(dim=0)
        raise KeyError(name)

    @torch.no_grad()
    def step(
        self, feature_stats: dict[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> dict[str, int]:
        if set(feature_stats) != set(self.LAYER_NAMES):
            raise ValueError(
                f"CBP feature statistics mismatch: expected {self.LAYER_NAMES}, "
                f"found {sorted(feature_stats)}"
            )
        replacements: dict[str, int] = {}
        self.last_gru_indices = torch.empty(
            0, dtype=torch.long, device=self.last_gru_indices.device
        )
        for name in self.LAYER_NAMES:
            mean, mean_abs = feature_stats[name]
            state = self.layers[name]
            if (
                mean.shape != state.mean_activation.shape
                or mean_abs.shape != state.mean_activation.shape
            ):
                raise ValueError(f"Invalid CBP statistics shape for {name}")
            state.ages.add_(1.0)
            state.mean_activation.mul_(self.config.decay_rate).add_(
                mean, alpha=1.0 - self.config.decay_rate
            )
            instantaneous = mean_abs * self._outgoing_magnitude(name)
            if instantaneous.ndim == 2:
                instantaneous = instantaneous.mean(dim=1)
            state.utility.mul_(self.config.decay_rate).add_(
                instantaneous, alpha=1.0 - self.config.decay_rate
            )
            eligible = torch.nonzero(
                state.ages > self.config.maturity_threshold, as_tuple=False
            ).flatten()
            state.replacement_accumulator += (
                self.config.replacement_rate * float(eligible.numel())
            )
            count = min(int(state.replacement_accumulator), int(eligible.numel()))
            if count == 0:
                replacements[name] = 0
                continue
            corrected_utility = self._bias_corrected(
                state.utility, self.config.decay_rate, state.ages
            )
            selected = eligible[
                torch.topk(-corrected_utility[eligible], k=count).indices
            ]
            corrected_mean = self._bias_corrected(
                state.mean_activation, self.config.decay_rate, state.ages
            )[selected].clone()
            self._replace(name, selected, corrected_mean)
            if name == "gru":
                self.last_gru_indices = selected.clone()
            state.utility[selected] = 0.0
            state.mean_activation[selected] = 0.0
            state.ages[selected] = 0.0
            state.replacement_accumulator -= count
            state.total_replacements += count
            replacements[name] = count
        return replacements

    @torch.no_grad()
    def _resample_rows(
        self, weight: nn.Parameter, rows: torch.Tensor, target_norms: torch.Tensor
    ) -> None:
        fresh = torch.randn_like(weight[rows]).flatten(1)
        fresh.div_(fresh.norm(dim=1, keepdim=True).clamp_min(torch.finfo(fresh.dtype).eps))
        fresh.mul_(target_norms[rows, None])
        weight[rows] = fresh.reshape_as(weight[rows])
        self.optimizer.reset_state(weight, (rows, ...))

    @torch.no_grad()
    def _replace(self, name: str, selected: torch.Tensor, mean: torch.Tensor) -> None:
        model = self.model
        if name == "conv1":
            incoming = model.encoder[0]
            outgoing = model.encoder[2]
            outgoing.weight[:, selected] = 0.0
            self.optimizer.reset_state(outgoing.weight, (slice(None), selected, ...))
            self._resample_rows(incoming.weight, selected, self._initial_row_norms[name])
            incoming.bias[selected] = 0.0
            self.optimizer.reset_state(incoming.bias, selected)
            return
        if name == "conv2":
            incoming = model.encoder[2]
            outgoing = model.pre_gru[0]
            pixels = model.grid_size * model.grid_size
            offsets = torch.arange(pixels, device=selected.device)
            columns = (selected[:, None] * pixels + offsets[None]).reshape(-1)
            outgoing.weight[:, columns] = 0.0
            self.optimizer.reset_state(outgoing.weight, (slice(None), columns))
            self._resample_rows(incoming.weight, selected, self._initial_row_norms[name])
            incoming.bias[selected] = 0.0
            self.optimizer.reset_state(incoming.bias, selected)
            return
        if name == "pre_gru":
            incoming = model.pre_gru[0]
            outgoing_weight = model.gru.weight_ih
            old = outgoing_weight[:, selected].clone()
            model.gru.bias_ih.add_(old @ mean)
            outgoing_weight[:, selected] = 0.0
            self.optimizer.reset_state(outgoing_weight, (slice(None), selected))
            self._resample_rows(incoming.weight, selected, self._initial_row_norms[name])
            incoming.bias[selected] = 0.0
            self.optimizer.reset_state(incoming.bias, selected)
            return
        if name == "gru":
            hidden = model.hidden_size
            gate_rows = torch.cat([selected + gate * hidden for gate in range(3)])
            old_recurrent = model.gru.weight_hh[:, selected].clone()
            old_policy = model.actor.weight[:, selected].clone()
            old_value = model.critic.weight[:, selected].clone()
            model.gru.bias_hh.add_(old_recurrent @ mean)
            model.actor.bias.add_(old_policy @ mean)
            model.critic.bias.add_(old_value @ mean)
            self._resample_rows(
                model.gru.weight_ih, gate_rows, self._initial_row_norms["gru_ih"]
            )
            self._resample_rows(
                model.gru.weight_hh, gate_rows, self._initial_row_norms["gru_hh"]
            )
            model.gru.bias_ih[gate_rows] = 0.0
            model.gru.bias_hh[gate_rows] = 0.0
            self.optimizer.reset_state(model.gru.bias_ih, gate_rows)
            self.optimizer.reset_state(model.gru.bias_hh, gate_rows)
            # GRU incoming rows and outgoing columns intersect in weight_hh.
            # Zero the columns *after* row generation so the new feature has
            # no outgoing influence, including no recurrent self-connection.
            model.gru.weight_hh[:, selected] = 0.0
            model.actor.weight[:, selected] = 0.0
            model.critic.weight[:, selected] = 0.0
            self.optimizer.reset_state(
                model.gru.weight_hh, (slice(None), selected)
            )
            self.optimizer.reset_state(model.actor.weight, (slice(None), selected))
            self.optimizer.reset_state(model.critic.weight, (slice(None), selected))
            return
        raise KeyError(name)

    def totals(self) -> dict[str, int]:
        return {name: self.layers[name].total_replacements for name in self.LAYER_NAMES}

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.FORMAT_VERSION,
            "config": dataclasses.asdict(self.config),
            "initial_row_norms": {
                name: value.clone() for name, value in self._initial_row_norms.items()
            },
            "layers": {
                name: {
                    "utility": state.utility.clone(),
                    "mean_activation": state.mean_activation.clone(),
                    "ages": state.ages.clone(),
                    "replacement_accumulator": state.replacement_accumulator,
                    "total_replacements": state.total_replacements,
                }
                for name, state in self.layers.items()
            },
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if payload.get("format_version") != self.FORMAT_VERSION:
            raise ValueError("Unsupported CBP checkpoint format")
        if payload.get("config") != dataclasses.asdict(self.config):
            raise ValueError("CBP checkpoint configuration mismatch")
        if set(payload.get("layers", {})) != set(self.LAYER_NAMES):
            raise ValueError("CBP checkpoint layer mismatch")
        if set(payload.get("initial_row_norms", {})) != set(self._initial_row_norms):
            raise ValueError("CBP checkpoint initialization metadata mismatch")
        for name, destination in self._initial_row_norms.items():
            value = payload["initial_row_norms"][name].to(destination.device)
            if value.shape != destination.shape:
                raise ValueError(f"CBP checkpoint initialization shape mismatch for {name}")
            destination.copy_(value)
        for name, state in self.layers.items():
            saved = payload["layers"][name]
            for key in ("utility", "mean_activation", "ages"):
                destination = getattr(state, key)
                value = saved[key].to(destination.device)
                if value.shape != destination.shape:
                    raise ValueError(f"CBP checkpoint shape mismatch for {name}.{key}")
                destination.copy_(value)
            state.replacement_accumulator = float(saved["replacement_accumulator"])
            state.total_replacements = int(saved["total_replacements"])

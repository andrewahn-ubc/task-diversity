from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Categorical

from .env import ACTION_COUNT, CompactObs
from .taskgen import object_features


def _layer_init(layer: nn.Module, std: float = math.sqrt(2.0)) -> nn.Module:
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, std)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
    return layer


class RecurrentActorCritic(nn.Module):
    def __init__(
        self,
        *,
        grid_size: int,
        object_feature_dim: int,
        hidden_size: int,
        walls: torch.Tensor,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.object_feature_dim = object_feature_dim
        self.hidden_size = hidden_size
        self.register_buffer("object_feature_table", object_features(object_feature_dim))
        self.register_buffer("walls", walls.float()[None, None])
        in_channels = object_feature_dim + 3
        self.encoder = nn.Sequential(
            _layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)),
            nn.Tanh(),
            _layer_init(nn.Conv2d(32, 32, kernel_size=3, padding=1)),
            nn.Tanh(),
            nn.Flatten(),
        )
        encoder_size = 32 * grid_size * grid_size
        self.pre_gru = nn.Sequential(
            _layer_init(
                nn.Linear(encoder_size + 2 * object_feature_dim + 3, hidden_size)
            ),
            nn.Tanh(),
        )
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        for name, parameter in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(parameter, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(parameter, 1.0)
        self.actor = _layer_init(nn.Linear(hidden_size, ACTION_COUNT), std=0.01)
        self.critic = _layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch_size, self.hidden_size), device=device)

    def _encode(self, obs: CompactObs) -> torch.Tensor:
        batch = obs.objects.shape[0]
        valid = obs.objects >= 0
        ids = obs.objects.clamp(min=0)
        object_grid = self.object_feature_table[ids] * valid[..., None]
        object_grid = object_grid.permute(0, 3, 1, 2)
        agent_grid = torch.zeros(
            (batch, 1, self.grid_size, self.grid_size),
            device=obs.objects.device,
            dtype=torch.float32,
        )
        rows = torch.arange(batch, device=obs.objects.device)
        agent_grid[rows, 0, obs.agent[:, 0], obs.agent[:, 1]] = 1.0
        object_present = valid.float()[:, None]
        walls = self.walls.expand(batch, -1, -1, -1)
        grid = torch.cat((walls, agent_grid, object_present, object_grid), dim=1)
        encoded_grid = self.encoder(grid)
        goal = self.object_feature_table[obs.goal]
        held_valid = obs.held >= 0
        held = self.object_feature_table[obs.held.clamp(min=0)] * held_valid[:, None]
        extras = torch.cat(
            (
                goal,
                held,
                held_valid.float()[:, None],
                obs.elapsed.float()[:, None],
                torch.ones((batch, 1), device=obs.objects.device),
            ),
            dim=1,
        )
        return self.pre_gru(torch.cat((encoded_grid, extras), dim=1))

    def forward_step(
        self, obs: CompactObs, hidden: torch.Tensor, episode_start: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = hidden * (~episode_start.bool()).float()[:, None]
        hidden = self.gru(self._encode(obs), hidden)
        return self.actor(hidden), self.critic(hidden).squeeze(-1), hidden

    def act(
        self,
        obs: CompactObs,
        hidden: torch.Tensor,
        episode_start: torch.Tensor,
        *,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value, next_hidden = self.forward_step(obs, hidden, episode_start)
        distribution = Categorical(logits=logits)
        if action is None:
            if deterministic:
                action = logits.argmax(dim=-1)
            elif generator is None:
                action = distribution.sample()
            else:
                action = torch.multinomial(
                    distribution.probs, 1, generator=generator
                ).squeeze(-1)
        return action, distribution.log_prob(action), distribution.entropy(), value, next_hidden

    def evaluate_sequence(
        self,
        obs: CompactObs,
        initial_hidden: torch.Tensor,
        episode_starts: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logprobs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        hidden = initial_hidden
        time_steps, batch_size = actions.shape
        flat_obs = CompactObs(
            *(value.reshape(time_steps * batch_size, *value.shape[2:]) for value in obs.as_tuple())
        )
        encoded = self._encode(flat_obs).reshape(time_steps, batch_size, self.hidden_size)
        for step in range(time_steps):
            hidden = hidden * (~episode_starts[step].bool()).float()[:, None]
            hidden = self.gru(encoded[step], hidden)
            logits = self.actor(hidden)
            value = self.critic(hidden).squeeze(-1)
            distribution = Categorical(logits=logits)
            logprobs.append(distribution.log_prob(actions[step]))
            entropies.append(distribution.entropy())
            values.append(value)
        return torch.stack(logprobs), torch.stack(entropies), torch.stack(values)

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        shared = list(self.encoder.parameters()) + list(self.pre_gru.parameters()) + list(
            self.gru.parameters()
        )
        return {
            "all": list(self.parameters()),
            "shared": shared,
            "policy": list(self.actor.parameters()),
            "value": list(self.critic.parameters()),
        }

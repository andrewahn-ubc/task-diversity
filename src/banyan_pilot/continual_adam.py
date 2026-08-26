from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch


class ContinualAdam(torch.optim.Optimizer):
    """Adam with element-wise step counters that CBP can selectively reset.

    Standard PyTorch Adam stores one scalar step counter per parameter tensor.
    Continual Backprop replaces individual neurons, so their incoming and
    outgoing parameter slices must receive fresh first/second moments *and*
    fresh bias-correction ages.  This optimizer is otherwise standard Adam.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid Adam betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        super().__init__(
            params,
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
            },
        )

    def _initialize_state(self, parameter: torch.nn.Parameter) -> dict[str, Any]:
        state = self.state[parameter]
        if not state:
            state["step"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
            state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(
                parameter, memory_format=torch.preserve_format
            )
        return state

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("ContinualAdam does not support sparse gradients")
                if group["weight_decay"]:
                    gradient = gradient.add(parameter, alpha=group["weight_decay"])
                state = self._initialize_state(parameter)
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                step.add_(1.0)
                exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                bias_correction1 = 1.0 - torch.pow(
                    torch.as_tensor(beta1, device=parameter.device, dtype=parameter.dtype), step
                )
                bias_correction2 = 1.0 - torch.pow(
                    torch.as_tensor(beta2, device=parameter.device, dtype=parameter.dtype), step
                )
                corrected_mean = exp_avg / bias_correction1
                corrected_variance = exp_avg_sq / bias_correction2
                denominator = corrected_variance.sqrt().add_(group["eps"])
                parameter.addcdiv_(corrected_mean, denominator, value=-group["lr"])
        return loss

    @torch.no_grad()
    def reset_state(self, parameter: torch.nn.Parameter, index: Any) -> None:
        """Reset Adam moments and bias-correction ages for one parameter slice."""

        state = self.state.get(parameter)
        if not state:
            return
        state["step"][index] = 0.0
        state["exp_avg"][index] = 0.0
        state["exp_avg_sq"][index] = 0.0

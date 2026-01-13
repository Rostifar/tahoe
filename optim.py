import math
import torch

from typing import Callable
from collections.abc import Callable, Iterable


def cosine_lr_scheduler(t: int, lr_max: float, lr_min: float, t_warmup: int, t_cos: int) -> float:
    assert t_warmup > 0 and t_cos > t_warmup, "Invalid params"
    if t < t_warmup:
        return (t / t_warmup) * lr_max
    elif t <= t_cos:
        angle = math.pi * (t - t_warmup) / (t_cos - t_warmup)
        return lr_min + 0.5 * (1 + math.cos(angle)) * (lr_max - lr_min)
    return lr_min


def clip_grads(params: Iterable, max_grad: float, eps: float = 1e-6) -> None:
    for param in params:
        if param.grad is None:
            continue
        norm = torch.norm(param.grad)
        if norm > max_grad:
            param.grad *= max_grad / (norm + eps)


class AdamW(torch.optim.Optimizer):
    def __init__(
        self, 
        params: Iterable,
        lr: float = 1e-3,
        betas: float = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ) -> None:
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None) -> float:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, betas, eps, weight_decay = (
                group["lr"], group["betas"], group["eps"], group["weight_decay"]
            )
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                # first and second moments, respectively
                m = state.get("m", torch.zeros_like(grad))
                v = state.get("v", torch.zeros_like(grad))
                t = state.get("t", 1)

                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * (grad**2)
                lr_t = lr * math.sqrt(1 - betas[1]**t) / (1 - betas[0]**t)
                p.data -= lr_t * (m / (torch.sqrt(v) + eps))
                p.data -= lr * weight_decay * p.data

                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
        return loss

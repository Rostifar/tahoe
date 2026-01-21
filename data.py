import os
import torch
import typing
import numpy as np
import torch.nn as nn
from collections.abc import Iterable


def yield_batch(
    x: np.array, 
    batch_size: int, 
    context_length: int, 
    device: str # "cpu" | "cuda:0" | "mps" | ...
) -> tuple[torch.Tensor, torch.Tensor]:
    # truncate to fit batch slice
    x = x[:batch_size * context_length + 1]
    shape = (batch_size, context_length)
    x = torch.tensor(x, dtype=torch.long).to(device)
    return x[:-1].view(*shape), x[1:].view(*shape)


def load_batches(
    x: np.array,
    batch_size: int, 
    context_length: int, 
    device: str,
    start_iteration: int = 1
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    batch_slice = batch_size * context_length
    # restart from previous iteration
    starting_pos = batch_slice * (start_iteration - 1)
    for i in range(starting_pos, len(x), batch_slice):
        if len(x) - i < batch_slice + 1:
            return
        yield yield_batch(x[i: i + batch_slice + 1], batch_size, context_length, device)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    payload = { 
        "model": model.state_dict(), 
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(payload, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device
) -> int:
    payload = torch.load(src, map_location=device)
    model.load_state_dict(payload["model"])
    if optimizer:
        optimizer.load_state_dict(payload["optimizer"])
    return payload["iteration"]

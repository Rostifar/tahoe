import os
import torch
import typing
import numpy as np
import torch.nn as nn

def yield_batch(
    x: np.array, 
    batch_size: int, 
    context_length: int, 
    device: str # "cpu" | "cuda:0" | "mps" | ...
) -> tuple[torch.Tensor, torch.Tensor]:
    # truncate to fit batch slice
    x = x[:batch_size * context_length + 1]
    shape = (batch_size, context_length)
    x = torch.tensor(x).to(device)
    return x[:-1].view(*shape), x[1:].view(*shape)

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
    optimizer: torch.optim.Optimizer
) -> int:
    payload = torch.load(src)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return payload["iteration"]

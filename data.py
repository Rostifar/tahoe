import torch
import numpy as np

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

import math
from regex import X
import torch
import torch.nn as nn

"""
Pieces to implement:
- Linear Module
- Embedding Module
"""

class Linear(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        # skip bias term, 
        self.W = nn.Parameter(torch.empty(out_dim, in_dim, dtype=dtype, device=device))
        std = math.sqrt(2. / (in_dim + out_dim))
        nn.init.trunc_normal_(self.W, mean=0., std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T


class Embedding(nn.Module):
    def __init__(
        self, 
        num_embeddings: int, 
        embed_dim: int, 
        device: torch.dtype | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.table = nn.Parameter(torch.empty(num_embeddings, embed_dim, dtype=dtype, device=device))
        nn.init.trunc_normal_(self.table, mean=0., std=1., a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.table[token_ids] 


class RMSNorm(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__() 
        self.d_model = d_model
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_model, "Invalid input type."
        in_dtype = x.dtype
        
        x = x.to(torch.float32)
        squares = torch.sum(x**2, dim=-1, keepdim=True)
        rms = torch.sqrt(self.eps + (squares / self.d_model))
        
        norm_x = x * (self.g / rms)
        return norm_x.to(in_dtype)

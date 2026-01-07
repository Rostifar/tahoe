import math
import torch
import torch.nn as nn


def softmax(x: torch.Tensor, dim: int=-1):
    x = torch.exp(x - x.max(dim=dim, keepdim=True).values)
    return x / x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    K: torch.Tensor,
    Q: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    d_k = K.shape[-1]
    scaled_dot = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scaled_dot = scaled_dot.masked_fill(~mask, float('-inf'))
    return torch.softmax(scaled_dot, dim=-1) @ V


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


class FCN(nn.Module):
    def __init__(
        self,
        d_model: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        kwargs = dict(device=device, dtype=dtype)
        
        # NB. round to nearest multiple of 64
        d_ff = int((8 / 3) * d_model)
        d_ff = 64 * round(d_ff / 64)
        self.sigmoid = nn.Sigmoid()
        self.w1 = Linear(in_dim=d_model, out_dim=d_ff, **kwargs)
        self.w2 = Linear(in_dim=d_model, out_dim=d_ff, **kwargs)
        self.w3 = Linear(in_dim=d_ff, out_dim=d_model, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.w1(x)
        silu = x_proj * self.sigmoid(x_proj)
        return self.w3(silu * self.w2(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self, 
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,  
    ) -> None:
        super().__init__()
        max_k = d_k // 2
        k = torch.arange(1, max_k + 1, device=device)
        
        freqs = 1.0 / (theta ** ((2 * k - 2) / d_k))  # Shape: (max_k,)
        pos = torch.arange(max_seq_len, device=device).unsqueeze(1)  # Shape: (max_seq_len, 1)
        angles = pos * freqs
        
        self.register_buffer(
            "cos",
            torch.cos(angles).repeat_interleave(2, dim=-1),
            persistent=False,
        )
        self.register_buffer(
            "sin",
            torch.sin(angles).repeat_interleave(2, dim=-1),
            persistent=False,
        )


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x1, x2 = x[..., ::2], x[..., 1::2]
        x_next = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return x * cos + x_next * sin


class MultiheadSelfAttention(nn.Module):
    """
    Naive: create separate modules for heads, concat results, and project.
    """
    def __init__(
        self, 
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = self.d_v = d_model // num_heads
        self.Q_proj = Linear(in_dim=d_model * num_heads, out_dim=self.d_k * num_heads, device=device, dtype=dtype)
        self.K_proj = Linear(in_dim=d_model * num_heads, out_dim=self.d_k * num_heads, device=device, dtype=dtype)
        self.V_proj = Linear(in_dim=d_model * num_heads, out_dim=self.d_v * num_heads, device=device, dtype=dtype)
        
        self.head_proj = Linear(in_dim=self.d_v * num_heads, out_dim=d_model, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.Q_proj(x)
        K = self.K_proj(x)
        V = self.V_proj(x)

        scaled_dot_product_attention(Q, K, V)
        
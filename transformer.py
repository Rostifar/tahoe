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
    def __init__(
        self, 
        d_model: int,
        num_heads: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = self.d_v = d_model // num_heads
        self.num_heads = num_heads
        
        self.Q_proj = Linear(in_dim=d_model, out_dim=self.d_k * num_heads, device=device, dtype=dtype)
        self.K_proj = Linear(in_dim=d_model, out_dim=self.d_k * num_heads, device=device, dtype=dtype)
        self.V_proj = Linear(in_dim=d_model, out_dim=self.d_v * num_heads, device=device, dtype=dtype)
        self.head_proj = Linear(in_dim=self.d_v * num_heads, out_dim=d_model, device=device, dtype=dtype)
        
        self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)
        self.head_proj = Linear(in_dim=self.d_v * num_heads, out_dim=d_model, device=device, dtype=dtype)
        self.register_buffer('mask', torch.tril(torch.ones(max_seq_len, max_seq_len, device=device)).bool())

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        *leading, d = x.shape
        return x.view(*leading, self.num_heads, d // self.num_heads).transpose(-2, -3)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        
        # (batch, seq_len, d_k * num_heads) -> (batch, num_heads, seq_len, d_k)
        Q = self._split_heads(self.Q_proj(x))
        K = self._split_heads(self.K_proj(x))
        V = self._split_heads(self.V_proj(x))

        positions = torch.arange(seq_len, device=x.device).expand(*Q.shape[:-2], seq_len)
        Q = self.rope(Q, token_positions=positions) 
        K = self.rope(K, token_positions=positions) 
        
        attn = scaled_dot_product_attention(K=K, Q=Q, V=V, mask=self.mask[:seq_len, :seq_len])
        # remove head dimension
        merged = attn.transpose(-2, -3).reshape(*x.shape[:-1], -1)
        return self.head_proj(merged)


class Block(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        theta: float,
        max_seq_len: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.pre_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype   
        )
        self.post_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.fcn = FCN(d_model, device=device, dtype=dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(self.pre_norm(x)) + x
        return self.fcn(self.post_norm(x)) + x


class Transformer(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        context_length: int, 
        num_layers: int,
        d_model: int,
        num_heads: int,
        theta: float,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        kwargs = dict(device=device, dtype=dtype)
        self.vocab_embed = Embedding(num_embeddings=vocab_size, embed_dim=d_model, **kwargs)
        self.blocks = nn.Sequential(*[
            Block(
                d_model=d_model,
                num_heads=num_heads,
                theta=theta,
                max_seq_len=context_length,
                d_ff=d_ff,
                **kwargs
            ) for _ in range(num_layers)
        ])
        self.post_norm = RMSNorm(d_model=d_model, **kwargs)
        self.lm_head = Linear(in_dim=d_model, out_dim=vocab_size, **kwargs)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vocab_embed(x)
        x = self.post_norm(self.blocks(x))
        return softmax(self.lm_head(x), dim=-1)

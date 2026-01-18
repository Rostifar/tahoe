import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def param_estimate(
    vocab_size: int,
    d_model: int,
    num_layers: int,
    d_ff: int
) -> int:
    return 2 * vocab_size * d_model + (2 * num_layers + 1) * d_model\
        + (4 * num_layers) * d_model**2 + 3 * num_layers * d_model * d_ff

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def estimate_flops(
    vocab_size: int,
    d_model: int,
    num_layers: int,
    d_ff: int,
    seq_len: int
) -> None:
    print("===Estimating Flops===")
    print(f"Parameters: " + ", ".join([f"{k}={v}" for k, v in locals().items() if k != 'self']))
    info = dict(
        attn_kqv_proj=num_layers * (6 * seq_len * d_model**2),
        attn_self_attn=num_layers * (4 * seq_len**2 * d_model),
        attn_head_proj=num_layers * 2 * seq_len * d_model**2,
        attn_fcn=num_layers * (6 * seq_len * d_model * d_ff),
        lm_proj=2 * seq_len * vocab_size * d_model
    )
    total_flops = sum(info.values())
    for k, v in info.items():
        proportion = v / total_flops if total_flops > 0 else 0
        print(f"{k} := {v / 10**12} TFLOPS ({proportion:.2%} of total)")
    print(f"Total TFLOPS: {total_flops / 10**12}\n")

def softmax(x: torch.Tensor, dim: int=-1):
    # substract max(..) for numerical stability, all elements will 
    # fall in the range [-inf, 0];
    # aside: softmax provides a nice prob dist since it handles negatives!
    x = torch.exp(x - x.max(dim=dim, keepdim=True).values)
    return x / x.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(
    K: torch.Tensor,
    Q: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    d_k, d_q = K.shape[-1], Q.shape[-1]
    assert d_k == d_q, f"Dim mismatch: d_k({d_k}) != d_q({d_q})"
    
    # K.T is needed since Q is row-oriented, broadcasting handles the rest;
    # aside: √d_k needed to bound variance, particularly when d_k >> 0
    scaled_dot = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scaled_dot = scaled_dot.masked_fill(~mask, float('-inf'))
    # V is on RHS as j-th column fills in the j-th component of weighted sum
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
        # skip bias term because that's what modern LLMs do
        kwargs = dict(device=device, dtype=dtype)
        self.W = nn.Parameter(torch.empty(out_dim, in_dim, **kwargs))

        # Xavier/Glorot: normalize to keep unit variance, even after summing 
        # ~ (in_dim + out_dim) elements.
        # Unit variance is necessary because variance is multiplicative across 
        # layers via independence: V[K-layers] ~ V[layer]^k
        # TODO: come back to this fact
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
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_model, f"Dim mismatch: {x.shape[-1]} != {self.d_model}."
        in_dtype = x.dtype
        
        # increase precision to prevent overflow from squares
        x = x.to(torch.float32)
        squares = torch.sum(x**2, dim=-1, keepdim=True)
        rms = torch.sqrt(self.eps + (squares / self.d_model))
        
        norm_x = x * (self.g / rms)
        return norm_x.to(in_dtype)


class FCN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        kwargs = dict(device=device, dtype=dtype)
        
        # Round to nearest multiple of 64
        if not d_ff:
            d_ff = int((8 / 3) * d_model)
            d_ff = 64 * round(d_ff / 64)
        self.w1 = Linear(in_dim=d_model, out_dim=d_ff, **kwargs)
        self.w2 = Linear(in_dim=d_model, out_dim=d_ff, **kwargs)
        self.w3 = Linear(in_dim=d_ff, out_dim=d_model, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.w1(x)
        # ~swish~
        silu = x_proj * F.sigmoid(x_proj)
        # ~glu~ using side projection
        glu = silu * self.w2(x)
        # out projection
        return self.w3(glu)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self, 
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,  
    ) -> None:
        super().__init__()
        pairs = d_k // 2
        k = torch.arange(1, pairs + 1, device=device)
        
        freqs = 1.0 / (theta ** ((2 * k - 2) / d_k))  # Shape: (pairs,)
        pos = torch.arange(1, max_seq_len + 1, device=device).unsqueeze(1)  # Shape: (max_seq_len, 1)
        # ~broadcast~ by (pairs,) -> (1, pairs) with freqs
        angles = pos * freqs
        
        # interleave needed for `d_k` operation
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
        # (..., max_seq_len, ...) -> (..., seq_len, ...)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        # derived by breaking down the pair-wise rotation to position-wise
        x1, x2 = x[..., ::2], x[..., 1::2]
        # negative pair-wise swap
        x_next = torch.stack([-x2, x1], dim=-1).flatten(-2)
        # derived from 2D rotation mat
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
        
        kwargs = dict(device=device, dtype=dtype)
        # TODO: expand this into a single (d_model, (3 * d_k + d_v) * num_heads) transformation
        self.Q_proj = Linear(in_dim=d_model, out_dim=self.d_k * num_heads, **kwargs)
        self.K_proj = Linear(in_dim=d_model, out_dim=self.d_k * num_heads, **kwargs)
        self.V_proj = Linear(in_dim=d_model, out_dim=self.d_v * num_heads, **kwargs)
        self.head_proj = Linear(in_dim=self.d_v * num_heads, out_dim=d_model, **kwargs)
        
        self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)
        self.head_proj = Linear(in_dim=self.d_v * num_heads, out_dim=d_model, **kwargs)
        
        self.register_buffer('mask', torch.tril(torch.ones(max_seq_len, max_seq_len, device=device)).bool())

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        *leading, embed_dim = x.shape
        # split out heads and transpose with seq_len dim
        return x.view(*leading, self.num_heads, embed_dim // self.num_heads).transpose(-2, -3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        
        # (batch, seq_len, d_k * num_heads) -> (batch, num_heads, seq_len, d_k)
        K = self._split_heads(self.K_proj(x))
        Q = self._split_heads(self.Q_proj(x))
        V = self._split_heads(self.V_proj(x))

        assert K.shape == Q.shape, "Dim Mismatch"
        positions = torch.arange(seq_len, device=x.device).expand(*Q.shape[:-2], seq_len)

        # apply positional embeddings
        Q = self.rope(Q, token_positions=positions) 
        K = self.rope(K, token_positions=positions) 
        
        attn = scaled_dot_product_attention(K=K, Q=Q, V=V, mask=self.mask[:seq_len, :seq_len])
        # remove head dimension
        cat_heads = attn.transpose(-2, -3).reshape(*x.shape[:-1], -1)
        return self.head_proj(cat_heads)


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
        kwargs = dict(device=device, dtype=dtype)
        self.pre_norm = RMSNorm(d_model, **kwargs)
        self.post_norm = RMSNorm(d_model, **kwargs)
        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            **kwargs
        )
        self.fcn = FCN(d_model, d_ff, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(self.pre_norm(x)) + x
        return self.fcn(self.post_norm(x)) + x


class Transformer(nn.Module):
    # Pre-Norm Transformer with MultiHead Self-Attention
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
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        kwargs = dict(device=device, dtype=dtype)
        self.context_length = context_length
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
        # return logits
        return self.lm_head(x)

def decode(
    model: nn.Module,
    prompt: list[int],
    stop_token: int,
    max_tokens: int,
    temperature: float = 0.8,
    top_p: float = 0.8,
) -> list[int]:
    assert 0 < top_p <= 1
    assert 0 < temperature

    prompt_len = len(prompt)
    tokens = torch.tensor(prompt).unsqueeze(0)
    
    for _ in range(max_tokens):
        context = tokens[:, -model.context_length:]
        logits = model(context).squeeze(0)[-1, :]
        probs = softmax(logits / temperature)
        probs, indices = torch.sort(probs, descending=True)
        
        cum_probs = torch.cumsum(probs, dim=-1)
        cutoff = int((cum_probs <= top_p).sum().item()) + 1

        top_indices = indices[:cutoff]
        top_probs = probs[:cutoff]
        top_probs = top_probs / top_probs.sum()
        token = top_indices[torch.multinomial(top_probs, 1).item()]
        tokens = torch.cat([tokens, token.view(1, 1)], dim=1)

        print(" > " + str(token.item()))
        if token.item() == stop_token:
            break
    return tokens.squeeze(0).tolist()

if __name__ == "__main__":
    GPT2_S =  dict(vocab_size = 50257, seq_len=1024, num_layers=12, d_model=768, d_ff=6400)
    estimate_flops(**GPT2_S)

    GPT2_M =  dict(vocab_size = 50257, seq_len=1024, num_layers=24, d_model=1024, d_ff=6400)
    estimate_flops(**GPT2_M)

    GPT2_XL = dict(vocab_size = 50257, seq_len=1024, num_layers=48, d_model=1600, d_ff=6400)
    estimate_flops(**GPT2_XL)

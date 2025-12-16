from turtle import forward
import torch
import torch.nn as nn
from torch.nn import LayerNorm, functional as F

# hyperparams
batch_size = 64
block_size = 256
blocks = 1
max_iters = 5000
eval_interval = 500
lr = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_heads = 1
n_embed = 384
# 

torch.manual_seed(1337)

with open("data/tinyshakespeare.txt", 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set[str](text)))
vocab_size = len(chars)

char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for i, c in enumerate(chars)}

def encode(string: str) -> list[int]:
    return [char_to_id[c] for c in string]

def decode(tokens: list[int]) -> str:
    return "".join([id_to_char[t] for t in tokens]) 

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split: str):
    data = train_data if split == "train" else val_data
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in idx])
    y = torch.stack([data[i+1: i + block_size + 1] for i in idx])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(*[
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed)
        ])
    
    def forward(self, X: torch.Tensor):
        return self.layers(X)

class Head(nn.Module):
    def __init__(self, head_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, X: torch.Tensor):
        _, T, _ = X.shape
        q = self.query(X) # (batch_size, block_size, head_size)
        k = self.key(X) # (batch_size, block_size, head_size)
        v = self.value(X) # (batch_size, block_size, n_embed)

        # normalized dot-product between QK
        wei = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5) # (batch_size, block_size, block_size)
        # mask out look-ahead tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v # (B, T, T) * (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embed)
    
    def forward(self, X: torch.Tensor):
        out = torch.cat([h(X) for h in self.heads], dim=-1)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, num_heads: int, n_embed: int) -> None:
        super().__init__()
        assert n_embed % num_heads == 0, "`n_embed` must be a multiple of `num_heads`"

        # Note: we need two of these!
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

        self.attn = MultiHeadAttention(num_heads=num_heads, head_size=n_embed // num_heads)
        self.ffn = FeedForward()
    
    def forward(self, X: torch.Tensor):
        # layer norm
        X = X + self.attn(self.ln1(X))
        return X + self.ffn(self.ln2(X))


class Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 0 FLOPS
        self.char_embed = nn.Embedding(vocab_size, n_embed)
        self.pos_embed = nn.Embedding(block_size, n_embed)
        
        # 
        self.ln = nn.LayerNorm(n_embed)
        self.blocks = nn.Sequential(*[Block(n_heads, n_embed) for _ in range(blocks)])
        self.fc = nn.Linear(n_embed, vocab_size)
    
    def forward(self, X: torch.Tensor, targets: torch.Tensor):
        """
        1. Character + position embeddings
        2. Attention Blocks
        3. Linear Layer
        4. Softmax
        """
        tok_embed = self.char_embed(X)
        pos_embed = self.pos_embed(torch.arange(block_size, device=device))
        
        x = tok_embed + pos_embed
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.fc(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            targets = targets.view(B*T)  
            logits = logits.view(B*T, C)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

model = Transformer()
m = model.to(device)

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

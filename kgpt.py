import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparams
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
lr = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
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

class Head(nn.Module):
    def __init__(self, head_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, X: torch.Tensor):
        # X is (batch_size, block_size, n_embed)
        B,T,C = X.shape
        q = self.query(X) # (batch_size, block_size, head_size)
        k = self.key(X) # (batch_size, block_size, head_size)
        v = self.value(X) # (batch_size, block_size, head_size)

        wei = (q @ k.transpose(-2, -1)) * k.shape[-1]**-0.5 # (batch_size, block_size, block_size)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v # (B, T, T) * (B, T, head_size)
        return out
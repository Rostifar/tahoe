import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparams
batch_size = 32
block_size = 8
max_iters = 3000
lr = 1e-3
eval_interval = 300
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
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

assert (encode("ab") == [char_to_id["a"], char_to_id["b"]])
assert (decode([char_to_id["a"], char_to_id["b"]]) == "ab")

data = torch.tensor(encode(text), dtype=torch.long)
assert (data.max().item() == vocab_size - 1 and data.min().item() == 0)

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

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # lookup table for bigram probabilities
        self.token_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, X: torch.tensor, targets: torch.Tensor | None = None):
        # (batch_size, block_size, vocab_size)
        logits = self.token_table(X)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            targets = targets.view(B*T)  
            logits = logits.view(B*T, C)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, X: torch.tensor, max_new_tokens: int):
        # X: (batch_size, T)
        for _ in range(max_new_tokens):
            logits, _ = self(X) # (batch_size, T, vocab_size)
            next_token_lh = F.softmax(logits[:, -1, :], dim=-1) # (batch_size, vocab_size)
            X_next = torch.multinomial(next_token_lh, num_samples=1)
            X = torch.cat((X, X_next), dim=1) # (batch_size, T+1)
        return X

model = BigramLanguageModel(vocab_size)
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

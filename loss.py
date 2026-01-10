import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # numerical stability
    logits = logits - logits.max(dim=-1, keepdim=True).values
    
    # (B, S, V) -> (B, S, 1)
    mass = torch.exp(logits).sum(dim=-1)
    mass = torch.log(mass)

    # (B, S, V) -> (B, S, 1) -> (1)
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (mass - target_logits).mean()


if __name__ == "__main__":
    logits = torch.tensor([[[0.001, 0.8, 0.00001, 0, 0], [10.0, 0.00001, 0.00003, 0, .0]]])
    targets = torch.tensor([[1, 0]])

    print(cross_entropy(logits, targets))
    print(F.cross_entropy(logits.view(-1, 5), targets.view(-1)))

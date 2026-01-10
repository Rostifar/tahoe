import torch
import torch.nn.functional as F

from loss import (
    cross_entropy
)

def test_cross_entropy():
    # Uniform logits
    (logits, targets) = (
        torch.tensor([
            [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]
        ]),
        torch.tensor([[0, 0]])
    )
    torch.testing.assert_close(
        cross_entropy(logits, targets),
        F.cross_entropy(logits.view(-1, 3), targets.view(-1))
    )

    # Single token
    (logits, targets) = (
        torch.tensor([[[2.0, 1.0, 0.5]]]),
        torch.tensor([[1]])
    )
    torch.testing.assert_close(
        cross_entropy(logits, targets),
        F.cross_entropy(logits.view(-1, 3), targets.view(-1))
    )

    # Batch size > 1
    (logits, targets) = (
        torch.tensor([
            [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
            [[0.5, 0.5, 0.5], [1.0, 0.0, -1.0]]
        ]),
        torch.tensor([[2, 0], [1, 2]])
    )
    torch.testing.assert_close(
        cross_entropy(logits, targets),
        F.cross_entropy(logits.view(-1, 3), targets.view(-1))
    )

    # High confidence, correct prediction
    (logits, targets) = (
        torch.tensor([[[10.0, -10.0, -10.0]]]),
        torch.tensor([[0]])
    )
    torch.testing.assert_close(
        cross_entropy(logits, targets),
        F.cross_entropy(logits.view(-1, 3), targets.view(-1))
    )

    # High confidence, wrong prediction
    (logits, targets) = (
        torch.tensor([[[-10.0, 10.0, -10.0]]]),
        torch.tensor([[0]])
    )
    torch.testing.assert_close(
        cross_entropy(logits, targets),
        F.cross_entropy(logits.view(-1, 3), targets.view(-1))
    )

    # Negative logits
    (logits, targets) = (
        torch.tensor([[[-1.0, -2.0, -3.0], [-0.5, -1.5, -2.5]]]),
        torch.tensor([[2, 0]])
    )
    torch.testing.assert_close(
        cross_entropy(logits, targets),
        F.cross_entropy(logits.view(-1, 3), targets.view(-1))
    )

    # Larger vocab and random values
    vocab_size = 50
    logits = torch.randn(2, 10, vocab_size)
    targets = torch.randint(0, vocab_size, (2, 10))
    torch.testing.assert_close(
        cross_entropy(logits, targets),
        F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    )

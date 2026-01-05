import pytest
import torch
from transformer import (
    Linear,
    Embedding,
    RMSNorm
)

def test_linear():
    layer = Linear(in_dim=10, out_dim=10)
    torch.testing.assert_close(layer.forward(torch.eye(10, 10)), layer.W.T)

    x = torch.randn(128, 256)
    layer = Linear(in_dim=256, out_dim=1024)
    out = layer.forward(x)
    assert out.shape == (128, 1024)
    torch.testing.assert_close(out, x @ layer.W.T)

    layer = Linear(in_dim=4, out_dim=5, dtype=torch.float16)
    assert layer.W.dtype == torch.float16
    x = torch.randn(2, 4, dtype=torch.float16)
    out = layer.forward(x)
    assert out.dtype == torch.float16


def test_embedding():
    embed = Embedding(num_embeddings=1000, embed_dim=924)
    out = embed.forward(torch.arange(0, 10))
    assert out.shape == (10, 924)
    torch.testing.assert_close(out, embed.table[:10])


def test_embedding_out_of_bounds():
    embed = Embedding(num_embeddings=1000, embed_dim=924)

    with pytest.raises(IndexError):
        embed.forward(torch.tensor([10001]))


def test_rms_norm():
    rms = RMSNorm(d_model=512)
    x = torch.ones(10, 10, 512)
    out = rms.forward(x)

    gt_rms = torch.nn.RMSNorm((512), eps=1e-5)
    torch.testing.assert_close(out, gt_rms(out))
    assert out.shape == (10, 10, 512)

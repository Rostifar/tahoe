import pytest
import torch
from transformer import (
    Linear,
    Embedding,
    RMSNorm,
    FCN,
    RotaryPositionalEmbedding,
    softmax,
    scaled_dot_product_attention
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


def test_fcn():
    fcn = FCN(d_model=129)
    assert fcn(torch.zeros(100, 10, 129)).shape == (100, 10, 129)
    assert fcn.w1.W.shape == (320, 129)

    model = FCN(d_model=256)
    x = torch.randn(4, 16, 256) * 10
    y = model(x)
    
    assert not torch.isnan(y).any().item()
    assert not torch.isinf(y).any().item()


def test_rotary():
    d_k = 8
    max_seq_len = 32
    theta = 10000.0
    rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len)
    x = torch.randn(1, d_k)
    token_pos = torch.tensor([0])
    out = rope(x, token_pos)
    assert out.shape == (1, d_k)

    x = torch.randn(4, d_k)
    positions = torch.arange(4)
    out = rope(x, positions)
    assert out.shape == (4, d_k)
    # Output should change with position
    diff_count = torch.count_nonzero(torch.abs(out[0] - out[1]) > 1e-6)
    assert diff_count > 0


def test_softmax():
    x = torch.tensor([1.0, 2.0, 3.0])
    sm = softmax(x)
    torch.testing.assert_close(sm, torch.nn.functional.softmax(x, dim=-1))

    x2 = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    sm2 = softmax(x2, dim=-1)
    torch.testing.assert_close(sm2, torch.nn.functional.softmax(x2, dim=-1))

    sm2_alt = softmax(x2, dim=0)
    torch.testing.assert_close(sm2_alt, torch.nn.functional.softmax(x2, dim=0))

    summed = sm2.sum(dim=-1)
    torch.testing.assert_close(summed, torch.ones_like(summed))

    large = torch.tensor([1234.0, 1235.0, 1236.0])
    sm_large = softmax(large)
    torch.testing.assert_close(sm_large, torch.nn.functional.softmax(large, dim=-1))


def test_scaled_dot_product_attention():
    torch.manual_seed(0)
    # Test with obvious Q, K, V (no mask)
    K = torch.eye(4)
    Q = torch.eye(4)
    V = torch.arange(4*4.).reshape(4, 4)
    out = scaled_dot_product_attention(K, Q, V)
    torch.testing.assert_close(out, V, atol=1e-5, rtol=0)

    # Q attends to K as identity (should output V), V all ones: should return all ones.
    V = torch.ones(4, 4)
    out2 = scaled_dot_product_attention(K, Q, V)
    torch.testing.assert_close(out2, torch.ones_like(out2), atol=1e-5, rtol=0)

    # Test with batch dimensions
    Q = torch.randn(2, 4, 8)
    K = torch.randn(2, 4, 8)
    V = torch.randn(2, 4, 8)
    result = scaled_dot_product_attention(K, Q, V)
    assert result.shape == (2, 4, 8)

    # Test with mask = causal
    Q = torch.randn(4, 8)
    K = torch.randn(4, 8)
    V = torch.randn(4, 8)
    seq_len = 4
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    out_masked = scaled_dot_product_attention(K, Q, V, mask=mask)
    assert out_masked.shape == (4, 8)

    # Output should differ if we mask out last token for last position (different row)
    mask_last_blocked = mask.clone()
    mask_last_blocked[-1, -1] = False
    out_masked2 = scaled_dot_product_attention(K, Q, V, mask=mask_last_blocked)
    assert not torch.equal(out_masked[-1], out_masked2[-1])

import pytest
import torch
from transformer import (
    Linear,
    Embedding,
    RMSNorm,
    FCN,
    RotaryPositionalEmbedding,
    softmax,
    scaled_dot_product_attention,
    MultiheadSelfAttention
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
    # Uniform attention (Q=K=0) → output is mean of V
    Q = torch.zeros(4, 8)
    K = torch.zeros(4, 8)
    V = torch.randn(4, 8)
    out = scaled_dot_product_attention(K, Q, V)
    expected = V.mean(dim=0, keepdim=True).expand_as(V)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    # Sharp attention (large-scale identity) → output ≈ V
    scale = 100.0
    K = torch.eye(4) * scale
    Q = torch.eye(4) * scale
    V = torch.arange(16.).reshape(4, 4)
    out = scaled_dot_product_attention(K, Q, V)
    torch.testing.assert_close(out, V, atol=1e-4, rtol=1e-4)

    # Batched inputs preserve shape
    Q = torch.randn(2, 4, 8)
    K = torch.randn(2, 4, 8)
    V = torch.randn(2, 4, 8)
    assert scaled_dot_product_attention(K, Q, V).shape == (2, 4, 8)

    # Causal mask changes output for later positions
    torch.manual_seed(42)
    Q = torch.randn(4, 8)
    K = torch.randn(4, 8)
    V = torch.randn(4, 8)
    mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    out_unmasked = scaled_dot_product_attention(K, Q, V)
    out_masked = scaled_dot_product_attention(K, Q, V, mask=mask)
    assert not torch.allclose(out_unmasked[1], out_masked[1])


def test_multi_head_self_attention():
    attn = MultiheadSelfAttention(d_model=64, num_heads=4, theta=10000.0, max_seq_len=128)

    assert attn.d_k == attn.d_v == 16
    assert attn.num_heads == 4
    assert attn.mask.shape == (128, 128)

    # bad dim
    with pytest.raises(AssertionError):
        MultiheadSelfAttention(d_model=65, num_heads=4, theta=10000.0, max_seq_len=128)

    # preserves shape
    x = torch.randn(2, 32, 64)
    assert attn(x).shape == x.shape

    # more shape testing
    for batch, seq in [(1, 1), (4, 16), (2, 128)]:
        x = torch.randn(batch, seq, 64)
        assert attn(x).shape == (batch, seq, 64)

    # split head testing
    x = torch.randn(2, 32, 64)
    out = attn._split_heads(x)
    assert out.shape == (2, 4, 32, 16)

    # grad flow
    x = torch.randn(2, 32, 64, requires_grad=True)
    attn(x).sum().backward()
    assert x.grad is not None and not torch.all(x.grad == 0)
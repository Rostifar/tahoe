import torch
import pytest
from optim import AdamW, cosine_lr_scheduler, clip_grads

def test_cosine_lr_scheduler():
    lr_max, lr_min, t_warmup, t_cos = 1.0, 0.1, 10, 110
    
    # Warmup: linear from 0 to lr_max
    assert cosine_lr_scheduler(0, lr_max, lr_min, t_warmup, t_cos) == 0.0
    assert cosine_lr_scheduler(5, lr_max, lr_min, t_warmup, t_cos) == 0.5
    assert cosine_lr_scheduler(10, lr_max, lr_min, t_warmup, t_cos) == lr_max
    
    # Cosine: decays from lr_max to lr_min
    assert cosine_lr_scheduler(60, lr_max, lr_min, t_warmup, t_cos) == pytest.approx(0.55)  # midpoint
    assert cosine_lr_scheduler(110, lr_max, lr_min, t_warmup, t_cos) == pytest.approx(lr_min)
    
    # Post-cosine: stays at lr_min
    assert cosine_lr_scheduler(200, lr_max, lr_min, t_warmup, t_cos) == lr_min

def test_clip_grads():
    # Test 1: Gradient exceeds max_grad, should be clipped
    p1 = torch.nn.Parameter(torch.zeros(3))
    p1.grad = torch.tensor([3.0, 4.0, 0.0])  # norm = 5
    clip_grads([p1], max_grad=2.5)
    assert torch.isclose(torch.norm(p1.grad), torch.tensor(2.5), atol=1e-5)

    # Test 2: Gradient below max_grad, should remain unchanged
    p2 = torch.nn.Parameter(torch.zeros(3))
    p2.grad = torch.tensor([1.0, 1.0, 1.0])  # norm ≈ 1.73
    original_grad = p2.grad.clone()
    clip_grads([p2], max_grad=5.0)
    assert torch.allclose(p2.grad, original_grad)

    # Test 3: Multiple params, mixed clipping
    p3 = torch.nn.Parameter(torch.zeros(2))
    p4 = torch.nn.Parameter(torch.zeros(2))
    p3.grad = torch.tensor([6.0, 8.0])  # norm = 10, should clip
    p4.grad = torch.tensor([1.0, 0.0])  # norm = 1, unchanged
    clip_grads([p3, p4], max_grad=5.0)
    assert torch.isclose(torch.norm(p3.grad), torch.tensor(5.0), atol=1e-5)
    assert torch.isclose(torch.norm(p4.grad), torch.tensor(1.0), atol=1e-5)


class TestAdamW:
    def test_basic_optimization(self):
        x = torch.tensor([5.0, -3.0], requires_grad=True)
        opt = AdamW([x], lr=0.1)
        for _ in range(100):
            opt.zero_grad()
            (x**2).sum().backward()
            opt.step()
        assert torch.allclose(x, torch.zeros(2), atol=0.1)


    def test_state_updates(self):
        x = torch.tensor([1.0], requires_grad=True)
        opt = AdamW([x], lr=0.01, betas=(0.9, 0.999))
        x.grad = torch.tensor([2.0])
        opt.step()
        
        state = opt.state[x]
        assert state["t"] == 2
        assert torch.isclose(state["m"], torch.tensor(0.2))   # (1-0.9)*2
        assert torch.isclose(state["v"], torch.tensor(0.004)) # (1-0.999)*4


    def test_weight_decay(self):
        x = torch.tensor([1.0], requires_grad=True)
        opt = AdamW([x], lr=0.1, weight_decay=0.1)
        x.grad = torch.zeros(1)
        opt.step()
        assert x.item() < 1.0


    def test_skips_none_grad(self):
        x = torch.tensor([1.0], requires_grad=True)
        opt = AdamW([x], lr=0.1)
        opt.step()  # x.grad is None
        assert x.item() == 1.0


    def test_vs_torch_adamw(self):
        torch.manual_seed(0)
        x1 = torch.tensor([1.0, -2.0], requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)
        opt1, opt2 = AdamW([x1], lr=0.01), torch.optim.AdamW([x2], lr=0.01)
        
        for _ in range(10):
            for o, x in [(opt1, x1), (opt2, x2)]:
                o.zero_grad()
                (x**2).sum().backward()
                o.step()
        assert torch.allclose(x1, x2, atol=1e-5)

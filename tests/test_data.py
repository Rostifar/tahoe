import torch
import pytest
import numpy as np
import torch.nn as nn
from data import (
    save_checkpoint,
    load_checkpoint,
    yield_batch
)

class DummyModule(nn.Module):
    def __init__(self, dim: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.param = nn.Parameter(torch.rand(dim, dim))

@pytest.fixture
def dummy_module():
    return DummyModule(10)

@pytest.fixture
def dummy_optimizer(dummy_module):
    return torch.optim.SGD(dummy_module.parameters(), lr=0.01)

def test_checkpoint(dummy_module, dummy_optimizer, tmp_path):
    # silly grad
    loss = dummy_module.param.sum()
    loss.backward()
    dummy_optimizer.step()

    # save
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(dummy_module, dummy_optimizer, iteration=42, out=path)
    
    assert path.exists()
    new_module = DummyModule(10)
    new_optimizer = torch.optim.SGD(new_module.parameters(), lr=0.01)

    iteration = load_checkpoint(path, new_module, new_optimizer, torch.device("cpu"))
    assert iteration == 42
    torch.testing.assert_close(new_module.param, dummy_module.param)
    assert new_optimizer.state_dict()['param_groups'] == dummy_optimizer.state_dict()['param_groups']

def test_yield_batch():
    x, y = yield_batch(np.arange(9), 2, 4, "cpu")
    torch.testing.assert_close(torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]), x)
    torch.testing.assert_close(torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), y)

    x, y = yield_batch(np.arange(15), 2, 4, "cpu")
    torch.testing.assert_close(torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]), x)
    torch.testing.assert_close(torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), y)

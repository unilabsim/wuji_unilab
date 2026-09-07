import pytest
import torch

from wuji_unilab.rl.distributions import SoftplusGaussianDistribution


def test_positive_exploration_floor_and_differentiability():
    distribution = SoftplusGaussianDistribution(20, init_std=0.5, min_std=0.2)
    output = torch.randn(4, 2, 20, requires_grad=True)
    distribution.update(output)
    assert torch.all(distribution.std >= 0.2)
    loss = distribution.entropy.mean()
    loss.backward()
    assert output.grad is not None and torch.isfinite(output.grad).all()
    assert torch.any(output.grad != 0)
    with pytest.raises(ValueError):
        SoftplusGaussianDistribution(20, init_std=0.1, min_std=0.2)

import pytest
import torch
import deepinv as dinv
from blurlabs.models.ram2 import _model_input


@pytest.mark.parametrize("scale", [1.0, 0.1, 0.01, 0.001, 0.0001])
def test_small_operator_norm_preserves_identity_solution(scale):
    x = torch.linspace(.1, .9, 4096).reshape(1, 1, 64, 64)
    physics = dinv.physics.LinearPhysics(A=lambda v: scale * v, A_adjoint=lambda v: scale * v)
    y = physics.A(x)
    with torch.no_grad():
        output, norm, maps, _ = _model_input(y, physics.A_adjoint(y), physics, torch.full_like(y, scale * .001), 3, 30)
    torch.testing.assert_close(output[:, -1:], x, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(norm, torch.full_like(norm, scale**2), atol=0, rtol=2e-4)
    assert torch.isfinite(maps).all()

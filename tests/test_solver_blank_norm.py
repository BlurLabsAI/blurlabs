import pytest
import torch
import deepinv as dinv
from blurlabs.models.ram2 import _model_input


@pytest.mark.parametrize("scale", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("blank", [True, False])
def test_operator_norm_uses_noise_rhs_for_blank_images(scale, blank):
    x = torch.full((2, 1, 64, 64), 0.5)
    if blank:
        x[0] = 0
    physics = dinv.physics.LinearPhysics(A=lambda v: scale * v, A_adjoint=lambda v: scale * v)
    y = physics.A(x)
    with torch.no_grad():
        output, norm, maps, _ = _model_input(y, physics.A_adjoint(y), physics, torch.full_like(y, scale * .01), 3, 10)
    torch.testing.assert_close(norm, torch.full_like(norm, scale**2), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(output[:, -1:], x, atol=2e-5, rtol=2e-5)
    assert torch.isfinite(maps).all()


def test_nonblank_measurement_keeps_its_original_norm_estimate():
    # Distinct singular subspaces make the noise estimate larger than the data estimate.
    diagonal = torch.ones(1, 1, 64, 64)
    diagonal[..., 32:, :] = 2
    x = torch.zeros_like(diagonal)
    x[..., :32, :] = .5
    noise = torch.zeros_like(x)
    noise[..., 32:, :] = .01
    physics = dinv.physics.LinearPhysics(A=lambda v: diagonal * v, A_adjoint=lambda v: diagonal * v)
    y = physics.A(x)
    with torch.no_grad():
        _, norm, _, _ = _model_input(y, physics.A_adjoint(y), physics, noise, 3, 10)
    # The existing nonblank estimator sees only the unit singular-value subspace.
    torch.testing.assert_close(norm, torch.ones_like(norm), atol=2e-5, rtol=2e-5)

"""Native CPU plumbing checks, with tiny untrained RAM2 networks."""
from dataclasses import replace

import pytest
import torch

import blurlabs.models.ram2 as ram2


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(ram2, "DEFAULT_RAM2_CONFIG", replace(
        ram2.DEFAULT_RAM2_CONFIG, features=(4, 8), block_numbers=(1, 1),
        unrolled_steps=2, max_iter=4,
    ))
    torch.manual_seed(24)
    return ram2.RAM2(pretrained=None)


@pytest.mark.parametrize("shape", [(1, 1, 65, 67), (2, 3, 31, 35),
                                  (1, 2, 17, 19), (2, 1, 64, 64)])
def test_constant_sigma_map_matches_scalar(model, shape):
    y = torch.linspace(0.1, 0.9, torch.tensor(shape).prod().item()).reshape(shape)
    with torch.no_grad():
        torch.manual_seed(12)
        scalar = model(y, sigma=0.1)
        torch.manual_seed(12)
        mapped = model(y, sigma=torch.full_like(y, 0.1))
    assert scalar.shape == mapped.shape == y.shape
    assert torch.isfinite(mapped).all()
    torch.testing.assert_close(mapped, scalar, rtol=0, atol=0)


def test_spatial_noise_map_replicates_boundary_without_changing_interior(model, monkeypatch):
    shape = (2, 3, 31, 35)
    y = torch.full(shape, 0.5)
    sigma = torch.linspace(0.01, 0.2, y.numel()).reshape(shape)
    original_sigma = sigma.clone()
    native_model_input = ram2._model_input
    recorded = []

    def record_input(y, aty, physics, noise, *args, **kwargs):
        recorded.append(noise.abs().clone())
        return native_model_input(y, aty, physics, noise, *args, **kwargs)

    monkeypatch.setattr(ram2, "_model_input", record_input)
    with torch.no_grad():
        output = model(y, sigma=sigma)
    noise = recorded[0]
    assert noise.shape == (2, 3, 64, 64)
    # Independent four-region oracle: original interior, first row, first
    # column, and the top-left corner, without calling a padding helper.
    torch.testing.assert_close(noise[..., 33:, 29:], sigma, rtol=0, atol=0)
    torch.testing.assert_close(
        noise[..., :33, 29:], sigma[..., :1, :].expand(-1, -1, 33, -1),
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        noise[..., 33:, :29], sigma[..., :, :1].expand(-1, -1, -1, 29),
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        noise[..., :33, :29], sigma[..., :1, :1].expand(-1, -1, 33, 29),
        rtol=0, atol=0,
    )
    torch.testing.assert_close(sigma, original_sigma, rtol=0, atol=0)
    assert output.shape == shape
    assert torch.isfinite(output).all()

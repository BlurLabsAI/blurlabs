"""Scalar noise representations must retain batch rather than spatial axes."""

import importlib
from dataclasses import replace

import deepinv as dinv
import pytest
import torch

ram2 = importlib.import_module("blurlabs.models.ram2")


@pytest.mark.parametrize("name", ["sigma", "gain", "l"])
@pytest.mark.parametrize("shape", [(), (1,), (1, 1), (1, 1, 1), (1, 1, 1, 1)])
def test_single_element_noise_tensor_is_scalar(name, shape):
    y = torch.ones(2, 1, 64, 64)
    physics = dinv.physics.Denoising()
    scalar = ram2._noise_level(0.1, physics, name, 1.0, y)
    shaped = ram2._noise_level(torch.full(shape, 0.1), physics, name, 1.0, y)
    assert shaped.shape == (2, 1, 1, 1)
    torch.testing.assert_close(shaped, scalar, rtol=0, atol=0)


def test_batch_vector_and_spatial_map_are_preserved():
    y = torch.ones(2, 1, 64, 64)
    physics = dinv.physics.Denoising()
    vector = torch.tensor([0.1, 0.2])
    result = ram2._noise_level(vector, physics, "sigma", 1.0, y)
    torch.testing.assert_close(result, vector[:, None, None, None], rtol=0, atol=0)
    spatial = torch.linspace(0.01, 0.2, y.numel()).reshape_as(y)
    torch.testing.assert_close(
        ram2._noise_level(spatial, physics, "sigma", 1.0, y), spatial, rtol=0, atol=0
    )


@pytest.fixture
def tiny_model(monkeypatch):
    monkeypatch.setattr(
        ram2,
        "DEFAULT_RAM2_CONFIG",
        replace(
            ram2.DEFAULT_RAM2_CONFIG,
            features=(4,),
            block_numbers=(0,),
            unrolled_steps=2,
            max_iter=5,
        ),
    )
    torch.manual_seed(17)
    return ram2.RAM2(pretrained=None).eval()


@pytest.mark.parametrize("height,width", [(64, 64), (65, 67)])
def test_native_gamma_scalar_representations_match(tiny_model, height, width):
    x = torch.full((2, 1, height, width), 0.5)
    results = []
    measurements = []
    for level in [torch.tensor(10.0), torch.full((1, 1, 1, 1), 10.0)]:
        physics = dinv.physics.Denoising(noise_model=dinv.physics.GammaNoise(l=level))
        torch.manual_seed(91)
        y = physics(x)
        assert y.shape == x.shape
        measurements.append(y)
        torch.manual_seed(71)
        with torch.no_grad():
            results.append(tiny_model(y, physics))
    torch.testing.assert_close(measurements[0], measurements[1], rtol=0, atol=0)
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
    assert results[1].isfinite().all()


def test_public_explicit_scalar_sigma_tensor_matches_number(tiny_model):
    y = torch.full((2, 1, 65, 67), 0.5)
    with torch.no_grad():
        torch.manual_seed(71)
        numeric = tiny_model(y, sigma=0.1)
        torch.manual_seed(71)
        tensor = tiny_model(y, sigma=torch.full((1, 1, 1, 1), 0.1))
    torch.testing.assert_close(numeric, tensor, rtol=0, atol=0)

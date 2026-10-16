"""Native CPU regression tests; no pretrained weights or downloads are used."""

import pytest
import torch

from blurlabs.models.ram2 import _AvgPool2d, _NAFBlock


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kernel", [1, 64])
def test_low_precision_box_pool_preserves_constant(dtype, kernel):
    image = torch.ones(1, 1, 1024, 1024, dtype=dtype)
    output = _AvgPool2d(kernel)(image)
    assert output.dtype == dtype
    assert output.shape == image.shape
    torch.testing.assert_close(output, image, rtol=0, atol=0)


def test_half_naf_zero_gain_remains_identity_on_large_image():
    # These are the shipping residual block's default zero beta/gamma gains.
    # Overflowing its spatial pooling must not turn the identity into NaNs.
    with torch.random.fork_rng():
        torch.manual_seed(3)
        block = _NAFBlock(2, 64).half()
    image = torch.ones(1, 2, 1024, 1024, dtype=torch.float16)
    image[:, 1] = 3
    with torch.no_grad():
        output = block(image)
    torch.testing.assert_close(output, image, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_float_pooling_preserves_dtype_shape_and_gradients(dtype):
    image = torch.full((1, 2, 65, 97), 0.5, dtype=dtype, requires_grad=True)
    output = _AvgPool2d(16)(image)
    assert output.dtype == dtype
    assert output.shape == image.shape
    torch.testing.assert_close(output, torch.full_like(image, 0.5), rtol=0, atol=0)
    output.sum().backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_odd_shape_pool_retains_gradients(dtype):
    image = torch.full((1, 2, 65, 97), 0.5, dtype=dtype, requires_grad=True)
    output = _AvgPool2d(16)(image)
    assert output.dtype == dtype
    assert output.shape == image.shape
    torch.testing.assert_close(output, torch.full_like(image, 0.5), rtol=0, atol=0)
    output.sum().backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert image.grad.abs().sum().item() > 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_global_pool_control(dtype):
    image = torch.ones(1, 2, 7, 11, dtype=dtype, requires_grad=True)
    output = _AvgPool2d(64)(image)
    assert output.dtype == dtype
    torch.testing.assert_close(output, torch.ones(1, 2, 1, 1, dtype=dtype))
    output.sum().backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()

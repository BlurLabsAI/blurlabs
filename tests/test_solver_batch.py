import pytest
import torch
import deepinv as dinv
from blurlabs.models.ram2 import _model_input, _PhysicsCropper


@pytest.mark.parametrize("cropped", [False, True])
def test_solver_preserves_per_image_physics_batch(cropped):
    torch.manual_seed(11)
    mask = dinv.physics.generator.BernoulliSplittingMaskGenerator(
        img_size=(1, 61, 63), split_ratio=0.7
    ).step(batch_size=2, seed=11)["mask"]
    x = torch.rand(2, 1, 61, 63)
    noise = mask * 0.01
    physics = dinv.physics.Inpainting(img_size=x.shape[1:], mask=mask)
    y = physics.A(x)
    if cropped:
        physics = _PhysicsCropper(physics, [3, 1])
    with torch.no_grad():
        actual = _model_input(y, physics.A_adjoint(y), physics, noise, 3, 10)
        singles = []
        for i in range(2):
            single = dinv.physics.Inpainting(img_size=x.shape[1:], mask=mask[i:i+1])
            if cropped:
                single = _PhysicsCropper(single, [3, 1])
            singles.append(_model_input(y[i:i+1], single.A_adjoint(y[i:i+1]), single, noise[i:i+1], 3, 10))
    for index, value in enumerate(actual):
        torch.testing.assert_close(value, torch.cat([s[index] for s in singles]), atol=2e-5, rtol=2e-5)


def test_shared_mask_batch_control():
    x = torch.full((2, 1, 64, 64), 0.5)
    physics = dinv.physics.Inpainting(img_size=x.shape[1:], mask=torch.ones(1, 1, 64, 64))
    with torch.no_grad():
        output, norm, _, _ = _model_input(x, physics.A_adjoint(x), physics, torch.full_like(x, .01), 3, 10)
    torch.testing.assert_close(output[:, -1:], x, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(norm, torch.ones_like(norm), atol=2e-5, rtol=2e-5)

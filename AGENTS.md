# Using RAM 2 as an agent

RAM 2 (`ram2.py`, class `RAM2`) reconstructs an image from measurements `y` of a known linear operator `physics`
(a [DeepInverse](https://deepinv.org) `LinearPhysics`) and the noise level. It is a single model for any such
problem: denoising, deblurring, inpainting, super-resolution, demosaicing, MRI, CT, and more. It is not trained per
task, so most of the work is in giving it the right inputs.

## Basic usage

```python
import deepinv as dinv
from blurlabs.models import RAM2

model = RAM2()  # downloads the weights; gated: request access on Hugging Face, then `hf auth login` or set HF_TOKEN
physics = dinv.physics.Blur(filter=dinv.physics.functional.blur.gaussian_blur(sigma=(2.0, 2.0)),
                            noise_model=dinv.physics.GaussianNoise(0.01))
# load data y here

x_hat = model(y, physics, verbose=True)  # y: (B, C, ...) measurements of physics
```

- Images are 2D with 1, 2 (real and imaginary parts, e.g. MRI) or 3 channels.
- Use `physics=None` to denoise: `model(y, sigma=0.1)`. Pass noise parameters by keyword; the second positional
  argument is `physics`, not `sigma`.
- Run inference under `torch.no_grad()` and `model.eval()` (set by `pretrained`).

## Run with `verbose=True` first, and act on what it reports

On a new problem or new data, always call the model once with `verbose=True` and read the output before trusting the
reconstruction. It prints one line per batch:

```
[RAM2] ||A||=[...], sigma=[...], gain=[...], l=[...], input 10%/90% quantiles=[(q10, q90), ...]
```

Check that `sigma`, `gain` and `l` are the noise levels you meant, and that `||A||` is plausible for the operator.
The quantiles are those of the image the network sees (a least-squares estimate from `y`). Then act on the messages:

- **"input 90% quantile ... is above 1"**: the signal is above the training range. Set `rescale_inputs` to the
  suggested value (about the 90% quantile); the model returns `a * f(y / a)`, so the output keeps the data's scale.
- **"input 10% quantile ... is below 0"** (1 or 3 channels) **or below -0.5** (2 channels): the data has an offset or
  a sign convention the model was not trained on. Fix the normalisation of the data, e.g. remove a negative
  background or flip the sign; `rescale_inputs` does not help here.
- **"the least-squares solve is far from converged"**: the operator is likely badly calibrated (wrong adjoint,
  wrong scaling of `A`, wrong noise level), or the problem is extremely ill-posed. Check `physics` with
  `physics.adjointness_test(x)` and `physics.compute_norm(x)` and fix it before anything else; a preconditioner can
  also help. Increasing `max_iter` (default 30) only helps when the operator itself is correct.

Repeat until the verbose output is clean. Only switch `verbose` off once it is.

## Noise levels

- Give the noise through `physics.noise_model` (e.g. `GaussianNoise(sigma)`, `PoissonGaussianNoise(gain, sigma)`) or
  through the keyword arguments `sigma`, `gain` (Poisson) and `l` (Gamma), which take precedence.
- `sigma` given alone is used as the exact noise standard deviation: a scalar, or a per-pixel map of the shape of `y`.
- The noise level matters: an underestimated noise level leaves noise and artefacts, an overestimated one
  oversmooths. If it is unknown, estimate it (e.g. from a flat region or from the k-space background) rather than
  guessing, and report the value used to the user.

## Specific problems

- **MRI** (`dinv.physics.MRI`, `dinv.physics.MultiCoilMRI`): the model was trained on phase-demodulated images rotated
  by `e^{i pi/4}`; this rotation is applied inside the model, so pass the data as it is. Coil maps must be given in
  `physics`; with estimated maps (e.g. ESPIRiT), make sure their phase is smooth. Lower `max_iter` (e.g. 20) on
  real data whose operator is not well calibrated.
- **Operators with large or small norm**: RAM 2 accepts `||A|| != 1`; do not rescale the operator, but keep the data
  in range with `rescale_inputs`.
- **Unknown operator** (blind problems): RAM 2 needs `physics`. Estimate the operator first (e.g. the blur kernel),
  then reconstruct; say so to the user, as the result depends on that estimate.

## Reporting to the user

State the physics, noise levels and `rescale_inputs` used, and any verbose warning that remained. Do not present a
reconstruction as reliable when the verbose output still warned about the input range or convergence.

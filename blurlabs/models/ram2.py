import math
import warnings
from collections import OrderedDict
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import deepinv as dinv
from deepinv.models import Reconstructor, Denoiser

from ._configs import DEFAULT_RAM2_CONFIG, resolve_ram2_pretrained


class RAM2(Reconstructor, Denoiser):
    def __init__(self, pretrained: str | None = "download"):
        r"""
        Reconstruct Anything Model (RAM) 2: a physics-conditioned network for solving linear inverse problems.

        RAM 2 is a foundation model for image reconstruction, trained on a large variety of linear inverse problems with different noise models.

        It is designed to solve problems of the form :math:`y = N(Ax)`, where :math:`N` is noising procedure (which can be a mixture of Gaussian, Poisson, or Gamma), :math:`A` is a linear operator, and :math:`x` the sought image.

        .. math::

            \hat{x} = R(y, A, \sigma)

        where :math:`\sigma` is the noise level.

        .. note::

            To download the pretrained weights, request access at https://huggingface.co/blurlabs/ram2, then run
            `hf auth login` or set `HF_TOKEN`.

        :param str, None pretrained: `"download"` to download the weights, a path to a checkpoint, or `None` for random
            initialisation. A checkpoint is a plain state dict of this model. The model is put in eval mode.
        """
        super().__init__()
        config, pretrained = resolve_ram2_pretrained(DEFAULT_RAM2_CONFIG, pretrained)

        self.config = config
        features = config.features
        block_numbers = config.block_numbers
        # This order determines the checkpoint names of the channel-specific heads.
        in_channels_list = config.supported_channels
        N = self.N = config.unrolled_steps
        self.max_iter = config.max_iter

        # plain container, to keep the checkpoint key names
        self.module_in = nn.Module()
        self.module_in.convin = _Heads(
            "conv", in_channels_list, lambda c: nn.Conv2d(2 * N * c, features[0], 3, padding=1, bias=False)
        )
        self.module_out = _Heads(
            "tail",
            in_channels_list,
            lambda c: nn.Sequential(OrderedDict(convin=nn.Conv2d(features[0], c, 3, padding=1, bias=False))),
        )

        def level(i: int) -> nn.Sequential:
            blocks = [
                _DRUNetBlock(features[i]) if i < 2 else _NAFBlock(features[i], 256 // 2**i)
                for _ in range(block_numbers[i])
            ]
            return nn.Sequential(OrderedDict(layers=nn.Sequential(*blocks)))

        scales = len(features)
        self.down_levels = nn.ModuleList(level(i) for i in range(scales - 1))
        self.up_levels = nn.ModuleList(level(i) for i in range(scales - 1))
        self.mid_level = level(scales - 1)

        self.downsamplers = nn.ModuleList(
            nn.Sequential(OrderedDict(conv_out=nn.Conv2d(f_in, f_out, 2, 2, bias=False)))
            for f_in, f_out in zip(features[:-1], features[1:])
        )
        for down in self.downsamplers:
            nn.init.orthogonal_(down.conv_out.weight, gain=0.2)
        self.upsamplers = nn.ModuleList(
            _Upsampler(features[i + 1], features[i], in_channels_list, scale=i) for i in range(scales - 1)
        )

        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained, map_location="cpu", weights_only=True))
            self.eval()

    def forward(
        self,
        y: torch.Tensor,
        physics: dinv.physics.LinearPhysics | None = None,
        sigma: torch.Tensor | float | None = None,
        gain: torch.Tensor | float | None = None,
        l: torch.Tensor | float | None = None,
        rescale_inputs: float | None = None,
        max_iter: int | None = None,
        verbose: bool = False,
    ) -> torch.Tensor:
        r"""
        Reconstruct an image from measurements `y` and a linear physics operator `physics`.

        .. note::

            The model performs best when the output is approximately between [0, 1], which can be
            achieved by rescaling the inputs with `rescale_inputs` (None by default).

        As well as the ``physics``, the model can receive either

        i) a set of **scalar** noise parameters `sigma`, `gain` and `l` (see DeepInverse docs for more details), where
            - `sigma` is the standard deviation of additive Gaussian noise,
            - `gain` is the gain of normalized Poisson noise, i.e. :math:`\gamma \mathcal{P}(\frac{x}{\gamma})`, and
            - `l` is the scale of multiplicative Gamma noise, i.e. :math:`\mathcal{G}(x; l, 1/l)`.

        ii) or a **per-pixel** noise map `sigma` of the same shape as `y`, indicating the standard deviation of the noise per measurement.

        .. note::

            The model can be used as a denoiser by setting `physics=None` and providing only `sigma`, `gain` and `l`.

        :param torch.Tensor y: input measurements of shape ``(B, C, ...)`` where `C` is number of channels, either 1, 2, or 3.
        :param deepinv.physics.LinearPhysics physics: linear physics operator with methods :meth:`A` and :meth:`A_adjoint`.
            The operator can have non-normalized norm, ie :math:`\|A\| \neq 1`. If `None`, the model is a denoiser
            (:math:`A = I`) and the noise is given by `sigma`, `gain` and `l` only.
        :param torch.Tensor | float sigma: standard deviation of additive Gaussian noise, or a per-pixel noise map of the same shape as `y`.
        :param torch.Tensor | float gain: gain of normalized Poisson noise.
        :param torch.Tensor | float l: scale of multiplicative Gamma noise.
        :param float | None rescale_inputs: rescale inputs by this factor, and rescale outputs back.
            This can be used to bring the inputs into the ideal range around [0, 1].
        :param int | None max_iter: maximum number of iterations for the least-squares solver (default 30).
            This can be lowered if the operator is not well calibrated.
        :param bool verbose: print the operator norm, noise levels and input range, warn if the input is outside the
            training range, and report if the least-squares solve is far from converged.
        :return: torch.Tensor  with the reconstructed image.
        """
        if physics is None:
            physics = dinv.physics.Denoising(noise_model=dinv.physics.ZeroNoise())
        # MRI was trained on phase-demodulated images rotated by e^{i pi/4}, so that both the real and
        # imaginary parts are mostly positive: reconstruct x e^{i pi/4} from y e^{i pi/4} and rotate the output back
        rotation = None
        if isinstance(physics, (dinv.physics.MRI, dinv.physics.MultiCoilMRI)):
            c = s = math.sqrt(0.5)
            rotation = torch.tensor([[c, -s], [s, c]], device=y.device, dtype=y.dtype)
            y = torch.einsum("ij,bj...->bi...", rotation, y)
        y_shape = y.shape
        exact_sigma = sigma is not None and gain is None and l is None

        img_shape = physics.img_shape if hasattr(physics, "img_shape") else physics.A_adjoint(y).shape[-3:]
        channels = img_shape[0]
        pad = [max(-s % 32, 64 - s) for s in img_shape[-2:]]
        if isinstance(physics, dinv.physics.Denoising):
            y = F.pad(y, (pad[1], 0, pad[0], 0))
        elif any(pad):
            physics = _PhysicsCropper(physics, pad)

        min_sigma = 1e-3
        sigma = _noise_level(sigma, physics, "sigma", min_sigma, y)
        gain = _noise_level(gain, physics, "gain", min_sigma**2, y)
        l = _noise_level(l, physics, "l", min_sigma ** (-2), y)
        # an exact sigma map follows a denoising measurement to the padded grid
        if exact_sigma and sigma.ndim == y.ndim and sigma.shape[2:] == y_shape[2:] and y.shape != y_shape:
            sigma = F.pad(sigma, (pad[1], 0, pad[0], 0))

        with torch.no_grad():
            Aty = physics.A_adjoint(y)
            e = (torch.rand(y.shape, device=y.device, dtype=y.dtype) > 0.5) * 2.0 - 1.0
            if exact_sigma:
                std = sigma
            else:  # robust Poisson-Gaussian-Gamma estimate from the noisy y
                std = (gain * (3 / 8 * gain + y.abs()) + y.pow(2) / (l + 1) + sigma.pow(2) + 1e-8).sqrt()
            # zero rows of A (e.g. outside an inpainting mask) are not measurements and noise there would inflate ||e||
            support = physics.A(torch.rand(Aty.shape, device=y.device, dtype=y.dtype)) != 0
            x, operator_norm, noise_maps, n_reached = _model_input(
                y,
                Aty,
                physics,
                std * e * support,
                self.N,
                self.max_iter if max_iter is None else max_iter,
            )

        scaling = 1.0 if rescale_inputs is None else rescale_inputs
        if verbose:
            def fmt(t: torch.Tensor) -> list[float]:
                return [round(v, 4) for v in t.flatten(1).float().mean(1).tolist()]

            # the most converged solution is model input
            image = x[:, -channels:, pad[0] :, pad[1] :].flatten(1).float() / scaling
            image = image[:, :: -(-image.numel() // 2**24)]
            q10, q90 = torch.quantile(image, torch.tensor([0.1, 0.9], device=image.device), dim=1)
            print(
                f"[RAM2] ||A||={fmt(operator_norm.sqrt())}, sigma={fmt(sigma)}, gain={fmt(gain)}, l={fmt(l)}, "
                f"input 10%/90% quantiles={[(round(a, 3), round(b, 3)) for a, b in zip(q10.tolist(), q90.tolist())]}"
            )

            low = -0.5 if channels == 2 else 0.0
            if (q90 > 1).any():
                warnings.warn(
                    f"RAM2 input 90% quantile {q90.max().item():.3g} is above 1, outside the training range: "
                    f"set rescale_inputs to about {scaling * q90.max().item():.3g}."
                )
            if (q10 < low).any():
                warnings.warn(
                    f"RAM2 input 10% quantile {q10.min().item():.3g} is below {low} for {channels}-channel images, "
                    "outside the training range: check the normalisation (offset or sign) of the data."
                )
            if (n_reached < self.N // 2).any():
                print(
                    f"[RAM2] the least-squares solve is far from converged: {n_reached.tolist()} of {self.N} residual "
                    "levels reached per image. The operator might be badly calibrated (e.g. a wrong adjoint or noise "
                    "level), or the problem very highly ill-posed. Consider using a pre-conditioner or improving the operator calibration."
                )

        x = self.module_in.convin(torch.cat([x, noise_maps], dim=1) / scaling, channels)

        skips = []
        for level, down in zip(self.down_levels, self.downsamplers):
            x = level(x)
            skips.append(x)
            x = down(x)
        x = self.mid_level(x)
        for level, up, skip in reversed(list(zip(self.up_levels, self.upsamplers, skips))):
            x = level(up(x, skip, physics, channels, operator_norm))

        x = self.module_out(x, channels)[..., pad[0] :, pad[1] :] * scaling
        return x if rotation is None else torch.einsum("ji,bj...->bi...", rotation, x)


class _Heads(nn.Module):
    def __init__(self, prefix: str, in_channels_list: tuple[int, ...], make: Callable[[int], nn.Module]):
        super().__init__()
        self.prefix = prefix
        self.in_channels_list = list(in_channels_list)
        for i, c in enumerate(in_channels_list):
            self.add_module(f"{prefix}{i}", make(c))

    def forward(self, x: torch.Tensor, channels: int) -> torch.Tensor:
        return getattr(self, f"{self.prefix}{self.in_channels_list.index(channels)}")(x)


class _DRUNetBlock(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1, bias=False)
        nn.init.orthogonal_(self.conv1.weight, gain=0.2)
        nn.init.orthogonal_(self.conv2.weight, gain=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(F.relu(self.conv1(x), inplace=True))


class _NAFBlock(nn.Module):
    def __init__(self, features: int, avg_pool_kernel: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, 2 * features, 1)
        self.conv2 = nn.Conv2d(2 * features, 2 * features, 3, padding=1, groups=2 * features)
        self.conv3 = nn.Conv2d(features, features, 1)
        self.sca = nn.Sequential(_AvgPool2d(avg_pool_kernel), nn.Conv2d(features, features, 1))
        self.conv4 = nn.Conv2d(features, 4 * features, 1)
        self.conv5 = nn.Conv2d(2 * features, features, 1)
        self.beta = nn.Parameter(torch.zeros(1, features, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, features, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # channel normalisation without centring
        u1, u2 = self.conv2(self.conv1(x / torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5))).chunk(2, dim=1)
        u = u1 * u2
        x = x + self.conv3(u * self.sca(u)) * self.beta.to(u.dtype)
        u1, u2 = self.conv4(x / torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)).chunk(2, dim=1)
        return x + self.conv5(u1 * u2) * self.gamma.to(x.dtype)


class _AvgPool2d(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = max(int(kernel_size), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        k = self.kernel_size
        if k >= h and k >= w:
            return F.adaptive_avg_pool2d(x, 1)

        # fast approximation: box filter on a grid subsampled by the largest divisor r <= 5
        r1 = next(r for r in (5, 4, 3, 2, 1) if h % r == 0)
        r2 = next(r for r in (5, 4, 3, 2, 1) if w % r == 0)
        k1, k2 = min(h // r1 - 1, k // r1), min(w // r2 - 1, k // r2)
        if k1 > 0 and k2 > 0:
            s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
            out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
            out = F.interpolate(out, scale_factor=(r1, r2))
        else:
            s = F.pad(x.cumsum(dim=-1).cumsum_(dim=-2), (1, 0, 1, 0))
            k1, k2 = min(h, k), min(w, k)
            out = (s[:, :, k1:, k2:] + s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2]) / (k1 * k2)

        dh, dw = h - out.shape[-2], w - out.shape[-1]
        return F.pad(out, (dw // 2, (dw + 1) // 2, dh // 2, (dh + 1) // 2), mode="replicate")


class _EncodingHead(nn.Module):
    def __init__(self, in_channels: int, features: int):
        super().__init__()
        self.convin = nn.Conv2d(in_channels, features, 1, bias=False)
        self.conv10 = nn.Conv2d(features, features, 1, bias=False)
        self.conv20 = nn.Conv2d(features, features, 1, bias=False)
        self.skipconv0 = nn.Conv2d(features, features, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convin(x)
        return self.conv20(F.relu(self.conv10(x))) + self.skipconv0(x)


class _Upsampler(nn.Module):
    def __init__(self, features_in: int, features_out: int, in_channels_list: tuple[int, ...], scale: int):
        super().__init__()
        self.scale = scale
        self.conv_out = nn.Conv2d(features_in, 4 * features_out, 1, bias=False)
        # plain container, to keep the checkpoint key names
        mc = self.measurement_conditioning = nn.Module()
        mc.decoding_conv = _Heads(
            "tail",
            in_channels_list,
            lambda c: nn.Sequential(OrderedDict(convin=nn.Conv2d(features_out, c * 4**scale, 1, bias=False))),
        )
        mc.encoding_conv = _Heads("head", in_channels_list, lambda c: _EncodingHead(2 * c * 4**scale, features_out))
        mc.gain = nn.Parameter(torch.zeros(1, features_out, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        physics: dinv.physics.LinearPhysics,
        channels: int,
        operator_norm: torch.Tensor,
    ) -> torch.Tensor:
        x = F.pixel_shuffle(self.conv_out(x), 2) + skip

        # measurement conditioning: decode a full-resolution image u, embed [u, A^T A u / ||A||^2] back at this scale
        mc = self.measurement_conditioning
        u = F.pixel_shuffle(mc.decoding_conv(x, channels), 2**self.scale)
        u = torch.cat([u, physics.A_adjoint_A(u) / operator_norm], dim=1)
        emb = F.relu(mc.encoding_conv(F.pixel_unshuffle(u, 2**self.scale), channels))
        return x + emb * mc.gain.to(emb.dtype)


class _PhysicsCropper(dinv.physics.LinearPhysics):
    def __init__(self, physics: dinv.physics.LinearPhysics, crop: list[int]):
        super().__init__(noise_model=physics.noise_model)
        self.base = physics
        self.crop = crop

    def A(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.base.A(x[..., self.crop[0] :, self.crop[1] :].contiguous(), **kwargs)

    def A_adjoint(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        return F.pad(self.base.A_adjoint(y, **kwargs), (self.crop[1], 0, self.crop[0], 0))


def _noise_level(
    value: torch.Tensor | float | None, physics: dinv.physics.LinearPhysics, name: str, default: float, y: torch.Tensor
) -> torch.Tensor:
    if value is None:
        value = getattr(getattr(physics, "noise_model", None), name, None)
    if value is None:
        value = default
    b = y.shape[0]
    if not isinstance(value, torch.Tensor):
        value = torch.ones(b, device=y.device, dtype=y.dtype) * value
    elif value.numel() == 1:
        value = torch.ones(b, device=y.device, dtype=y.dtype) * value.to(device=y.device, dtype=y.dtype)
    value = value.to(device=y.device, dtype=y.dtype)
    return value.view(b, *[1] * (y.ndim - 1)) if value.ndim <= 1 else value


def _model_input(
    y: torch.Tensor,
    Aty: torch.Tensor,
    physics: dinv.physics.LinearPhysics,
    noise: torch.Tensor,
    N: int,
    max_iter: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # iterative least-squares solve on the measurements and the noise realisations stacked along the batch
    # the iterate is saved when cond(A) ||Ax - y|| / ||e|| crosses each of N thresholds
    b = Aty.shape[0]
    n_samples = noise.shape[0] // b
    n_total = 1 + n_samples
    dtype, device = Aty.dtype, Aty.device
    sig_shape = Aty.shape[1:]
    scalar = (n_total * b,) + (1,) * (Aty.ndim - 1)
    meas_scalar = (n_total * b,) + (1,) * (y.ndim - 1)
    main_scalar = (b,) + (1,) * (Aty.ndim - 1)

    def norm(t: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(t, dim=tuple(range(1, t.ndim)), keepdim=True)

    noise_norm = noise.pow(2).reshape(b, n_samples, -1).sum(dim=2).sqrt().mean(1).to(dtype).view(main_scalar)

    # updates are in place to save memory, the solve runs without autograd
    y_all = torch.cat([y, noise.to(dtype=dtype, device=device)])
    b_norm = norm(y_all)
    b_zero = b_norm < 1e-10
    b_norm = b_norm.masked_fill(b_zero, 1.0)
    u = y_all.div_(b_norm)
    b_norm, b_zero = b_norm.view(scalar), b_zero.view(scalar)

    # Keep per-image physics parameters on their original batch grid for each RHS.
    def apply_physics(operator: Callable, value: torch.Tensor) -> torch.Tensor:
        return torch.cat([operator(block) for block in value.split(b)], dim=0)

    # solver state
    x = torch.zeros(n_total * b, *sig_shape, dtype=dtype, device=device)
    v = apply_physics(physics.A_adjoint, u)
    alpha = norm(v).clamp_min(eps).view(scalar)
    v.div_(alpha)
    alphabar, zetabar = alpha.clone(), alpha.clone()
    rho, rhobar, cbar = (torch.ones(scalar, dtype=dtype, device=device) for _ in range(3))
    sbar = torch.zeros(scalar, dtype=dtype, device=device)
    rho_max = torch.zeros(scalar, dtype=dtype, device=device)
    rho_min = torch.full(scalar, float("inf"), dtype=dtype, device=device)
    h, hbar = v, torch.zeros_like(v)

    thresholds = (10 ** torch.linspace(-2, 0.5, N, device=device)).to(dtype).flip(0)
    threshold_idx = torch.zeros(b, dtype=torch.long, device=device)
    solutions = [torch.zeros(b, *sig_shape, dtype=dtype, device=device) for _ in range(N)]
    variances = [torch.zeros(b, *sig_shape, dtype=dtype, device=device) for _ in range(N)]
    active = torch.ones(scalar, dtype=torch.bool, device=device)

    # fixed iteration count, without data-dependent early exit, so that torch.compile generates fast code
    for i in range(int(max_iter)):
        u = apply_physics(physics.A, v).addcmul_(alpha.view(meas_scalar), u, value=-1)
        beta = norm(u).clamp_min(eps)
        u.div_(beta)
        beta = beta.view(scalar)

        v_new = apply_physics(physics.A_adjoint, u).addcmul_(beta, v, value=-1)
        alpha_new = norm(v_new).clamp_min(eps)
        v = v_new.div_(alpha_new)
        alpha_new = alpha_new.view(scalar)

        rho_rhobar_old = (rho * rhobar).clamp_min(eps)
        rho = torch.sqrt(alphabar * alphabar + beta * beta).clamp_min(eps)
        c = alphabar / rho
        thetanew = beta / rho * alpha_new
        alphabar = c * alpha_new

        thetabar = sbar * rho
        rhobar = torch.sqrt((cbar * rho) ** 2 + thetanew**2).clamp_min(eps)
        cbar_new = cbar * rho / rhobar
        sbar = thetanew / rhobar
        zeta = cbar_new * zetabar
        zetabar = -sbar * zetabar
        cbar = cbar_new

        # rho approximates the singular values of A
        rho_max = torch.maximum(rho_max, rho)
        if i >= 1:
            rho_min = torch.minimum(rho_min, rho)

        hbar_scale = thetabar * rho / rho_rhobar_old
        x_step = (zeta / (rho * rhobar).clamp_min(eps)) * active.to(zeta.dtype)
        hbar.mul_(-hbar_scale).add_(h)
        x.addcmul_(hbar, x_step)
        h.mul_(-(thetanew / rho)).add_(v)
        alpha = alpha_new

        # ||Ax - y|| = |zetabar| ||y||
        cond = (rho_max[:b] / rho_min[:b].clamp_min(eps)).clamp_min(1.0)
        normalized_error = (zetabar.abs() * b_norm)[:b] / (noise_norm / cond).clamp_min(eps)
        normalized_error = normalized_error.view(b, -1).amax(dim=1)

        solution = x[:b] * b_norm[:b]
        variance = (x[b:] * b_norm[b:]).view(n_samples, b, *sig_shape).pow(2).mean(dim=0)
        for k in range(N):
            crossed = (normalized_error < thresholds[k]) & (threshold_idx == k)
            mask = crossed.view(main_scalar)
            solutions[k] = torch.where(mask, solution, solutions[k])
            variances[k] = torch.where(mask, variance, variances[k])
            threshold_idx = threshold_idx + crossed.long()
        active = (threshold_idx < N).repeat(n_total).view(scalar)

    # thresholds never crossed take the last iterate
    x.mul_(b_norm)
    b_zero = b_zero[:b]
    last = x[:b].masked_fill_(b_zero, 0.0)
    last_variance = x[b:].view(n_samples, b, *sig_shape).pow(2).mean(dim=0)
    for k in range(N):
        not_filled = (threshold_idx <= k).view(main_scalar)
        solutions[k] = torch.where(not_filled, last, solutions[k]).masked_fill(b_zero, 0.0)
        variances[k] = torch.where(not_filled, last_variance, variances[k]).masked_fill(b_zero, 0.0)

    return torch.cat(solutions, dim=1), rho_max[:b].pow(2), torch.cat(variances, dim=1).sqrt(), threshold_idx

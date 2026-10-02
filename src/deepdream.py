"""Model-agnostic DeepDream: octave-based gradient ascent on activations of a
named layer, used here to visualise what a lens classifier (or Zoobot) has
learned to respond to in simulated single-lens images.

Includes optional gradient blurring + a total-variation penalty -- the
standard DeepDream regularizers. Without them, both CNNs and (especially)
ViTs tend to collapse into saturated/adversarial high-frequency noise rather
than interpretable structure, since plain gradient ascent has no pressure
toward smoothness on its own."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _gaussian_kernel(sigma: float, device) -> torch.Tensor:
    radius = max(1, int(3 * sigma))
    ax = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k1d = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    k1d /= k1d.sum()
    return k1d[:, None] @ k1d[None, :]


def _blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return img
    kernel = _gaussian_kernel(sigma, img.device).to(img.dtype)
    k = kernel.shape[-1]
    weight = kernel.expand(img.shape[1], 1, k, k)
    return F.conv2d(img, weight, padding=k // 2, groups=img.shape[1])


def _tv_loss(img: torch.Tensor) -> torch.Tensor:
    dh = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean()
    dw = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean()
    return dh + dw


class DeepDreamer:
    def __init__(self, model: torch.nn.Module, layer_name: str, device: torch.device):
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self._activation = None
        layer = self.model.get_submodule(layer_name)
        layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self._activation = output

    def dream(self, seed_image: torch.Tensor, channel: int | None = None,
              iterations: int = 40, lr: float = 0.08, octaves: int = 3,
              octave_scale: float = 1.4, jitter: int = 2,
              tv_weight: float = 0.0, grad_blur_sigma: float = 0.0,
              image_blur_every: int = 0, image_blur_sigma: float = 0.0) -> torch.Tensor:
        """seed_image: (1, C, H, W) float tensor in [0, 1]. Returns the dreamed
        image at the original resolution, still in roughly [0, 1].

        tv_weight/grad_blur_sigma/image_blur_every/image_blur_sigma default to
        off (0) for backward compatibility; pass non-zero values to suppress
        high-frequency noise -- recommended for small/undertrained models."""
        img = seed_image.clone().to(self.device)

        octave_imgs = [img]
        for _ in range(octaves - 1):
            h, w = octave_imgs[-1].shape[-2:]
            new_size = (max(1, int(h / octave_scale)), max(1, int(w / octave_scale)))
            octave_imgs.append(F.interpolate(octave_imgs[-1], size=new_size,
                                              mode="bilinear", align_corners=False))
        octave_imgs = octave_imgs[::-1]  # coarsest (smallest) first

        detail = torch.zeros_like(octave_imgs[0])
        result = octave_imgs[0]
        for octave_img in octave_imgs:
            if detail.shape[-2:] != octave_img.shape[-2:]:
                detail = F.interpolate(detail, size=octave_img.shape[-2:],
                                        mode="bilinear", align_corners=False)
            working = (octave_img + detail).clone().detach().requires_grad_(True)

            for i in range(iterations):
                sx, sy = np.random.randint(-jitter, jitter + 1, size=2) if jitter > 0 else (0, 0)
                shifted = torch.roll(working, shifts=(int(sy), int(sx)), dims=(2, 3))

                self.model(shifted)
                activation = self._activation
                if channel is not None:
                    activation = activation[:, channel]
                loss = activation.mean()
                if tv_weight > 0:
                    loss = loss - tv_weight * _tv_loss(shifted)
                loss.backward()

                grad = working.grad.data
                if grad_blur_sigma > 0:
                    grad = _blur(grad, grad_blur_sigma)
                grad = grad / (grad.std() + 1e-8)
                working.data = working.data + lr * grad
                if image_blur_every and (i + 1) % image_blur_every == 0:
                    working.data = _blur(working.data, image_blur_sigma)
                working.data.clamp_(0.0, 1.0)
                working.grad.zero_()

            result = working.detach()
            detail = result - octave_img

        return result.cpu()

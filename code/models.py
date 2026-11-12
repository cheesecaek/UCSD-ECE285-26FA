"""DCGAN model definitions used by the training and evaluation scripts."""

from __future__ import annotations

import torch
from torch import nn


class Generator(nn.Module):
    """Map a latent vector ``(N, nz, 1, 1)`` to a 64x64 RGB image."""

    def __init__(self, latent_dim: int = 128, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, c * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(c * 8, c * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(c * 2, c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(True),
            nn.ConvTranspose2d(c, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """Return one real/fake logit for every 64x64 RGB image."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(3, c, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 4, c * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image).flatten()


def dcgan_weight_init(module: nn.Module) -> None:
    """Initialization recommended in the original DCGAN paper/tutorial."""

    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())

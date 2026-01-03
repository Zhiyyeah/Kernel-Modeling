from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class KernelVAE(nn.Module):
    def __init__(self, latent_dim: int = 10, hidden_dim: int = 128, kernel_size: int = 21) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size

        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc_fc = nn.Linear(32 * kernel_size * kernel_size, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_fc2 = nn.Linear(hidden_dim, kernel_size * kernel_size)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.enc_conv(x)
        h = h.view(h.size(0), -1)
        h = F.relu(self.enc_fc(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        h = F.relu(self.dec_fc1(z))
        h = self.dec_fc2(h)
        h = h.view(-1, 1, self.kernel_size, self.kernel_size)
        h = torch.softmax(h.view(h.size(0), -1), dim=-1).view_as(h)
        return h

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @torch.no_grad()
    def encode_only(self, x: Tensor) -> Tensor:
        mu, _ = self.encode(x)
        return mu


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual, inplace=True)
        return out


class KernelPredictor(nn.Module):
    def __init__(self, in_channels: int = 5, base_channels: int = 64, latent_dim: int = 10, num_blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*(ResidualBlock(base_channels) for _ in range(num_blocks)))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(base_channels, latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        feat = self.stem(x)
        feat = self.blocks(feat)
        pooled = self.pool(feat).flatten(1)
        code = self.head(pooled)
        return code

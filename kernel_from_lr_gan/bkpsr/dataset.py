from __future__ import annotations

import glob
import math
import os
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def _meshgrid(size: int, device: torch.device) -> Tuple[Tensor, Tensor]:
    coords = torch.arange(size, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    return yy - (size - 1) / 2.0, xx - (size - 1) / 2.0


def generate_anisotropic_gaussian_kernel(
    size: int = 21,
    sigma_x: float = 1.6,
    sigma_y: float = 2.4,
    theta: float = 0.0,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Create a single anisotropic Gaussian kernel normalized to sum=1."""
    device = device or torch.device("cpu")
    yy, xx = _meshgrid(size, device)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rot_x = cos_t * xx + sin_t * yy
    rot_y = -sin_t * xx + cos_t * yy
    exp_term = (rot_x ** 2) / (2 * sigma_x ** 2) + (rot_y ** 2) / (2 * sigma_y ** 2)
    kernel = torch.exp(-exp_term)
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    return kernel.unsqueeze(0)  # [1, H, W]


def _load_npy_files(folder: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(folder, "*.npy")))
    if not files:
        raise FileNotFoundError(f"未在 {folder} 找到 .npy 文件")
    return files


class KernelGenerationDataset(Dataset):
    """Stage 1: generate random anisotropic Gaussian kernels on-the-fly."""

    def __init__(
        self,
        num_samples: int = 50000,
        size: int = 21,
        sigma_range: Tuple[float, float] = (0.8, 3.5),
        device: Optional[torch.device] = None,
    ) -> None:
        self.num_samples = num_samples
        self.size = size
        self.sigma_range = sigma_range
        self.device = device or torch.device("cpu")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tensor:  # noqa: ARG002 - idx unused
        sigma_x = np.random.uniform(*self.sigma_range)
        sigma_y = np.random.uniform(*self.sigma_range)
        theta = np.random.uniform(0, 2 * math.pi)
        kernel = generate_anisotropic_gaussian_kernel(
            size=self.size,
            sigma_x=float(sigma_x),
            sigma_y=float(sigma_y),
            theta=float(theta),
            device=self.device,
        )
        return kernel


class DegradationDataset(Dataset):
    """Stage 2: degrade HR npy patches with random kernels and return LR + kernel code."""

    def __init__(
        self,
        hr_folder: str,
        encoder: Callable[[Tensor], Tensor],
        scale: int = 4,
        kernel_size: int = 21,
        sigma_range: Tuple[float, float] = (0.8, 3.5),
        patch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.hr_files = _load_npy_files(hr_folder)
        self.encoder = encoder
        self.scale = scale
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
        self.patch_size = patch_size
        self.device = device or torch.device("cpu")

    def __len__(self) -> int:
        return len(self.hr_files)

    def _random_hr_patch(self, arr: np.ndarray) -> Tensor:
        tensor = torch.from_numpy(arr.astype(np.float32))
        if self.patch_size is None:
            return tensor
        _, h, w = tensor.shape
        if h < self.patch_size or w < self.patch_size:
            raise ValueError(f"Patch尺寸 {h}x{w} 小于裁剪尺寸 {self.patch_size}")
        y0 = torch.randint(0, h - self.patch_size + 1, (1,)).item()
        x0 = torch.randint(0, w - self.patch_size + 1, (1,)).item()
        return tensor[:, y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        hr = np.load(self.hr_files[idx])  # [C,H,W]
        hr_t = self._random_hr_patch(hr).to(self.device)

        sigma_x = np.random.uniform(*self.sigma_range)
        sigma_y = np.random.uniform(*self.sigma_range)
        theta = np.random.uniform(0, 2 * math.pi)
        kernel = generate_anisotropic_gaussian_kernel(
            size=self.kernel_size,
            sigma_x=float(sigma_x),
            sigma_y=float(sigma_y),
            theta=float(theta),
            device=self.device,
        )

        c = hr_t.shape[0]
        k = kernel.to(hr_t.device)
        k = k / k.sum().clamp_min(1e-8)
        weight = k.expand(c, 1, self.kernel_size, self.kernel_size)
        padding = self.kernel_size // 2
        lr = torch.nn.functional.conv2d(
            hr_t.unsqueeze(0),
            weight=weight,
            bias=None,
            stride=self.scale,
            padding=padding,
            groups=c,
        ).squeeze(0)

        with torch.no_grad():
            code = self.encoder(k.unsqueeze(0)).squeeze(0)

        return {"lr": lr, "kernel": k, "kernel_code": code}

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import List, Sequence, Optional


class ConditionEncoder(nn.Module):
    """
    轻量条件编码器：将输入 Patch 编码为低维向量，用于生成每层的尺度系数。
    仅对卷积核做逐通道尺度调制，保持生成链的线性特性。
    """
    def __init__(
        self,
        in_ch: int = 5,
        mid_ch: int = 32,
        ks: Sequence[int] = (7, 5, 3, 1, 1, 1),
        scale_gain: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.ks = list(ks)
        # 对应生成链每层的输出通道数（前五层 mid_ch，最后一层 1）
        self.layer_out_channels = [mid_ch, mid_ch, mid_ch, mid_ch, mid_ch, 1]
        self.total_scales = in_ch * sum(self.layer_out_channels)
        self.scale_gain = scale_gain

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(in_features=64, out_features=self.total_scales)

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        # x: [B,C,H,W]
        h = self.encoder(x)  # [B,64,H/4,W/4]
        h = torch.mean(input=h, dim=(2, 3))  # [B,64]
        raw = self.fc(h)  # [B,total_scales]
        # 将拉平的尺度向量切分为 per-band / per-layer
        scales: List[List[torch.Tensor]] = []
        start = 0
        for _ in range(self.in_ch):
            band_scales: List[torch.Tensor] = []
            for out_c in self.layer_out_channels:
                end = start + out_c
                # 约束尺度在 [0.9, 1.1] 左右的范围，避免数值暴涨
                s = 1.0 + self.scale_gain * torch.tanh(raw[:, start:end])  # [B,out_c]
                band_scales.append(s)
                start = end
            scales.append(band_scales)
        return scales


class DynamicMultiBandLinearGenerator(nn.Module):
    """
    动态多波段线性生成器：
    - 基础卷积核作为可学习参数
    - 条件编码器输出逐层逐通道尺度，动态调制卷积核
    - 深度线性结构保持可提取性
    """
    def __init__(
        self,
        in_ch: int = 5,
        mid_ch: int = 32,
        ks: Sequence[int] = (7, 5, 3, 1, 1, 1),
        scale_gain: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.ks = list(ks)
        self.layer_out_channels = [mid_ch, mid_ch, mid_ch, mid_ch, mid_ch, 1]

        # 基础卷积权重（仍可学习），后续由条件编码器输出的尺度进行调制
        weight_bands: List[nn.ParameterList] = []
        for _ in range(in_ch):
            band_weights = nn.ParameterList()
            in_c = 1
            for idx, ksize in enumerate(self.ks):
                out_c = self.layer_out_channels[idx]
                w = nn.Parameter(torch.randn(out_c, in_c, ksize, ksize) * 0.01)
                band_weights.append(w)
                in_c = out_c
            weight_bands.append(band_weights)
        self.weight_bands = nn.ModuleList(weight_bands)

        # 8 倍下采样
        self.down1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.down2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.down3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.condition_encoder = ConditionEncoder(in_ch=in_ch, mid_ch=mid_ch, ks=ks, scale_gain=scale_gain)

    def _apply_conv_chain(
        self,
        x: torch.Tensor,
        band_weights: nn.ParameterList,
        band_scales: List[torch.Tensor],
    ) -> torch.Tensor:
        h = x  # [1,1,H,W]
        for idx, (w_base, ksize) in enumerate(zip(band_weights, self.ks)):
            scale = band_scales[idx]  # [1,out_c]
            w = w_base * scale.view(-1, 1, 1, 1)
            pad = ksize // 2
            h = F.pad(h, pad=(pad, pad, pad, pad), mode='reflect')
            h = F.conv2d(input=h, weight=w, bias=None)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        scales = self.condition_encoder(x)  # List[band][layer] each [B,out_c]
        outs: List[torch.Tensor] = []
        batch_size = x.shape[0]
        for b in range(batch_size):
            band_outs: List[torch.Tensor] = []
            for band in range(self.in_ch):
                h = x[b:b+1, band:band+1]
                band_weights = self.weight_bands[band]
                band_scales = [s[b:b+1] for s in scales[band]]  # 取当前样本
                h = self._apply_conv_chain(h, band_weights, band_scales)
                band_outs.append(h)
            outs.append(torch.cat(tensors=band_outs, dim=1))
        return torch.cat(tensors=outs, dim=0)

    @torch.no_grad()
    def extract_effective_kernels(self, x: Optional[torch.Tensor] = None, reduce_batch: bool = True) -> torch.Tensor:
        """
        提取等效模糊核：
        - 若提供 x，则使用其条件尺度生成对应的动态核；可对批次求平均。
        - 若 x 为 None，则返回使用单位尺度（未调制）的核。
        返回: [C,kH,kW] 或 [B,C,kH,kW]
        """
        device = self.weight_bands[0][0].device
        dtype = self.weight_bands[0][0].dtype
        if x is None:
            batch_size = 1
            scales: List[List[torch.Tensor]] = []
            for _ in range(self.in_ch):
                band_scales = [torch.ones(1, out_c, device=device, dtype=dtype) for out_c in self.layer_out_channels]
                scales.append(band_scales)
        else:
            scales = self.condition_encoder(x)
            batch_size = x.shape[0]

        def conv_kernel(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            a = A.unsqueeze(dim=0).unsqueeze(dim=0)
            b = torch.flip(input=B, dims=[0, 1]).unsqueeze(dim=0).unsqueeze(dim=0)
            pad_h = b.shape[-2] - 1
            pad_w = b.shape[-1] - 1
            y = F.conv2d(input=a, weight=b, padding=(pad_h, pad_w))
            return y.squeeze(dim=0).squeeze(dim=0)

        def compose_kernel(sample_idx: int) -> torch.Tensor:
            kernel_list: List[torch.Tensor] = []
            for band in range(self.in_ch):
                weights = []
                for layer_idx, (w_base, ksize) in enumerate(zip(self.weight_bands[band], self.ks)):
                    s = scales[band][layer_idx][sample_idx:sample_idx+1].view(-1, 1, 1, 1)
                    w = w_base * s  # [out,in,k,k]
                    weights.append(w)
                K_cur = weights[0]
                for W in weights[1:]:
                    C_out, C_mid, kH, kW = W.shape
                    _, C_in, _, _ = K_cur.shape
                    K_next: List[torch.Tensor] = []
                    for co in range(C_out):
                        row: List[torch.Tensor] = []
                        for ci in range(C_in):
                            acc = None
                            for cm in range(C_mid):
                                kA = W[co, cm]
                                kB = K_cur[cm, ci]
                                kk = conv_kernel(kA, kB)
                                acc = kk if acc is None else acc + kk
                            row.append(acc)
                        row_stacked = torch.stack(tensors=row, dim=0)
                        K_next.append(row_stacked)
                    K_cur = torch.stack(tensors=K_next, dim=0)
                k = K_cur.mean(dim=(0, 1))
                k = torch.clamp(input=k, min=0)
                s_sum = k.sum()
                if s_sum <= 1e-12:
                    s_sum = torch.tensor(data=1.0, device=k.device, dtype=k.dtype)
                k = k / s_sum
                kernel_list.append(k)
            return torch.stack(tensors=kernel_list, dim=0)

        kernels = [compose_kernel(i) for i in range(batch_size)]  # [B,C,kH,kW]
        kernels_stacked = torch.stack(tensors=kernels, dim=0)
        if reduce_batch:
            return kernels_stacked.mean(dim=0)
        return kernels_stacked

    @torch.no_grad()
    def extract_merged_kernel(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        ks = self.extract_effective_kernels(x=x, reduce_batch=True)  # [C,kH,kW]
        return ks.mean(dim=0)


class NoiseEstimator(nn.Module):
    """
    可学习噪声估计器：输出 per-channel 的高斯噪声强度 sigma，并叠加噪声。
    """
    def __init__(self, channels: int = 5, init_sigma: float = 0.01, sigma_max: float = 0.2) -> None:
        super().__init__()
        self.channels = channels
        self.sigma_max = sigma_max
        init = torch.full(size=(channels,), fill_value=init_sigma)
        self.log_sigma = nn.Parameter(torch.log(init))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = torch.clamp(torch.exp(self.log_sigma), min=1e-5, max=self.sigma_max)  # [C]
        noise = torch.randn_like(x) * sigma.view(1, -1, 1, 1)
        return x + noise, sigma


class DegradationModel(nn.Module):
    """
    组合动态模糊生成器与噪声估计器，输出干净与含噪低清图。
    """
    def __init__(
        self,
        in_ch: int = 5,
        mid_ch: int = 32,
        ks: Sequence[int] = (7, 5, 3, 1, 1, 1),
        scale_gain: float = 0.1,
        noise_init: float = 0.01,
        noise_max: float = 0.2,
    ) -> None:
        super().__init__()
        self.generator = DynamicMultiBandLinearGenerator(in_ch=in_ch, mid_ch=mid_ch, ks=ks, scale_gain=scale_gain)
        self.noise_estimator = NoiseEstimator(channels=in_ch, init_sigma=noise_init, sigma_max=noise_max)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clean = self.generator(x)
        noisy, sigma = self.noise_estimator(clean)
        return clean, noisy, sigma


class PatchDiscriminator(nn.Module):
    """
    全卷积 Patch 判别器，和原版保持一致。
    """
    def __init__(self, in_ch: int = 5, base_ch: int = 64, num_blocks: int = 4):
        super().__init__()
        layers: List[nn.Module] = []
        layers.append(spectral_norm(nn.Conv2d(in_channels=in_ch, out_channels=base_ch, kernel_size=7, stride=1, padding=3)))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        ch = base_ch
        for _ in range(num_blocks):
            layers.append(spectral_norm(nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=1, stride=1, padding=0)))
            layers.append(nn.BatchNorm2d(num_features=ch))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        layers.append(spectral_norm(nn.Conv2d(in_channels=ch, out_channels=1, kernel_size=1, stride=1, padding=0)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = DynamicMultiBandLinearGenerator(in_ch=5, mid_ch=32).to(device)
    D = PatchDiscriminator(in_ch=5).to(device)
    x = torch.rand(4, 5, 64, 64, device=device)
    clean = G(x)
    print(f"G out: {tuple(clean.shape)}")
    ks = G.extract_effective_kernels(x)
    km = G.extract_merged_kernel(x)
    print(f"kernels: {tuple(ks.shape)} merged: {tuple(km.shape)} sum(merged)={km.sum().item():.4f}")
    s = D(clean)
    print(f"D out: {tuple(s.shape)}")

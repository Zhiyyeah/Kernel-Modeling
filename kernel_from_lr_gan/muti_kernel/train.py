from __future__ import annotations

import os
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from networks import DegradationModel, PatchDiscriminator
from loss import lsgan_d_loss, lsgan_g_loss, kernel_regularization, noise_reg_loss


def load_patches_from_folder(patch_dir: str) -> tuple[list, int]:
    patch_files = sorted(glob.glob(os.path.join(patch_dir, '*.npy')))
    if len(patch_files) == 0:
        raise ValueError(f"在 {patch_dir} 中没有找到 .npy 文件")
    first_patch = np.load(patch_files[0])
    original_size = first_patch.shape[1]
    print(f"找到 {len(patch_files)} 个patch文件")
    print(f"原始patch尺寸: {first_patch.shape}")
    return patch_files, original_size


def sample_patches_from_files(
    patch_files: list,
    batch_size: int,
    target_size: int = 128,
    original_size: int = 128,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    selected_indices = torch.randint(low=0, high=len(patch_files), size=(batch_size,))
    patches = []
    for idx in selected_indices:
        patch = np.load(patch_files[idx.item()])  # [5,H,W]
        patch_tensor = torch.from_numpy(patch.astype(np.float32))
        if torch.isnan(patch_tensor).any():
            nan_count = torch.isnan(patch_tensor).sum().item()
            nan_ratio = nan_count / patch_tensor.numel() * 100
            raise ValueError(
                f"Patch文件包含NaN值: {patch_files[idx.item()]}\n"
                f"NaN像素数: {nan_count}/{patch_tensor.numel()} ({nan_ratio:.2f}%)\n"
                f"这表示patch质量不足，应该在生成阶段就被过滤掉。"
            )
        if original_size != target_size:
            H, W = patch_tensor.shape[-2], patch_tensor.shape[-1]
            max_y = H - target_size
            max_x = W - target_size
            if max_y <= 0 or max_x <= 0:
                raise ValueError(
                    f"Patch尺寸 {H}x{W} 小于目标尺寸 {target_size}x{target_size}，无法裁剪"
                )
            y0 = torch.randint(0, max_y + 1, (1,)).item()
            x0 = torch.randint(0, max_x + 1, (1,)).item()
            patch_tensor = patch_tensor[:, y0:y0+target_size, x0:x0+target_size]
        patches.append(patch_tensor)
    result = torch.stack(tensors=patches, dim=0)
    if device is not None:
        result = result.to(device)
    return result


def ascii_kernel(k: torch.Tensor, size: int = 11) -> str:
    k2 = k.unsqueeze(dim=0).unsqueeze(dim=0)
    k2 = F.interpolate(input=k2, size=(size, size), mode='bilinear', align_corners=False)
    k2 = k2[0, 0]
    chars = " .:-=+*#%@"
    mx = k2.max().item() + 1e-12
    out_lines = []
    for i in range(size):
        line = ''
        for j in range(size):
            v = k2[i, j].item() / mx
            idx = min(int(v * (len(chars) - 1)), len(chars) - 1)
            line += chars[idx]
        out_lines.append(line)
    return '\n'.join(out_lines)


def kernel_metrics(k: torch.Tensor) -> dict:
    kH, kW = k.shape
    thresh = k.max() * 0.05
    sparsity = float((k > thresh).float().mean().item())
    yy, xx = torch.meshgrid(torch.arange(kH, device=k.device), torch.arange(kW, device=k.device), indexing='ij')
    mass = k + 1e-12
    cy = (yy.float() * mass).sum() / mass.sum()
    cx = (xx.float() * mass).sum() / mass.sum()
    center_y = (kH - 1) / 2.0
    center_x = (kW - 1) / 2.0
    center_offset = float(((cy - center_y) ** 2 + (cx - center_x) ** 2).sqrt().item())
    return {
        'k_shape': f'{kH}x{kW}',
        'k_sum': float(k.sum().item()),
        'k_max': float(k.max().item()),
        'k_min': float(k.min().item()),
        'k_std': float(k.std().item()),
        'sparsity': sparsity,
        'center_offset': center_offset,
    }


def generator_weight_stats(G: torch.nn.Module) -> str:
    vals = []
    for b, chain in enumerate(G.weight_bands):
        w0 = chain[0]
        w_last = chain[-1]
        vals.append(f"B{b}(L0n={w0.norm().item():.3f},Ln={w_last.norm().item():.3f})")
    return ' '.join(vals)


def main() -> None:
    use_cpu = True
    device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu else 'cpu')
    print(f'Using device: {device}')

    patch_dir = r'/Users/zy/Downloads/GOCI-2/patches_all256' 
    print(f'使用patch文件夹: {patch_dir}')
    patch_files, original_patch_size = load_patches_from_folder(patch_dir)

    iters = 3000
    patch_size = 256
    batch_size = 8
    lr_rate = 1e-4
    outdir = './muti_kernel/kernelgan_out'
    log_every = 100
    kernel_log_every = 300
    mini_log_every = 10
    save_intermediate = True
    verbose = True
    target_sigma = torch.tensor([0.56, 0.73, 0.84, 0.63, 0.20], device=device)
    noise_reg_weight = 20.0

    print(f'将从 {len(patch_files)} 个文件中随机采样')
    print(f'原始patch尺寸: {original_patch_size}x{original_patch_size}')
    print(f'目标patch尺寸: {patch_size}x{patch_size}')

    model = DegradationModel(in_ch=5, mid_ch=32, ks=(7, 5, 3, 1, 1, 1), scale_gain=0.1, noise_init=0.3, noise_max=1.2)
    model = model.to(device)
    D = PatchDiscriminator(in_ch=5, base_ch=64, num_blocks=4).to(device)

    opt_D = optim.Adam(params=D.parameters(), lr=lr_rate, betas=(0.5, 0.999))
    opt_G = optim.Adam(params=model.parameters(), lr=lr_rate, betas=(0.5, 0.999))

    prev_k = None

    progress = tqdm(range(iters), desc='Training', unit='iter')
    for t in progress:
        iter_idx = t + 1

        def log_stage(stage: str) -> None:
            progress.set_postfix_str(f'{stage}', refresh=False)

        log_stage('sampling HR patches')
        patches = sample_patches_from_files(
            patch_files=patch_files,
            batch_size=batch_size,
            target_size=patch_size,
            original_size=original_patch_size,
            device=device,
        )
        log_stage('sampling real LR patches')
        real_ds = sample_patches_from_files(
            patch_files=patch_files,
            batch_size=batch_size,
            target_size=32,
            original_size=original_patch_size,
            device=device,
        )

        log_stage('running degradation model')
        clean_fake, fake_ds, sigma = model(patches)

        # 训练 D
        D.train(); model.train()
        log_stage('updating discriminator')
        pred_real = D(real_ds)
        pred_fake = D(fake_ds.detach())
        loss_D = lsgan_d_loss(pred_real, pred_fake)
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()

        # 训练 G（动态模糊 + 噪声）
        log_stage('updating generator')
        pred_fake = D(fake_ds)
        loss_G_adv = lsgan_g_loss(pred_fake)
        ks_band = model.generator.extract_effective_kernels(patches, reduce_batch=True)  # [C,kH,kW]
        reg_list = [kernel_regularization(k=ks_band[i], alpha=0.5, beta=0.5, gamma=5.0, delta=1.0) for i in range(ks_band.shape[0])]
        loss_reg = torch.mean(input=torch.stack(tensors=reg_list))
        loss_noise = noise_reg_loss(sigma, target=target_sigma, mode='l2')
        loss_G = loss_G_adv + loss_reg + noise_reg_weight * loss_noise
        opt_G.zero_grad(); loss_G.backward(); opt_G.step()

        if iter_idx % mini_log_every == 0:
            tqdm.write(f"Iter {iter_idx}/{iters} | D: {loss_D.item():.4f} | G_adv: {loss_G_adv.item():.4f} | Noise: {loss_noise.item():.4f} | sigma_mean: {sigma.mean().item():.4f}")
        if iter_idx % log_every == 0:
            extra = generator_weight_stats(model.generator) if verbose else ''
            tqdm.write(f"[LOG] Iter {iter_idx}/{iters} | D: {loss_D.item():.4f} | G_adv: {loss_G_adv.item():.4f} | Reg: {loss_reg.item():.4f} | Noise: {loss_noise.item():.4f} {extra}")

        if iter_idx % kernel_log_every == 0:
            ks_all = model.generator.extract_effective_kernels(patches, reduce_batch=False)  # [B,C,kH,kW]
            k_merged = ks_all.mean(dim=(0, 1))
            delta = 0.0
            if prev_k is not None:
                delta = torch.norm(k_merged - prev_k).item()
            prev_k = k_merged.detach().clone()
            km = kernel_metrics(k_merged)
            tqdm.write(
                f"  [Kernel] shape={km['k_shape']} sum={km['k_sum']:.4f} max={km['k_max']:.4f} min={km['k_min']:.4f} "
                f"std={km['k_std']:.4f} sparsity(>5%max)={km['sparsity']:.3f} center_offset={km['center_offset']:.3f} delta_L2={delta:.5f}"
            )
            if verbose:
                tqdm.write("  [Kernel ASCII merged]\n" + ascii_kernel(k_merged))
                band_max = ' '.join([f'b{i}_max={ks_all[:, i].max().item():.3f}' for i in range(min(ks_all.shape[1], 3))])
                tqdm.write(f"  [Bands] {band_max} | sigma_mean={sigma.mean().item():.4f}")
            if save_intermediate:
                os.makedirs(outdir, exist_ok=True)
                np.save(os.path.join(outdir, f'kernel_iter{t+1}.npy'), k_merged.cpu().numpy())
                np.save(os.path.join(outdir, f'kernel_per_band_iter{t+1}.npy'), ks_all.mean(dim=0).cpu().numpy())

    ks_final = model.generator.extract_effective_kernels(patches, reduce_batch=False).mean(dim=0).cpu().numpy()
    k_final_merged = ks_final.mean(axis=0)
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, 'kernel_per_band.npy'), ks_final)
    np.save(os.path.join(outdir, 'kernel_merged.npy'), k_final_merged)
    print(f"Saved per-band kernels -> kernel_per_band.npy ({ks_final.shape}), merged -> kernel_merged.npy sum={k_final_merged.sum():.6f}")


if __name__ == "__main__":
    main()

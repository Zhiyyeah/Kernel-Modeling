import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# 加载结果
save_dir = './moe_kernels'

# 加载所有核和噪声
kernels = []
sigmas = []

for k_idx in range(10):
    kernel = np.load(os.path.join(save_dir, f'kernel_{k_idx}.npy'))  # [5, 13, 13]
    sigma = np.load(os.path.join(save_dir, f'sigma_{k_idx}.npy'))    # [5]
    kernels.append(kernel)
    sigmas.append(sigma)

kernels = np.array(kernels)  # [10, 5, 13, 13]
sigmas = np.array(sigmas)    # [10, 5]

print(f"Kernels shape: {kernels.shape}")
print(f"Sigmas shape: {sigmas.shape}")

# ==========================================
# 1. 可视化所有 10 个核 (取平均波段)
# ==========================================
fig, axes = plt.subplots(2, 5, figsize=(16, 8))
fig.suptitle('10 Degradation Kernels (Mean across 5 bands)', fontsize=16, fontweight='bold')

for k_idx in range(10):
    ax = axes[k_idx // 5, k_idx % 5]
    # 对 5 个波段求平均
    kernel_mean = kernels[k_idx].mean(axis=0)  # [13, 13]
    im = ax.imshow(kernel_mean, cmap='viridis')
    ax.set_title(f'Kernel {k_idx}', fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('./moe_kernels/kernels_visualization.png', dpi=150, bbox_inches='tight')
print("✓ Saved kernels_visualization.png")
plt.show()

# ==========================================
# 2. 可视化所有 5 个波段的核 (for kernel 0 as example)
# ==========================================
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle('Kernel 0: All 5 Bands', fontsize=14, fontweight='bold')

for band_idx in range(5):
    ax_top = axes[0, band_idx]
    ax_bot = axes[1, band_idx]
    
    kernel_band = kernels[0, band_idx]  # [13, 13]
    
    # 第一行：热力图
    im = ax_top.imshow(kernel_band, cmap='RdBu_r')
    ax_top.set_title(f'Band {band_idx}', fontweight='bold')
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    plt.colorbar(im, ax=ax_top)
    
    # 第二行：3D 柱状图（简单表示）
    X, Y = np.meshgrid(range(13), range(13))
    ax_bot.contourf(X, Y, kernel_band, levels=15, cmap='RdBu_r')
    ax_bot.set_title(f'Band {band_idx} Contour')
    ax_bot.set_xticks([])
    ax_bot.set_yticks([])

plt.tight_layout()
plt.savefig('./moe_kernels/kernel_0_bands_detail.png', dpi=150, bbox_inches='tight')
print("Saved kernel_0_bands_detail.png")
plt.show()

# ==========================================
# 3. 噪声水平对比 (Sigma across kernels)
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：每个核的平均噪声
sigma_mean = sigmas.mean(axis=1)  # [10]
ax = axes[0]
bars = ax.bar(range(10), sigma_mean, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xlabel('Kernel Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Noise Sigma', fontsize=12, fontweight='bold')
ax.set_title('Average Noise Level per Kernel', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 右图：每个波段的噪声分布 (Box plot)
ax = axes[1]
data_for_box = [sigmas[:, band_idx] for band_idx in range(5)]
bp = ax.boxplot(data_for_box, labels=[f'Band {i}' for i in range(5)], patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightgreen')
    patch.set_alpha(0.7)
ax.set_ylabel('Noise Sigma', fontsize=12, fontweight='bold')
ax.set_title('Noise Distribution across Bands', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('./moe_kernels/noise_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved noise_analysis.png")
plt.show()

# ==========================================
# 4. 热力图矩阵：所有核的所有波段
# ==========================================
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(10, 5, hspace=0.4, wspace=0.3)

for k_idx in range(10):
    for band_idx in range(5):
        ax = fig.add_subplot(gs[k_idx, band_idx])
        kernel_2d = kernels[k_idx, band_idx]  # [13, 13]
        im = ax.imshow(kernel_2d, cmap='viridis')
        ax.set_xticks([])
        ax.set_yticks([])
        if band_idx == 0:
            ax.set_ylabel(f'K{k_idx}', fontweight='bold', fontsize=10)
        if k_idx == 0:
            ax.set_title(f'B{band_idx}', fontweight='bold', fontsize=10)

fig.suptitle('All 10 Kernels × 5 Bands Heatmap', fontsize=16, fontweight='bold')
plt.savefig('./moe_kernels/all_kernels_heatmap.png', dpi=150, bbox_inches='tight')
print("✓ Saved all_kernels_heatmap.png")
plt.show()

# ==========================================
# 5. 打印统计信息
# ==========================================
print("\n" + "="*60)
print("KERNEL AND NOISE STATISTICS")
print("="*60)
print(f"\nKernels shape: {kernels.shape} (10 kernels, 5 bands, 13×13 spatial)")
print(f"Sigmas shape: {sigmas.shape} (10 kernels, 5 bands)\n")

print("Kernel statistics (across spatial dimensions):")
for k_idx in range(10):
    kernel_flat = kernels[k_idx].flatten()
    print(f"  Kernel {k_idx}: min={kernel_flat.min():.4f}, max={kernel_flat.max():.4f}, "
          f"mean={kernel_flat.mean():.4f}, std={kernel_flat.std():.4f}")

print("\nNoise sigma statistics (per kernel):")
for k_idx in range(10):
    sigma = sigmas[k_idx]
    print(f"  Kernel {k_idx}: {sigma} → mean={sigma.mean():.4f}")

print("\nNoise sigma statistics (per band):")
for band_idx in range(5):
    sigma_band = sigmas[:, band_idx]
    print(f"  Band {band_idx}: min={sigma_band.min():.4f}, max={sigma_band.max():.4f}, "
          f"mean={sigma_band.mean():.4f}")

print("\n" + "="*60)
print("All visualizations saved to: ./moe_kernels/")
print("="*60)

# ==========================================
# 6. 噪声可视化（单独）
# ==========================================
# 6.1 噪声箱线图（每核 × 5 波段）
fig, ax = plt.subplots(figsize=(10, 5))
data_for_box = [sigmas[i] for i in range(sigmas.shape[0])]
bp = ax.boxplot(data_for_box, labels=[f'K{i}' for i in range(sigmas.shape[0])], patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.7)
ax.set_ylabel('Sigma', fontsize=12, fontweight='bold')
ax.set_title('Noise Sigma per Kernel (Boxplot across 5 bands)', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('./moe_kernels/noise_box_per_kernel.png', dpi=150, bbox_inches='tight')
print("✓ Saved noise_box_per_kernel.png")
plt.show()

# 6.2 噪声热力图：核 × 波段
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(sigmas, cmap='YlOrRd', aspect='auto')
ax.set_xlabel('Band', fontsize=12, fontweight='bold')
ax.set_ylabel('Kernel', fontsize=12, fontweight='bold')
ax.set_title('Noise Sigma Heatmap (Kernels × Bands)', fontsize=13, fontweight='bold')
ax.set_xticks(range(5))
ax.set_xticklabels([f'B{i}' for i in range(5)])
ax.set_yticks(range(sigmas.shape[0]))
ax.set_yticklabels([f'K{i}' for i in range(sigmas.shape[0])])
plt.colorbar(im, ax=ax, label='Sigma')
plt.tight_layout()
plt.savefig('./moe_kernels/noise_heatmap.png', dpi=150, bbox_inches='tight')
print("✓ Saved noise_heatmap.png")
plt.show()

# 6.3 噪声直方图：所有核、所有波段展开
fig, ax = plt.subplots(figsize=(8, 4))
flat_sigma = sigmas.flatten()
ax.hist(flat_sigma, bins=40, color='steelblue', alpha=0.8)
ax.set_title('Histogram of All Noise Sigmas', fontsize=13, fontweight='bold')
ax.set_xlabel('Sigma value')
ax.set_ylabel('Count')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('./moe_kernels/noise_hist_all.png', dpi=150, bbox_inches='tight')
print("✓ Saved noise_hist_all.png")
plt.show()

# ==========================================
# 6. 核/噪声的两两差异矩阵（L2 距离）
# ==========================================
kernel_flat = kernels.reshape(kernels.shape[0], -1)
kernel_diff = np.linalg.norm(kernel_flat[:, None, :] - kernel_flat[None, :, :], axis=-1)

sigma_flat = sigmas.reshape(sigmas.shape[0], -1)
sigma_diff = np.linalg.norm(sigma_flat[:, None, :] - sigma_flat[None, :, :], axis=-1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
im0 = axes[0].imshow(kernel_diff, cmap='magma')
axes[0].set_title('Pairwise L2 Distance of Kernels')
axes[0].set_xlabel('Kernel idx')
axes[0].set_ylabel('Kernel idx')
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(sigma_diff, cmap='magma')
axes[1].set_title('Pairwise L2 Distance of Noise Sigmas')
axes[1].set_xlabel('Kernel idx')
axes[1].set_ylabel('Kernel idx')
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('./moe_kernels/pairwise_distance.png', dpi=150, bbox_inches='tight')
print("✓ Saved pairwise_distance.png")
plt.show()

print("\n核方差（展平后）:")
for i in range(kernels.shape[0]):
    var = kernel_flat[i].var()
    print(f"  Kernel {i}: var={var:.6e}")

print("\n噪声方差（展平后）:")
for i in range(sigmas.shape[0]):
    var = sigma_flat[i].var()
    print(f"  Sigma {i}: var={var:.6e}")

import numpy as np
import matplotlib.pyplot as plt
import os

save_dir = './moe_kernels'

# 加载所有噪声
sigmas = []
for k_idx in range(10):
    sigma = np.load(os.path.join(save_dir, f'sigma_{k_idx}.npy'))
    sigmas.append(sigma)

sigmas = np.array(sigmas)  # [10, 5]

print("\n" + "="*70)
print("NOISE SIGMA 详细数据")
print("="*70)

print("\n【按核索引排列】")
print("Kernel ID | Band 0    | Band 1    | Band 2    | Band 3    | Band 4    | 平均值")
print("-" * 85)
for k_idx in range(10):
    sigma_vals = sigmas[k_idx]
    mean_val = sigma_vals.mean()
    print(f"Kernel {k_idx} | {sigma_vals[0]:9.6f} | {sigma_vals[1]:9.6f} | {sigma_vals[2]:9.6f} | {sigma_vals[3]:9.6f} | {sigma_vals[4]:9.6f} | {mean_val:9.6f}")

print("\n【按波段统计】")
print("Band ID | Min        | Max        | Mean       | Std")
print("-" * 65)
for band_idx in range(5):
    sigma_band = sigmas[:, band_idx]
    print(f"Band {band_idx} | {sigma_band.min():10.6f} | {sigma_band.max():10.6f} | {sigma_band.mean():10.6f} | {sigma_band.std():10.6f}")

print("\n" + "="*70)

# ==========================================
# 可视化：噪声热力图
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：噪声矩阵热力图 (10 kernels × 5 bands)
ax = axes[0]
im = ax.imshow(sigmas, cmap='YlOrRd', aspect='auto')
ax.set_xlabel('Band Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Kernel Index', fontsize=12, fontweight='bold')
ax.set_title('Noise Sigma Heatmap (10 Kernels × 5 Bands)', fontsize=13, fontweight='bold')
ax.set_xticks(range(5))
ax.set_yticks(range(10))
ax.set_xticklabels([f'B{i}' for i in range(5)])
ax.set_yticklabels([f'K{i}' for i in range(10)])

# 添加数值标注
for k_idx in range(10):
    for band_idx in range(5):
        text = ax.text(band_idx, k_idx, f'{sigmas[k_idx, band_idx]:.3f}',
                      ha="center", va="center", color="black", fontsize=8)

plt.colorbar(im, ax=ax, label='Sigma Value')

# 右图：每个核的平均噪声 + 每个波段的平均噪声
ax = axes[1]
x_pos = np.arange(10)
sigma_per_kernel = sigmas.mean(axis=1)
ax.bar(x_pos, sigma_per_kernel, color='coral', alpha=0.7, edgecolor='darkred', linewidth=1.5, label='Per Kernel')
ax.axhline(sigmas.mean(), color='blue', linestyle='--', linewidth=2, label=f'Global Mean: {sigmas.mean():.4f}')
ax.set_xlabel('Kernel Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Noise Sigma', fontsize=12, fontweight='bold')
ax.set_title('Average Noise Level per Kernel', fontsize=13, fontweight='bold')
ax.set_xticks(range(10))
ax.set_xticklabels([f'K{i}' for i in range(10)])
ax.grid(axis='y', alpha=0.3)
ax.legend(fontsize=10)

# 在柱子上显示数值
for i, v in enumerate(sigma_per_kernel):
    ax.text(i, v + 0.002, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('./moe_kernels/noise_sigma_detailed.png', dpi=150, bbox_inches='tight')
print("✓ 噪声可视化已保存到: ./moe_kernels/noise_sigma_detailed.png")
plt.show()

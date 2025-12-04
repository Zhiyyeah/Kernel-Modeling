import os
import numpy as np
import matplotlib.pyplot as plt
import glob

def visualize_kernels(kernel_dir='../../kernelgan_out', output_path=None, dpi=300):
    """
    可视化kernelgan_out文件夹中的所有模糊核
    
    参数:
        kernel_dir (str): 包含.npy核文件的文件夹路径
        output_path (str, optional): 输出图片路径，默认保存到kernel_dir
        dpi (int): 输出图片分辨率，默认300
    """
    # 查找所有.npy文件
    kernel_files = sorted(glob.glob(os.path.join(kernel_dir, '*.npy')))
    
    if len(kernel_files) == 0:
        print(f"在 {kernel_dir} 中没有找到 .npy 文件")
        return
    
    print(f"找到 {len(kernel_files)} 个核文件:")
    for f in kernel_files:
        print(f"  - {os.path.basename(f)}")
    
    # 加载所有核
    kernels = []
    labels = []
    for f in kernel_files:
        k = np.load(f)
        kernels.append(k)
        labels.append(os.path.basename(f).replace('.npy', ''))
        print(f"\n{os.path.basename(f)}:")
        print(f"  形状: {k.shape}")
        print(f"  总和: {k.sum():.6f}")
        print(f"  最大值: {k.max():.6f}")
        print(f"  最小值: {k.min():.6f}")
        print(f"  均值: {k.mean():.6f}")
    
    # 创建可视化
    n_kernels = len(kernels)
    
    # 计算子图布局
    if n_kernels <= 4:
        nrows, ncols = 1, n_kernels
        figsize = (5 * n_kernels, 5)
    elif n_kernels <= 8:
        nrows, ncols = 2, (n_kernels + 1) // 2
        figsize = (5 * ncols, 5 * nrows)
    else:
        ncols = 4
        nrows = (n_kernels + ncols - 1) // ncols
        figsize = (5 * ncols, 5 * nrows)
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    # 确保axes是数组
    if n_kernels == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # 绘制每个核
    for idx, (kernel, label) in enumerate(zip(kernels, labels)):
        ax = axes[idx]
        
        # 使用热力图显示核
        im = ax.imshow(kernel, cmap='hot', interpolation='nearest')
        ax.set_title(f'{label}\nShape: {kernel.shape}, Sum: {kernel.sum():.4f}', 
                    fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Kernel Value', fontsize=8)
        
        # 在核上标注数值（如果核不太大）
        if kernel.shape[0] <= 13 and kernel.shape[1] <= 13:
            for i in range(kernel.shape[0]):
                for j in range(kernel.shape[1]):
                    text_color = 'white' if kernel[i, j] > kernel.max() * 0.5 else 'black'
                    ax.text(j, i, f'{kernel[i, j]:.3f}', 
                           ha='center', va='center', 
                           fontsize=6, color=text_color)
    
    # 隐藏多余的子图
    for idx in range(n_kernels, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Kernel Visualization from {os.path.basename(kernel_dir)}', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 保存图片
    if output_path is None:
        output_path = os.path.join(kernel_dir, 'kernels_visualization.png')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"\n可视化结果已保存至: {output_path}")
    
    plt.show()
    
    # 绘制核的演化（如果有多个迭代核）
    iter_kernels = [(k, l) for k, l in zip(kernels, labels) if 'iter' in l]
    if len(iter_kernels) > 1:
        print("\n绘制核演化图...")
        plot_kernel_evolution(iter_kernels, kernel_dir)


def plot_kernel_evolution(iter_kernels, kernel_dir):
    """
    绘制核在训练过程中的演化
    
    参数:
        iter_kernels (list): (kernel, label) 元组列表
        kernel_dir (str): 输出目录
    """
    # 按迭代次数排序
    def extract_iter(label):
        import re
        match = re.search(r'iter(\d+)', label)
        return int(match.group(1)) if match else 0
    
    iter_kernels = sorted(iter_kernels, key=lambda x: extract_iter(x[1]))
    
    n_iters = len(iter_kernels)
    fig, axes = plt.subplots(1, n_iters, figsize=(5 * n_iters, 5))
    
    if n_iters == 1:
        axes = [axes]
    
    # 找到所有核的全局最大值，用于统一颜色范围
    vmax = max([k.max() for k, _ in iter_kernels])
    
    for idx, (kernel, label) in enumerate(iter_kernels):
        ax = axes[idx]
        im = ax.imshow(kernel, cmap='hot', interpolation='nearest', vmin=0, vmax=vmax)
        
        iter_num = extract_iter(label)
        ax.set_title(f'Iteration {iter_num}\nSum: {kernel.sum():.4f}', 
                    fontsize=12, fontweight='bold')
        ax.axis('off')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if idx == n_iters - 1:
            cbar.set_label('Kernel Value', fontsize=10)
    
    plt.suptitle('Kernel Evolution During Training', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(kernel_dir, 'kernel_evolution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"核演化图已保存至: {output_path}")
    
    plt.show()


def compare_kernels_stats(kernel_dir='../../kernelgan_out'):
    """
    比较不同迭代核的统计特性
    
    参数:
        kernel_dir (str): 包含.npy核文件的文件夹路径
    """
    kernel_files = sorted(glob.glob(os.path.join(kernel_dir, 'kernel_iter*.npy')))
    
    if len(kernel_files) == 0:
        print("没有找到迭代核文件")
        return
    
    # 提取迭代次数和统计数据
    import re
    stats = []
    
    for f in kernel_files:
        k = np.load(f)
        match = re.search(r'iter(\d+)', os.path.basename(f))
        iter_num = int(match.group(1)) if match else 0
        
        # 计算质心
        h, w = k.shape
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        total_mass = k.sum()
        if total_mass > 0:
            center_y = (y_coords * k).sum() / total_mass
            center_x = (x_coords * k).sum() / total_mass
            center_offset = np.sqrt((center_y - (h-1)/2)**2 + (center_x - (w-1)/2)**2)
        else:
            center_offset = 0
        
        stats.append({
            'iter': iter_num,
            'sum': k.sum(),
            'max': k.max(),
            'std': k.std(),
            'center_offset': center_offset
        })
    
    stats = sorted(stats, key=lambda x: x['iter'])
    
    # 绘制统计图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    iters = [s['iter'] for s in stats]
    
    # 总和
    axes[0, 0].plot(iters, [s['sum'] for s in stats], 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Iteration', fontsize=12)
    axes[0, 0].set_ylabel('Kernel Sum', fontsize=12)
    axes[0, 0].set_title('Kernel Sum (should → 1.0)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=1.0, color='r', linestyle='--', label='Target')
    axes[0, 0].legend()
    
    # 最大值
    axes[0, 1].plot(iters, [s['max'] for s in stats], 'o-', linewidth=2, markersize=8, color='orange')
    axes[0, 1].set_xlabel('Iteration', fontsize=12)
    axes[0, 1].set_ylabel('Max Value', fontsize=12)
    axes[0, 1].set_title('Kernel Max Value', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 标准差
    axes[1, 0].plot(iters, [s['std'] for s in stats], 'o-', linewidth=2, markersize=8, color='green')
    axes[1, 0].set_xlabel('Iteration', fontsize=12)
    axes[1, 0].set_ylabel('Std Dev', fontsize=12)
    axes[1, 0].set_title('Kernel Standard Deviation', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 质心偏移
    axes[1, 1].plot(iters, [s['center_offset'] for s in stats], 'o-', linewidth=2, markersize=8, color='purple')
    axes[1, 1].set_xlabel('Iteration', fontsize=12)
    axes[1, 1].set_ylabel('Center Offset (pixels)', fontsize=12)
    axes[1, 1].set_title('Kernel Center Offset', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Kernel Statistics Evolution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(kernel_dir, 'kernel_stats.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"统计图已保存至: {output_path}")
    
    plt.show()


def compare_band_kernels(kernels_all_bands, band_names, kernel_dir, title_suffix=''):
    """
    比较不同波段核的统计特性
    
    参数:
        kernels_all_bands (np.ndarray): [n_bands, H, W] 所有波段的核
        band_names (list): 波段名称列表
        kernel_dir (str): 输出目录
        title_suffix (str): 标题后缀
    """
    n_bands = kernels_all_bands.shape[0]
    
    # 计算每个波段的统计信息
    stats = []
    for i in range(n_bands):
        k = kernels_all_bands[i]
        h, w = k.shape
        
        # 计算质心
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        total_mass = k.sum()
        if total_mass > 0:
            center_y = (y_coords * k).sum() / total_mass
            center_x = (x_coords * k).sum() / total_mass
            center_offset = np.sqrt((center_y - (h-1)/2)**2 + (center_x - (w-1)/2)**2)
        else:
            center_offset = 0
        
        stats.append({
            'band': band_names[i],
            'sum': k.sum(),
            'max': k.max(),
            'std': k.std(),
            'center_offset': center_offset
        })
    
    # 绘制对比图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    bands = [s['band'] for s in stats]
    x_pos = np.arange(len(bands))
    
    # 总和
    axes[0, 0].bar(x_pos, [s['sum'] for s in stats], color='steelblue', alpha=0.8)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(bands, rotation=45)
    axes[0, 0].set_ylabel('Kernel Sum', fontsize=12)
    axes[0, 0].set_title('Kernel Sum by Band', fontsize=12, fontweight='bold')
    axes[0, 0].axhline(y=1.0, color='r', linestyle='--', label='Target', linewidth=2)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].legend()
    
    # 最大值
    axes[0, 1].bar(x_pos, [s['max'] for s in stats], color='orange', alpha=0.8)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(bands, rotation=45)
    axes[0, 1].set_ylabel('Max Value', fontsize=12)
    axes[0, 1].set_title('Kernel Max Value by Band', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 标准差
    axes[1, 0].bar(x_pos, [s['std'] for s in stats], color='green', alpha=0.8)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(bands, rotation=45)
    axes[1, 0].set_ylabel('Std Dev', fontsize=12)
    axes[1, 0].set_title('Kernel Standard Deviation by Band', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 质心偏移
    axes[1, 1].bar(x_pos, [s['center_offset'] for s in stats], color='purple', alpha=0.8)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(bands, rotation=45)
    axes[1, 1].set_ylabel('Center Offset (pixels)', fontsize=12)
    axes[1, 1].set_title('Kernel Center Offset by Band', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    title = 'Kernel Statistics Comparison Across Bands'
    if title_suffix:
        title += f' - {title_suffix}'
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_name = 'kernels_band_comparison.png'
    if title_suffix and 'Iter' in title_suffix:
        import re
        match = re.search(r'(\d+)', title_suffix)
        if match:
            output_name = f'kernels_band_comparison_iter{match.group(1)}.png'
    
    output_path = os.path.join(kernel_dir, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"波段对比图已保存至: {output_path}")
    
    plt.show()


def main():
    # 设置核文件夹路径
    kernel_dir = 'kernelgan_out'
    
    print("="*60)
    print("Kernel Visualization Tool")
    print("="*60)
    
    # 检查是否有分波段核文件（最终版本）
    per_band_file = os.path.join(kernel_dir, 'kernel_per_band.npy')
    if os.path.exists(per_band_file):
        print("\n找到最终分波段核文件...")
        visualize_per_band_kernels(per_band_file, kernel_dir, title_suffix='Final')
    else:
        print("\n未找到 kernel_per_band.npy（训练可能未完成）")
    
    # 检查是否有中间保存的分波段核文件
    per_band_iter_files = sorted(glob.glob(os.path.join(kernel_dir, 'kernel_per_band_iter*.npy')))
    if len(per_band_iter_files) > 0:
        print(f"\n找到 {len(per_band_iter_files)} 个中间分波段核文件")
        # 显示最新的中间核
        latest_per_band = per_band_iter_files[-1]
        print(f"可视化最新的分波段核: {os.path.basename(latest_per_band)}")
        import re
        match = re.search(r'iter(\d+)', os.path.basename(latest_per_band))
        iter_num = match.group(1) if match else 'Unknown'
        visualize_per_band_kernels(latest_per_band, kernel_dir, title_suffix=f'Iter {iter_num}')
    
    # 可视化所有合并核
    print("\n" + "="*60)
    print("可视化中间保存的合并核")
    print("="*60)
    visualize_kernels(kernel_dir=kernel_dir, dpi=300)
    
    # 比较核的统计特性
    print("\n" + "="*60)
    print("Kernel Statistics Comparison")
    print("="*60)
    compare_kernels_stats(kernel_dir=kernel_dir)


def visualize_per_band_kernels(per_band_file, kernel_dir, title_suffix=''):
    """
    可视化每个波段的核
    
    参数:
        per_band_file (str): kernel_per_band.npy文件路径
        kernel_dir (str): 输出目录
        title_suffix (str): 标题后缀（如 'Final' 或 'Iter 300'）
    """
    # 加载分波段核 [5, H, W]
    kernels_all_bands = np.load(per_band_file)
    n_bands = kernels_all_bands.shape[0]
    
    print(f"分波段核形状: {kernels_all_bands.shape}")
    print(f"波段数量: {n_bands}")
    
    band_names = ['443nm', '490nm', '555nm', '660nm', '865nm']
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 找到全局最大值用于统一颜色范围
    vmax = kernels_all_bands.max()
    
    # 绘制每个波段的核
    for i in range(n_bands):
        ax = axes[i]
        kernel = kernels_all_bands[i]
        
        im = ax.imshow(kernel, cmap='hot', interpolation='nearest', vmin=0, vmax=vmax)
        ax.set_title(f'Band {i}: {band_names[i]}\nSum={kernel.sum():.4f}, Max={kernel.max():.4f}', 
                    fontsize=12, fontweight='bold')
        ax.axis('off')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Value', fontsize=8)
        
        # 在核上标注数值（如果核不太大）
        if kernel.shape[0] <= 13 and kernel.shape[1] <= 13:
            for ii in range(kernel.shape[0]):
                for jj in range(kernel.shape[1]):
                    text_color = 'white' if kernel[ii, jj] > vmax * 0.5 else 'black'
                    ax.text(jj, ii, f'{kernel[ii, jj]:.3f}', 
                           ha='center', va='center', 
                           fontsize=5, color=text_color)
    
    # 绘制平均核
    ax = axes[5]
    kernel_mean = kernels_all_bands.mean(axis=0)
    im = ax.imshow(kernel_mean, cmap='hot', interpolation='nearest', vmin=0, vmax=vmax)
    ax.set_title(f'Mean Kernel (All Bands)\nSum={kernel_mean.sum():.4f}, Max={kernel_mean.max():.4f}', 
                fontsize=12, fontweight='bold')
    ax.axis('off')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Value', fontsize=8)
    
    if kernel_mean.shape[0] <= 13 and kernel_mean.shape[1] <= 13:
        for ii in range(kernel_mean.shape[0]):
            for jj in range(kernel_mean.shape[1]):
                text_color = 'white' if kernel_mean[ii, jj] > vmax * 0.5 else 'black'
                ax.text(jj, ii, f'{kernel_mean[ii, jj]:.3f}', 
                       ha='center', va='center', 
                       fontsize=5, color=text_color)
    
    title = f'Per-Band Kernels (5 Spectral Bands)'
    if title_suffix:
        title += f' - {title_suffix}'
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_name = 'kernels_per_band.png'
    if title_suffix and 'Iter' in title_suffix:
        import re
        match = re.search(r'(\d+)', title_suffix)
        if match:
            output_name = f'kernels_per_band_iter{match.group(1)}.png'
    
    output_path = os.path.join(kernel_dir, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"分波段核可视化已保存至: {output_path}")
    
    plt.show()
    
    # 比较不同波段核的差异
    compare_band_kernels(kernels_all_bands, band_names, kernel_dir, title_suffix)


if __name__ == "__main__":
    main()

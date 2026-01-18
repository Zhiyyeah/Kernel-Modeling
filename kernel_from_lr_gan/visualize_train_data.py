"""
可视化训练数据集中的HR和LR对比
- 读取训练数据文件（包含hr和lr组）
- 随机抽取最多30个样本进行可视化
- 展示5个波段的HR (256×256) vs LR (32×32) 对比

运行示例：
    python kernel_from_lr_gan/visualize_train_data.py
    python kernel_from_lr_gan/visualize_train_data.py --max_samples 10
"""
import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from tqdm import tqdm

# 默认配置
TRAIN_DATA_DIR = r"H:\Landsat\patch_output_nc\train_data_with_noise"
OUTPUT_DIR = r"H:\Landsat\patch_output_nc\vis_train_data"
BAND_NAMES = ['L_TOA_443', 'L_TOA_490', 'L_TOA_555', 'L_TOA_660', 'L_TOA_865']
MAX_SAMPLES = 30


def load_group_bands(nc_path: str, group_name: str) -> np.ndarray:
    """加载指定组的5个波段，返回 (5, H, W)"""
    with Dataset(nc_path, 'r') as ds:
        if group_name not in ds.groups:
            raise ValueError(f"组 {group_name} 不存在于 {nc_path}")
        grp = ds.groups[group_name]
        bands = []
        for b in BAND_NAMES:
            if b not in grp.variables:
                raise ValueError(f"波段 {b} 不存在于组 {group_name}")
            arr = grp.variables[b][:]
            if isinstance(arr, np.ma.MaskedArray):
                arr = arr.filled(np.nan)
            bands.append(np.array(arr, dtype=np.float32))
        return np.stack(bands, axis=0)  # (5, H, W)


def plot_hr_lr_compare(hr: np.ndarray, lr: np.ndarray, fname: str, output_dir: str):
    """
    绘制HR vs LR对比图
    hr: (5, 256, 256)
    lr: (5, 32, 32)
    """
    n_bands = len(BAND_NAMES)
    fig, axes = plt.subplots(2, n_bands, figsize=(4*n_bands, 8))
    
    for j, band_name in enumerate(BAND_NAMES):
        hr_band = hr[j, :, :]  # (256, 256)
        lr_band = lr[j, :, :]  # (32, 32)
        
        # 计算共享的颜色范围
        vmin = np.nanmin([hr_band.min(), lr_band.min()])
        vmax = np.nanmax([hr_band.max(), lr_band.max()])
        
        # 绘制HR
        ax = axes[0, j]
        im = ax.imshow(hr_band, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f"HR {band_name}\n{hr_band.shape}")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 绘制LR
        ax = axes[1, j]
        im = ax.imshow(lr_band, cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(f"LR {band_name}\n{lr_band.shape}")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(f"Training Data: {fname}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, fname.replace('.nc', '.png'))
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_train_data(train_dir: str, output_dir: str, max_samples: int = 30, seed: int = 42):
    """可视化训练数据"""
    random.seed(seed)
    
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"训练数据目录不存在: {train_dir}")
    
    # 获取所有训练数据文件
    nc_files = [f for f in os.listdir(train_dir) if f.endswith('.nc')]
    if not nc_files:
        raise FileNotFoundError(f"目录中没有.nc文件: {train_dir}")
    
    # 随机抽样
    random.shuffle(nc_files)
    nc_files = nc_files[:max_samples]
    
    print(f"开始可视化 {len(nc_files)} 个训练样本...")
    success_count = 0
    
    for fname in tqdm(nc_files, desc="生成可视化", unit="file"):
        nc_path = os.path.join(train_dir, fname)
        
        try:
            # 加载HR和LR
            hr = load_group_bands(nc_path, 'hr')  # (5, 256, 256)
            lr = load_group_bands(nc_path, 'lr')  # (5, 32, 32)
            
            # 验证形状
            if hr.shape != (5, 256, 256):
                print(f"\n⚠️  {fname}: HR形状 {hr.shape} 异常，跳过")
                continue
            
            if lr.shape != (5, 32, 32):
                print(f"\n⚠️  {fname}: LR形状 {lr.shape} 异常，跳过")
                continue
            
            # 绘制对比图
            plot_hr_lr_compare(hr, lr, fname, output_dir)
            success_count += 1
            
        except Exception as e:
            print(f"\n❌ 处理 {fname} 失败: {e}")
            continue
    
    print(f"\n✅ 可视化完成:")
    print(f"   - 成功生成: {success_count} 张图片")
    print(f"   - 输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="可视化训练数据HR vs LR")
    parser.add_argument('--train_dir', type=str, default=TRAIN_DATA_DIR,
                        help='训练数据目录')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='可视化输出目录')
    parser.add_argument('--max_samples', type=int, default=MAX_SAMPLES,
                        help='最多可视化的样本数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    visualize_train_data(
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

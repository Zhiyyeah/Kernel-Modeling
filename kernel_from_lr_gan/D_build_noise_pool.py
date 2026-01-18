"""
从GOCI-2去噪数据中构建噪声池：
- 噪声 = geophysical_data - denoised
- 每个文件随机抽取指定数量的32×32噪声块
- 保存为 (N, 5, 32, 32) 的 .npy 文件

运行示例：
    python kernel_from_lr_gan/build_noise_pool.py
    python kernel_from_lr_gan/build_noise_pool.py --samples_per_file 3 --patch_size 64
"""
import os
import argparse
import random
import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm

# 默认配置
GOCI_DIR = r"H:\GOCI-2\patch_output_nc\patches_denoised"
OUTPUT_FILE = r"H:\GOCI-2\patch_output_nc\noise_pool/goci_noise_pool.npy"
METADATA_FILE = r"H:\GOCI-2\patch_output_nc\noise_pool/goci_noise_metadata.npy"

BAND_NAMES = ['L_TOA_443', 'L_TOA_490', 'L_TOA_555', 'L_TOA_660', 'L_TOA_865']


def load_group_bands(nc_path: str, group_name: str) -> np.ndarray:
    """加载指定组的5个波段，返回 (5, H, W)"""
    with Dataset(nc_path, 'r') as ds:
        if group_name not in ds.groups:
            raise ValueError(f"组 {group_name} 不存在于 {nc_path}")
        grp = ds.groups[group_name]
        bands = []
        for b in BAND_NAMES:
            arr = grp.variables[b][:]
            if isinstance(arr, np.ma.MaskedArray):
                arr = arr.filled(np.nan)
            bands.append(np.array(arr, dtype=np.float32))
        return np.stack(bands, axis=0)  # (5, H, W)


def random_crop(data: np.ndarray, crop_size: int, n_samples: int) -> list[np.ndarray]:
    """从 (C, H, W) 中随机裁剪 n_samples 个 (C, crop_size, crop_size) 块"""
    _, H, W = data.shape
    if H < crop_size or W < crop_size:
        raise ValueError(f"图像尺寸 {H}×{W} 小于裁剪尺寸 {crop_size}")
    
    patches = []
    for _ in range(n_samples):
        top = random.randint(0, H - crop_size)
        left = random.randint(0, W - crop_size)
        patch = data[:, top:top+crop_size, left:left+crop_size]
        patches.append(patch)
    return patches


def build_noise_pool(
    goci_dir: str,
    output_file: str,
    metadata_file: str,
    samples_per_file: int = 1,
    patch_size: int = 32,
    seed: int = 42
):
    """构建噪声池"""
    random.seed(seed)
    np.random.seed(seed)
    
    if not os.path.isdir(goci_dir):
        raise FileNotFoundError(f"GOCI目录不存在: {goci_dir}")
    
    nc_files = [f for f in os.listdir(goci_dir) if f.endswith('.nc')]
    if not nc_files:
        raise FileNotFoundError(f"目录中没有.nc文件: {goci_dir}")
    
    all_noise_patches = []
    metadata = []  # 记录来源文件和索引
    
    print(f"开始处理 {len(nc_files)} 个文件，每文件采样 {samples_per_file} 个 {patch_size}×{patch_size} 噪声块...")
    
    for fname in tqdm(nc_files, desc="提取噪声", unit="file"):
        nc_path = os.path.join(goci_dir, fname)
        try:
            # 加载原始和去噪数据
            geo_data = load_group_bands(nc_path, 'geophysical_data')  # (5, H, W)
            denoised_data = load_group_bands(nc_path, 'denoised')      # (5, H, W)
            
            # 计算噪声
            noise = geo_data - denoised_data  # (5, H, W)
            
            # 随机裁剪
            noise_patches = random_crop(noise, patch_size, samples_per_file)
            all_noise_patches.extend(noise_patches)
            
            # 记录元数据
            for i in range(samples_per_file):
                metadata.append({
                    'source_file': fname,
                    'patch_id': i,
                    'patch_size': patch_size
                })
        
        except Exception as e:
            print(f"\n⚠️  处理 {fname} 失败: {e}")
            continue
    
    if not all_noise_patches:
        raise RuntimeError("未成功提取任何噪声块")
    
    # 转换为numpy数组 (N, 5, patch_size, patch_size)
    noise_pool = np.stack(all_noise_patches, axis=0)
    
    # 保存
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, noise_pool)
    np.save(metadata_file, metadata)
    
    # 统计信息
    print(f"\n✅ 噪声池构建完成:")
    print(f"   - 总样本数: {noise_pool.shape[0]}")
    print(f"   - 数组形状: {noise_pool.shape}")
    print(f"   - 文件大小: {os.path.getsize(output_file) / 1024**2:.2f} MB")
    print(f"   - 保存路径: {output_file}")
    print(f"   - 元数据:   {metadata_file}")
    
    # 统计每个波段的噪声特性
    print(f"\n📊 噪声统计 (各波段):")
    for i, band in enumerate(BAND_NAMES):
        band_noise = noise_pool[:, i, :, :]
        print(f"   {band:12s}: mean={np.nanmean(band_noise):+.6f}, "
              f"std={np.nanstd(band_noise):.6f}, "
              f"min={np.nanmin(band_noise):+.6f}, "
              f"max={np.nanmax(band_noise):+.6f}")


def main():
    parser = argparse.ArgumentParser(description="构建GOCI-2噪声池")
    parser.add_argument('--goci_dir', type=str, default=GOCI_DIR,
                        help='GOCI去噪数据目录')
    parser.add_argument('--output_file', type=str, default=OUTPUT_FILE,
                        help='输出.npy文件路径')
    parser.add_argument('--metadata_file', type=str, default=METADATA_FILE,
                        help='元数据.npy文件路径')
    parser.add_argument('--samples_per_file', type=int, default=1,
                        help='每个文件采样的噪声块数量')
    parser.add_argument('--patch_size', type=int, default=32,
                        help='噪声块大小（默认32×32）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    build_noise_pool(
        goci_dir=args.goci_dir,
        output_file=args.output_file,
        metadata_file=args.metadata_file,
        samples_per_file=args.samples_per_file,
        patch_size=args.patch_size,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

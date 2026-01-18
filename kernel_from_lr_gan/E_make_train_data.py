"""
从 Landsat 模糊数据 + GOCI 噪声池构建训练数据集
- 输入: H:\Landsat\patch_output_nc\patches_denoised_blurred (包含 denoised, blurred 组)
- 噪声池: H:\GOCI-2\patch_output_nc\noise_pool\goci_noise_pool.npy
- 输出结构:
  * hr: denoised 组 (5, 256, 256)
  * lr: blurred + 随机噪声 (5, 32, 32)
  * navigation_data: 经纬度信息

运行示例:
    python kernel_from_lr_gan/E_make_train_data.py
    python kernel_from_lr_gan/E_make_train_data.py --output_dir "D:/train_data"
"""
import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from tqdm import tqdm

# 默认配置
INPUT_DIR = r"H:\Landsat\patch_output_nc\patches_denoised_blurred"
NOISE_POOL_PATH = r"H:\GOCI-2\patch_output_nc\noise_pool\goci_noise_pool.npy"
OUTPUT_DIR = r"H:\Landsat\patch_output_nc\train_data_with_noise"
VIS_DIR = r"H:\Landsat\patch_output_nc\vis_train_generation"

BAND_NAMES = ['L_TOA_443', 'L_TOA_490', 'L_TOA_555', 'L_TOA_660', 'L_TOA_865']
MAX_VIS_SAMPLES = 30


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


def load_navigation_data(nc_path: str) -> dict:
    """加载导航数据"""
    with Dataset(nc_path, 'r') as ds:
        if 'navigation_data' not in ds.groups:
            raise ValueError(f"navigation_data 组不存在于 {nc_path}")
        nav_grp = ds.groups['navigation_data']
        nav_data = {}
        for var_name in ['latitude', 'longitude']:
            if var_name in nav_grp.variables:
                arr = nav_grp.variables[var_name][:]
                if isinstance(arr, np.ma.MaskedArray):
                    arr = arr.filled(np.nan)
                nav_data[var_name] = np.array(arr, dtype=np.float32)
        return nav_data


def add_noise(blurred: np.ndarray, noise_pool: np.ndarray) -> np.ndarray:
    """
    为模糊图像添加随机噪声
    blurred: (5, 32, 32)
    noise_pool: (N, 5, 32, 32)
    返回: (5, 32, 32)
    """
    idx = np.random.randint(0, len(noise_pool))
    noise = noise_pool[idx]  # (5, 32, 32)
    return blurred + noise


def save_training_sample(output_path: str, hr: np.ndarray, lr: np.ndarray, nav_data: dict):
    """
    保存训练样本到NetCDF文件
    hr: (5, 256, 256)
    lr: (5, 32, 32)
    nav_data: {'latitude': array, 'longitude': array}
    """
    with Dataset(output_path, 'w', format='NETCDF4') as ds:
        # 创建 hr 组
        hr_grp = ds.createGroup('hr')
        hr_grp.createDimension('band', 5)
        hr_grp.createDimension('y', hr.shape[1])
        hr_grp.createDimension('x', hr.shape[2])
        
        for i, band_name in enumerate(BAND_NAMES):
            var = hr_grp.createVariable(band_name, 'f4', ('y', 'x'), zlib=True, complevel=4)
            var[:] = hr[i, :, :]
        
        # 创建 lr 组
        lr_grp = ds.createGroup('lr')
        lr_grp.createDimension('band', 5)
        lr_grp.createDimension('y', lr.shape[1])
        lr_grp.createDimension('x', lr.shape[2])
        
        for i, band_name in enumerate(BAND_NAMES):
            var = lr_grp.createVariable(band_name, 'f4', ('y', 'x'), zlib=True, complevel=4)
            var[:] = lr[i, :, :]
        
        # 创建 navigation_data 组
        if nav_data:
            nav_grp = ds.createGroup('navigation_data')
            for key, value in nav_data.items():
                if value is not None and value.size > 0:
                    dims = []
                    for j, dim_size in enumerate(value.shape):
                        dim_name = f'{key}_dim_{j}'
                        if dim_name not in nav_grp.dimensions:
                            nav_grp.createDimension(dim_name, dim_size)
                        dims.append(dim_name)
                    var = nav_grp.createVariable(key, 'f4', tuple(dims), zlib=True, complevel=4)
                    var[:] = value


def plot_comparison(hr: np.ndarray, blurred: np.ndarray, lr_noisy: np.ndarray, fname: str, vis_dir: str):
    """
    绘制HR、Blurred、Noise、Blurred+Noise四者对比图
    hr: (5, 256, 256)
    blurred: (5, 32, 32)
    lr_noisy: (5, 32, 32)
    """
    n_bands = len(BAND_NAMES)
    fig, axes = plt.subplots(4, n_bands, figsize=(4*n_bands, 16))
    
    # 计算噪声
    noise = lr_noisy - blurred  # (5, 32, 32)
    
    for j, band_name in enumerate(BAND_NAMES):
        hr_band = hr[j, :, :]          # (256, 256)
        blur_band = blurred[j, :, :]   # (32, 32)
        noise_band = noise[j, :, :]    # (32, 32)
        noisy_band = lr_noisy[j, :, :] # (32, 32)
        
        # HR使用其自己的颜色范围
        hr_vmin, hr_vmax = hr_band.min(), hr_band.max()
        
        # LR相关的使用共享颜色范围
        lr_vmin = np.nanmin([blur_band.min(), noisy_band.min()])
        lr_vmax = np.nanmax([blur_band.max(), noisy_band.max()])
        
        # 噪声使用对称颜色范围（以0为中心）
        noise_abs_max = np.nanmax(np.abs(noise_band))
        noise_vmin, noise_vmax = -noise_abs_max, noise_abs_max
        
        # 绘制HR (denoised)
        ax = axes[0, j]
        im = ax.imshow(hr_band, cmap='viridis', vmin=hr_vmin, vmax=hr_vmax)
        ax.set_title(f"HR (denoised)\n{band_name}\n{hr_band.shape}", fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 绘制Blurred (无噪声)
        ax = axes[1, j]
        im = ax.imshow(blur_band, cmap='viridis', vmin=lr_vmin, vmax=lr_vmax, interpolation='nearest')
        ax.set_title(f"LR (blurred)\n{band_name}\n{blur_band.shape}", fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 绘制Noise
        ax = axes[2, j]
        im = ax.imshow(noise_band, cmap='RdBu_r', vmin=noise_vmin, vmax=noise_vmax, interpolation='nearest')
        ax.set_title(f"Noise\n{band_name}\n{noise_band.shape}", fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 绘制Blurred + Noise
        ax = axes[3, j]
        im = ax.imshow(noisy_band, cmap='viridis', vmin=lr_vmin, vmax=lr_vmax, interpolation='nearest')
        ax.set_title(f"LR (blurred+noise)\n{band_name}\n{noisy_band.shape}", fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(f"Training Data Generation: {fname}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(vis_dir, exist_ok=True)
    out_path = os.path.join(vis_dir, fname.replace('.nc', '.png'))
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_files(input_dir: str, noise_pool_path: str, output_dir: str, vis_dir: str = None, 
                  max_vis: int = 30, seed: int = 42):
    """处理所有文件"""
    np.random.seed(seed)
    
    # 检查输入目录
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    # 加载噪声池
    if not os.path.isfile(noise_pool_path):
        raise FileNotFoundError(f"噪声池文件不存在: {noise_pool_path}")
    
    print(f"加载噪声池: {noise_pool_path}")
    noise_pool = np.load(noise_pool_path)  # (N, 5, 32, 32)
    print(f"噪声池形状: {noise_pool.shape}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有输入文件
    nc_files = [f for f in os.listdir(input_dir) if f.endswith('.nc')]
    if not nc_files:
        raise FileNotFoundError(f"输入目录中没有.nc文件: {input_dir}")
    
    print(f"开始处理 {len(nc_files)} 个文件...")
    success_count = 0
    fail_count = 0
    
    # 随机选择要可视化的文件
    vis_files = set()
    if vis_dir and max_vis > 0:
        random.seed(seed)
        vis_files = set(random.sample(nc_files, min(max_vis, len(nc_files))))
        print(f"将为 {len(vis_files)} 个样本生成可视化...")
    
    for fname in tqdm(nc_files, desc="生成训练数据", unit="file"):
        input_path = os.path.join(input_dir, fname)
        
        # 输出文件名: 移除 _blurred 后缀，添加 _train
        base_name = fname.replace('_denoised_blurred.nc', '_train.nc')
        if base_name == fname:  # 如果没有替换成功，用其他方式
            base_name = fname.replace('.nc', '_train.nc')
        output_path = os.path.join(output_dir, base_name)
        
        try:
            # 加载数据
            hr = load_group_bands(input_path, 'denoised')    # (5, 256, 256)
            blurred = load_group_bands(input_path, 'blurred') # (5, 32, 32)
            nav_data = load_navigation_data(input_path)
            
            # 验证形状
            if hr.shape[1] != 256 or hr.shape[2] != 256:
                print(f"\n⚠️  {fname}: HR形状 {hr.shape} 不是 (5,256,256)，跳过")
                fail_count += 1
                continue
            
            if blurred.shape[1] != 32 or blurred.shape[2] != 32:
                print(f"\n⚠️  {fname}: Blurred形状 {blurred.shape} 不是 (5,32,32)，跳过")
                fail_count += 1
                continue
            
            # 添加噪声
            lr = add_noise(blurred, noise_pool)  # (5, 32, 32)
            
            # 保存
            save_training_sample(output_path, hr, lr, nav_data)
            
            # 可视化（如果在选中列表中）
            if fname in vis_files:
                try:
                    plot_comparison(hr, blurred, lr, fname, vis_dir)
                except Exception as vis_err:
                    print(f"\n⚠️  可视化 {fname} 失败: {vis_err}")
            
            success_count += 1
            
        except Exception as e:
            print(f"\n❌ 处理 {fname} 失败: {e}")
            fail_count += 1
            continue
    
    print(f"\n✅ 处理完成:")
    print(f"   - 成功: {success_count}")
    print(f"   - 失败: {fail_count}")
    print(f"   - 输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="构建带噪声的训练数据集")
    parser.add_argument('--input_dir', type=str, default=INPUT_DIR,
                        help='Landsat模糊数据目录')
    parser.add_argument('--noise_pool', type=str, default=NOISE_POOL_PATH,
                        help='GOCI噪声池.npy文件路径')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='输出训练数据目录')
    parser.add_argument('--vis_dir', type=str, default=VIS_DIR,
                        help='可视化输出目录')
    parser.add_argument('--max_vis', type=int, default=MAX_VIS_SAMPLES,
                        help='最多可视化的样本数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    process_files(
        input_dir=args.input_dir,
        noise_pool_path=args.noise_pool,
        output_dir=args.output_dir,
        vis_dir=args.vis_dir,
        max_vis=args.max_vis,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

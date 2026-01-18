"""
将训练好的模糊核应用于Landsat数据进行降分辨率
- 输入: Landsat NC文件 (任意分辨率，如128x128或256x256)
- 输出: 降分辨率后的影像 (输入分辨率 / 下采样倍数，默认8倍)
  例如: 128x128 -> 16x16 (8倍)，或 256x256 -> 32x32 (8倍)
"""
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil


def load_kernel(kernel_path):
    """
    加载训练好的模糊核
    参数:
        kernel_path (str): 核文件路径 (.npy)
    返回:
        torch.Tensor: 模糊核，形状 [n_bands, kH, kW] 或 [kH, kW]
    """
    kernel = np.load(kernel_path)
    kernel_tensor = torch.from_numpy(kernel.astype(np.float32))
    
    print(f"加载模糊核: {os.path.basename(kernel_path)}")
    print(f"  形状: {kernel_tensor.shape}")
    print(f"  总和: {kernel_tensor.sum().item():.6f}")
    
    return kernel_tensor


def load_landsat_nc(nc_path):
    """
    读取Landsat NC文件
    参数:
        nc_path (str): NC文件路径
    返回:
        torch.Tensor: 影像数据 [n_bands, H, W]
        list: 波段名称列表
    """
    # 读取NC文件
    with Dataset(nc_path, 'r') as ds:
        # 固定读取 hr 组及其规定的波段
        grp = ds.groups['denoised']
        band_names = ['L_TOA_443', 'L_TOA_490', 'L_TOA_555', 'L_TOA_660', 'L_TOA_865']
        all_bands_data = []

        for band_name in band_names:
            data = grp.variables[band_name][:]
            if isinstance(data, np.ma.MaskedArray):
                data = data.filled(fill_value=np.nan)
            all_bands_data.append(data.astype(np.float32))

        # 堆叠成 [C, H, W] 的数组
        img_array = np.stack(all_bands_data, axis=0)
        img_tensor = torch.from_numpy(img_array)
    
    print(f"影像形状: {img_tensor.shape}")
    print(f"数值范围: [{img_tensor.min().item():.2f}, {img_tensor.max().item():.2f}]")
    
    return img_tensor, band_names


def apply_kernel_degradation(img, kernel, downscale_factor: int = 8):
    """
    使用模糊核对影像进行模糊，并下采样（默认 8 倍，256->32）。
    
    参数:
        img (torch.Tensor): 输入影像 [C, H, W]
        kernel (torch.Tensor): 模糊核 [C, kH, kW] 或 [kH, kW]
        downscale_factor (int): 下采样倍数，默认8。
    
    返回:
        torch.Tensor: 模糊并下采样后的影像 [C, H/downscale_factor, W/downscale_factor]
    """
    C, H, W = img.shape
    
    # 确保核的维度正确
    if kernel.ndim == 2:
        # 单核，复制到所有通道
        kernel = kernel.unsqueeze(0).repeat(C, 1, 1)  # [C, kH, kW]
    elif kernel.ndim == 3:
        # 多波段核
        assert kernel.shape[0] == C, f"核的波段数({kernel.shape[0]})与影像波段数({C})不匹配"
    
    kH, kW = kernel.shape[-2:]
    
    # 归一化核（确保每个波段的核总和为1）
    kernel_normalized = kernel.clone()
    for i in range(C):
        kernel_sum = kernel[i].sum()
        if kernel_sum > 0:
            kernel_normalized[i] = kernel[i] / kernel_sum
    
    # 重塑核为卷积权重格式 [out_channels, in_channels, kH, kW]
    # 对于逐通道卷积，out_channels = in_channels = C, groups = C
    conv_kernel = kernel_normalized.unsqueeze(1)  # [C, 1, kH, kW]
    
    # 添加batch维度
    img_batch = img.unsqueeze(0)  # [1, C, H, W]
    
    # 使用复制填充避免零填充造成的边框伪影
    pad_h = kH // 2
    pad_w = kW // 2
    img_batch_padded = F.pad(img_batch, (pad_w, pad_w, pad_h, pad_h), mode='replicate')
    
    # 应用模糊核（分组卷积，每个通道独立），卷积本身不再额外padding
    blurred = F.conv2d(
        input=img_batch_padded,
        weight=conv_kernel,
        padding=0,
        groups=C
    )  # [1, C, H, W]

    # 下采样：连续 2x 平均池化实现 8x 缩放
    lr_img = blurred
    for _ in range(int(np.log2(downscale_factor))):
        lr_img = F.avg_pool2d(lr_img, kernel_size=2, stride=2)

    return lr_img.squeeze(0)  # [C, H/downscale_factor, W/downscale_factor]


def process_landsat_folder(landsat_dir, kernel_path, output_dir):
    """
    批量处理Landsat文件夹中的所有NC文件
    
    参数:
        landsat_dir (str): Landsat NC文件夹路径
        kernel_path (str): 模糊核文件路径
        output_dir (str): 输出文件夹路径
    """
    # 加载模糊核
    kernel = load_kernel(kernel_path)
    
    # 查找所有NC文件
    nc_files = sorted(glob.glob(os.path.join(landsat_dir, '*.nc')))
    
    if len(nc_files) == 0:
        print(f"在 {landsat_dir} 中没有找到 .nc 文件")
        return
    
    print(f"\n找到 {len(nc_files)} 个Landsat NC文件")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个文件
    for idx, nc_file in enumerate(tqdm(nc_files, desc='Applying kernel', unit='file')):
        print(f"\n{'='*60}")
        print(f"处理 [{idx+1}/{len(nc_files)}]: {os.path.basename(nc_file)}")
        print('='*60)
        
        try:
            # 读取Landsat数据
            img, band_names = load_landsat_nc(nc_file)
            
            # 应用模糊核（不再下采样），仅保存模糊结果
            blurred_img = apply_kernel_degradation(img, kernel, downscale_factor=8)
            
            print(f"模糊+下采样结果: {img.shape} -> {blurred_img.shape}")
            
            # 构造输出文件名：原始文件名 + _blurred.nc
            base_name = os.path.splitext(os.path.basename(nc_file))[0]
            new_nc_path = os.path.join(output_dir, f"{base_name}_blurred.nc")
            
            # 复制原始文件到新路径后再写入（避免覆盖原文件）
            shutil.copy(nc_file, new_nc_path)
            
            # 打开新NC文件进行编辑
            with Dataset(new_nc_path, 'a', format='NETCDF4') as ds:
                # -------- 保存模糊后的影像到 blurred 组 --------
                # 创建 blurred 维度（下采样后尺寸，如 32x32）
                if 'y_blurred' not in ds.dimensions:
                    ds.createDimension('y_blurred', blurred_img.shape[1])
                if 'x_blurred' not in ds.dimensions:
                    ds.createDimension('x_blurred', blurred_img.shape[2])

                # 创建或获取 blurred 组
                grp_blur = ds.groups['blurred'] if 'blurred' in ds.groups else ds.createGroup('blurred')

                blurred_data = blurred_img.numpy()
                for c, band_name in enumerate(band_names[:blurred_img.shape[0]]):
                    if band_name not in grp_blur.variables:
                        var = grp_blur.createVariable(band_name, 'f4', ('y_blurred', 'x_blurred'), zlib=True)
                    else:
                        var = grp_blur.variables[band_name]
                    var[:] = blurred_data[c, :, :]
                    var.long_name = f"Blurred TOA Radiance at {band_name.split('_')[-1]} nm"
                    var.units = "W m-2 sr-1 um-1"

                # 添加或更新全局属性
                ds.history = f"Original HR patch with added blurred group (applied blur kernel, 8x downsampled)"
            
            print(f"已保存: {new_nc_path}")
            
            # 可视化对比（可选）：HR vs Blurred (downsampled)
            if idx < 30:  # 只可视化前30个
                visualize_comparison(img, blurred_img, band_names, output_dir, 
                                    os.path.basename(nc_file).replace('.nc', ''))
        
        except Exception as e:
            print(f"处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"处理完成！结果保存在: {output_dir}")
    print('='*60)


def visualize_comparison(hr_img, blurred_img, band_names, output_dir, filename_prefix):
    """
    可视化高分辨率与模糊后(下采样)影像的对比
    
    参数:
        hr_img (torch.Tensor): 高分辨率影像 [C, H, W]
        blurred_img (torch.Tensor): 模糊后影像（已下采样） [C, H_d, W_d]
        band_names (list): 波段名称
        output_dir (str): 输出目录
        filename_prefix (str): 文件名前缀
    """
    C = hr_img.shape[0]
    n_show = min(C, 5)  # 最多显示5个波段
    
    fig, axes = plt.subplots(2, n_show, figsize=(5*n_show, 10))
    if n_show == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(n_show):
        # 计算该波段的全局范围（HR和模糊的联合范围）
        hr_band = hr_img[i].numpy()
        blur_band = blurred_img[i].numpy()
        vmin = min(hr_band.min(), blur_band.min())
        vmax = max(hr_band.max(), blur_band.max())
        
        # 高分辨率
        ax = axes[0, i]
        im = ax.imshow(hr_band, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_title(f'HR - {band_names[i]}\n{hr_band.shape}', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 模糊后
        ax = axes[1, i]
        im = ax.imshow(blur_band, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_title(f'Blurred - {band_names[i]}\n{blur_band.shape}', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle(f'{filename_prefix} - HR vs Blurred (downsampled)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{filename_prefix}_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"对比图已保存: {output_path}")


def main():
    # ========== 配置参数 ==========
    
    # Landsat数据文件夹（NC文件）
    landsat_dir = r"H:\Landsat\patch_output_nc\patches_denoised"  # NC文件目录
    
    # 模糊核文件路径
    kernel_path = r"output\kernelgan_out_denoised_single_kernel\kernel_per_band_iter2400.npy"  # 或其他核文件
    
    # 输出文件夹
    output_dir = r"H:\Landsat\patch_output_nc\patches_denoised_blurred"
    
    # Landsat波段名称（自动生成，无需手动指定）
    
    # ========== 执行处理 ==========
    
    print("="*60)
    print("Landsat降分辨率处理工具")
    print("="*60)
    print(f"Landsat文件夹: {landsat_dir}")
    print(f"模糊核: {kernel_path}")
    print(f"输出目录: {output_dir}")
    print("="*60)
    
    # 检查核文件是否存在
    if not os.path.exists(kernel_path):
        print(f"\n错误: 核文件不存在: {kernel_path}")
        print("请先训练模型生成模糊核，或指定正确的核文件路径")
        return
    
    # 检查Landsat文件夹是否存在
    if not os.path.exists(landsat_dir):
        print(f"\n错误: Landsat文件夹不存在: {landsat_dir}")
        print("请修改 landsat_dir 变量为实际的Landsat数据路径")
        return
    
    # 处理文件夹
    process_landsat_folder(landsat_dir, kernel_path, output_dir)


if __name__ == "__main__":
    main()

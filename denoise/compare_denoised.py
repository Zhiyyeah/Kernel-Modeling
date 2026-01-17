"""
去噪前后对比可视化工具
对比原始波段和去噪后波段的效果，并显示残差
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path


def compare_denoised(file_path, band_index=0, save_fig=False, output_dir=None):
    """
    对比去噪前后的波段效果
    
    参数:
        file_path: 去噪后的nc文件路径（包含denoised组）
        band_index: 波段索引 (0-4, 默认0表示第一个波段)
        save_fig: 是否保存图像
        output_dir: 图像保存目录
    """
    file_path = Path(file_path)
    
    # 波段名称列表
    band_names = ['L_TOA_443', 'L_TOA_490', 'L_TOA_555', 'L_TOA_660', 'L_TOA_865']
    
    if band_index < 0 or band_index >= len(band_names):
        raise ValueError(f"波段索引必须在 0-{len(band_names)-1} 之间")
    
    band_name = band_names[band_index]
    
    print(f"文件: {file_path.name}")
    print(f"对比波段: {band_name} (索引 {band_index})")
    
    try:
        # 读取原始数据 (geophysical_data 组)
        print("\n读取原始数据...")
        ds_original = xr.open_dataset(file_path, group='geophysical_data')
        original_band = ds_original[band_name].values
        
        # 读取去噪后数据 (denoised 组)
        print("读取去噪数据...")
        ds_denoised = xr.open_dataset(file_path, group='denoised')
        denoised_band = ds_denoised[band_name].values
        
        # 读取去噪参数
        h_factor = ds_denoised.attrs.get('h_factor', 'N/A')
        sigma = ds_denoised.attrs.get(f'{band_name}_sigma', 'N/A')
        h_val = ds_denoised.attrs.get(f'{band_name}_h', 'N/A')
        
        print(f"\n去噪参数:")
        print(f"  h_factor: {h_factor}")
        print(f"  sigma: {sigma:.6f}" if isinstance(sigma, (int, float)) else f"  sigma: {sigma}")
        print(f"  h: {h_val:.6f}" if isinstance(h_val, (int, float)) else f"  h: {h_val}")
        
        # 计算残差
        residual = original_band - denoised_band
        
        # 计算统计信息
        orig_mean = np.nanmean(original_band)
        orig_std = np.nanstd(original_band)
        denoised_mean = np.nanmean(denoised_band)
        denoised_std = np.nanstd(denoised_band)
        
        valid_mask = ~np.isnan(residual)
        residual_clean = residual[valid_mask]
        if len(residual_clean) > 0:
            rmse = np.sqrt(np.mean(residual_clean**2))
            residual_std = np.std(residual_clean)
        else:
            rmse = 0
            residual_std = 0
        
        print(f"\n统计信息:")
        print(f"  原始图像 - Mean: {orig_mean:.6f}, Std: {orig_std:.6f}")
        print(f"  去噪图像 - Mean: {denoised_mean:.6f}, Std: {denoised_std:.6f}")
        print(f"  残差 - RMSE: {rmse:.6f}, Std: {residual_std:.6f}")
        
        # 绘图
        print("\n生成对比图...")
        fig = plt.figure(figsize=(20, 6))
        
        # 统一色阶范围 (使用2%-98%拉伸)
        vmin = np.nanpercentile(original_band, 2)
        vmax = np.nanpercentile(original_band, 98)
        
        # 总标题
        fig.suptitle(f'compare of denoised: {band_name} | RMSE: {rmse:.6f}', 
                     fontsize=16, fontweight='bold')
        
        # 子图1: 原始图像
        ax1 = plt.subplot(1, 3, 1)
        im1 = ax1.imshow(original_band, cmap='viridis', vmin=vmin, vmax=vmax)
        ax1.set_title(f'Original Imagery \nMean: {orig_mean:.5f}, Std: {orig_std:.5f}')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        ax1.axis('off')
        
        # 子图2: 去噪图像
        ax2 = plt.subplot(1, 3, 2)
        im2 = ax2.imshow(denoised_band, cmap='viridis', vmin=vmin, vmax=vmax)
        ax2.set_title(f'Denoised Imagery \nMean: {denoised_mean:.5f}, Std: {denoised_std:.5f}')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        ax2.axis('off')
        
        # 子图3: 残差图
        ax3 = plt.subplot(1, 3, 3)
        res_limit = residual_std * 3  # 限制显示范围
        im3 = ax3.imshow(residual, cmap='coolwarm', vmin=-res_limit, vmax=res_limit)
        ax3.set_title(f'Residual (Original - Denoised)\nStd: {residual_std:.5f}')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        ax3.axis('off')
        
        plt.tight_layout()
        
        # 保存或显示
        if save_fig:
            if output_dir is None:
                output_dir = file_path.parent / "comparison_plots"
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_filename = f"{file_path.stem}_{band_name}_comparison.png"
            output_path = output_dir / output_filename
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ 图像已保存: {output_path}")
            plt.close()
        else:
            plt.show()
        
        # 关闭数据集
        ds_original.close()
        ds_denoised.close()
        
        return True
        
    except Exception as e:
        print(f"\n✗ 错误: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="去噪前后对比可视化工具")
    parser.add_argument("file_path", type=str, 
                        help="去噪后的NC文件路径（包含denoised组）")
    parser.add_argument("--band", type=int, default=0, 
                        help="波段索引 (0-4): 0=443nm, 1=490nm, 2=555nm, 3=660nm, 4=865nm (默认: 0)")
    parser.add_argument("--save", action="store_true",
                        help="保存图像而不是显示")
    parser.add_argument("--output", type=str, default=None,
                        help="图像保存目录 (默认: 文件同目录下的comparison_plots)")
    args = parser.parse_args()
    
    print("="*70)
    print("去噪前后对比工具")
    print("="*70)
    
    success = compare_denoised(
        file_path=args.file_path,
        band_index=args.band,
        save_fig=args.save,
        output_dir=args.output
    )
    
    if not success:
        exit(1)
    
    print("\n" + "="*70)
    print("完成!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

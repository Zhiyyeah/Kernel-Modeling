from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cv2  # 需要引入 opencv

def read_nc(file_path:str, group_name, band_names = ['L_TOA_443', 'L_TOA_490', 'L_TOA_555', 'L_TOA_660', 'L_TOA_865']):
    """读取NC文件"""
    data = []
    for band_name in band_names:
        # 打开数据集
        band_data = xr.open_dataset(file_path, group=group_name)[band_name]
        data.append(band_data)
    
    # 合并波段
    data_con = xr.concat(data, dim='band')
    # 将0值设为NaN (注意：去噪前我们需要处理这些NaN)
    data_con = data_con.where(data_con != 0, np.nan)
    return data_con 

def denoise_band(band_data, h_val=10):
    """
    专门针对单波段的去噪函数
    
    参数:
        band_data: xarray DataArray 或 numpy array (包含 NaN)
        h_val: 去噪强度，越大越平滑，但也越容易丢失细节。
               对于估计模糊核，建议设为 10 左右。
    """
    # 1. 转为 numpy 数组
    img = band_data.values if hasattr(band_data, 'values') else band_data
    
    # 2. 处理 NaN 值 (OpenCV 无法处理 NaN)
    # 创建一个掩膜，标记哪里是有效值
    valid_mask = ~np.isnan(img)
    # 将 NaN 替换为 0 (或者该波段的最小值/均值，防止去噪边缘出错)
    img_filled = np.nan_to_num(img, nan=0.0)
    
    # 3. 数据类型转换 (关键步骤)
    # GOCI 的 L_TOA 是浮点数。为了获得最佳去噪效果，建议先归一化到 0-255
    min_val = np.min(img_filled[valid_mask]) # 只算有效区域的最小值
    max_val = np.max(img_filled[valid_mask])
    
    # 线性拉伸到 0-255 (Uint8)
    if max_val - min_val == 0:
        return img_filled # 防止除以0
        
    img_uint8 = ((img_filled - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    # 4. 应用非局部均值去噪 (Non-local Means Denoising)
    # 这是针对单一图像最经典的去噪算法，非常适合去除高斯噪声
    img_denoised = cv2.fastNlMeansDenoising(img_uint8, None, h=h_val, templateWindowSize=7, searchWindowSize=21)
    
    # 5. (可选) 如果需要，可以把数据再转回原来的浮点范围
    # img_result = img_denoised.astype(np.float32) / 255.0 * (max_val - min_val) + min_val
    
    # 这里我们直接返回去噪后的 uint8 图像，因为 KernelGAN 等网络通常也喜欢吃 0-255 的图
    return img_denoised

def main() -> None:
    parser = argparse.ArgumentParser(description="Read a .npy file and print basic info")
    # 注意：这里你写的是 .npy file，但代码逻辑是 read_nc，如果是 nc 文件请确保路径正确
    parser.add_argument("file_path", type=str, help="Path to .nc file") 
    args = parser.parse_args()

    path = args.file_path
    print(f"Loading: {path}")
    
    # 1. 读取数据
    arr = read_nc(path, 'geophysical_data')
    
    # 打印基础信息
    bands, height, width = arr.shape
    print(f"Bands: {bands}, Height: {height}, Width: {width}")
    
    # 2. 提取第一波段
    first_band = arr[0] # 这是 L_TOA_443
    print(f"Processing First Band: {first_band.shape}")

    # 3. 执行去噪 (核心修改)
    # 这里的 h=10 是去噪强度，你可以根据上一轮我们算出的 sigma 来调整
    # 如果 sigma 很大，h 就要设大一点
    denoised_img = denoise_band(first_band, h_val=10)

    # 4. 可视化对比
    plt.figure(figsize=(12, 6))
    
    # 原图 (含 NaN, 用 xarray 的 plot 方便自动处理 NaN)
    plt.subplot(1, 2, 1)
    # 为了显示正常，我们只画有效值部分，由于 first_band 是 xarray对象，直接用 imshow 需要处理
    plt.imshow(first_band, cmap='viridis') 
    plt.title("Original (Raw GOCI-2)")
    plt.colorbar()
    
    # 去噪图
    plt.subplot(1, 2, 2)
    plt.imshow(denoised_img, cmap='viridis')
    plt.title("Denoised (NLM, h=10)")
    plt.colorbar()
    
    plt.tight_layout()
    
    # 创建保存目录
    save_path = Path("denoise/denoise_result")
    save_path.mkdir(exist_ok=True)
    
    plt.savefig(save_path / "denoise_comparison.png", dpi=300)
    print(f"Result saved to {save_path / 'denoise_comparison.png'}")
    
    # 5. 保存去噪后的数据供后续使用 (例如给 KernelGAN)
    # np.save(save_path / "clean_img_for_kernel.npy", denoised_img)

if __name__ == "__main__":
    main()
import numpy as np
import glob
import os

def analyze_radiance_stats(patch_dir, num_samples=100):
    """
    分析辐亮度 Patch 的统计特性，用于确定噪声正则项的 target。
    """
    # 获取所有 .npy 文件
    patch_files = sorted(glob.glob(os.path.join(patch_dir, '*.npy')))
    
    if len(patch_files) == 0:
        print(f"错误: 在路径 {patch_dir} 下未找到 .npy 文件。")
        return

    # 随机选择一部分样本进行分析（或者分析全部）
    num_samples = min(num_samples, len(patch_files))
    selected_files = patch_files[:num_samples]
    
    print(f"正在分析 {num_samples} 个 patch 文件...")

    channel_means = []
    channel_stds = []

    for f in selected_files:
        try:
            # 加载数据 [5, H, W]
            data = np.load(f)
            
            # 计算当前 patch 每个通道的均值和标准差
            # axis=(1,2) 表示在 H 和 W 维度上统计
            m = np.nanmean(data, axis=(1, 2))
            s = np.nanstd(data, axis=(1, 2))
            
            channel_means.append(m)
            channel_stds.append(s)
        except Exception as e:
            print(f"跳过文件 {f}: {e}")

    # 转换为 numpy 数组进行全局汇总
    all_means = np.array(channel_means)  # [N, 5]
    all_stds = np.array(channel_stds)    # [N, 5]

    # 计算 5 个通道的全局平均统计量
    avg_mean_per_band = np.mean(all_means, axis=0)
    avg_std_per_band = np.mean(all_stds, axis=0)

    print("\n" + "="*50)
    print(f"辐亮度数据统计结果 (基于 {num_samples} 个样本):")
    print("="*50)
    print(f"{'波段':<10} | {'平均辐亮度 (Mean)':<20} | {'平均标准差 (Noise Std)':<20}")
    print("-" * 55)
    
    for i in range(5):
        print(f"Band {i:<7} | {avg_mean_per_band[i]:<20.6f} | {avg_std_per_band[i]:<20.6f}")
    
    print("-" * 55)
    
    # 给出建议
    global_avg_std = np.mean(avg_std_per_band)
    print(f"全波段平均 Std (建议的 sigma 初始值): {global_avg_std:.6f}")
    print("="*50)

if __name__ == "__main__":
    # 请确保此路径与你 train.py 中的路径一致
    # 如果你在本地运行，请修改为你的实际路径
    patch_path = '/Users/zy/Downloads/GOCI-2/patches_all256' 
    analyze_radiance_stats(patch_path)
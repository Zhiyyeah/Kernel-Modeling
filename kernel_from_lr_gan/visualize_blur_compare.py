"""
随机抽取最多30个样本，对比模糊前(denoised)与模糊后(blurred)影像，生成PNG。
- 原始NC目录: H:\Landsat\patch_output_nc\patches_denoised
- 模糊NC目录: H:\Landsat\patch_output_nc\patches_denoised_blurred
- 输出目录:    H:\Landsat\patch_output_nc\vis_blurred_compare

每个样本输出一张图片，展示5个波段（443/490/555/660/865）的 HR 与 Blurred 对比。
"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from tqdm import tqdm

# 配置路径
ORIG_DIR = r"H:\Landsat\patch_output_nc\patches_denoised"
BLUR_DIR = r"H:\Landsat\patch_output_nc\patches_denoised_blurred"
OUT_DIR = r"H:\Landsat\patch_output_nc\vis_blurred_compare"

# 展示的波段（可根据需要调整）
SHOW_BANDS = ['L_TOA_443', 'L_TOA_490', 'L_TOA_555', 'L_TOA_660', 'L_TOA_865']
MAX_SAMPLES = 30

def load_group_vars(nc_path: str, group_name: str, band_names: list[str]) -> dict[str, np.ndarray]:
    with Dataset(nc_path, 'r') as ds:
        if group_name not in ds.groups:
            raise ValueError(f"组 {group_name} 不存在于 {nc_path}")
        grp = ds.groups[group_name]
        data = {}
        for b in band_names:
            arr = grp.variables[b][:]
            if isinstance(arr, np.ma.MaskedArray):
                arr = arr.filled(np.nan)
            data[b] = np.array(arr, dtype=np.float32)
        return data

def plot_compare(orig_data: dict[str, np.ndarray], blur_data: dict[str, np.ndarray], fname: str):
    n = len(SHOW_BANDS)
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for j, band in enumerate(SHOW_BANDS):
        hr = orig_data[band]
        bl = blur_data[band]
        vmin = np.nanmin([hr.min(), bl.min()])
        vmax = np.nanmax([hr.max(), bl.max()])

        ax = axes[0, j]
        im = ax.imshow(hr, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f"HR {band}\n{hr.shape}")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1, j]
        im = ax.imshow(bl, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f"Blurred {band}\n{bl.shape}")
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(fname, fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, fname + '.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    if not os.path.isdir(ORIG_DIR):
        raise FileNotFoundError(f"原始目录不存在: {ORIG_DIR}")
    if not os.path.isdir(BLUR_DIR):
        raise FileNotFoundError(f"模糊目录不存在: {BLUR_DIR}")

    # 取所有模糊文件，尝试匹配对应原始文件（去掉 _blurred 后缀）
    blur_files = [f for f in os.listdir(BLUR_DIR) if f.endswith('.nc')]
    if not blur_files:
        raise FileNotFoundError("模糊目录中没有 .nc 文件")

    random.shuffle(blur_files)
    blur_files = blur_files[:MAX_SAMPLES]

    for fname in tqdm(blur_files, desc="Plotting", unit="file"):
        blur_path = os.path.join(BLUR_DIR, fname)
        base = fname.replace('_blurred.nc', '')
        orig_path = os.path.join(ORIG_DIR, base + '.nc')
        if not os.path.exists(orig_path):
            print(f"跳过，无匹配原始文件: {fname}")
            continue
        try:
            orig = load_group_vars(orig_path, 'denoised', SHOW_BANDS)
            blur = load_group_vars(blur_path, 'blurred', SHOW_BANDS)
            plot_compare(orig, blur, base)
        except Exception as e:
            print(f"处理 {fname} 失败: {e}")
            continue

    print(f"完成，可视化保存在: {OUT_DIR}")


if __name__ == "__main__":
    main()

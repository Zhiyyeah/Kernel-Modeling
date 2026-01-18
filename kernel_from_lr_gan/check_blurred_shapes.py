"""
检查 H:\Landsat\patch_output_nc\patches_denoised_blurred 中每个 .nc 文件的 blurred 组：
- 是否存在 blurred 组
- 是否包含 5 个波段变量
- 每个波段的形状是否为 32x32

运行：
    python kernel_from_lr_gan/check_blurred_shapes.py
或指定解释器：
    & D:/ProgramFiles/miniconda3/envs/torch_gpu/python.exe d:/Py_Code/Kernel-Modeling-Super-Resolution/kernel_from_lr_gan/check_blurred_shapes.py
"""
import os
from netCDF4 import Dataset

BLUR_DIR = r"H:\Landsat\patch_output_nc\patches_denoised_blurred"
EXPECTED_SHAPE = (32, 32)
EXPECTED_BANDS = ['L_TOA_443', 'L_TOA_490', 'L_TOA_555', 'L_TOA_660', 'L_TOA_865']


def check_file(nc_path: str) -> tuple[bool, str]:
    if not os.path.isfile(nc_path):
        return False, "不是文件"
    try:
        with Dataset(nc_path, 'r') as ds:
            if 'blurred' not in ds.groups:
                return False, "缺少 blurred 组"
            grp = ds.groups['blurred']

            bands = [b for b in grp.variables.keys() if b in grp.variables]
            # 如果存在 EXPECTED_BANDS，则按预期顺序检查，否则检查全部
            targets = EXPECTED_BANDS if EXPECTED_BANDS else bands

            missing = [b for b in targets if b not in grp.variables]
            if missing:
                return False, f"缺少波段: {missing}"

            shapes = {}
            for b in targets:
                var = grp.variables[b]
                shapes[b] = var.shape
                if var.shape != EXPECTED_SHAPE:
                    return False, f"波段 {b} 形状为 {var.shape}, 期望 {EXPECTED_SHAPE}"

            if len(targets) != 5:
                return False, f"波段数量为 {len(targets)}, 期望 5"
            return True, "OK"
    except Exception as e:  # noqa: BLE001
        return False, f"读取失败: {e}"


def main():
    if not os.path.isdir(BLUR_DIR):
        raise FileNotFoundError(f"目录不存在: {BLUR_DIR}")

    nc_files = [f for f in os.listdir(BLUR_DIR) if f.endswith('.nc')]
    if not nc_files:
        raise FileNotFoundError("目录中没有 .nc 文件")

    total = len(nc_files)
    ok = 0
    bad = []
    for fname in nc_files:
        path = os.path.join(BLUR_DIR, fname)
        good, msg = check_file(path)
        if good:
            ok += 1
        else:
            bad.append((fname, msg))

    print(f"总计: {total}, 通过: {ok}, 异常: {len(bad)}")
    if bad:
        print("异常列表:")
        for fname, msg in bad:
            print(f"  {fname}: {msg}")


if __name__ == "__main__":
    main()

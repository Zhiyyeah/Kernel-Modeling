#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Landsat 8/9 C2 L1 顶层大气：辐亮度/反射率 计算（显式 NoData 版）

核心函数：
    calc_landsat_toa(root: str, bands: list[int], mode: str = "rad") -> str
    rad_out_path = calc_landsat_toa(example_root, example_bands, mode="rad")
参数：
    - root: 影像所在目录（包含 *_MTL.txt 与各波段 *_B{n}.TIF）
    - bands: 需要处理的波段列表，例如 [1,2,3,4,5]
    - mode: "rad" 计算辐亮度，"ref" 计算反射率（默认 "rad"）
作用：
    - 读取 MTL 定标系数
    - 对给定波段计算 TOA（按模式选择），无效像元写入 -9999.0 并设置 GeoTIFF nodata
    - 输出多波段 GeoTIFF，返回并打印输出文件路径
"""

import os
import math
import glob
from pathlib import Path
from typing import Optional, Union
import numpy as np
import rasterio
from rasterio.warp import transform as rio_transform
import netCDF4 as nc


def calc_landsat_toa(
    root,
    bands,
    mode="rad",
    out_dir: Optional[Union[str, Path]] = None,
    log_csv: Optional[Union[str, Path]] = None,
):
    """计算 Landsat TOA 并输出为 NetCDF (保持原始投影/分辨率, 不做重投影)。

    流程:
      1. 读取 MTL 参数
      2. 逐波段计算辐亮度或反射率 (按 mode)
      3. 校验所有波段尺寸与仿射变换一致 (若不一致抛错)
      4. 基于第一个波段 transform 生成二维中心点投影坐标 (x,y) 网格 (原始 CRS 单位)
      5. 写出 NetCDF 文件

    注意: 不输出 latitude/longitude；若需要经纬度请后续另行重投影或坐标转换。
    """
    # 1. 常量与参数
    NODATA_VAL = -9999.0
    BAND_WAVELENGTHS = {1:443, 2:482, 3:561, 4:655, 5:865, 6:1609, 7:2200, 8:590, 9:1373, 10:10895, 11:12005}
    BAND_NAMES = {443:'L_TOA_443', 482:'L_TOA_490', 561:'L_TOA_555', 655:'L_TOA_660', 865:'L_TOA_865'}
    print(f"数据目录: {root}")
    print(f"计算模式: {mode}")

    # 2. 读取MTL参数
    mtl_path = next((os.path.join(root, fn) for fn in os.listdir(root) if fn.upper().endswith("_MTL.TXT")), None)
    print(f"_MTL.txt路径：{mtl_path}")
    if mtl_path is None:
        raise FileNotFoundError("未找到MTL文件（*_MTL.txt）")
    kv = {}
    with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if " = " in line:
                k, v = line.strip().split(" = ", 1)
                kv[k.strip()] = v.strip().strip('"')
    sun_elev = float(kv["SUN_ELEVATION"])
    product_id = kv.get("LANDSAT_PRODUCT_ID", "Landsat_C2_L1")

    # 3. 读取并计算 TOA (无重投影) —— 改为字典存储，仅保存数组数据
    #    band_data: { wavelength(int): ndarray }
    band_data = {}
    first_transform = None
    first_width = None
    first_height = None
    first_crs = None
    for b in bands:
        suffix = f"_B{b}.TIF"
        band_path = next((os.path.join(root, fn) for fn in os.listdir(root) if fn.endswith(suffix) or fn.lower().endswith(suffix.lower())), None)
        if band_path is None:
            raise FileNotFoundError(f"未找到波段 {b} 文件（*{suffix}）")
        with rasterio.open(band_path) as src:
            src_data = src.read(1)
            mask_invalid = (src_data == 0)
            if mode == "ref":
                M = float(kv[f"REFLECTANCE_MULT_BAND_{b}"])
                A = float(kv[f"REFLECTANCE_ADD_BAND_{b}"])
                rho_prime = M * src_data.astype(np.float32) + A
                sin_elev = math.sin(math.radians(sun_elev))
                if sin_elev <= 0:
                    sin_elev = 1e-6
                arr = rho_prime / sin_elev
            else:
                M = float(kv[f"RADIANCE_MULT_BAND_{b}"])
                A = float(kv[f"RADIANCE_ADD_BAND_{b}"])
                arr = M * src_data.astype(np.float32) + A
            arr = arr.astype(np.float32, copy=False)
            arr[mask_invalid] = NODATA_VAL
            print(f"波段 {b} 读取完成, 形状: {arr.shape}, 无效像元数: {np.sum(mask_invalid)}")

            transform = src.transform
            width, height = src.width, src.height
            #print(f"width: {width}, height: {height}")
            wl = BAND_WAVELENGTHS[b]
            band_data[wl] = arr
            if first_transform is None:
                first_transform = transform
                first_width = width
                first_height = height
                first_crs = src.crs

    # 4. 生成二维投影坐标网格（以第一个波段为准）
    if first_transform is None:
        raise RuntimeError("未读取到任何波段数据，band_data 为空")
    width = first_width
    height = first_height
    src_crs = first_crs
    print(f"第一个波段 CRS: {src_crs}, width: {width}, height: {height}")
    # 4.1 计算像元中心的经纬度 (WGS84)。注：仅转换坐标，不对数据做重采样，仍保持原始行列网格。
    a, b, c, d, e, f = first_transform.a, first_transform.b, first_transform.c, first_transform.d, first_transform.e, first_transform.f
    north_up = (abs(b) < 1e-12 and abs(d) < 1e-12)
    lon_arr = np.empty((height, width), dtype=np.float32)
    lat_arr = np.empty((height, width), dtype=np.float32)
    if north_up:
        print("检测到北向上影像（无旋转/剪切），使用高效逐行转换")
        # 高效逐行：x 只与列相关，y 只与行相关
        col_idx = np.arange(width, dtype=np.float32) + 0.5
        x_vec_base = c + col_idx * a  # x = a*col + c
        for r in range(height):
            y_row = f + (r + 0.5) * e  # y = e*row + f (e 为负值通常表示北向)
            y_vec = np.full(width, y_row, dtype=np.float32)
            lon_vec, lat_vec = rio_transform(src_crs, 'EPSG:4326', x_vec_base, y_vec)
            lon_arr[r, :] = lon_vec
            lat_arr[r, :] = lat_vec
    else:
        print("检测到非北向上影像")
        # 有旋转/剪切，退回通用方法（占内存但更可靠）
        col_inds, row_inds = np.meshgrid(np.arange(width), np.arange(height))
        xs_full, ys_full = rasterio.transform.xy(first_transform, row_inds, col_inds, offset='center')
        xs_full = np.asarray(xs_full, dtype=np.float32).ravel()
        ys_full = np.asarray(ys_full, dtype=np.float32).ravel()
        lon_flat, lat_flat = rio_transform(src_crs, 'EPSG:4326', xs_full, ys_full)
        lon_arr[:, :] = np.asarray(lon_flat, dtype=np.float32).reshape((height, width))
        lat_arr[:, :] = np.asarray(lat_flat, dtype=np.float32).reshape((height, width))
    print(f"WGS84 坐标转换完成, lon/lat 形状: {lon_arr.shape}/{lat_arr.shape}")


    # 5. 写入 NetCDF
    out_dir_path = Path(out_dir) if out_dir is not None else Path("output/img/1_Lt/nc")
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_nc_path = out_dir_path / f"{product_id}_TOA_{mode.upper()}_B{'-'.join(map(str,bands))}_native.nc"
    ds = nc.Dataset(str(out_nc_path), 'w', format='NETCDF4')

    nav_grp = ds.createGroup('navigation_data')
    geo_grp = ds.createGroup('geophysical_data')
    nav_grp.createDimension('y', height)
    nav_grp.createDimension('x', width)
    geo_grp.createDimension('y', height)
    geo_grp.createDimension('x', width)

    lat_var = nav_grp.createVariable('latitude', 'f4', ('y','x'), zlib=True)
    lon_var = nav_grp.createVariable('longitude', 'f4', ('y','x'), zlib=True)
    lat_var[:, :] = lat_arr
    lon_var[:, :] = lon_arr
    lon_var.long_name = 'longitude'
    lat_var.long_name = 'latitude'
    lon_var.units = 'degrees_east'
    lat_var.units = 'degrees_north'
    lon_var.standard_name = 'longitude'
    lat_var.standard_name = 'latitude'

    for wl, arr in band_data.items():
        if wl in BAND_NAMES:
            vname = BAND_NAMES[wl]
            v = geo_grp.createVariable(vname, 'f4', ('y','x'), zlib=True, fill_value=NODATA_VAL)
            v[:, :] = arr
            v.long_name = f"TOA_{mode}_{wl}nm"
            v.units = 'W·m-2·sr-1·μm-1' if mode == 'rad' else '1'

    # 全局属性
    ds.product_id = product_id
    ds.source_crs = src_crs.to_wkt() if hasattr(src_crs, 'to_wkt') else str(src_crs)
    ds.history = f"Generated native grid data; pixel center coordinates transformed to WGS84; radiometry mode={mode}"
    ds.coordinates_crs = 'EPSG:4326'
    ds.data_crs = src_crs.to_string() if hasattr(src_crs, 'to_string') else str(src_crs)
    ds.close()

    print(f"[OK] 输出完成: {out_nc_path}")
    print(f"   - 原始 CRS: {src_crs}")
    print(f"   - 结构: navigation_data/x,y; geophysical_data/L_TOA_***")
    print(out_nc_path)
    
    return str(out_nc_path)


if __name__ == "__main__":
    # 示例用法（请按需修改路径与波段后再运行）
    path = '/Users/zy/Python_code/My_Git/match_cor/Imagery_WaterLand'
    path = '/Users/zy/Python_code/My_Git/match_cor/Imagery_WaterLand'

    # 先匹配所有以 LC08 或 LC09 开头的条目
    all_matches = glob.glob(os.path.join(path, 'LC0[89]*'))

    # 然后筛选出文件夹
    L_files = [item for item in all_matches if os.path.isdir(item)]

    print(f"找到 {len(L_files)} 个 Landsat 文件夹")
    for L_file in L_files:
        example_bands = [1, 2, 3, 4, 5]
        rad_out_path = calc_landsat_toa(L_file, example_bands, out_dir = '/Users/zy/Downloads/Landsat',mode="rad")
import numpy as np
import netCDF4 as nc

arr= nc.Dataset(r"H:\Landsat\patch_output_nc\patches\LC08_L1TP_116035_20240829_20240905_02_T1_TOA_RAD_B1-2-3-4-5_native_002_013.nc")
print(arr.geophysical_data.variables)
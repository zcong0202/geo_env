# Input necessary Package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import xarray as xr

# File paths
base_path = r'C:\Users\congz\Downloads\Climate_Model_Data'
hist_file = 'tas_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_185001-194912.nc'
ssp119_file = 'tas_Amon_GFDL-ESM4_ssp119_r1i1p1f1_gr1_201501-210012.nc'
ssp245_file = 'tas_Amon_GFDL-ESM4_ssp245_r1i1p1f1_gr1_201501-210012.nc'
ssp585_file = 'tas_Amon_GFDL-ESM4_ssp585_r1i1p1f1_gr1_201501-210012.nc'

# Loading datasets
hist_dset = xr.open_dataset(f'{base_path}/{hist_file}')
ssp119_dset = xr.open_dataset(f'{base_path}/{ssp119_file}')
ssp245_dset = xr.open_dataset(f'{base_path}/{ssp245_file}')
ssp585_dset = xr.open_dataset(f'{base_path}/{ssp585_file}')

# Explore the dataset


pdb.set_trace()

# Calculate historical mean (1850-1900)
mean_1850_1900 = np.mean(hist_dset['tas'].sel(time=slice('18500101', '19001231')), axis=0)
mean_1850_1900 = np.array(mean_1850_1900)

# Calculate future means (2071-2100) for each scenario
mean_2071_2100_ssp119 = np.mean(ssp119_dset['tas'].sel(time=slice('20710101', '21001231')), axis=0)
mean_2071_2100_ssp245 = np.mean(ssp245_dset['tas'].sel(time=slice('20710101', '21001231')), axis=0)
mean_2071_2100_ssp585 = np.mean(ssp585_dset['tas'].sel(time=slice('20710101', '21001231')), axis=0)

# Calculate temperature differences
temp_diff_ssp119 = mean_2071_2100_ssp119 - mean_1850_1900
temp_diff_ssp245 = mean_2071_2100_ssp245 - mean_1850_1900
temp_diff_ssp585 = mean_2071_2100_ssp585 - mean_1850_1900

# Plot historical mean
plt.figure()
plt.imshow(mean_1850_1900, origin='lower')  
plt.colorbar(label='Temperature (K)')
plt.title('Historical Mean Temperature (1850-1900)')
plt.savefig('mean_1850_1900.png', dpi=300)
plt.close()

# Plot SSP1-1.9 difference
plt.figure()
plt.imshow(temp_diff_ssp119, origin='lower')  
plt.colorbar(label='Temperature Difference (K)')
plt.title('Temperature Change SSP1-1.9 (2071-2100)')
plt.savefig('temp_diff_ssp119.png', dpi=300)
plt.close()

# Plot SSP2-4.5 difference
plt.figure()
plt.imshow(temp_diff_ssp245, origin='lower')  
plt.colorbar(label='Temperature Difference (K)')
plt.title('Temperature Change SSP2-4.5 (2071-2100)')
plt.savefig('temp_diff_ssp245.png', dpi=300)
plt.close()

# Plot SSP5-8.5 difference
plt.figure()
plt.imshow(temp_diff_ssp585, origin='lower')  
plt.colorbar(label='Temperature Difference (K)')
plt.title('Temperature Change SSP5-8.5 (2071-2100)')
plt.savefig('temp_diff_ssp585.png', dpi=300)
plt.close()
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
from collections import namedtuple
import warnings
warnings.filterwarnings('ignore')

# Output directory
output_dir = "/Users/cz/Downloads/Data/Output"
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# Part 1: Data Preparation and Loading
# =============================================================================

temp_126_file = "/Users/cz/Downloads/Data/Temp126.nc"
temp_370_file = "/Users/cz/Downloads/Data/Temp370.nc"
rain_126_file = "/Users/cz/Downloads/Data/Rainfall126.nc"
rain_370_file = "/Users/cz/Downloads/Data/Rainfall370.nc"
humi_126_file = "/Users/cz/Downloads/Data/Humidity126.nc"
humi_370_file = "/Users/cz/Downloads/Data/Humidity370.nc"

# Load datasets
try:
    ds_temp_126 = xr.open_dataset(temp_126_file)
    ds_temp_370 = xr.open_dataset(temp_370_file)
    ds_rain_126 = xr.open_dataset(rain_126_file)
    ds_rain_370 = xr.open_dataset(rain_370_file)
    ds_humi_126 = xr.open_dataset(humi_126_file)
    ds_humi_370 = xr.open_dataset(humi_370_file)
    
    print("All datasets loaded successfully")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# =============================================================================
# Part 2: Trend Analysis Functions
# =============================================================================

def hamed_rao_mk_test(x, alpha=0.05):
    n = len(x)
    s = 0
    for k in range(n-1):
        for j in range(k+1, n):
            s += np.sign(x[j] - x[k])
    
    var_s = n*(n-1)*(2*n+5)/18
    ties = np.unique(x, return_counts=True)[1]
    for t in ties:
        var_s -= t*(t-1)*(2*t+5)/18
    
    n_eff = n
    if n > 10:
        acf = [1] + [np.corrcoef(x[:-i], x[i:])[0,1] for i in range(1, n//4)]
        acf_filtered = [r for r in acf if not (np.isnan(r) or np.isinf(r))]
        if acf_filtered:
            n_eff = n / (1 + 2 * sum((n-i)/n * acf_filtered[i] for i in range(1, len(acf_filtered))))
            var_s *= n_eff / n
    
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    
    p = 2 * (1 - norm.cdf(abs(z)))
    h = abs(z) > norm.ppf(1-alpha/2)
    
    Trend = namedtuple('Trend', ['trend', 'h', 'p', 'z', 's'])
    trend = 'increasing' if s > 0 else 'decreasing' if s < 0 else 'no trend'
    return Trend(trend=trend, h=h, p=p, z=z, s=s)

def sens_slope(x, y):
    slopes = []
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    return np.median(slopes)

# =============================================================================
# Part 3: Data Processing and Analysis
# =============================================================================

def process_temp_data(ds, convert_temp=True):
    var_name = list(ds.data_vars)[0]
    temp_mean = ds[var_name].mean(dim=[d for d in ds[var_name].dims if d not in ['time']])
    if convert_temp:
        temp_mean = temp_mean - 273.15
    annual_mean = temp_mean.groupby('time.year').mean('time')
    return annual_mean

def process_rain_data(ds):
    var_name = list(ds.data_vars)[0]
    rain_mean = ds[var_name].mean(dim=[d for d in ds[var_name].dims if d not in ['time']])
    if hasattr(ds[var_name], 'units') and ds[var_name].units.lower() in ['kg m-2 s-1', 'kg/m2/s']:
        rain_mean = rain_mean * 86400
    annual_mean = rain_mean.groupby('time.year').mean('time')
    return annual_mean

# Process data
temp_126_annual = process_temp_data(ds_temp_126)
temp_370_annual = process_temp_data(ds_temp_370)
rain_126_annual = process_rain_data(ds_rain_126)
rain_370_annual = process_rain_data(ds_rain_370)

# =============================================================================
# Part 4: Visualization and Results
# =============================================================================

# Temperature trend analysis
years_126 = temp_126_annual.year.values
years_370 = temp_370_annual.year.values

temp_126_trend = hamed_rao_mk_test(temp_126_annual.values)
temp_370_trend = hamed_rao_mk_test(temp_370_annual.values)

temp_126_slope = sens_slope(years_126, temp_126_annual.values)
temp_370_slope = sens_slope(years_370, temp_370_annual.values)

# SSP1-RCP2.6 temperature plot
plt.figure(figsize=(10, 6))
plt.plot(years_126, temp_126_annual, 'bo-', alpha=0.7, label='Annual Mean Daily Temperature')
y_126_trend = years_126 * temp_126_slope + (temp_126_annual.values[0] - temp_126_slope * years_126[0])
plt.plot(years_126, y_126_trend, 'b--', linewidth=2, label='Trend Line')

plt.title('SSP1-RCP2.6: Annual Mean Daily Temperature (2015-2100)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.grid(True, alpha=0.3)

plt.text(0.02, 0.98, f"Trend: {temp_126_trend.trend}\np-value: {temp_126_trend.p:.4f}\n" +
         f"Significant: {'Yes' if temp_126_trend.h else 'No'}\nSlope: {temp_126_slope:.4f}°C/year", 
         transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7),
         verticalalignment='top')

plt.legend(loc='upper right', framealpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'temperature_trend_SSP1-RCP2.6.png'), dpi=300)
plt.close()

# SSP3-RCP7.0 temperature plot
plt.figure(figsize=(10, 6))
plt.plot(years_370, temp_370_annual, 'ro-', alpha=0.7, label='Annual Mean Daily Temperature')
y_370_trend = years_370 * temp_370_slope + (temp_370_annual.values[0] - temp_370_slope * years_370[0])
plt.plot(years_370, y_370_trend, 'r--', linewidth=2, label='Trend Line')

plt.title('SSP3-RCP7.0: Annual Mean Daily Temperature (2015-2100)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.grid(True, alpha=0.3)

plt.text(0.02, 0.98, f"Trend: {temp_370_trend.trend}\np-value: {temp_370_trend.p:.4f}\n" +
         f"Significant: {'Yes' if temp_370_trend.h else 'No'}\nSlope: {temp_370_slope:.4f}°C/year", 
         transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7),
         verticalalignment='top')

plt.legend(loc='upper right', framealpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'temperature_trend_SSP3-RCP7.0.png'), dpi=300)
plt.close()

# Precipitation trend analysis
rain_126_trend = hamed_rao_mk_test(rain_126_annual.values)
rain_370_trend = hamed_rao_mk_test(rain_370_annual.values)

rain_126_slope = sens_slope(years_126, rain_126_annual.values)
rain_370_slope = sens_slope(years_370, rain_370_annual.values)

# SSP1-RCP2.6 precipitation plot
plt.figure(figsize=(10, 6))
plt.plot(years_126, rain_126_annual, 'bo-', alpha=0.7, label='Annual Mean Daily Precipitation')
y_126_trend = years_126 * rain_126_slope + (rain_126_annual.values[0] - rain_126_slope * years_126[0])
plt.plot(years_126, y_126_trend, 'b--', linewidth=2, label='Trend Line')

plt.title('SSP1-RCP2.6: Annual Mean Daily Precipitation (2015-2100)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Precipitation (mm/day)', fontsize=12)
plt.grid(True, alpha=0.3)

plt.text(0.02, 0.98, f"Trend: {rain_126_trend.trend}\np-value: {rain_126_trend.p:.4f}\n" +
         f"Significant: {'Yes' if rain_126_trend.h else 'No'}\nSlope: {rain_126_slope:.4f} mm/day per year", 
         transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7),
         verticalalignment='top')

plt.legend(loc='upper right', framealpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'precipitation_trend_SSP1-RCP2.6.png'), dpi=300)
plt.close()

# SSP3-RCP7.0 precipitation plot
plt.figure(figsize=(10, 6))
plt.plot(years_370, rain_370_annual, 'ro-', alpha=0.7, label='Annual Mean Daily Precipitation')
y_370_trend = years_370 * rain_370_slope + (rain_370_annual.values[0] - rain_370_slope * years_370[0])
plt.plot(years_370, y_370_trend, 'r--', linewidth=2, label='Trend Line')

plt.title('SSP3-RCP7.0: Annual Mean Daily Precipitation (2015-2100)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Precipitation (mm/day)', fontsize=12)
plt.grid(True, alpha=0.3)

plt.text(0.02, 0.98, f"Trend: {rain_370_trend.trend}\np-value: {rain_370_trend.p:.4f}\n" +
         f"Significant: {'Yes' if rain_370_trend.h else 'No'}\nSlope: {rain_370_slope:.4f} mm/day per year", 
         transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7),
         verticalalignment='top')

plt.legend(loc='upper right', framealpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'precipitation_trend_SSP3-RCP7.0.png'), dpi=300)
plt.close()

# =============================================================================
# Part 5: Maximum Daily Analysis
# =============================================================================

# Process maximum daily temperature per year
def process_max_temp_data(ds, convert_temp=True):
    var_name = list(ds.data_vars)[0]
    # Spatial average across all grid cells
    temp_mean = ds[var_name].mean(dim=[d for d in ds[var_name].dims if d not in ['time']])
    if convert_temp:
        temp_mean = temp_mean - 273.15
    # Group by year and find maximum daily value for each year
    annual_max = temp_mean.groupby('time.year').max('time')
    return annual_max

# Process maximum daily precipitation per year
def process_max_rain_data(ds):
    var_name = list(ds.data_vars)[0]
    # Spatial average across all grid cells
    rain_mean = ds[var_name].mean(dim=[d for d in ds[var_name].dims if d not in ['time']])
    if hasattr(ds[var_name], 'units') and ds[var_name].units.lower() in ['kg m-2 s-1', 'kg/m2/s']:
        rain_mean = rain_mean * 86400
    # Group by year and find maximum daily value for each year
    annual_max = rain_mean.groupby('time.year').max('time')
    return annual_max

# Process max data
temp_126_max = process_max_temp_data(ds_temp_126)
temp_370_max = process_max_temp_data(ds_temp_370)
rain_126_max = process_max_rain_data(ds_rain_126)
rain_370_max = process_max_rain_data(ds_rain_370)

# Maximum Temperature trend analysis
temp_126_max_trend = hamed_rao_mk_test(temp_126_max.values)
temp_370_max_trend = hamed_rao_mk_test(temp_370_max.values)

temp_126_max_slope = sens_slope(years_126, temp_126_max.values)
temp_370_max_slope = sens_slope(years_370, temp_370_max.values)

# SSP1-RCP2.6 maximum temperature plot
plt.figure(figsize=(10, 6))
plt.plot(years_126, temp_126_max, 'bo-', alpha=0.7, label='Annual Maximum Daily Temperature')
y_126_max_trend = years_126 * temp_126_max_slope + (temp_126_max.values[0] - temp_126_max_slope * years_126[0])
plt.plot(years_126, y_126_max_trend, 'b--', linewidth=2, label='Trend Line')

plt.title('SSP1-RCP2.6: Annual Maximum Daily Temperature (2015-2100)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Maximum Temperature (°C)', fontsize=12)
plt.grid(True, alpha=0.3)

plt.text(0.02, 0.98, f"Trend: {temp_126_max_trend.trend}\np-value: {temp_126_max_trend.p:.4f}\n" +
         f"Significant: {'Yes' if temp_126_max_trend.h else 'No'}\nSlope: {temp_126_max_slope:.4f}°C/year", 
         transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7),
         verticalalignment='top')

plt.legend(loc='upper right', framealpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'max_temperature_trend_SSP1-RCP2.6.png'), dpi=300)
plt.close()

# SSP3-RCP7.0 maximum temperature plot
plt.figure(figsize=(10, 6))
plt.plot(years_370, temp_370_max, 'ro-', alpha=0.7, label='Annual Maximum Daily Temperature')
y_370_max_trend = years_370 * temp_370_max_slope + (temp_370_max.values[0] - temp_370_max_slope * years_370[0])
plt.plot(years_370, y_370_max_trend, 'r--', linewidth=2, label='Trend Line')

plt.title('SSP3-RCP7.0: Annual Maximum Daily Temperature (2015-2100)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Maximum Temperature (°C)', fontsize=12)
plt.grid(True, alpha=0.3)

plt.text(0.02, 0.98, f"Trend: {temp_370_max_trend.trend}\np-value: {temp_370_max_trend.p:.4f}\n" +
         f"Significant: {'Yes' if temp_370_max_trend.h else 'No'}\nSlope: {temp_370_max_slope:.4f}°C/year", 
         transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7),
         verticalalignment='top')

plt.legend(loc='upper right', framealpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'max_temperature_trend_SSP3-RCP7.0.png'), dpi=300)
plt.close()

# Maximum Precipitation trend analysis
rain_126_max_trend = hamed_rao_mk_test(rain_126_max.values)
rain_370_max_trend = hamed_rao_mk_test(rain_370_max.values)

rain_126_max_slope = sens_slope(years_126, rain_126_max.values)
rain_370_max_slope = sens_slope(years_370, rain_370_max.values)

# SSP1-RCP2.6 maximum precipitation plot
plt.figure(figsize=(10, 6))
plt.plot(years_126, rain_126_max, 'bo-', alpha=0.7, label='Annual Maximum Daily Precipitation')
y_126_max_trend = years_126 * rain_126_max_slope + (rain_126_max.values[0] - rain_126_max_slope * years_126[0])
plt.plot(years_126, y_126_max_trend, 'b--', linewidth=2, label='Trend Line')

plt.title('SSP1-RCP2.6: Annual Maximum Daily Precipitation (2015-2100)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Maximum Precipitation (mm/day)', fontsize=12)
plt.grid(True, alpha=0.3)

plt.text(0.02, 0.98, f"Trend: {rain_126_max_trend.trend}\np-value: {rain_126_max_trend.p:.4f}\n" +
         f"Significant: {'Yes' if rain_126_max_trend.h else 'No'}\nSlope: {rain_126_max_slope:.4f} mm/day per year", 
         transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7),
         verticalalignment='top')

plt.legend(loc='upper right', framealpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'max_precipitation_trend_SSP1-RCP2.6.png'), dpi=300)
plt.close()

# SSP3-RCP7.0 maximum precipitation plot
plt.figure(figsize=(10, 6))
plt.plot(years_370, rain_370_max, 'ro-', alpha=0.7, label='Annual Maximum Daily Precipitation')
y_370_max_trend = years_370 * rain_370_max_slope + (rain_370_max.values[0] - rain_370_max_slope * years_370[0])
plt.plot(years_370, y_370_max_trend, 'r--', linewidth=2, label='Trend Line')

plt.title('SSP3-RCP7.0: Annual Maximum Daily Precipitation (2015-2100)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Maximum Precipitation (mm/day)', fontsize=12)
plt.grid(True, alpha=0.3)

plt.text(0.02, 0.98, f"Trend: {rain_370_max_trend.trend}\np-value: {rain_370_max_trend.p:.4f}\n" +
         f"Significant: {'Yes' if rain_370_max_trend.h else 'No'}\nSlope: {rain_370_max_slope:.4f} mm/day per year", 
         transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7),
         verticalalignment='top')

plt.legend(loc='upper right', framealpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'max_precipitation_trend_SSP3-RCP7.0.png'), dpi=300)
plt.close()

# =============================================================================
# Part 6: Wet Bulb Temperature Calculation and Analysis
# =============================================================================

def calculate_wet_bulb_temperature(temp_k, rh_percent):
    """
    Calculate wet bulb temperature using Stull's formula (2011)
    
    Args:
        temp_k: Temperature in Kelvin
        rh_percent: Relative humidity in percent
        
    Returns:
        Wet bulb temperature in Kelvin
    """
    # Convert temperature from Kelvin to Celsius for calculations
    temp_c = temp_k - 273.15
    
    # Calculation using Stull's method (2011) - accurate to within 0.3°C
    wbt_c = temp_c * np.arctan(0.151977 * (rh_percent + 8.313659)**0.5) + \
            np.arctan(temp_c + rh_percent) - np.arctan(rh_percent - 1.676331) + \
            0.00391838 * (rh_percent)**(3/2) * np.arctan(0.023101 * rh_percent) - 4.686035
    
    # Convert back to Kelvin for consistency
    wbt_k = wbt_c + 273.15
    
    return wbt_k

# Calculate wet bulb temperature for both scenarios
for scenario in [
    ('126', ds_temp_126, ds_humi_126), 
    ('370', ds_temp_370, ds_humi_370)
]:
    scenario_name, temp_ds, humi_ds = scenario
    
    print(f"Calculating wet bulb temperature for SSP-RCP{scenario_name}...")
    
    # Get variable names
    temp_var_name = list(temp_ds.data_vars)[0]
    humi_var_name = list(humi_ds.data_vars)[0]
    
    print(f"Temperature variable: {temp_var_name}")
    print(f"Humidity variable: {humi_var_name}")
    
    # Process datasets
    temp_data = temp_ds[temp_var_name]
    humi_data = humi_ds[humi_var_name]
    
    # Calculate wet bulb temperature
    wet_bulb_k = calculate_wet_bulb_temperature(temp_data, humi_data)
    
    # Convert to Celsius for analysis and visualization
    wet_bulb_c = wet_bulb_k - 273.15
    
    # Create a dataset
    wet_bulb_ds = xr.Dataset(
        data_vars={
            'wet_bulb_temp': (wet_bulb_c.dims, wet_bulb_c.values, 
                             {'units': 'degC', 'long_name': 'Wet Bulb Temperature'})
        },
        coords=wet_bulb_c.coords,
        attrs={'description': f'Wet bulb temperature for SSP-RCP{scenario_name} scenario',
               'calculation_method': "Stull's method (2011)"}
    )
    
    # Save to NetCDF
    wet_bulb_file = os.path.join(output_dir, f'wet_bulb_temp_{scenario_name}.nc')
    wet_bulb_ds.to_netcdf(wet_bulb_file)
    print(f"Saved wet bulb temperature to {wet_bulb_file}")
    
    # Calculate spatial mean
    wb_spatial_mean = wet_bulb_c.mean(dim=[d for d in wet_bulb_c.dims if d not in ['time']])
    
    # Calculate annual averages
    wb_annual_mean = wb_spatial_mean.groupby('time.year').mean('time')
    
    # Calculate annual maximum
    wb_annual_max = wb_spatial_mean.groupby('time.year').max('time')
    
    # Get years
    years = wb_annual_mean.year.values
    
    # Trend analysis for annual mean wet bulb temperature
    wb_mean_trend = hamed_rao_mk_test(wb_annual_mean.values)
    wb_mean_slope = sens_slope(years, wb_annual_mean.values)
    
    # Trend analysis for annual maximum wet bulb temperature
    wb_max_trend = hamed_rao_mk_test(wb_annual_max.values)
    wb_max_slope = sens_slope(years, wb_annual_max.values)
    
    # Plot annual mean wet bulb temperature
    plt.figure(figsize=(10, 6))
    color = 'bo-' if scenario_name == '126' else 'ro-'
    plt.plot(years, wb_annual_mean, color, alpha=0.7, label='Annual Mean Daily Wet Bulb Temperature')
    
    # Add trend line
    y_trend = years * wb_mean_slope + (wb_annual_mean.values[0] - wb_mean_slope * years[0])
    trend_color = 'b--' if scenario_name == '126' else 'r--'
    plt.plot(years, y_trend, trend_color, linewidth=2, label='Trend Line')
    
    plt.title(f'SSP-RCP{scenario_name}: Annual Mean Daily Wet Bulb Temperature (2015-2100)', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Wet Bulb Temperature (°C)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.text(0.02, 0.98, f"Trend: {wb_mean_trend.trend}\np-value: {wb_mean_trend.p:.4f}\n" +
             f"Significant: {'Yes' if wb_mean_trend.h else 'No'}\nSlope: {wb_mean_slope:.4f}°C/year", 
             transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7),
             verticalalignment='top')
    
    plt.legend(loc='upper right', framealpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'wet_bulb_mean_trend_SSP-RCP{scenario_name}.png'), dpi=300)
    plt.close()
    
    # Plot annual maximum wet bulb temperature
    plt.figure(figsize=(10, 6))
    plt.plot(years, wb_annual_max, color, alpha=0.7, label='Annual Maximum Daily Wet Bulb Temperature')
    
    # Add trend line
    y_trend = years * wb_max_slope + (wb_annual_max.values[0] - wb_max_slope * years[0])
    plt.plot(years, y_trend, trend_color, linewidth=2, label='Trend Line')
    
    plt.title(f'SSP-RCP{scenario_name}: Annual Maximum Daily Wet Bulb Temperature (2015-2100)', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Maximum Wet Bulb Temperature (°C)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.text(0.02, 0.98, f"Trend: {wb_max_trend.trend}\np-value: {wb_max_trend.p:.4f}\n" +
             f"Significant: {'Yes' if wb_max_trend.h else 'No'}\nSlope: {wb_max_slope:.4f}°C/year", 
             transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7),
             verticalalignment='top')
    
    plt.legend(loc='upper right', framealpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'wet_bulb_max_trend_SSP-RCP{scenario_name}.png'), dpi=300)
    plt.close()
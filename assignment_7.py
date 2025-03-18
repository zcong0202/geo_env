import os
import xarray as xr
import glob
import re
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
from shapely.geometry import mapping
import pandas as pd

# PART 1: LOAD DATA
datasets = [
    {
        "name": "Precipitation",
        "folder": r"C:\Users\congz\OneDrive - KAUST\Desktop(DELL)\Erse316\ERSE316 ASSIGNMENT\AS7\Data\Precipitation",
        "pattern": r'era5_OLR_(\d{4})_total_precipitation'
    },
    {
        "name": "Runoff",
        "folder": r"C:\Users\congz\OneDrive - KAUST\Desktop(DELL)\Erse316\ERSE316 ASSIGNMENT\AS7\Data\Runoff",
        "pattern": r'ambientera5_OLR_(\d{4})_total_runoff'
    },
    {
        "name": "Evaporation",
        "folder": r"C:\Users\congz\OneDrive - KAUST\Desktop(DELL)\Erse316\ERSE316 ASSIGNMENT\AS7\Data\Total_Evaporation",
        "pattern": r'era5_OLR_(\d{4})_total_evaporation'
    }
]

# Load datasets
raster_data = {}
for dataset in datasets:
    files = glob.glob(os.path.join(dataset['folder'], "*.nc"))
    data_dict = {}
    for file_path in files:
        match = re.search(dataset['pattern'], os.path.basename(file_path))
        if match:
            year = int(match.group(1))
            data_dict[year] = xr.open_dataset(file_path)
    raster_data[dataset['name']] = data_dict

# Load Saudi Arabia shapefile
shapefile_folder = r"C:\Users\congz\OneDrive - KAUST\Desktop(DELL)\Erse316\ERSE316 ASSIGNMENT\AS7\Data\Saudi_Shape_File\Saudi_Shape_File"
saudi_shapefile_path = os.path.join(shapefile_folder, "Saudi_Shape.shp")
saudi_country = gpd.read_file(saudi_shapefile_path)

# Clip raster data
clipped_data = {}
for dataset_name in raster_data:
    clipped_data[dataset_name] = {}
    for year, ds in raster_data[dataset_name].items():
        ds_with_dims = ds.rio.set_spatial_dims(x_dim='longitude', y_dim='latitude')
        ds_with_crs = ds_with_dims.rio.write_crs("EPSG:4326")
        clipped_ds = ds_with_crs.rio.clip(saudi_country.geometry.apply(mapping), drop=True)
        clipped_data[dataset_name][year] = clipped_ds

# Process time series
def process_dataset(dataset_name, start_year=2000, end_year=2020):
    available_years = sorted([year for year in clipped_data[dataset_name].keys() 
                             if start_year <= year <= end_year])
    
    monthly_results = []
    yearly_results = []
    yearly_monthly_data = {}
    
    for year in available_years:
        ds = clipped_data[dataset_name][year]
        main_var = list(ds.data_vars)[0]
        time_dim = next((dim for dim in ds.dims if dim in ['time', 'valid_time']), None)
        
        monthly_grouped_data = ds[main_var].groupby(f"{time_dim}.month").sum(time_dim)
        
        yearly_total = 0
        yearly_monthly_values = []
        
        for month in monthly_grouped_data.month.values:
            monthly_spatial_mean = monthly_grouped_data.sel(month=month).mean(
                dim=['latitude', 'longitude']).values.item()
            
            if hasattr(ds[main_var], 'units') and 'm' in ds[main_var].units.lower():
                monthly_spatial_mean *= 1000
            
            if dataset_name == "Evaporation" and monthly_spatial_mean < 0:
                monthly_spatial_mean = abs(monthly_spatial_mean)
            
            monthly_results.append({
                'date': pd.Timestamp(year=year, month=int(month), day=15),
                'value': monthly_spatial_mean
            })
            
            yearly_total += monthly_spatial_mean
            yearly_monthly_values.append(monthly_spatial_mean)
        
        yearly_results.append({'year': year, 'value': yearly_total})
        yearly_monthly_data[year] = yearly_monthly_values
    
    monthly_df = pd.DataFrame(monthly_results).sort_values('date')
    yearly_df = pd.DataFrame(yearly_results)
    
    # Visualization with year data labels
    plt.figure(figsize=(16, 8))
    
    monthly_continuous_time = [date.year + (date.month-1)/12 for date in monthly_df['date']]
    
    plt.plot(monthly_continuous_time, monthly_df['value'], 'b-', linewidth=1.0, 
             label=f'Monthly {dataset_name.lower()}')
    plt.plot(yearly_df['year'], yearly_df['value'], 'r--', linewidth=2.0, 
             marker='o', markersize=6, label=f'Yearly {dataset_name.lower()}')
    
    # Add data labels to yearly points
    for x, y in zip(yearly_df['year'], yearly_df['value']):
        plt.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                     xytext=(0,10), ha='center', fontsize=8)
    
    plt.grid(True, linestyle='-', alpha=0.3)
    plt.title(f'Monthly and Yearly {dataset_name} (2000-2020)', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel(f'{dataset_name} (mm)', fontsize=14)
    plt.legend(loc='upper left')
    plt.xticks(np.arange(min(yearly_df['year']), max(yearly_df['year'])+1), rotation=45)
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_TimeSeries_Combined_2000_2020.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return monthly_df, yearly_df, yearly_monthly_data

# Process each dataset
precip_monthly_df, precip_yearly_df, precip_monthly_data = process_dataset("Precipitation")
evap_monthly_df, evap_yearly_df, evap_monthly_data = process_dataset("Evaporation")
runoff_monthly_df, runoff_yearly_df, runoff_monthly_data = process_dataset("Runoff")

# Annual comparison plot with all points labeled
plt.figure(figsize=(16, 10))

plt.plot(precip_yearly_df['year'], precip_yearly_df['value'], 'b-', linewidth=2.0, marker='o', label='Precipitation')
plt.plot(evap_yearly_df['year'], evap_yearly_df['value'], 'r-', linewidth=2.0, marker='s', label='Evaporation')
plt.plot(runoff_yearly_df['year'], runoff_yearly_df['value'], 'g-', linewidth=2.0, marker='^', label='Runoff')

# Add value labels for all datasets
for x, y in zip(precip_yearly_df['year'], precip_yearly_df['value']):
    plt.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontsize=8)

for x, y in zip(evap_yearly_df['year'], evap_yearly_df['value']):
    plt.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                 xytext=(0,-15), ha='center', fontsize=8)

for x, y in zip(runoff_yearly_df['year'], runoff_yearly_df['value']):
    plt.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontsize=8)

plt.grid(True, linestyle='--', alpha=0.7)
plt.title('Annual Hydrological Components (2000-2020)', fontsize=18)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Amount (mm)', fontsize=14)
plt.legend(loc='upper right', fontsize=12)
plt.xticks(precip_yearly_df['year'], precip_yearly_df['year'], rotation=45)
plt.ylim(bottom=0)

plt.figtext(0.02, 0.02, 'Hydrological Components\nSaudi Arabia\nData: ERA5 Reanalysis', 
            fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('Annual_Hydrological_Components_2000_2020.png', dpi=300, bbox_inches='tight')
plt.show()

# Water balance analysis
water_balance = pd.DataFrame({
    'Year': precip_yearly_df['year'],
    'Precipitation': precip_yearly_df['value'],
    'Evaporation': evap_yearly_df['value'],
    'Runoff': runoff_yearly_df['value']
})

water_balance['Balance'] = water_balance['Precipitation'] - water_balance['Evaporation'] - water_balance['Runoff']

# Create monthly water balance
monthly_dates = sorted(list(set(precip_monthly_df['date']).intersection(
    set(evap_monthly_df['date'])).intersection(set(runoff_monthly_df['date']))))
monthly_water_balance = pd.DataFrame({'date': monthly_dates})

for date in monthly_dates:
    precip_value = precip_monthly_df.loc[precip_monthly_df['date'] == date, 'value'].values[0]
    evap_value = evap_monthly_df.loc[evap_monthly_df['date'] == date, 'value'].values[0]
    runoff_value = runoff_monthly_df.loc[runoff_monthly_df['date'] == date, 'value'].values[0]
    
    idx = monthly_water_balance.index[monthly_water_balance['date'] == date].tolist()[0]
    monthly_water_balance.loc[idx, 'precipitation'] = precip_value
    monthly_water_balance.loc[idx, 'evaporation'] = evap_value
    monthly_water_balance.loc[idx, 'runoff'] = runoff_value
    monthly_water_balance.loc[idx, 'balance'] = precip_value - evap_value - runoff_value

monthly_water_balance = monthly_water_balance.sort_values('date')
monthly_years = [date.year + (date.month-1)/12 for date in monthly_water_balance['date']]

# Plot water balance trends
plt.figure(figsize=(16, 8))

plt.plot(monthly_years, monthly_water_balance['balance'], 'b-', linewidth=1.0, alpha=0.6, label='Monthly Water Balance')
plt.plot(water_balance['Year'], water_balance['Balance'], 'r--', linewidth=2.0, marker='o', markersize=6, label='Yearly Water Balance')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.8)

plt.grid(True, linestyle='--', alpha=0.7)
plt.title('Monthly and Yearly Water Balance (P - E - R) in Saudi Arabia (2000-2020)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Water Balance (mm)', fontsize=14)
plt.legend(loc='upper right', fontsize=12)

for x, y in zip(water_balance['Year'], water_balance['Balance']):
    color = 'green' if y >= 0 else 'red'
    plt.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                 xytext=(0, 10 if y >= 0 else -15), ha='center', 
                 fontsize=9, color=color, weight='bold')

plt.xticks(np.arange(min(water_balance['Year']), max(water_balance['Year'])+1), 
           np.arange(min(water_balance['Year']), max(water_balance['Year'])+1).astype(int), 
           rotation=45)

plt.figtext(0.02, 0.02, 
            'Water Balance = Precipitation - Evaporation - Runoff\nPositive values indicate water surplus\nNegative values indicate water deficit', 
            fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('Water_Balance_Trends_2000_2020.png', dpi=300, bbox_inches='tight')
plt.show()

# Create stacked area chart
plt.figure(figsize=(16, 10))

years = water_balance['Year']
precip = water_balance['Precipitation']
evap = water_balance['Evaporation']
runoff = water_balance['Runoff']

plt.fill_between(years, 0, precip, alpha=0.7, label='Precipitation', color='blue')
plt.fill_between(years, 0, evap, alpha=0.7, label='Evaporation', color='red')
plt.fill_between(years, 0, runoff, alpha=0.7, label='Runoff', color='green')
plt.plot(years, water_balance['Balance'], 'k-', linewidth=2.5, marker='o', markersize=8, label='Water Balance')
plt.axhline(y=0, color='k', linestyle='--', linewidth=1.0)

plt.title('Water Balance Components and Net Balance (2000-2020)', fontsize=18)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Amount (mm)', fontsize=14)
plt.legend(loc='upper right', fontsize=12)
plt.xticks(years, years.astype(int), rotation=45)

for x, y in zip(years, water_balance['Balance']):
    color = 'green' if y >= 0 else 'red'
    plt.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                 xytext=(0, 10 if y >= 0 else -20), ha='center', 
                 fontsize=9, color=color, weight='bold')

plt.figtext(0.02, 0.02, 'Water Balance Components\nSaudi Arabia\nData: ERA5 Reanalysis', 
            fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('Water_Balance_Components_2000_2020.png', dpi=300, bbox_inches='tight')
plt.show()


# 1. MONTHLY COMPARISON CHART - All three components
plt.figure(figsize=(18, 10))

# Create continuous time axis for monthly data
precip_monthly_time = [date.year + (date.month-1)/12 for date in precip_monthly_df['date']]
evap_monthly_time = [date.year + (date.month-1)/12 for date in evap_monthly_df['date']]
runoff_monthly_time = [date.year + (date.month-1)/12 for date in runoff_monthly_df['date']]

plt.plot(precip_monthly_time, precip_monthly_df['value'], 'b-', linewidth=1.0, 
         alpha=0.7, label='Monthly Precipitation')
plt.plot(evap_monthly_time, evap_monthly_df['value'], 'r-', linewidth=1.0, 
         alpha=0.7, label='Monthly Evaporation')
plt.plot(runoff_monthly_time, runoff_monthly_df['value'], 'g-', linewidth=1.0, 
         alpha=0.7, label='Monthly Runoff')

plt.grid(True, linestyle='--', alpha=0.4)
plt.title('Monthly Hydrological Components Comparison (2000-2020)', fontsize=18)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Amount (mm)', fontsize=14)
plt.legend(loc='upper right', fontsize=12)
plt.ylim(bottom=0)

# Add shading for better visibility between years
min_year = min(min(precip_monthly_df['date']).year, min(evap_monthly_df['date']).year, min(runoff_monthly_df['date']).year)
max_year = max(max(precip_monthly_df['date']).year, max(evap_monthly_df['date']).year, max(runoff_monthly_df['date']).year)

for year in range(min_year, max_year+1, 2):
    plt.axvspan(year, year+1, color='gray', alpha=0.1)

plt.xticks(np.arange(min_year, max_year+1), rotation=45)

plt.figtext(0.02, 0.02, 'Monthly Hydrological Components\nSaudi Arabia\nData: ERA5 Reanalysis', 
            fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('Monthly_Hydrological_Components_Comparison_2000_2020.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. ANNUAL SEASONAL CYCLES - Monthly averages by component
# Calculate average value for each month across all years
def calculate_monthly_averages(df):
    df['month'] = df['date'].dt.month
    return df.groupby('month')['value'].mean().reset_index()

precip_monthly_avg = calculate_monthly_averages(precip_monthly_df)
evap_monthly_avg = calculate_monthly_averages(evap_monthly_df)
runoff_monthly_avg = calculate_monthly_averages(runoff_monthly_df)

# Create seasonal cycle comparison chart
plt.figure(figsize=(14, 8))

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
x = np.arange(len(months))

plt.plot(x, precip_monthly_avg['value'], 'b-', linewidth=2.5, marker='o', label='Precipitation')
plt.plot(x, evap_monthly_avg['value'], 'r-', linewidth=2.5, marker='s', label='Evaporation')
plt.plot(x, runoff_monthly_avg['value'], 'g-', linewidth=2.5, marker='^', label='Runoff')

# Add data labels
for i, (p, e, r) in enumerate(zip(precip_monthly_avg['value'], evap_monthly_avg['value'], runoff_monthly_avg['value'])):
    plt.annotate(f'{p:.1f}', (i, p), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontsize=8, color='blue')
    plt.annotate(f'{e:.1f}', (i, e), textcoords="offset points", 
                 xytext=(0,-15), ha='center', fontsize=8, color='red')
    plt.annotate(f'{r:.1f}', (i, r), textcoords="offset points", 
                 xytext=(0,10), ha='center', fontsize=8, color='green')

plt.grid(True, linestyle='--', alpha=0.7)
plt.title('Average Monthly Hydrological Cycle (2000-2020)', fontsize=18)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Average Amount (mm)', fontsize=14)
plt.legend(loc='upper right', fontsize=12)
plt.xticks(x, months)
plt.ylim(bottom=0)

# Add light shading for seasons
plt.axvspan(-0.5, 1.5, color='skyblue', alpha=0.15, label='Winter')  # Dec-Feb
plt.axvspan(1.5, 4.5, color='lightgreen', alpha=0.15, label='Spring')  # Mar-May
plt.axvspan(4.5, 7.5, color='yellow', alpha=0.15, label='Summer')  # Jun-Aug
plt.axvspan(7.5, 10.5, color='orange', alpha=0.15, label='Fall')  # Sep-Nov
plt.axvspan(10.5, 11.5, color='skyblue', alpha=0.15)  # Dec (winter again)

plt.figtext(0.02, 0.02, 'Average Monthly Hydrological Components\nSaudi Arabia\nData: ERA5 Reanalysis', 
            fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('Monthly_Seasonal_Cycle_Comparison_2000_2020.png', dpi=300, bbox_inches='tight')
plt.show()
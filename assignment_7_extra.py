import os
import xarray as xr
import glob
import re
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
from shapely.geometry import mapping, Point
import pandas as pd
import matplotlib.colors as colors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.patheffects as PathEffects
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.ticker as mticker

# PART 1: LOAD RASTER DATA

# Define dataset information
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

# Load all datasets
raster_data_all = {}
raster_data_2020 = {}

for dataset in datasets:
    # Load all years data
    files = glob.glob(os.path.join(dataset['folder'], "*.nc"))
    data_dict = {}
    
    for file_path in files:
        match = re.search(dataset['pattern'], os.path.basename(file_path))
        if match:
            year = int(match.group(1))
            data = xr.open_dataset(file_path)
            data_dict[year] = data
            
            # Store 2020 data separately
            if year == 2020:
                raster_data_2020[dataset['name']] = data
    
    raster_data_all[dataset['name']] = data_dict


# PART 2: LOAD AND PROCESS SAUDI ARABIA PROVINCES

shapefile_folder = r"C:\Users\congz\OneDrive - KAUST\Desktop(DELL)\Erse316\ERSE316 ASSIGNMENT\AS7\Data\KSAshapefile"
saudi_provinces = gpd.read_file(os.path.join(shapefile_folder, "gadm41_SAU_1.shp"))

# Filter out islands by finding mainland provinces
saudi_provinces['area'] = saudi_provinces.geometry.area
mainland_area_threshold = saudi_provinces['area'].max() * 0.01
saudi_mainland = saudi_provinces[saudi_provinces['area'] > mainland_area_threshold]

# PART 3: CALCULATE WATER BALANCE BY PROVINCE

province_results = {}

for idx, row in saudi_mainland.iterrows():
    province_name = row['NAME_1']
    
    # Create GeoDataFrame for this province
    province_gdf = gpd.GeoDataFrame(geometry=[row.geometry], crs=saudi_mainland.crs)
    province_values = {}
    
    # Process each dataset
    for dataset_name, ds in raster_data_2020.items():
        # Prepare and clip data
        ds_with_dims = ds.rio.set_spatial_dims(x_dim='longitude', y_dim='latitude')
        ds_with_crs = ds_with_dims.rio.write_crs("EPSG:4326")
        clipped_ds = ds_with_crs.rio.clip(province_gdf.geometry.apply(mapping), drop=True)
        
        # Extract and calculate values
        main_var = list(clipped_ds.data_vars)[0]
        time_dim = next((dim for dim in clipped_ds.dims if dim in ['time', 'valid_time']), None)
        summed_ds = clipped_ds[main_var].sum(dim=time_dim)
        mean_value = summed_ds.mean().values.item() * 1000  # Convert to mm
        
        # Handle negative evaporation
        if dataset_name == "Evaporation" and mean_value < 0:
            mean_value = abs(mean_value)
        
        province_values[dataset_name] = mean_value
    
    # Calculate water balance
    province_values['Water_Balance'] = (
        province_values['Precipitation'] - 
        province_values['Evaporation'] - 
        province_values['Runoff']
    )
    
    province_results[province_name] = province_values

# Create DataFrame from results
water_balance_df = pd.DataFrame.from_dict(province_results, orient='index')
water_balance_df.to_csv('Saudi_Arabia_Water_Balance_By_Province_2020.csv')

# PART 4: VISUALIZE WATER BALANCE MAP

# Add water balance data to GeoDataFrame
saudi_mainland['Water_Balance'] = saudi_mainland['NAME_1'].map({k: v['Water_Balance'] for k, v in province_results.items()})

# Create custom colormap and normalization
cmap = colors.LinearSegmentedColormap.from_list(
    'water_balance_cmap', 
    [(0.8, 0, 0), (1, 0.5, 0.5), (1, 1, 1), (0.5, 0.7, 1), (0, 0.3, 0.8)],
    N=256
)

min_value = min(saudi_mainland['Water_Balance'].min(), -20)
max_value = max(saudi_mainland['Water_Balance'].max(), 20)
abs_max = max(abs(min_value), abs(max_value))
norm = colors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

# Create map
fig, ax = plt.subplots(figsize=(15, 12))

saudi_mainland.plot(
    column='Water_Balance',
    ax=ax,
    cmap=cmap,
    norm=norm,
    legend=True,
    legend_kwds={
        'label': 'Water Balance (mm)',
        'orientation': 'horizontal',
        'shrink': 0.6,
        'fraction': 0.04,
        'pad': 0.01,
        'aspect': 40
    }
)

# Add province boundaries
saudi_mainland.boundary.plot(ax=ax, linewidth=1.0, color='black')

# Add province labels
for idx, row in saudi_mainland.iterrows():
    centroid = row.geometry.centroid
    txt = ax.text(
        centroid.x, centroid.y, 
        f"{row['NAME_1']}\n{row['Water_Balance']:.1f} mm",
        ha='center', va='center', 
        fontsize=8, fontweight='bold'
    )
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='white')])

plt.title('Annual Water Balance by Province in Saudi Arabia (2020)', fontsize=16)
plt.figtext(0.02, 0.01, 'Water Balance = Precipitation - Evaporation - Runoff\nPositive values indicate water surplus\nNegative values indicate water deficit\nData source: ERA5 Reanalysis', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig('Saudi_Arabia_Water_Balance_Map_2020.png', dpi=300, bbox_inches='tight')
plt.show()

# PART 5: CREATE COMPONENT MAPS

# Add component data to GeoDataFrame
for component in ['Precipitation', 'Evaporation', 'Runoff']:
    saudi_mainland[component] = saudi_mainland['NAME_1'].map({k: v[component] for k, v in province_results.items()})

# Create component maps
components = ['Precipitation', 'Evaporation', 'Runoff']
component_cmaps = {
    'Precipitation': 'Blues',
    'Evaporation': 'Reds',
    'Runoff': 'Greens'
}

fig, axes = plt.subplots(1, 3, figsize=(20, 8))

for i, component in enumerate(components):
    saudi_mainland.plot(
        column=component,
        ax=axes[i],
        cmap=component_cmaps[component],
        legend=True,
        legend_kwds={
            'label': f'{component} (mm)',
            'orientation': 'horizontal',
            'shrink': 0.6,
            'fraction': 0.05,
            'pad': 0.01,
            'aspect': 20
        }
    )
    
    saudi_mainland.boundary.plot(ax=axes[i], linewidth=0.8, color='black')
    axes[i].set_title(f'Annual {component} (2020)', fontsize=14)
    axes[i].set_xticks([])
    axes[i].set_yticks([])

plt.suptitle('Hydrological Components by Province in Saudi Arabia (2020)', fontsize=16, y=0.95)
plt.figtext(0.02, 0.01, 'Data source: ERA5 Reanalysis', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
plt.tight_layout(rect=[0, 0.02, 1, 0.95])
plt.savefig('Saudi_Arabia_Hydrological_Components_Map_2020.png', dpi=300, bbox_inches='tight')
plt.show()

# PART 6: EXTRACT POINT TIME SERIES

# Define points for Jizan and Ha'il
jizan_row = saudi_mainland[saudi_mainland['NAME_1'].str.contains('Jīzān|Jizan|Jazan', regex=True)]
hail_row = saudi_mainland[saudi_mainland['NAME_1'].str.contains('Ḥaʼil|Hail|Ha\'il', regex=True)]

# Get representative points
jizan_point = jizan_row.iloc[0].geometry.representative_point()
jizan_lon, jizan_lat = jizan_point.x, jizan_point.y

hail_point = hail_row.iloc[0].geometry.representative_point()
hail_lon, hail_lat = hail_point.x, hail_point.y

# Extract time series function
def extract_time_series(location_name, lon, lat):
    # Extract each dataset
    time_series = {}
    for dataset_name, dataset_dict in raster_data_all.items():
        all_times = []
        all_values = []
        
        for year, ds in dataset_dict.items():
            main_var = list(ds.data_vars)[0]
            time_dim = next((dim for dim in ds.dims if dim in ['time', 'valid_time']), None)
            
            ds_point = ds.sel(longitude=lon, latitude=lat, method='nearest')
            times = ds_point[time_dim].values
            values = ds_point[main_var].values
            
            all_times.extend(times)
            all_values.extend(values)
        
        # Create DataFrame
        df = pd.DataFrame({'time': all_times, 'value': all_values})
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')
        
        # Unit conversion
        df['value'] = df['value'] * 1000  # Convert to mm
        
        # Handle evaporation
        if dataset_name == 'Evaporation':
            df['value'] = df['value'].abs()
        
        # Group by day (resample to daily values by summing)
        df['date'] = df['time'].dt.date
        daily_df = df.groupby('date')['value'].sum().reset_index()
        daily_df['date'] = pd.to_datetime(daily_df['date'])
            
        time_series[dataset_name] = daily_df
    
    # Merge datasets - now using 'date' instead of 'time'
    merged_df = time_series['Precipitation'][['date', 'value']].rename(columns={'value': 'Precipitation'})
    for dataset_name in ['Evaporation', 'Runoff']:
        merged_df = pd.merge(
            merged_df, 
            time_series[dataset_name][['date', 'value']].rename(columns={'value': dataset_name}),
            on='date', 
            how='outer'
        )
    
    # Calculate water balance
    merged_df['Water_Balance'] = merged_df['Precipitation'] - merged_df['Evaporation'] - merged_df['Runoff']
    merged_df = merged_df.sort_values('date')
    
    # Save to CSV
    merged_df.to_csv(f'ERA5_Point_Daily_TimeSeries_{location_name}.csv', index=False)
    
    return merged_df

# Extract time series for both locations
jizan_df = extract_time_series('Jizan', jizan_lon, jizan_lat)
hail_df = extract_time_series('Hail', hail_lon, hail_lat)

# PART 7: CORRELATION ANALYSIS

# Variables for correlation
cols_for_corr = ['Precipitation', 'Evaporation', 'Runoff', 'Water_Balance']

# Calculate correlations and p-values
def calculate_correlation(df):
    # Calculate Pearson correlation
    corr_matrix = df[cols_for_corr].corr(method='pearson')
    
    # Calculate p-values
    p_matrix = pd.DataFrame(np.zeros((len(cols_for_corr), len(cols_for_corr))), 
                           index=cols_for_corr, columns=cols_for_corr)
    
    for i, col1 in enumerate(cols_for_corr):
        for j, col2 in enumerate(cols_for_corr):
            corr, p = pearsonr(df[col1].dropna(), df[col2].dropna())
            p_matrix.loc[col1, col2] = p
    
    return corr_matrix, p_matrix

# Calculate correlation for both locations
jizan_corr, jizan_p = calculate_correlation(jizan_df)
hail_corr, hail_p = calculate_correlation(hail_df)


# PART 8: CREATE CORRELATION HEATMAPS

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plot Jizan correlation
sns.heatmap(jizan_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
            linewidths=0.5, fmt='.2f', square=True, ax=ax1)

# Add significance stars
for i, col1 in enumerate(jizan_corr.columns):
    for j, col2 in enumerate(jizan_corr.columns):
        if i != j:  # Skip diagonal
            p = jizan_p.loc[col1, col2]
            if p < 0.001:
                ax1.text(j + 0.5, i + 0.85, '***', ha='center', va='center', color='black', fontsize=12)
            elif p < 0.01:
                ax1.text(j + 0.5, i + 0.85, '**', ha='center', va='center', color='black', fontsize=12)
            elif p < 0.05:
                ax1.text(j + 0.5, i + 0.85, '*', ha='center', va='center', color='black', fontsize=12)

ax1.set_title('Daily Correlation Matrix at Jizan', fontsize=16)

# Plot Ha'il correlation
sns.heatmap(hail_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
            linewidths=0.5, fmt='.2f', square=True, ax=ax2)

# Add significance stars
for i, col1 in enumerate(hail_corr.columns):
    for j, col2 in enumerate(hail_corr.columns):
        if i != j:  # Skip diagonal
            p = hail_p.loc[col1, col2]
            if p < 0.001:
                ax2.text(j + 0.5, i + 0.85, '***', ha='center', va='center', color='black', fontsize=12)
            elif p < 0.01:
                ax2.text(j + 0.5, i + 0.85, '**', ha='center', va='center', color='black', fontsize=12)
            elif p < 0.05:
                ax2.text(j + 0.5, i + 0.85, '*', ha='center', va='center', color='black', fontsize=12)

ax2.set_title("Daily Correlation Matrix at Ha'il", fontsize=16)

plt.suptitle('Correlation Analysis of Daily Hydrological Components', fontsize=18)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig('ERA5_Point_Daily_Correlation_Jizan_vs_Hail.png', dpi=300, bbox_inches='tight')
plt.show()
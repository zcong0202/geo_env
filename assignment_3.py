# Input necessary Package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import xarray as xr
import tools

# Read the CVS file
df_isd = tools.read_isd_csv(r'C:\Users\congz\OneDrive - KAUST\Desktop(DELL)\Erse316\ERSE316 ASSIGNMENT\AS3\DATA\41024099999.csv')
plot = df_isd.plot(title="ISD data for Jeddah")
plt.show()
# Read the NetCDF file
ssp245 = xr.open_dataset(r'C:\Users\congz\OneDrive - KAUST\Desktop(DELL)\Erse316\ERSE316 ASSIGNMENT\AS3\DATA\tas_Amon_GFDL-ESM4_ssp245_r1i1p1f1_gr1_201501-210012.nc')
history_data = xr.open_dataset(r'C:\Users\congz\OneDrive - KAUST\Desktop(DELL)\Erse316\ERSE316 ASSIGNMENT\AS3\DATA\tas_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_185001-194912.nc')

# View CVS dataframe
print("\nFirst 5 rows of the data:")
print(df_isd.head())

# View CVS basic information about the dataset
print("\nDataset information:")
print(df_isd.info())

# Convert CVS DATA dewpoint to relative humidity
df_isd['RH'] = tools.dewpoint_to_rh(df_isd['DEW'].values, df_isd['TMP'].values)

# Calculate hourly Heat Index
df_isd['HI'] = tools.gen_heat_index(df_isd['TMP'].values, df_isd['RH'].values)
df_isd['HI_F'] = df_isd['HI'] * 9/5 + 32 # Cover index c to f

# Calculate overall maximum HI
max_hi_c = df_isd['HI'].max()
max_hi_f = df_isd['HI_F'].max()
max_hi_time = df_isd['HI'].idxmax()

print(f"\nMaximum Heat Index: {max_hi_c:.2f}°C ({max_hi_f:.2f}°F)")
print(f"Occurred at: {max_hi_time}")

# Print conditions at specific time
print("\nConditions at 2024-08-10 11:00:00:")
conditions = df_isd.loc["2024-08-10 11:00:00"]
print(f"Temperature: {conditions['TMP']:.2f}°C")
print(f"Relative Humidity: {conditions['RH']:.2f}%")
print(f"Heat Index: {conditions['HI']:.2f}°C")

# Calculate daily HI
daily_mean = df_isd.resample('D').mean()
daily_max_hi = df_isd['HI'].resample('D').max()  

# Create figure for daily mean and max HI
plt.figure(figsize=(15, 8))

# Plot both daily mean and max Heat Index
plt.plot(daily_mean.index, daily_mean['HI'], 'b-', label='Daily Mean HI', linewidth=2)
plt.plot(daily_max_hi.index, daily_max_hi, 'r-', label='Daily Max HI', linewidth=2)

# Add reference lines
mean_hi = daily_mean['HI'].mean()
max_hi = daily_max_hi.mean()
plt.axhline(y=mean_hi, color='b', linestyle='--', 
            label=f'Mean HI Average: {mean_hi:.1f}°C', alpha=0.7)
plt.axhline(y=max_hi, color='r', linestyle='--', 
            label=f'Max HI Average: {max_hi:.1f}°C', alpha=0.7)
plt.title('Daily Mean and Maximum Heat Index in Jeddah', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Heat Index (°C)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Extract data for August 7-14, 2024
start_date = '2024-08-07'
end_date = '2024-08-14'
aug_data = daily_mean[start_date:end_date]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

# First subplot: Temperature and Heat Index
ax1.plot(aug_data.index, aug_data['TMP'], 'b-', label='Mean Temperature', linewidth=2, marker='o')
ax1.plot(aug_data.index, aug_data['HI'], 'r-', label='Mean Heat Index', linewidth=2, marker='s')
ax1.axhline(y=35, color='orange', linestyle='--', label='35°C Benchmark', linewidth=1.5)

ax1.set_title('Daily Temperature and Heat Index in Jeddah (Aug 7-14, 2024)', fontsize=14)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Temperature (°C)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(fontsize=10)
ax1.tick_params(axis='x', rotation=45)

# Second subplot: Relative Humidity
ax2.plot(aug_data.index, aug_data['RH'], 'g-', label='Mean Relative Humidity', linewidth=2, marker='o')
ax2.set_title('Daily Relative Humidity in Jeddah (Aug 7-14, 2024)', fontsize=14)
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Relative Humidity (%)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(fontsize=10)
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Print statistics
print("\nHeat Index Statistics:")
print("\nDaily Mean Heat Index:")
print(f"Maximum: {daily_mean['HI'].max():.2f}°C")
print(f"Average: {daily_mean['HI'].mean():.2f}°C")
print(f"Date of Maximum: {daily_mean['HI'].idxmax()}")
print("\nDaily Maximum Heat Index:")
print(f"Maximum: {daily_max_hi.max():.2f}°C")
print(f"Average: {daily_max_hi.mean():.2f}°C")
print(f"Date of Maximum: {daily_max_hi.idxmax()}")

# Calculate number of days with dangerous heat conditions
dangerous_mean = (daily_mean['HI'] > 35).sum()
dangerous_max = (daily_max_hi > 35).sum()
print(f"\nDays with mean HI > 35°C: {dangerous_mean}")
print(f"Days with max HI > 35°C: {dangerous_max}")


# Input Jeddah coordinates from CVS
jeddah_lat = 21.679564
jeddah_lon = 39.156536

# Find nearest grid points in the datasets
# For SSP245 data (2071-2100)
# First select the time slice, then select the nearest coordinates
future_temp = ssp245.sel(time=slice('20710101', '21001231'))
future_temp = future_temp.sel(
    lat=jeddah_lat,
    lon=jeddah_lon,
    method='nearest'
)
future_mean = future_temp.tas.mean().values - 273.15  # Convert Kelvin to Celsius

# For historical data (1850-1900)
# First select the time slice, then select the nearest coordinates
historical_temp = history_data.sel(time=slice('18500101', '19001231'))
historical_temp = historical_temp.sel(
    lat=jeddah_lat,
    lon=jeddah_lon,
    method='nearest'
)
historical_mean = historical_temp.tas.mean().values - 273.15  # Convert Kelvin to Celsius

# Calculate temperature difference
temp_difference = future_mean - historical_mean

# Print results
print(f"\nMean Temperature Analysis for Jeddah:")
print(f"Historical Period (1850-1900): {historical_mean:.2f}°C")
print(f"Future Period (2071-2100): {future_mean:.2f}°C")
print(f"Temperature Difference: {temp_difference:.2f}°C")

# Get the projected warming from our previous calculation
projected_warming = temp_difference

# Apply projected warming to air temperature
df_isd['TMP_projected'] = df_isd['TMP'] + projected_warming

# Recalculate RH with the new projected temperature using the original dewpoint
df_isd['RH_adjusted'] = tools.dewpoint_to_rh(df_isd['DEW'].values, df_isd['TMP_projected'].values)

# Recalculate heat index with adjusted temperature and adjusted RH
df_isd['HI_adjusted'] = tools.gen_heat_index(df_isd['TMP_projected'].values, df_isd['RH_adjusted'].values)

# Find maximum values
original_hi_max = df_isd['HI'].max()
adjusted_hi_max = df_isd['HI_adjusted'].max()

# Calculate increase in HI max value
increase_in_hi_max = adjusted_hi_max - original_hi_max

# Find when these maxima occur and print the conditions at those times
original_hi_max_time = df_isd['HI'].idxmax()
adjusted_hi_max_time = df_isd['HI_adjusted'].idxmax()

# Print results formatted to 2 decimal points
print("\nHeat Index Analysis Results:")
print(f"Original HI max: {original_hi_max:.2f}")
print(f"Adjusted HI max: {adjusted_hi_max:.2f}")
print(f"Increase in HI max value: {increase_in_hi_max:.2f}")
print(original_hi_max_time)
print(adjusted_hi_max_time)


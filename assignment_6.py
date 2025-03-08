# Import necessary libraries
import xarray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load two datasets using the exact paths provided
instant_path = r"C:\Users\congz\OneDrive - KAUST\Desktop(DELL)\Erse316\ERSE316 ASSIGNMENT\AS6\data\data_stream-oper_stepType-instant.nc"
accum_path = r"C:\Users\congz\OneDrive - KAUST\Desktop(DELL)\Erse316\ERSE316 ASSIGNMENT\AS6\data\data_stream-oper_stepType-accum.nc"

dset_instant = xarray.open_dataset(instant_path)
dset_accum = xarray.open_dataset(accum_path)

# 2. Check variables in the files
print("Instantaneous file variables:", list(dset_instant.variables))
print("Accumulated file variables:", list(dset_accum.variables))

# 3. Extract relevant variables and convert to numpy arrays
# Extract temperature and coordinate data from the instantaneous file
t2m = np.array(dset_instant.variables['t2m'])
latitude_instant = np.array(dset_instant.variables['latitude'])
longitude_instant = np.array(dset_instant.variables['longitude'])
time_dt_instant = np.array(dset_instant.variables['valid_time'])

# Extract precipitation and coordinate data from the accumulated file
tp = np.array(dset_accum.variables['tp'])
latitude_accum = np.array(dset_accum.variables['latitude'])
longitude_accum = np.array(dset_accum.variables['longitude'])
time_dt_accum = np.array(dset_accum.variables['valid_time'])

# Check if coordinates and times match between the two files
coords_match = (np.array_equal(latitude_instant, latitude_accum) and 
                np.array_equal(longitude_instant, longitude_accum))
times_match = np.array_equal(time_dt_instant, time_dt_accum)

print("Coordinates match between files:", coords_match)
print("Time arrays match between files:", times_match)

# Use one set of coordinate and time data 
latitude = latitude_instant
longitude = longitude_instant
time_dt = time_dt_instant

# Print data shapes for verification
print("Temperature data shape:", t2m.shape)
print("Precipitation data shape:", tp.shape)
print("Latitude shape:", latitude.shape)
print("Longitude shape:", longitude.shape)
print("Time shape:", time_dt.shape)

# 4. Unit conversion
# Convert temperature from K to °C
t2m = t2m - 273.15

# Convert precipitation from m/h to mm/h
tp = tp * 1000

# 5. Check data dimensions, if 4D calculate mean
if t2m.ndim == 4:
    t2m = np.nanmean(t2m, axis=1)
if tp.ndim == 4:
    tp = np.nanmean(tp, axis=1)

# 6. Create a Pandas DataFrame containing time series data for both temperature and precipitation
time_pd = pd.to_datetime(time_dt)
df_era5 = pd.DataFrame(index=time_pd)
df_era5['t2m'] = t2m[:,3,2]
df_era5['tp'] = tp[:,3,2]
print(df_era5.head())

# 7. Plot the time series
plt.figure(figsize=(12, 6))
df_era5.plot()
plt.xlabel('Time')
plt.ylabel('Value (°C for t2m, mm/h for tp)')
plt.grid(True)
plt.show()

# 8. Resample data to annual time step and calculate mean precipitation
annual_precip = df_era5['tp'].resample('YE').mean() * 24 * 365.25

# Calculate multi-year average precipitation
mean_annual_precip = np.nanmean(annual_precip)

print("\nAnnual precipitation (mm/y):")
print(annual_precip)
print(f"\nMean annual precipitation: {mean_annual_precip:.2f} mm/year")

# Visualize annual precipitation
plt.figure(figsize=(8, 5))
annual_precip.plot(kind='bar')
plt.axhline(y=mean_annual_precip, color='r', linestyle='-', label=f'Mean: {mean_annual_precip:.2f} mm/year')
plt.xlabel('Year')
plt.ylabel('Precipitation (mm/year)')
plt.legend()
plt.grid(True, axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 9. Calculate Potential Evaporation (PE)
tmin = df_era5['t2m'].resample('D').min().values
tmax = df_era5['t2m'].resample('D').max().values
tmean = df_era5['t2m'].resample('D').mean().values

# Set geographic latitude and date
lat = 21.25  # latitude
doy = df_era5['t2m'].resample('D').mean().index.dayofyear  # day of year

# Import tools module and calculate potential evaporation
import tools
pe = tools.hargreaves_samani_1982(tmin, tmax, tmean, lat, doy)

# 10. Plot the potential evaporation time series
ts_index = df_era5['t2m'].resample('D').mean().index

plt.figure(figsize=(12, 6))
plt.plot(ts_index, pe, label='Potential Evaporation')
plt.xlabel('Time')
plt.ylabel('Potential evaporation (mm d⁻¹)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 11. Calculate and print average daily potential evaporation
mean_daily_pe = np.nanmean(pe)
print(f"Mean daily potential evaporation: {mean_daily_pe:.2f} mm/day")

# 12. Calculate annual potential evaporation
annual_pe = mean_daily_pe * 365.25
print(f"Mean annual potential evaporation: {annual_pe:.2f} mm/year")

# 13. Calculate annual evaporation volume from the reservoir
reservoir_area = 1.6  # reservoir area in square kilometers
reservoir_area_m2 = reservoir_area * 1_000_000  # convert to square meters

# Annual evaporation volume (cubic meters)
annual_evaporation_volume_m3 = annual_pe / 1000 * reservoir_area_m2  # convert mm to m

# Express in different units
annual_evaporation_volume_million_m3 = annual_evaporation_volume_m3 / 1_000_000  # million cubic meters

print(f"Reservoir area: {reservoir_area:.2f} km²")
print(f"Mean annual potential evaporation: {annual_pe:.2f} mm/year")
print(f"Annual reservoir evaporation volume: {annual_evaporation_volume_m3:.2f} m³/year")
print(f"Annual reservoir evaporation volume: {annual_evaporation_volume_million_m3:.2f} million m³/year")
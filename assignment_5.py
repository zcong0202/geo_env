# Import required libraries
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Load file
file_path = r'C:\Users\congz\OneDrive - KAUST\Desktop(DELL)\Erse316\ERSE316 ASSIGNMENT\AS5\data\GRIDSAT-B1.2009.11.25.00.v02r01.nc'
dset = xr.open_dataset(file_path)
# Load the longwave infrared data 
IR = np.array(dset.variables['irwin_cdr']).squeeze()
# Check the dimensions of the array
print("IR data shape:", IR.shape)
# Flip the data vertically to correct orientation
IR = np.flipud(IR)

# Apply scale and offset to get brightness temperatures in kelvin
IR = IR*0.01+200

# Convert from kelvin to degrees Celsius
IR = IR-273.15

# Plot the data with a colorbar
plt.figure(1)
plt.imshow(IR, extent=[-180.035, 180.035, -70.035, 70.035], aspect='auto')
cbar = plt.colorbar()
cbar.set_label('Brightness temperature (degrees Celsius)')

# Mark Jeddah's location on the map
jeddah_lat = 21.5
jeddah_lon = 39.2
plt.scatter(jeddah_lon, jeddah_lat, color='red', marker='o', label='Jeddah')
plt.legend()

plt.show()
print(dset.attrs)

# Calculate brightness temperature at Jeddah's location for 00:00 UTC
# Get latitude and longitude arrays
lats = np.linspace(70.035, -70.035, IR.shape[0])
lons = np.linspace(-180.035, 180.035, IR.shape[1])

# Find the grid point closest to Jeddah's location
lat_idx = np.abs(lats - jeddah_lat).argmin()
lon_idx = np.abs(lons - jeddah_lon).argmin()

# Extract brightness temperature at Jeddah's location for 00:00 UTC
jeddah_temp_00 = IR[lat_idx, lon_idx]
print(f"\nBrightness temperature at Jeddah's location at 2009-11-25 00:00 UTC: {jeddah_temp_00:.2f}°C")

# Define file paths for other time points
time_points = ['03:00', '06:00', '09:00', '12:00']
file_paths = [
    r'C:\Users\congz\OneDrive - KAUST\Desktop(DELL)\Erse316\ERSE316 ASSIGNMENT\AS5\data\GRIDSAT-B1.2009.11.25.03.v02r01.nc',
    r'C:\Users\congz\OneDrive - KAUST\Desktop(DELL)\Erse316\ERSE316 ASSIGNMENT\AS5\data\GRIDSAT-B1.2009.11.25.06.v02r01.nc',
    r'C:\Users\congz\OneDrive - KAUST\Desktop(DELL)\Erse316\ERSE316 ASSIGNMENT\AS5\data\GRIDSAT-B1.2009.11.25.09.v02r01.nc',
    r'C:\Users\congz\OneDrive - KAUST\Desktop(DELL)\Erse316\ERSE316 ASSIGNMENT\AS5\data\GRIDSAT-B1.2009.11.25.12.v02r01.nc'
]

# Store brightness temperatures for all time points
jeddah_temps = [jeddah_temp_00]
valid_times = ['00:00']

# Load data for other time points
for i, file_path in enumerate(file_paths):
    try:
        # Load dataset
        temp_dset = xr.open_dataset(file_path)
        
        # Load infrared data
        temp_IR = np.array(temp_dset.variables['irwin_cdr']).squeeze()
        
        # Flip data to correct orientation
        temp_IR = np.flipud(temp_IR)
        
        # Apply scale and offset to get brightness temperatures in kelvin
        temp_IR = temp_IR*0.01+200
        
        # Convert to degrees Celsius
        temp_IR = temp_IR-273.15
        
        # Extract brightness temperature at Jeddah's location
        jeddah_temp = temp_IR[lat_idx, lon_idx]
        
        # Store brightness temperature and time point
        jeddah_temps.append(jeddah_temp)
        valid_times.append(time_points[i])
        
        print(f"Brightness temperature at Jeddah's location at 2009-11-25 {time_points[i]} UTC: {jeddah_temp:.2f}°C")
        
        # Close dataset
        temp_dset.close()
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")

# Find the time point with the lowest temperature
if jeddah_temps:
    min_temp = min(jeddah_temps)
    min_idx = jeddah_temps.index(min_temp)
    min_time = valid_times[min_idx]
    print(f"\nLowest brightness temperature at Jeddah's location: {min_temp:.2f}°C, occurring at {min_time} UTC")

# Plot the time series of brightness temperatures
plt.figure(figsize=(10, 6))
plt.plot(valid_times, jeddah_temps, 'o-', color='blue')
plt.scatter(min_time, min_temp, color='red', s=100, zorder=5, label=f'Minimum Temperature: {min_temp:.2f}°C at {min_time}')
plt.xlabel('Time (UTC)')
plt.ylabel('Brightness Temperature (°C)')
plt.show()



# Define AutoEstimator relationship parameters
A = 1.1183e11  
b = 3.6382e-2  
c = 1.2        

# Convert temperature from Celsius back to Kelvin
jeddah_temps_kelvin = [temp + 273.15 for temp in jeddah_temps]

# Calculate rainfall rate using AutoEstimator formula
jeddah_rainfall = []
for temp_kelvin in jeddah_temps_kelvin:
    rainfall_rate = A * np.exp(-b * (temp_kelvin**c))
    jeddah_rainfall.append(rainfall_rate)

# Print rainfall rate for each time point
print("\nRainfall rate estimation results:")
for i, time in enumerate(valid_times):
    print(f"Time: {time} UTC, Temperature: {jeddah_temps[i]:.2f}°C ({jeddah_temps_kelvin[i]:.2f}K), Rainfall Rate: {jeddah_rainfall[i]:.4f} mm/h")

# Find the time point with maximum rainfall rate
max_rainfall = max(jeddah_rainfall)
max_rainfall_idx = jeddah_rainfall.index(max_rainfall)
max_rainfall_time = valid_times[max_rainfall_idx]
print(f"\nMaximum rainfall rate: {max_rainfall:.4f} mm/h, occurring at {max_rainfall_time} UTC")

# Plot rainfall rate time series
plt.figure(figsize=(10, 6))
plt.plot(valid_times, jeddah_rainfall, 'o-', color='green')
plt.scatter(max_rainfall_time, max_rainfall, color='red', s=100, zorder=5, 
           label=f'Maximum Rainfall: {max_rainfall:.4f} mm/h at {max_rainfall_time}')
plt.xlabel('Time (UTC)')
plt.ylabel('Rainfall Rate (mm/h)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

git add assignment_5.py
git commit -m "Add assignment 5: Analysis of 2009 Jeddah rainfall event using GridSat data"
git push origin main
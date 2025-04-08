import xarray as xr
import geopandas as gpd
import numpy as np
import scipy.optimize as opt
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

#os.chdir(os.path.abspath(''))
os.chdir(os.path.dirname(__file__))
print(os.getcwd())
## ---Part 1: Pre-Processing ERA5 dataset ---

# Clip each variable using the shapefile
def load_and_clip(nc_file, var_name, gdf):
    ds = xr.open_dataset(nc_file)
    ds = ds.rio.write_crs("EPSG:4326")  # Ensure correct CRS
    clipped = ds.rio.clip(gdf.geometry, gdf.crs, drop=True)
    return clipped[var_name]

# Load the watershed shapefile
shapefile_path = r"C:\Users\congz\OneDrive - KAUST\Desktop(DELL)\Erse316\ERSE316 ASSIGNMENT\AS8\DATA\WS_3/WS_3.shp"
gdf = gpd.read_file(shapefile_path)

# Load the NetCDF files (precipitation, ET, runoff)
precip_file = r"C:\Users\congz\OneDrive - KAUST\Desktop(DELL)\Erse316\ERSE316 ASSIGNMENT\AS8\DATA\era5_OLR_2001_total_precipitation.nc"
et_file = r"C:\Users\congz\OneDrive - KAUST\Desktop(DELL)\Erse316\ERSE316 ASSIGNMENT\AS8\DATA\era5_OLR_2001_total_evaporation.nc"
runoff_file = r"C:\Users\congz\OneDrive - KAUST\Desktop(DELL)\Erse316\ERSE316 ASSIGNMENT\AS8\DATA\ambientera5_OLR_2001_total_runoff.nc"


# Extract variables
# Load and clip each dataset,unit conversion: meters to mm
P_grid = load_and_clip(precip_file, "tp", gdf) * 1000.0
ET_grid = load_and_clip(et_file, "e", gdf) * 1000.0
Q_grid = load_and_clip(runoff_file, "ro", gdf) * 1000.0

# Compute area-averaged values
P = P_grid.mean(dim=["latitude", "longitude"]).values
ET = ET_grid.mean(dim=["latitude", "longitude"]).values
Q_obs = Q_grid.mean(dim=["latitude", "longitude"]).values

# Ensure ET is positive
ET = np.where(ET < 0.0, -ET, ET) 

ET = np.where(ET < 0.0, -ET, ET)
print("P:", P)
print("ET:", ET)
print("Q_obs:", Q_obs)

#plot
# Create time array for 2001
start_date_2001 = datetime(2001, 1, 1)
hours_2001 = [start_date_2001 + timedelta(hours=i) for i in range(len(P))]

# Create figure and subplots for 2001
fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle('Hourly Watershed Variables - 2001', fontsize=16)

# Plot precipitation (2001)
ax[0].plot(hours_2001, P, 'b-', linewidth=1)
ax[0].set_ylabel('Precipitation (mm/hr)')
ax[0].grid(True, linestyle='--', alpha=0.7)
ax[0].set_title('Precipitation')

# Plot evaporation (2001)
ax[1].plot(hours_2001, ET, 'g-', linewidth=1)
ax[1].set_ylabel('Evaporation (mm/hr)')
ax[1].grid(True, linestyle='--', alpha=0.7)
ax[1].set_title('Total Evaporation')

# Plot runoff (2001)
ax[2].plot(hours_2001, Q_obs, 'r-', linewidth=1)
ax[2].set_ylabel('Runoff (mm/hr)')
ax[2].set_xlabel('Time')
ax[2].grid(True, linestyle='--', alpha=0.7)
ax[2].set_title('Runoff')

# Format x-axis with month labels
for axis in ax:
    axis.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    axis.xaxis.set_major_locator(mdates.MonthLocator())

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('watershed_variables_2001.png', dpi=300)
plt.show()

# Load the NetCDF files for 2002 (precipitation, ET, runoff)
precip_file2002 = r"C:\Users\congz\OneDrive - KAUST\Desktop(DELL)\Erse316\ERSE316 ASSIGNMENT\AS8\DATA\era5_OLR_2002_total_precipitation.nc"
et_file2002 = r"C:\Users\congz\OneDrive - KAUST\Desktop(DELL)\Erse316\ERSE316 ASSIGNMENT\AS8\DATA\era5_OLR_2002_total_evaporation.nc"
runoff_file2002 = r"C:\Users\congz\OneDrive - KAUST\Desktop(DELL)\Erse316\ERSE316 ASSIGNMENT\AS8\DATA\ambientera5_OLR_2002_total_runoff.nc"

# Extract variables
P_grid2002 = load_and_clip(precip_file2002, "tp", gdf) * 1000.0
ET_grid2002 = load_and_clip(et_file2002, "e", gdf) * 1000.0
Q_grid2002 = load_and_clip(runoff_file2002, "ro", gdf) * 1000.0

# Compute area-averaged values
P2002 = P_grid2002.mean(dim=["latitude", "longitude"]).values
ET2002 = ET_grid2002.mean(dim=["latitude", "longitude"]).values
Q_obs2002 = Q_grid2002.mean(dim=["latitude", "longitude"]).values

# Ensure ET is positive
ET2002 = np.where(ET2002 < 0.0, -ET2002, ET2002)

# Create time array for 2002
start_date_2002 = datetime(2002, 1, 1)
hours_2002 = [start_date_2002 + timedelta(hours=i) for i in range(len(P2002))]

# Create figure and subplots for 2002
fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle('Hourly Watershed Variables - 2002', fontsize=16)

# Plot precipitation (2002)
ax[0].plot(hours_2002, P2002, 'b-', linewidth=1)
ax[0].set_ylabel('Precipitation (mm/hr)')
ax[0].grid(True, linestyle='--', alpha=0.7)
ax[0].set_title('Precipitation')

# Plot evaporation (2002)
ax[1].plot(hours_2002, ET2002, 'g-', linewidth=1)
ax[1].set_ylabel('Evaporation (mm/hr)')
ax[1].grid(True, linestyle='--', alpha=0.7)
ax[1].set_title('Total Evaporation')

# Plot runoff (2002)
ax[2].plot(hours_2002, Q_obs2002, 'r-', linewidth=1)
ax[2].set_ylabel('Runoff (mm/hr)')
ax[2].set_xlabel('Time')
ax[2].grid(True, linestyle='--', alpha=0.7)
ax[2].set_title('Runoff')

# Format x-axis with month labels
for axis in ax:
    axis.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    axis.xaxis.set_major_locator(mdates.MonthLocator())

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('watershed_variables_2002.png', dpi=300)
plt.show()


## --- Part 2: Model setup and calibration ---

# Rainfall-runoff model (finite difference approximation)
def simulate_runoff(k, P, ET, dt=1):
    n = len(P)
    Q_sim = np.zeros(n)
    Q_sim[0] = Q_obs[0]
    
    for t in range(2, n):
        Q_t = (Q_sim[t-1] + (P[t] - ET[t]) * dt) / (1 + dt/k)
        Q_sim[t] = max(0,Q_t) # Ensure non-negative runoff

    return (Q_sim)

# Define the objective (KGE) function
def kge(Q_obs, Q_sim):
    r = np.corrcoef(Q_obs, Q_sim)[0, 1]
    alpha = np.std(Q_sim) / np.std(Q_obs)
    beta = np.mean(Q_sim) / np.mean(Q_obs)
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    #print interative matrix, if needed
    #print(r, alpha, beta, kge)
    return (kge, r, alpha, beta)

# Create the list of k values and run the model to get simulated runoff and performance index
k_testlist = np.arange(0.15,0.3,0.15 )
#k_testlist = 0.15
Q_sim_all = np.empty([len(P), len(k_testlist)])
PerfIndex_all = np.empty([len(k_testlist), 5]) #for k, kge, r, alpha, beta

n=0
for k in k_testlist:
    Qsim = simulate_runoff(k, P, ET)
    Q_sim_all[:,n] = Qsim
    
    PerfIndex = kge(Q_obs, Qsim)
    PerfIndex_all[n,0] = k
    PerfIndex_all[n,1:] = PerfIndex
    n += 1

#print (Q_sim_all)
print (PerfIndex_all)

# Model Validation (with given k=0.15)
print("\n--- Part 2: Model Validation with k=0.15 ---")
fixed_k = 0.15
Q_sim_fixed = simulate_runoff(fixed_k, P, ET)
kge_fixed = kge(Q_obs, Q_sim_fixed)

print(f"Validation with fixed k={fixed_k}:")
print(f"KGE: {kge_fixed[0]:.3f}")
print(f"Correlation coefficient (r): {kge_fixed[1]:.3f}")
print(f"Relative variability (α): {kge_fixed[2]:.3f}")
print(f"Bias (β): {kge_fixed[3]:.3f}")


# Plot comparison of observed and simulated runoff for 2001 (validation with fixed k)
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(hours_2001, Q_obs, 'r-', linewidth=1, label='Observed Runoff')
ax.plot(hours_2001, Q_sim_fixed, 'b--', linewidth=1, label='Simulated Runoff')
ax.set_ylabel('Runoff (mm/hr)')
ax.set_xlabel('Time')
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_title(f'2001 - Validation with Fixed k = {fixed_k} (KGE = {kge_fixed[0]:.3f})')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.tight_layout()
plt.savefig('runoff_validation_fixed_k.png', dpi=300)
plt.show()

# Scatter plot for validation with fixed k
plt.figure(figsize=(8, 8))
plt.scatter(Q_obs, Q_sim_fixed, alpha=0.5)
max_val = max(np.max(Q_obs), np.max(Q_sim_fixed))
plt.plot([0, max_val], [0, max_val], 'r--')
plt.xlabel('Observed Runoff (mm/hr)')
plt.ylabel('Simulated Runoff (mm/hr)')
plt.title(f'2001 - Validation with Fixed k = {fixed_k} (KGE = {kge_fixed[0]:.3f})')
plt.grid(True, linestyle='--', alpha=0.7)
plt.text(0.05, 0.9, f'KGE = {kge_fixed[0]:.3f}', transform=plt.gca().transAxes)
plt.text(0.05, 0.85, f'r = {kge_fixed[1]:.3f}', transform=plt.gca().transAxes)
plt.text(0.05, 0.8, f'α = {kge_fixed[2]:.3f}', transform=plt.gca().transAxes)
plt.text(0.05, 0.75, f'β = {kge_fixed[3]:.3f}', transform=plt.gca().transAxes)
plt.tight_layout()
plt.savefig('runoff_scatter_fixed_k.png', dpi=300)
plt.show()

## KGE Optimization ##

# Objective function for optimization
def objective(k, P, ET, Q_obs):
    Q_sim = simulate_runoff(k, P, ET)
    kge_model = kge(Q_obs, Q_sim)
    return (1.0 - kge_model[0])

# Optimize k using KGE
res = opt.minimize_scalar(objective, bounds=(0.1, 2), args=(P, ET, Q_obs), method='bounded')
print(res)

# Best k value
best_k = res.x
Q_sim = simulate_runoff(best_k, P, ET)
print(f"Optimized k: {best_k:.3f}")


# Plot comparison of observed and simulated runoff for 2001 (calibration)
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(hours_2001, Q_obs, 'r-', linewidth=1, label='Observed Runoff')
ax.plot(hours_2001, Q_sim, 'b--', linewidth=1, label='Simulated Runoff')
ax.set_ylabel('Runoff (mm/hr)')
ax.set_xlabel('Time')
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_title(f'2001 - Calibration Results (k = {best_k:.3f}, KGE = {kge(Q_obs, Q_sim)[0]:.3f})')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.tight_layout()
plt.savefig('runoff_calibration_2001.png', dpi=300)
plt.show()

# Scatter plot for 2001 calibration
plt.figure(figsize=(8, 8))
plt.scatter(Q_obs, Q_sim, alpha=0.5)
max_val = max(np.max(Q_obs), np.max(Q_sim))
plt.plot([0, max_val], [0, max_val], 'r--')
plt.xlabel('Observed Runoff (mm/hr)')
plt.ylabel('Simulated Runoff (mm/hr)')
plt.title(f'2001 - Calibration (KGE = {kge(Q_obs, Q_sim)[0]:.3f})')
plt.grid(True, linestyle='--', alpha=0.7)
kge_vals = kge(Q_obs, Q_sim)
plt.text(0.05, 0.9, f'KGE = {kge_vals[0]:.3f}', transform=plt.gca().transAxes)
plt.text(0.05, 0.85, f'r = {kge_vals[1]:.3f}', transform=plt.gca().transAxes)
plt.text(0.05, 0.8, f'α = {kge_vals[2]:.3f}', transform=plt.gca().transAxes)
plt.text(0.05, 0.75, f'β = {kge_vals[3]:.3f}', transform=plt.gca().transAxes)
plt.tight_layout()
plt.savefig('runoff_scatter_2001.png', dpi=300)
plt.show()

# compare with 2002
# --- Validation ---

# Load the NetCDF files for validation (precipitation, ET, runoff)
precip_fileVal = r"C:\Users\congz\OneDrive - KAUST\Desktop(DELL)\Erse316\ERSE316 ASSIGNMENT\AS8\DATA\era5_OLR_2002_total_precipitation.nc"
et_fileVal = r"C:\Users\congz\OneDrive - KAUST\Desktop(DELL)\Erse316\ERSE316 ASSIGNMENT\AS8\DATA\era5_OLR_2002_total_evaporation.nc"
runoff_fileVal = r"C:\Users\congz\OneDrive - KAUST\Desktop(DELL)\Erse316\ERSE316 ASSIGNMENT\AS8\DATA\ambientera5_OLR_2002_total_runoff.nc"

P_gridVal = load_and_clip(precip_fileVal, "tp", gdf) * 1000.0
ET_gridVal = load_and_clip(et_fileVal, "e", gdf) * 1000.0
Q_gridVal = load_and_clip(runoff_fileVal, "ro", gdf) * 1000.0

# Compute area-averaged values
P_v = P_gridVal.mean(dim=["latitude", "longitude"]).values
ET_v = ET_gridVal.mean(dim=["latitude", "longitude"]).values
Q_obs_v = Q_gridVal.mean(dim=["latitude", "longitude"]).values

# Ensure ET is positive
ET_v = np.where(ET_v < 0.0, -ET_v, ET_v) 

Q_sim_v = simulate_runoff(best_k, P_v, ET_v)
print(Q_obs_v, Q_sim_v)
kge_v = kge(Q_obs_v, Q_sim_v)

print(f"KGE for validation: {kge_v[0]:.3f}")

# Validation with 2002 data
# Create time array for 2002 validation
start_date_2002_val = datetime(2002, 1, 1)
hours_2002_val = [start_date_2002_val + timedelta(hours=i) for i in range(len(P_v))]

# Report validation metrics
kge_vals_v = kge(Q_obs_v, Q_sim_v)
print(f"KGE: {kge_vals_v[0]:.3f}")
print(f"Correlation coefficient (r): {kge_vals_v[1]:.3f}")
print(f"Relative variability (α): {kge_vals_v[2]:.3f}")
print(f"Bias (β): {kge_vals_v[3]:.3f}")



# Plot comparison of observed and simulated runoff for 2002 (validation)
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(hours_2002_val, Q_obs_v, 'r-', linewidth=1, label='Observed Runoff')
ax.plot(hours_2002_val, Q_sim_v, 'b--', linewidth=1, label='Simulated Runoff')
ax.set_ylabel('Runoff (mm/hr)')
ax.set_xlabel('Time')
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_title(f'2002 - Validation Results (k = {best_k:.3f}, KGE = {kge_vals_v[0]:.3f})')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.tight_layout()
plt.savefig('runoff_validation_2002.png', dpi=300)
plt.show()

# Scatter plot for 2002 validation
plt.figure(figsize=(8, 8))
plt.scatter(Q_obs_v, Q_sim_v, alpha=0.5)
max_val = max(np.max(Q_obs_v), np.max(Q_sim_v))
plt.plot([0, max_val], [0, max_val], 'r--')
plt.xlabel('Observed Runoff (mm/hr)')
plt.ylabel('Simulated Runoff (mm/hr)')
plt.title(f'2002 - Validation (KGE = {kge_vals_v[0]:.3f})')
plt.grid(True, linestyle='--', alpha=0.7)
plt.text(0.05, 0.9, f'KGE = {kge_vals_v[0]:.3f}', transform=plt.gca().transAxes)
plt.text(0.05, 0.85, f'r = {kge_vals_v[1]:.3f}', transform=plt.gca().transAxes)
plt.text(0.05, 0.8, f'α = {kge_vals_v[2]:.3f}', transform=plt.gca().transAxes)
plt.text(0.05, 0.75, f'β = {kge_vals_v[3]:.3f}', transform=plt.gca().transAxes)
plt.tight_layout()
plt.savefig('runoff_scatter_2002.png', dpi=300)
plt.show()




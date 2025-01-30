#Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import xarray as xr
#Data Loading Section
##Open the SRTM DEM dataset using xarray
dset = xr.open_dataset(r'C:\Users\congz\OneDrive - KAUST\Desktop(DELL)\ERSE316 ASSIGNMENT\N21E039.SRTMGL1_NC.nc')
pdb.set_trace() # Debugging breakpoint to inspect dataset
#Data Processing
##Convert DEM variable to numpy array for analysis
DEM = np.array(dset.variables['SRTMGL1_DEM'])
pdb.set_trace()
##Check dimensions of DEM array
DEM.shape
#Visualization Section 
##Create elevation map using imshow
plt.imshow(DEM)
##Add colorbar showing elevation scale
cbar = plt.colorbar()
cbar.set_label('ELevation (m asl)')
##Display the plot
plt.show()
##Save high resolution figure
plt.savefig('assignment_1.png', dpi=300)
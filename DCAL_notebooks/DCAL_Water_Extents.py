# Code behind module for DCAL_Water_Extents.ipynb

################################
##
## Import Statments
##
################################

# Import standard Python modules
import sys
import datacube
import numpy as np  

# Import DCAL utilities containing function definitions used generally across DCAL
sys.path.append('../DCAL_utils')


################################
##
## Function Definitions
##
################################

# GD duplicated in DCAL_Spectral_Products ?
def threshold_count(da, min_threshold, max_threshold, mask = None):
    def count_not_nans(arr):
        return np.count_nonzero(~np.isnan(arr))
    
    in_threshold = np.logical_and( da.values > min_threshold, da.values < max_threshold)
    
    total_non_cloudy = count_not_nans(da.values) if mask is None else np.sum(mask) 
    
    return dict(total = np.size(da.values),
                total_non_cloudy = total_non_cloudy,
                inside = np.nansum(in_threshold),
                outside = total_non_cloudy - np.nansum(in_threshold)
               )    

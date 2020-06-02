# Code behind module for DCAL_Vegetation_Change.ipynb

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

# Define NDVI function
def NDVI(dataset):
    return (dataset.nir - dataset.red)/(dataset.nir + dataset.red)

# GD duplicate - to library?
# Define threshold count function
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

# GD duplicate - to library?
# Define threshold percentage function
def threshold_percentage(da, min_threshold, max_threshold, mask = None):
    counts = threshold_count(da, min_threshold, max_threshold, mask = mask)
    return dict(percent_inside_threshold = (counts["inside"]   / counts["total"]) * 100.0,
                percent_outside_threshold = (counts["outside"] / counts["total"]) * 100.0,
                percent_clouds = ( 100.0-counts["total_non_cloudy"] / counts["total"] * 100.0))
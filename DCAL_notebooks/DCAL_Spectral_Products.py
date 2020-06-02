# Code behind module for DCAL_Spectral_Products.ipynb

################################
##
## Import Statments
##
################################

# Import standard Python modules
import sys
import datacube
import matplotlib.pyplot as plt
import numpy as np  
from matplotlib.ticker import FuncFormatter

# Import DCAL utilities containing function definitions used generally across DCAL
sys.path.append('../DCAL_utils')


################################
##
## Function Definitions
##
################################

# GD These indicies could be made common in a library...?
# Normalized Difference Built-up Index (NDBI)
def NDBI(dataset):
    return (dataset.swir1 - dataset.nir)/(dataset.swir1 + dataset.nir)
    
# Normalized Difference Vegetation Index (NDVI)
def NDVI(dataset):
    return (dataset.nir - dataset.red)/(dataset.nir + dataset.red)

# Normalized Difference Water Index (NDWI)
def NDWI(dataset):
    return (dataset.green - dataset.nir)/(dataset.green + dataset.nir)

# Soil Adjusted Vegetation Index (SAVI)
def SAVI(dataset):
    return (dataset.nir - dataset.red)/(dataset.nir + dataset.red + 0.5)*1.5

# Enhanced Vegetation Index (EVI) 
def EVI(dataset):
    return 2.5*(dataset.nir - dataset.red)/(dataset.nir + 6.0*dataset.red - 7.5*dataset.blue + 1.0)

# Threshold plot function.
def threshold_plot(da, min_threshold, max_threshold, mask = None, width = 10, *args, **kwargs): 
    color_in    = np.array([255,0,0])
    color_out   = np.array([0,0,0])
    color_cloud = np.array([255,255,255])
    
    array = np.zeros((*da.values.shape, 3)).astype(np.int16)
    
    inside  = np.logical_and(da.values > min_threshold, da.values < max_threshold)
    outside = np.invert(inside)
    masked  = np.zeros(da.values.shape).astype(bool) if mask is None else mask
    
    array[inside] =  color_in
    array[outside] = color_out
    array[masked] =  color_cloud

    def figure_ratio(ds, fixed_width = 10):
        width = fixed_width
        height = len(ds.latitude) * (fixed_width / len(ds.longitude))
        return (width, height)


    fig, ax = plt.subplots(figsize = figure_ratio(da,fixed_width = width))
    
    lat_formatter = FuncFormatter(lambda y_val, tick_pos: "{0:.3f}".format(da.latitude.values[tick_pos] ))
    lon_formatter = FuncFormatter(lambda x_val, tick_pos: "{0:.3f}".format(da.longitude.values[tick_pos]))

    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    
    plt.title("Threshold: {} < x < {}".format(min_threshold, max_threshold))
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    plt.imshow(array, *args, **kwargs)
    plt.axis('off')
    plt.show()

# GD duplicated in DCAL_Water_Extents ?    
# Threshold count function.
def threshold_count(da, min_threshold, max_threshold, mask = None):
    def count_not_nans(arr):
        return np.count_nonzero(~np.isnan(arr))
    
    in_threshold = np.logical_and( da.values > min_threshold, da.values < max_threshold)
    
    total_non_cloudy = count_not_nans(da.values) if mask is None else np.sum(mask.values)
    
    return dict(total = np.size(da.values),
                total_non_cloudy = total_non_cloudy,
                inside = np.nansum(in_threshold),
                outside = total_non_cloudy - np.nansum(in_threshold)
               )    

# Threshold percentage function.
def threshold_percentage(da, min_threshold, max_threshold, mask = None):
    counts = threshold_count(da, min_threshold, max_threshold, mask = mask)
    return dict(percent_inside_threshold = (counts["inside"]   / counts["total"]) * 100.0,
                percent_outside_threshold = (counts["outside"] / counts["total"]) * 100.0,
                percent_clouds = ( 100.0-counts["total_non_cloudy"] / counts["total"] * 100.0))    
    
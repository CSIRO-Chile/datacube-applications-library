# Code behind module for Water_Extent_Time_Series.ipynb

################################
##
## Import Statments
##
################################

# Import standard Python modules
import datacube
import sys
import numpy as np
import xarray as xr

# Import DCAL utilities containing function definitions used generally across DCAL
sys.path.append('..') 
sys.path.append('../DCAL_utils') 

# Import clean mask functions
from DCAL_utils.clean_mask import landsat_qa_clean_mask, landsat_clean_mask_invalid

# Import aggregate scale function
from DCAL_utils.aggregate import xr_scale_res

# Import sort function
from DCAL_utils.sort import xarray_sortby_coord

# Import DCAL Utils SPECIAL data cube load function
from DCAL_utils_special.dc_load import match_dim_sizes

################################
##
## Function Definitions
##
################################

def load_for_time_range(platforms, products, time_extents, dc, lon, lat, clear_px_thresh):
    measurements = ['red', 'blue', 'green', 'nir', 'swir1', 'swir2', 'pixel_qa']
    matching_abs_res, same_dim_sizes = match_dim_sizes(dc, products, lon, lat)
    datasets = {}
    clean_masks = {}
    for platform, product in zip(platforms, products):
        # Load the dataset.
        dataset = dc.load(platform=platform, product=product, lat=lat, lon=lon, 
                          time=time_extents, measurements=measurements)
        if len(dataset.dims) == 0: # The dataset is empty.
            continue
        datasets[product] = dataset
        # Get the clean mask.
        clean_mask = (landsat_qa_clean_mask(dataset, platform) &
                      ((dataset != -9999).to_array().all('variable')) &
                      landsat_clean_mask_invalid(dataset))\
                     .astype(np.uint8)
        dataset = dataset.drop('pixel_qa')    
        # Discard acquisitions with insufficient data.
        acq_times_to_keep = dataset.time.values[(clean_mask.mean(['latitude', 'longitude']) > clear_px_thresh).values]
        dataset = dataset.sel(time=acq_times_to_keep)
        clean_mask = clean_mask.sel(time=acq_times_to_keep)
        # If needed, scale the datasets and clean masks to the same size in the x and y dimensions.
        if not same_dim_sizes:    
            dataset = xr_scale_res(dataset, abs_res=matching_abs_res)
            clean_mask = xr_scale_res(clean_mask, abs_res=matching_abs_res)
        clean_mask = clean_mask.astype(np.bool)
        # Clean the data.
        dataset = dataset.astype(np.float16).where(clean_mask)
        datasets[product], clean_masks[product] = dataset, clean_mask
    # Combine everything.
    if len(datasets) > 0:
        dataset = xarray_sortby_coord(xr.concat(list(datasets.values()), dim='time'), coord='time')
        clean_mask = xarray_sortby_coord(xr.concat(list(clean_masks.values()), dim='time'), coord='time')
    else:
        dataset = xr.Dataset()
        clean_mask = xr.DataArray(np.empty((0,), dtype=np.bool))
    return dataset, clean_mask


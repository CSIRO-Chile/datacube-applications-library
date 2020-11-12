# Code behind module for DCAL_Cloud_Statistics.ipynb

################################
##
## Import Statments
##
################################

# Import standard Python modules
import sys
import datacube
import numpy as np
import pandas as pd

import ipywidgets as widgets
from ipywidgets import Button, Layout
from IPython.display import display
import functools

# Import DCAL utilities containing function definitions used generally across DCAL
sys.path.append('../DCAL_utils')

# Import Mosaic functions
from dc_mosaic import ls8_unpack_qa, ls7_unpack_qa

# Import clean mask functions
from clean_mask import landsat_qa_clean_mask, landsat_clean_mask_invalid


################################
##
## Function Definitions
##
################################

def build_cloud_coverage_table_landsat(product,
                                       platform,
                                       latitude,
                                       longitude,
                                       time     = None,
                                       dc       = None,
                                       extra_band = 'green'):
    dc = dc if dc is not None else datacube.Datacube(app = "")
    
    load_params = dict(platform=platform,
                       product=product,
                       latitude = latitude,
                       longitude = longitude,
                       measurements = [extra_band, 'pixel_qa'],
                       group_by='solar_day')
    
    if time is not None: 
        load_params["time"] = time
    
    landsat_dataset = dc.load(**load_params)
    clean_mask = landsat_qa_clean_mask(landsat_dataset, platform=platform) & \
                 (landsat_dataset != -9999).to_array().all('variable') & \
                 landsat_clean_mask_invalid(landsat_dataset)
    landsat_dataset = landsat_dataset.where(clean_mask)
    
    times = list(landsat_dataset.time.values)
    scene_slice_list = list(map(lambda t: landsat_dataset.sel(time = str(t)), times))
    
    clean_mask_list = [clean_mask.sel(time=str(time)).values for time in clean_mask.time.values]
    no_data_mask_list = list(map(lambda ds: (ds[extra_band]==-9999).values, scene_slice_list))
    # Calculate the percentage of all pixels which are not cloud.
    percentage_list = [clean_mask.mean()*100 for clean_mask in clean_mask_list]
    clean_pixel_count_list = list(map(np.sum, clean_mask_list))
    
    data = {"times": times,
            "clean_percentage": percentage_list,
            "clean_count": clean_pixel_count_list }
    
    return landsat_dataset, pd.DataFrame(data=data, columns = ["times", "clean_percentage", "clean_count"])
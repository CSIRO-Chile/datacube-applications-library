# Code behind module for DCAL_Vegetation_Phenology.ipynb

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
from dc_time import _n64_datetime_to_scalar, _scalar_to_n64_datetime


################################
##
## Function Definitions
##
################################

def TIMESAT_stats(dataarray, time_dim='time'):
    """
    For a 1D array of values for a vegetation index - for which higher values tend to 
    indicate more vegetation - determine several statistics:
    1. Beginning of Season (BOS): The time index of the beginning of the growing season.
        (The downward inflection point before the maximum vegetation index value)
    2. End of Season (EOS): The time index of the end of the growing season.
        (The upward inflection point after the maximum vegetation index value)
    3. Middle of Season (MOS): The time index of the maximum vegetation index value.
    4. Length of Season (EOS-BOS): The time length of the season (index difference).
    5. Base Value (BASE): The minimum vegetation index value.
    6. Max Value (MAX): The maximum vegetation index value (the value at MOS).
    7. Amplitude (AMP): The difference between BASE and MAX.
    
    Parameters
    ----------
    dataarray: xarray.DataArray
        The 1D array of non-NaN values to determine the statistics for.
    time_dim: string
        The name of the time dimension in `dataarray`.

    Returns
    -------
    stats: dict
        A dictionary mapping statistic names to values.
    """
    assert time_dim in dataarray.dims, "The parameter `time_dim` is \"{}\", " \
        "but that dimension does not exist in the data.".format(time_dim)
    stats = {}
    data_np_arr = dataarray.values
    time_np_arr = _n64_datetime_to_scalar(dataarray[time_dim].values)
    data_inds = np.arange(len(data_np_arr))
    
    # Obtain the first and second derivatives.
    fst_deriv = np.gradient(data_np_arr, time_np_arr)
    pos_fst_deriv = fst_deriv > 0
    neg_fst_deriv = 0 > fst_deriv
    snd_deriv = np.gradient(fst_deriv, time_np_arr)
    pos_snd_deriv = snd_deriv > 0
    neg_snd_deriv = 0 > snd_deriv
    
    # Determine MOS.
    # MOS is the index of the highest value.
    idxmos = np.argmax(data_np_arr)
    stats['Middle of Season'] = idxmos
    
    data_inds_before_mos = data_inds[:idxmos]
    data_inds_after_mos = data_inds[idxmos:]
    
    # Determine BOS.
    # BOS is the last negative inflection point before the MOS.
    # If that point does not exist, choose the first positive
    # first derivative point before the MOS. If that point does
    # not exist, the BOS is the MOS (there is no point before the MOS in this case).
    snd_deriv_neg_infl = np.concatenate((np.array([False]), neg_snd_deriv[1:] & ~neg_snd_deriv[:-1]))
    if snd_deriv_neg_infl[data_inds_before_mos].sum() > 0:
        idxbos = data_inds_before_mos[len(data_inds_before_mos) - 1 - 
                                      np.argmax(snd_deriv_neg_infl[data_inds_before_mos][::-1])]
    elif pos_fst_deriv[data_inds_before_mos].sum() > 0:
        idxbos = np.argmax(pos_fst_deriv[data_inds_before_mos])
    else:
        idxbos = idxmos
    stats['Beginning of Season'] = idxbos
    
    # Determine EOS.    
    # EOS is the first positive inflection point after the MOS.
    # If that point does not exist, choose the last negative
    # first derivative point after the MOS. If that point does
    # not exist, the EOS is the MOS (there is no point after the MOS in this case).
    snd_deriv_pos_infl = np.concatenate((np.array([False]), pos_snd_deriv[1:] & ~pos_snd_deriv[:-1]))
    if snd_deriv_pos_infl[data_inds_after_mos].sum() > 0:
        idxeos = data_inds_after_mos[np.argmax(snd_deriv_pos_infl[data_inds_after_mos])]
    elif neg_fst_deriv[data_inds_after_mos].sum() > 0:
        idxeos = np.argmax(neg_fst_deriv[data_inds_after_mos])
    else:
        idxeos = idxmos
    stats['End of Season'] = idxeos
    
    # Determine EOS-BOS.
    stats['Length of Season'] = idxeos - idxbos
    # Determine BASE.
    stats['Base Value'] = data_np_arr.min()
    # Determine MAX.
    stats['Max Value'] = data_np_arr.max()
    # Determine AMP.
    stats['Amplitude'] = stats['Max Value'] - stats['Base Value']
    
    return stats




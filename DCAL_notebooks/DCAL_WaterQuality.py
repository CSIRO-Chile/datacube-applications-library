# Code behind module for DCAL_WaterQuality.ipynb

################################
##
## Import Statments
##
################################

# Import standard Python modules
import sys
import datacube

# Import DCAL utilities containing function definitions used generally across DCAL
sys.path.append('../DCAL_utils')


################################
##
## Function Definitions
##
################################

# Define LYM7 function
def LYM7(dataset):
        return (3983 * ((dataset.green + dataset.red)*0.0001/2)**1.6246)

# Define LYM8 function
def LYM8(dataset):
        return (3957 * ((dataset.green + dataset.red)*0.0001/2)**1.6436)

# Define SPM function
def SPM_QIU(dataset):
        return (10**(2.26*(dataset.red/dataset.green)**3 - 
                     5.42*(dataset.red/dataset.green)**2 +
                     5.58*(dataset.red/dataset.green) - 0.72) - 1.43)
    
# Define NDSSI function
def NDSSI(dataset):
        return ((dataset.blue-dataset.nir)/(dataset.blue+dataset.nir))
    
# Define QUANG8 function
def QUANG8(dataset):
        return (380.32 * (dataset.red)*0.0001 - 1.7826)
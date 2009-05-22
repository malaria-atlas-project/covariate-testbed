# Author: Anand Patil
# Date: 6 Feb 2009
# License: Creative Commons BY-NC-SA
####################################


import numpy as np
import pymc as pm
from st_cov_fun import my_st
import gc

__all__ = ['transform_bin_data']

def transform_bin_data(pos, neg):
    return pm.logit((pos+1.)/(pos+neg+2.))
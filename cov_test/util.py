# Author: Anand Patil
# Date: 6 Feb 2009
# License: Creative Commons BY-NC-SA
####################################


import numpy as np
import pymc as pm
from st_cov_fun import my_st
import gc
from map_utils import combine_st_inputs as combine_input_data

__all__ = ['transform_bin_data', 'st_mean_comp', 'my_st']

def transform_bin_data(pos, neg):
    return pm.logit((pos+1.)/(pos+neg+2.))

def st_mean_comp(x, m_const, t_coef):
    lon = x[:,0]
    lat = x[:,1]
    t = x[:,2]
    return m_const + t_coef * t
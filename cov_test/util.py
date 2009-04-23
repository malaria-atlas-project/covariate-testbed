# Author: Anand Patil
# Date: 6 Feb 2009
# License: Creative Commons BY-NC-SA
####################################


import numpy as np
import pymc as pm
from st_cov_fun import my_st
import gc

__all__ = ['transform_bin_data', 'st_mean_comp', 'combine_input_data', 'my_st']

def transform_bin_data(pos, neg):
    return pm.logit((pos+1.)/(pos+neg+2.))

def st_mean_comp(x, m_const, t_coef):
    lon = x[:,0]
    lat = x[:,1]
    t = x[:,2]
    return m_const + t_coef * t

def combine_input_data(lon,lat,t):
    # Convert latitude and longitude from degrees to radians.
    lon = lon*np.pi/180.
    lat = lat*np.pi/180.

    # Convert time to end year - 2009 (no sense forcing mu to adjust by too much).
    t = t - 2009
    
    # Make lon, lat, t triples.
    data_mesh = np.vstack((lon, lat, t)).T 
    return data_mesh

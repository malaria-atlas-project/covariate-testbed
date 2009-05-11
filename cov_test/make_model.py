# Author: Anand Patil
# Date: 6 Feb 2009
# License: Creative Commons BY-NC-SA
####################################

import numpy as np
import pymc as pm
from st_cov_fun import my_st
from util import *
import gc
from map_utils import basic_st_submodel

__all__ = ['make_model','f_name','x_name','nugget_name','f_has_nugget','metadata_keys']

def make_model(pos,neg,lon,lat,t,covariate_values,cpus=1,lockdown=False):
    """
    pos : Number positive
    neg : Number negative
    lon : longitude
    lat : latitude
    t : time
    covariate_values : {'ndvi': array, 'rainfall': array} etc.
    cpus : int
    """

    # =====================
    # = Create PyMC model =
    # =====================    
    
    # V_shift = pm.Exponential('V_shift',.1,value=1.)
    # V = V_shift + .1
    # V.__name__ = 'V'
    # V.trace=True
    
    V = pm.Exponential('V',.1,value=1.)
    
    init_OK = False
    while not init_OK:
        # Create covariance and MV-normal F if model is spatial.   
        try:
            st_sub = basic_st_submodel(lon, lat, t, covariate_values, cpus)        

            # The evaluation of the Covariance object, plus the nugget.
            @pm.deterministic
            def nug_C_eval(C_eval = st_sub['C_eval'], V=V):
                """nug_C_eval = function(C_eval, V)"""
                return C_eval + V*np.eye(len(lon))
                                            
            # The field evaluated at the uniquified data locations
            data = pm.MvNormalCov('data',st_sub['M_eval'],nug_C_eval,value=transform_bin_data(pos,neg),observed=True)

            init_OK = True
        except pm.ZeroProbability, msg:
            print 'Trying again: %s'%msg
            init_OK = False
            gc.collect()
            
    out = locals()
    out.update(st_sub)
    return out
    
f_name = 'data'
x_name = 'logp_mesh'
f_has_nugget = True
nugget_name = 'V'
metadata_keys = []
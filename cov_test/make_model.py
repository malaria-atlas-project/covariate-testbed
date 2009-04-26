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

__all__ = ['make_model']
    
def make_model(d,lon,lat,t,covariate_values,cpus=1,lockdown=False):
    """
    d : transformed ('gaussian-ish') data
    lon : longitude
    lat : latitude
    t : time
    covariate_values : {'ndvi': array, 'rainfall': array} etc.
    cpus : int
    """

    # =====================
    # = Create PyMC model =
    # =====================    
    # log_V = pm.Uninformative('log_V', value=0)
    # V = pm.Lambda('V', lambda lv = log_V: np.exp(lv))        
    V = pm.Exponential('V',.1,value=1.)
    
    init_OK = False
    while not init_OK:
        # Create covariance and MV-normal F if model is spatial.   
        try:
            st_sub = basic_st_submodel(lon, lat, t, covariate_values, cpus)        

            # The evaluation of the Covariance object, plus the nugget.
            @pm.deterministic(trace=False)
            def S_eval(C_eval=st_sub['C_eval'], V=V):
                out = C_eval
                out += V*np.eye(len(lon))
                try:
                    return np.linalg.cholesky(out)
                except np.linalg.LinAlgError:
                    return None
                    
            @pm.potential
            def check_pd(s=S_eval):
                if s is None:
                    return -np.inf
                else:
                    return 0.
                                            
            # The field evaluated at the uniquified data locations
            data = pm.MvNormalChol('f',st_sub['M_eval'],S_eval,value=d,observed=True)

            init_OK = True
        except pm.ZeroProbability, msg:
            print 'Trying again: %s'%msg
            init_OK = False
            gc.collect()

    return locals()
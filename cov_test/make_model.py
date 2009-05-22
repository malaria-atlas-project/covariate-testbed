# Author: Anand Patil
# Date: 6 Feb 2009
# License: Creative Commons BY-NC-SA
####################################

import numpy as np
import pymc as pm
from st_cov_fun import my_st
from util import *
import gc
from generic_mbg import *

__all__ = ['make_model','f_name','x_name','nugget_name','f_has_nugget','metadata_keys','postproc']

def make_model(pos,neg,lon,lat,t,covariate_values,cpus=1):
    """
    pos : Number positive
    neg : Number negative
    lon : longitude
    lat : latitude
    t : time
    covariate_values : {'ndvi': array, 'rainfall': array} etc.
    cpus : int
    """
    logp_mesh = combine_st_inputs(lon,lat,t)
    data_mesh = logp_mesh

    # =====================
    # = Create PyMC model =
    # =====================
    
    init_OK = False
    while not init_OK:
        # Create covariance and MV-normal F if model is spatial.   
        try:
            V = pm.Exponential('V',1,value=.01,observed=True)
            

            inc = pm.CircVonMises('inc', 0,0)
            sqrt_ecc = pm.Uniform('sqrt_ecc',0,.95)
            ecc = pm.Lambda('ecc', lambda s=sqrt_ecc: s**2)
            amp = pm.Exponential('amp',.1,value=1.)
            scale = pm.Exponential('scale',1.,value=1.)
            scale_t = pm.Exponential('scale_t',.1,value=.1)
            t_lim_corr = pm.Uniform('t_lim_corr',0,.95)

            @pm.stochastic(__class__ = pm.CircularStochastic, lo=0, hi=1)
            def sin_frac(value=.1):
                return 0.

            M, M_eval = trivial_means(logp_mesh)

            # A constraint on the space-time covariance parameters that ensures temporal correlations are 
            # always between -1 and 1.
            @pm.potential
            def st_constraint(sd=.5, sf=sin_frac, tlc=t_lim_corr):    
                if -sd >= 1./(-sf*(1-tlc)+tlc):
                    return -np.Inf
                else:
                    return 0.

            # A Deterministic valued as a Covariance object. Uses covariance my_st, defined above. 
            @pm.deterministic
            def C(amp=amp,scale=scale,inc=inc,ecc=ecc,scale_t=scale_t, t_lim_corr=t_lim_corr, sin_frac=sin_frac):
                return pm.gp.FullRankCovariance(my_st, amp=amp, scale=scale, inc=inc, ecc=ecc,st=scale_t, sd=.5,
                                                tlc=t_lim_corr, sf = sin_frac, n_threads=cpus)

            covariate_dict, C_eval = cd_and_C_eval(covariate_values, C, logp_mesh)        

            # The evaluation of the Covariance object, plus the nugget.
            @pm.deterministic(trace=False)
            def nug_C_eval(C_eval = C_eval, V=V):
                """nug_C_eval = function(C_eval, V)"""
                return C_eval + V*np.eye(len(lon))
                                            
            # The field evaluated at the uniquified data locations
            data = pm.MvNormalCov('data',M_eval,nug_C_eval,value=transform_bin_data(pos,neg),observed=True)
            data_val = data.value

            init_OK = True
        except pm.ZeroProbability, msg:
            print 'Trying again: %s'%msg
            init_OK = False
            gc.collect()
            
    return locals()
    
f_name = 'data_val'
x_name = 'logp_mesh'
f_has_nugget = True
nugget_name = 'V'
postproc = invlogit
metadata_keys = ['data_val']
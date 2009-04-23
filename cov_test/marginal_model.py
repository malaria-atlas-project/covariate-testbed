# Author: Anand Patil
# Date: 6 Feb 2009
# License: Creative Commons BY-NC-SA
####################################


import numpy as np
import pymc as pm
from st_cov_fun import my_st
from util import *
import gc

__all__ = ['make_model']
    
def make_model(d,lon,lat,t,covariate_values,cpus=1,prior_var=np.inf):
    """
    d : transformed ('gaussian-ish') data
    lon : longitude
    lat : latitude
    t : time
    covariate_values : {'ndvi': array, 'rainfall': array} etc.
    cpus : int
    """
        
    logp_mesh = combine_input_data(lon,lat,t)
    covariate_names = covariate_values.keys()

    u = np.asmatrix(np.empty((len(covariate_names)+2, logp_mesh.shape[0])))
    u[0,:] = 1.
    u[2,:] = logp_mesh[:,2]
    for i in xrange(len(covariate_names)):
        u[i+2,:] = covariate_values[covariate_names[i]]
    
    # =====================
    # = Create PyMC model =
    # =====================    
    init_OK = False
    while not init_OK:    

        # log_V = pm.Uninformative('log_V', value=0)
        # V = pm.Lambda('V', lambda lv = log_V: np.exp(lv))        
        V = pm.Exponential('V',.1,value=1.)

        inc = pm.CircVonMises('inc', 0,0)

        @pm.stochastic(__class__ = pm.CircularStochastic, lo=0, hi=1)
        def sqrt_ecc(value=.1):
            return 0.
        ecc = pm.Lambda('ecc', lambda s=sqrt_ecc: s**2)

        # log_amp = pm.Uninformative('log_amp', value=0)
        # amp = pm.Lambda('amp', lambda la = log_amp: np.exp(la))
        amp = pm.Exponential('amp',.1,value=1.)

        # log_scale = pm.Uninformative('log_scale', value=0)
        # scale = pm.Lambda('scale', lambda ls = log_scale: np.exp(ls))
        scale = pm.Exponential('scale',.1,value=1.)

        # log_scale_t = pm.Uninformative('log_scale_t', value=0)
        # scale_t = pm.Lambda('scale_t', lambda ls = log_scale_t: np.exp(ls))
        scale_t = pm.Exponential('scale_t',.1,value=.1)
        
        @pm.stochastic(__class__ = pm.CircularStochastic, lo=0, hi=1)
        def t_lim_corr(value=.2):
            return 0.

        @pm.stochastic(__class__ = pm.CircularStochastic, lo=0, hi=1)
        def sin_frac(value=.1):
            return 0.
            
        # Create covariance and MV-normal F if model is spatial.   
        try:
            # A constraint on the space-time covariance parameters that ensures temporal correlations are 
            # always between -1 and 1.
            @pm.potential
            def st_constraint(sd=.5, sf=sin_frac, tlc=t_lim_corr):    
                if -sd >= 1./(-sf*(1-tlc)+tlc):
                    return -np.Inf
                else:
                    return 0.

            # A Deterministic valued as a Covariance object. Uses covariance my_st, defined above. 
            @pm.deterministic(trace=True)
            def C(amp=amp,scale=scale,inc=inc,ecc=ecc,scale_t=scale_t, t_lim_corr=t_lim_corr, sin_frac=sin_frac):
                return pm.gp.FullRankCovariance(my_st, amp=amp, scale=scale, inc=inc, ecc=ecc,st=scale_t, sd=.5,
                                                tlc=t_lim_corr, sf = sin_frac, n_threads=cpus)
            
            # The evaluation of the Covariance object, plus the nugget.
            @pm.deterministic(trace=False)
            def S_eval(C=C, V=V, u=u):
                out = np.asarray(C(logp_mesh, logp_mesh))
                out += V*np.eye(logp_mesh.shape[0])
                out += u.T*u
                if np.any(np.isnan(out)):
                    return None
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
            data = pm.MvNormalChol('f',np.zeros(logp_mesh.shape[0]),S_eval,value=d,observed=True)
            

            init_OK = True
        except pm.ZeroProbability, msg:
            print 'Trying again: %s'%msg
            init_OK = False
            gc.collect()

    return locals()
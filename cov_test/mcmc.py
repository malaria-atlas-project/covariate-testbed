# Author: Anand Patil
# Date: 6 Feb 2009
# License: Creative Commons BY-NC-SA
####################################


import numpy as np
import pymc as pm
from make_model import make_model, transform_bin_data, CovariateStepper

__all__ = ['MCMC_obj']        
        
def MCMC_obj(pos,neg,lon,lat,t,cv,cpus,dbname=None,lockdown=False,**kwds):
    """
    Creates an MCMC object around the model for transformed PR at sampled locations.
    """
    d=transform_bin_data(pos,neg)
    while True:
        print 'Trying to create model'
        try:
            if dbname is not None:
                M = pm.MCMC(make_model(d,lon,lat,t,cv,cpus,lockdown), db='hdf5', dbname=dbname, dbcomplevel=1, dbcomplib='zlib')
            else:
                M = pm.MCMC(make_model(d,lon,lat,t,cv,cpus,lockdown))
            break
        except np.linalg.LinAlgError:
            pass
    M.db._h5file.createArray('/','lon',lon)
    M.db._h5file.createArray('/','lat',lat)
    M.db._h5file.createArray('/','t',t)
    M.db._h5file.createArray('/','data',d)
    for name, val in cv.iteritems():
        M.db._h5file.createArray('/',name+'_value',val)
        
    # Special Gibbs step method for covariates
    M.use_step_method(CovariateStepper, M.covariate_dict, M.m_const, t-2009, M.t_coef, M.M_eval, M.S_eval, M.data)
    # mean_params = [v[0] for v in M.covariate_dict.values()] + [M.m_const, M.t_coef]
    # M.use_step_method(pm.AdaptiveMetropolis, mean_params, scales=dict(zip(mean_params, [.001]*len(mean_params))), **kwds)

    if lockdown:
        cov_params = [M.sqrt_ecc, M.inc]
    else:
        cov_params = [M.V, M.sqrt_ecc, M.amp, M.scale, M.scale_t, M.t_lim_corr, M.inc]
    M.use_step_method(pm.AdaptiveMetropolis, cov_params, scales=dict(zip(cov_params, [.001]*len(cov_params))), **kwds)

    S = M.step_method_dict[M.m_const][0]

    return M, S

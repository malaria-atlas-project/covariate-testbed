# Author: Anand Patil
# Date: 6 Feb 2009
# License: Creative Commons BY-NC-SA
####################################


import numpy as np
import pymc as pm
from make_model import make_model, transform_bin_data
from map_utils import add_standard_metadata

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
    
    add_standard_metadata(M, M.logp_mesh, M.covariate_dict, lon=lon, lat=lat, t=t, data=M.data.value)
    
    cov_params = list(M.stochastics)
    M.use_step_method(pm.AdaptiveMetropolis, cov_params, scales=dict(zip(cov_params, [.001]*len(cov_params))), **kwds)
    S = M.step_method_dict[M.inc][0]

    return M,S
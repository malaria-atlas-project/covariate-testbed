# Author: Anand Patil
# Date: 6 Feb 2009
# License: Creative Commons BY-NC-SA
####################################


import numpy as np
import pymc as pm
from make_model import make_model, transform_bin_data

__all__ = ['MCMC_obj']

class CovariateStepper(pm.StepMethod):
    
    def __init__(self, covariate_dict, m_const, t, t_coef, M_eval, sig, d):
        self.m_const = m_const
        self.t_coef=t_coef
        self.M = M_eval
        self.sig = sig
        self.d = d.value
        
        cvv = covariate_dict.values()
        self.beta = pm.Container([self.m_const, self.t_coef]+[v[0] for v in cvv])
        self.x = np.vstack((np.ones((1,len(t))), np.atleast_2d(t), np.asarray([v[1] for v in cvv])))
    
        pm.StepMethod.__init__(self, self.beta)
    
    def step(self):

        pri_sig = np.asarray(self.sig.value)
        lo = pm.gp.trisolve(pri_sig, self.x.T, uplo='L').T
        post_tau = np.dot(lo,lo.T)
        l = np.linalg.cholesky(post_tau)
        
        post_C = pm.gp.trisolve(l, np.eye(l.shape[0]),uplo='L')
        post_C = pm.gp.trisolve(l.T, post_C, uplo='U')
        
        post_mean = np.dot(lo, pm.gp.trisolve(pri_sig, self.d, uplo='L'))
        post_mean = pm.gp.trisolve(l, post_mean, uplo='L')
        post_mean = pm.gp.trisolve(l.T, post_mean, uplo='U')
        
        new_val = pm.rmv_normal_cov(post_mean, post_C).squeeze()
        
        [b.set_value(nv) for (b,nv) in zip(self.beta, new_val)]
        
        
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

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
        
        self.beta = pm.Container([self.m_const, self.t_coef]+[v[0] for v in covariate_dict.values()])
        self.x = np.vstack((np.ones((1,len(t))), np.atleast_2d(t), np.asmatrix([v[1] for v in covariate_dict.values()])))
    
        pm.StepMethod.__init__(self, self.beta)
    
    def step(self):
        post_tau_sig = pm.gp.trisolve(self.sig.value.T, self.x.T, uplo='U').T
        x_tau = pm.gp.trisolve(self.sig.value, post_tau_sig.T, uplo='L').T
        post_tau = np.dot(post_tau_sig, post_tau_sig.T)
        post_tau_sig = np.linalg.cholesky(post_tau)
        
        post_mean = np.dot(np.dot(post_tau, x_tau), self.d)
        new_val = np.asarray(np.dot(post_tau_sig, np.random.normal(size=self.x.shape[0])) + post_mean).squeeze()
        
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
    # M.use_step_method(CovariateStepper, M.covariate_dict, M.m_const, t, M.t_coef, M.M_eval, M.sig, M.data)
    # Adaptive Metropolis step method for covariance parameters
    M.use_step_method(pm.AdaptiveMetropolis, list(M.stochastics), scales=dict(zip(M.stochastics, [.001]*len(M.stochastics))), **kwds)
    # if lockdown:
    #     M.use_step_method(pm.AdaptiveMetropolis, [M.sqrt_ecc, M.inc], **kwds)
    # else:
    #     M.use_step_method(pm.AdaptiveMetropolis, [M.tau, M.sqrt_ecc, M.amp, M.scale, M.scale_t, M.t_lim_corr, M.inc], **kwds)
    S = M.step_method_dict[M.m_const][0]

    return M, S

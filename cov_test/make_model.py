# Author: Anand Patil
# Date: 6 Feb 2009
# License: Creative Commons BY-NC-SA
####################################


import numpy as np
import pymc as pm
from st_cov_fun import my_st
import gc

__all__ = ['transform_bin_data','make_model','combine_input_data','st_mean_comp']

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
    
def make_model(d,lon,lat,t,covariate_values,cpus=1,lockdown=False):
    """
    d : transformed ('gaussian-ish') data
    lon : longitude
    lat : latitude
    t : time
    covariate_values : {'ndvi': array, 'rainfall': array} etc.
    cpus : int
    """
        
    logp_mesh = combine_input_data(lon,lat,t)

    # =====================
    # = Create PyMC model =
    # =====================    
    init_OK = False
    while not init_OK:
                
        # Make coefficients for the covariates.
        m_const = pm.Uninformative('m_const', value=0.)
        t_coef = pm.Uninformative('t_coef',value=.1)        
        covariate_dict = {}
        for cname, cval in covariate_values.iteritems():
            this_coef = pm.Uninformative(cname + '_coef', value=0.)
            covariate_dict[cname] = (this_coef, cval)

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
        ecc = pm.Lambda('ecc', lambda s=sqrt_ecc: s**2)

        @pm.stochastic(__class__ = pm.CircularStochastic, lo=0, hi=1)
        def sin_frac(value=.1):
            return 0.
        
        if lockdown:
            for p in [V, amp, scale, scale_t, t_lim_corr, sin_frac]:
                p._observed=True
    
        # The mean of the field
        @pm.deterministic(trace=True)
        def M(mc=m_const, tc=t_coef):
            return pm.gp.Mean(st_mean_comp, m_const = mc, t_coef = tc)
        
        # The mean, evaluated  at the observation points, plus the covariates    
        @pm.deterministic(trace=False)
        def M_eval(M=M, lpm=logp_mesh, cv=covariate_dict):
            out = M(lpm)
            for c in cv.itervalues():
                out += c[0]*c[1]
            return out

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
            def S_eval(C=C, V=V):
                out = np.asarray(C(logp_mesh, logp_mesh))
                out += V*np.eye(logp_mesh.shape[0])
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
            data = pm.MvNormalChol('f',M_eval,S_eval,value=d,observed=True)

            init_OK = True
        except pm.ZeroProbability, msg:
            print 'Trying again: %s'%msg
            init_OK = False
            gc.collect()

    return locals()
    
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

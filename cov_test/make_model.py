import numpy as np
import pymc as pm
from st_cov_fun import my_st
import gc

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
        

def st_mean_comp(x, m_const, t_coef):
    lon = x[:,0]
    lat = x[:,1]
    t = x[:,2]
    return m_const + t_coef * t


def create_model(d,lon,lat,t,covariate_values,cpus=1):
    """
    d : transformed ('gaussian-ish') data
    lon : longitude
    lat : latitude
    t : time
    covariate_values : {'ndvi': array, 'rainfall': array} etc.
    cpus : int
    """
        
    # Convert latitude and longitude from degrees to radians.
    lon = lon*np.pi/180.
    lat = lat*np.pi/180.

    # Convert time to end year - 2009 (no sense forcing mu to adjust by too much).
    t = t - 2009
    
    # Make lon, lat, t triples.
    data_mesh = np.vstack((lon, lat, t)).T 
    logp_mesh = data_mesh  

    # =====================
    # = Create PyMC model =
    # =====================    
    init_OK = False
    while not init_OK:
                
        # Make coefficients for the covariates.
        m_const = pm.Uninformative('m_const', value=-3.)
        t_coef = pm.Uninformative('t_coef',value=.1)        
        covariate_dict = {}
        for cname, cval in covariate_values.iteritems():
            this_coef = pm.Uninformative(cname + '_coef', value=0.)
            covariate_dict[cname] = (this_coef, cval)

        tau = pm.Gamma('tau', value=2., alpha=.001, beta=.001/.25)
        V = pm.Lambda('V', lambda tau=tau:1./tau)        
        nonmod_inc = pm.Uninformative('nonmod_inc', value=.5)
        inc = pm.Lambda('inc', lambda nonmod_inc = nonmod_inc: nonmod_inc % np.pi)
        sqrt_ecc = pm.Uniform('sqrt_ecc', value=.1, lower=0., upper=1.)
        ecc = pm.Lambda('ecc', lambda s=sqrt_ecc: s**2)
        amp = pm.Exponential('amp',.1)
        scale = pm.Exponential('scale',.1)        
        scale_t = pm.Exponential('scale_t', .1)
        t_lim_corr = pm.Uniform('t_lim_corr',0,1,value=.8)
        sin_frac = pm.Uniform('sin_frac',0,1)
    
        # The mean of the field
        @pm.deterministic
        def M(m=m_const, tc=t_coef):
            return pm.gp.Mean(st_mean_comp, m_const = m, t_coef = tc)
        
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
            @pm.deterministic
            def C(amp=amp,scale=scale,inc=inc,ecc=ecc,scale_t=scale_t, t_lim_corr=t_lim_corr, sin_frac=sin_frac):
                return pm.gp.FullRankCovariance(my_st, amp=amp, scale=scale, inc=inc, ecc=ecc,st=scale_t, sd=.5,
                                                tlc=t_lim_corr, sf = sin_frac, n_threads=cpus)

            # The evaluation of the Covariance object, plus the nugget.
            @pm.deterministic(trace=False)
            def C_eval(C=C, V=V):
                nug = np.atleast_2d(V*np.ones(logp_mesh.shape[0]))
                out = C(logp_mesh, logp_mesh)
                out.ravel()[0,::logp_mesh.shape[0]+1]+=nug
                return  out
                
            @pm.deterministic(trace=False)
            def sig(c=C_eval):
                try:
                    return np.linalg.cholesky(c)
                except np.linalg.LinAlgError:
                    return None
            
            # FIXME: You need to make PyMC give ZeroProbabilities priority over any other errors encountered during a computation.    
            @pm.potential
            def posdef_check(sig=sig):
                if sig is None:
                    return -np.inf
                else:
                    return 0
            
            # The field evaluated at the uniquified data locations
            data = pm.MvNormalChol('f',M_eval,sig,value=d, observed=True)

            init_OK = True
        except pm.ZeroProbability, msg:
            print 'Trying again: %s'%msg
            init_OK = False
            gc.collect()

    return locals()

def MCMC_obj(d,lon,lat,t,cv,cpus,dbname=None,**kwds):
    while True:
        print 'Trying to create model'
        try:
            if dbname is not None:
                M = pm.MCMC(create_model(d,lon,lat,t,cv,cpus), db='hdf5', dbname=dbname, dbcomplevel=1, dbcomplib='zlib')
            else:
                M = pm.MCMC(create_model(d,lon,lat,t,cv,cpus))
            break
        except np.linalg.LinAlgError:
            pass
    # Special Gibbs step method for covariates
    M.use_step_method(CovariateStepper, M.covariate_dict, M.m_const, t, M.t_coef, M.M_eval, M.sig, M.data)
    # Adaptive Metropolis step method for covariance parameters
    M.use_step_method(pm.AdaptiveMetropolis, [M.tau, M.sqrt_ecc, M.amp, M.scale, M.scale_t, M.t_lim_corr], **kwds)
    S = M.step_method_dict[M.m_const][0]
    
    return M, S
    

if __name__ == '__main__':
    # create_model(d,lon,lat,t,covariate_values)
    N=50
    names = ['rain','temp','ndvi']
    d=np.random.normal(size=N)
    lon=np.random.normal(size=N)
    lat=np.random.normal(size=N)
    t=np.random.normal(size=N)
    cv = {}
    for name in names:
        cv[name] = np.random.normal(size=N)
    M, S = MCMC_obj(d,lon,lat,t,cv,8,'trial')
    M.isample(10000,0,10)
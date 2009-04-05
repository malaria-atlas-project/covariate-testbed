import tables as tb
import pymc as pm
import numpy as np
from make_model import *

__all__ = ['trace_to_prediction']

def trace_to_prediction(tracefile, lon, lat, t, covariate_values, ntot, burn=0):
    """
    Converts a previously-produced HDF5 archive to samples from the predictive distribution
    of PR at unsampled locations.
    """
    hf = tb.openFile(tracefile)
    tr = hf.root.chain0.PyMCsamples.cols
    ntr = len(hf.root.chain0.PyMCsamples)
    
    if ntr<=burn:
        raise ValueError, 'Burnin of %i requested but length of trace is %i'%(burn, ntr)
    
    nper = np.zeros(ntr,dtype=int)
    nper[burn:] = pm.rmultinomial(ntot,np.ones(ntr-burn)/(ntr-burn))
    
    pred_mesh = combine_input_data(lon,lat,t)
    data_mesh = combine_input_data(hf.root.lon[:],hf.root.lat[:],hf.root.t[:])
    data = hf.root.data[:]
    
    samps = np.empty((ntot, len(t)))
    sofar = 0
    
    for i in xrange(burn, ntr):
        M=hf.root.chain0.group0.M[i]
        C=hf.root.chain0.group0.C[i]
        C = pm.gp.NearlyFullRankCovariance(C.eval_fun, **C.params)
        V=tr.V[i]
        
        pc = np.zeros(pred_mesh.shape[0])
        dc = np.zeros(data_mesh.shape[0])        
        for name, val in covariate_values.iteritems():
            coef = getattr(tr,name+'_coef')[i]
            pc += val * coef
            dc += coef * getattr(hf.root, name+'_value')[:]        
        
        pm.gp.observe(M,C,data_mesh,data-dc,V)
        
        for j in xrange(nper[i]):
            samp = pm.gp.Realization(M,C)(pred_mesh)
            samp += np.random.normal(size=pred_mesh.shape[0])*np.sqrt(V)
            samp += pc
            samps[sofar,:] = pm.flib.invlogit(samp)
            sofar += 1
            
    return samps
            
        
    
if __name__ == '__main__':
    
    
    N=20
    names = ['rain','temp','ndvi']
    lon=np.random.normal(size=N)
    lat=np.random.normal(size=N)
    t=np.random.normal(size=N)
    
    cv = {}
    for name in names:
        cv[name] = np.random.normal(size=N)
    
    s=trace_to_prediction('trial',lon,lat,t,cv,2000,20)
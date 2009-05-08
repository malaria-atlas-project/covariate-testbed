import sys, os
import numpy as np
import pymc as pm
import pylab as pl
from make_model import st_mean_comp, my_st
n_data = 900
n_pred = 100

# make_model(d,lon,lat,t,covariate_values)
names = ['rain','temp','ndvi']
vals = {'rain': .2,
        'temp': -.2,
        'ndvi': .4}

mc = -.4
tc = .6
V = .1

lon=np.random.normal(size=n_data+n_pred)
lat=np.random.normal(size=n_data+n_pred)
t=np.random.normal(size=n_data+n_pred)+2009

cv = {}
for name in names:
    cv[name] = np.random.normal(size=n_data+n_pred)#np.ones(n_data)
    
M = pm.gp.Mean(st_mean_comp, m_const = mc, t_coef = tc)
C = pm.gp.FullRankCovariance(my_st, amp=1, scale=1, inc=np.pi/4, ecc=.9,st=.1, sd=.5, tlc=.2, sf = .1)

dm = np.vstack((lon,lat,t)).T

M_eval = M(dm)
C_eval = C(dm,dm)

f = pm.rmv_normal_cov(M_eval, C_eval) + np.random.normal(size=n_data+n_pred)*np.sqrt(V)
f += np.sum([cv[name]*vals[name] for name in names],axis=0)
p = pm.flib.invlogit(f)
ns = 100
pos = pm.rbinomial(ns, p)
neg = ns - pos

ra_data = np.rec.fromarrays((pos[:n_data], neg[:n_data], lon[:n_data], lat[:n_data], t[:n_data]) + tuple([cv[name][:n_data] for name in names]), names=['pos','neg','lon','lat','t']+names)
pl.rec2csv(ra_data,'test_data.csv')

ra_pred = np.rec.fromarrays((pos[n_pred:], neg[n_pred:], lon[n_pred:], lat[n_pred:], t[n_pred:]) + tuple([cv[name][n_pred:] for name in names]), names=['pos','neg','lon','lat','t']+names)
pl.rec2csv(ra_pred,'test_pred.csv')


os.system('cov-test-infer test_data.csv 10000 10 2 test')
os.system('cov-test-predict test test_pred.csv 1000 100')

# ra_data = pl.csv2rec('test_data.csv')
# ra_pred = pl.csv2rec('test_pred.csv')
samps = np.fromfile('test_samps.csv',sep=',').reshape((n_pred,-1))

pos_pred = pos[n_data:]
neg_pred = neg[n_data:]
p_pred = (pos_pred+1.)/(pos_pred+neg_pred+2.)

quants = np.empty(samps.shape[0])
for i in xrange(samps.shape[0]):
    quants[i] = np.sum(samps[i,:]<p_pred[i])/float(samps.shape[1])
    
quants.tofile('test_quants.csv',sep=',')
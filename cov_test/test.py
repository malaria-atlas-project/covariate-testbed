import sys, os
import numpy as np
import pymc as pm
import pylab as pl
from make_model import st_mean_comp, my_st
n_data = 90
n_pred = 10

on = 1

names = ['rain','temp','ndvi','m','t']
vals = {'rain': .2,
        'temp': -.2,
        'ndvi': .4,
        'm' : 1,
        't' : -.5}
# names = ['m','t']
# vals = {'m':3,'t':0}
V = .01

lon=np.random.normal(size=n_data+n_pred)
lat=np.random.normal(size=n_data+n_pred)
t=np.random.normal(size=n_data+n_pred)

cv = {}
if len(names)>2:
    for name in names[:-2]:
        cv[name] = np.random.normal(size=n_data+n_pred)*on#np.ones(n_data)
cv['m'] = np.ones(n_data+n_pred)*on
cv['t'] = t*on
    
C = pm.gp.FullRankCovariance(my_st, amp=1, scale=1, inc=np.pi/4, ecc=.3,st=.1, sd=.5, tlc=.2, sf = .1)

dm = np.vstack((lon,lat,t)).T

C_eval = C(dm,dm)

f = pm.rmv_normal_cov(np.sum([cv[name]*vals[name] for name in names],axis=0), C_eval) + np.random.normal(size=n_data+n_pred)*np.sqrt(V)
p = pm.flib.invlogit(f)
ns = 100
pos = pm.rbinomial(ns, p)
neg = ns - pos

print p

ra_data = np.rec.fromarrays((pos[:n_data], neg[:n_data], lon[:n_data], lat[:n_data]) + tuple([cv[name][:n_data] for name in names]), names=['pos','neg','lon','lat']+names)
pl.rec2csv(ra_data,'test_data.csv')

ra_pred = np.rec.fromarrays((pos[n_data:], neg[n_data:], lon[n_data:], lat[n_data:]) + tuple([cv[name][n_data:] for name in names]), names=['pos','neg','lon','lat']+names)
pl.rec2csv(ra_pred,'test_pred.csv')

os.system('infer cov_test test_db test_data.csv -t 10 -n 8 -i 100000')
# os.system('cov-test-predict test test_pred.csv 1000 100')
# 
# # ra_data = pl.csv2rec('test_data.csv')
# # ra_pred = pl.csv2rec('test_pred.csv')
# samps = np.fromfile('test_samps.csv',sep=',').reshape((n_pred,-1))
# 
# pos_pred = pos[n_data:]
# neg_pred = neg[n_data:]
# p_pred = (pos_pred+1.)/(pos_pred+neg_pred+2.)
# 
# quants = np.empty(samps.shape[0])
# for i in xrange(samps.shape[0]):
#     quants[i] = np.sum(samps[i,:]<p_pred[i])/float(samps.shape[1])
#     
# quants.tofile('test_quants.csv',sep=',')import sys, os
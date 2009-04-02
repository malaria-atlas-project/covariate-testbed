"""
Utility functions for transforming data to something that can be modeled as Gaussian.
"""

import numpy as np
import pymc as pm

def transform_bin_data(pos, neg):
    return pm.logit((pos+1.)/(pos+neg+2.))
    
def transform_pois_data(count):
    return np.log(count)
    

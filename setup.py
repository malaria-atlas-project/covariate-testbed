# Author: Anand Patil
# Date: 6 Feb 2009
# License: Creative Commons BY-NC-SA
####################################

from setuptools import setup
from numpy.distutils.misc_util import Configuration
import os
config = Configuration('cov_test',parent_package=None,top_path=None)

config.packages = ["cov_test"]
if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**(config.todict()))
    
for f in ['cov-test-infer','cov-test-predict']:
    os.system('chmod ugo+x %s'%f)
    os.system('cp %s /usr/local/bin'%f)
#-*- coding: utf-8 -*-

from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup
import sys
import os

def configuration(parent_package='', top_path=None):
    config = Configuration('tseries', parent_package, top_path)
    config.add_subpackage('timefn')
    return config

if __name__ == '__main__':
    setup(configuration=configuration)

# end of file

#-*- coding: utf-8 -*-

import numpy as np
from numpy.distutils.core import setup
import subprocess
import shutil
import sys
import os

def configuration(parent_package='', top_path=None):

    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('iceutils')
    config.get_version('iceutils/version.py')

    return config

if __name__ == '__main__':

    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    # List of scripts to install
    scripts = [
        'bin/ice_stack_invert.py',
        'bin/ice_crop_stack.py',
        'bin/ice_interactive_trace.py',
        'bin/ice_resample.py',
        'bin/ice_view_stack_mean.py',
        'bin/ice_explore_stack.py',
        'bin/ice_info.py'
    ]

    # Run build
    try:
        setup(name='iceutils',
              maintainer='Bryan Riel',
              author='Bryan Riel',
              author_email='bryanvriel@gmail.com',
              scripts=scripts,
              configuration=configuration)
    finally:
        del sys.path[0]
        os.chdir(old_path)
    
# end of file

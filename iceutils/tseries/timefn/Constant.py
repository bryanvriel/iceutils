#!/usr/bin/env python3

############################################################
# Program is part of GIAnT v2.0                            #
# Copyright 2013, by the California Institute of Technology#
# Contact: earthdef@gps.caltech.edu                        #
############################################################


from .BasisFn import BasisFn


class Constant(BasisFn):
    '''Class to represent a step function or a table function.

    .. math::
        
        (t \geq t_{min} ) and (t \leq t_{max} )


    Notes
    -----

    This is a user interface to the base class BasisFn.
    '''

    fnname = 'Constant'

    def __init__(self, **kwargs):
        '''Constructor for constant function.
        
        Parameters
        ----------

        tmin : str (or) datetime.datetime
            Left edge of active time period (keyword)
        tmax : str (or) datetime.datetime
            Right edge of active time period (keyword)

        '''
        
        BasisFn.__init__(self, **kwargs)

    
    def computeRaw(self, tinput):
        ''' Returns constant value of 1.
        '''
        return 1.0


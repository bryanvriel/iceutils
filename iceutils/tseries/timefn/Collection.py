#!/usr/bin/env python3

############################################################
# Program is part of GIAnT v2.0                            #
# Copyright 2013, by the California Institute of Technology#
# Contact: earthdef@gps.caltech.edu                        #
############################################################

from .BasisFn import BasisFn
import copy
import numpy as np
from .observations import _normalize_observations, _seconds_per_unit

class TimefnCollection(object):
    '''Class to represent a collection of time functions.

    Users are meant to interact with a collection when building their temporal model matrices. Holds a list of BasisFn objects. 
    '''

    def __init__(self):
        '''Constructor for a timefn collection.

        Initializes an empty collection.
        '''
        
        self.data = []

    def __len__(self):
        '''Return number of functions held by collection.
        '''
        return len(self.data)

    def __iadd__(self, coll):
        '''Add one collection to another in place.

        Parameters
        ----------
        coll : TimefnCollection (or) BasisFn

        If input is a TimefnCollection, appends the list held by current object with functions in the input.

        If input is a BasisFn, appends the input to the list held by current object directly.
        '''

        if isinstance(coll, TimefnCollection):
            for value in coll.data:
                if value in self.data:
                    raise ValueError('{0} already exists in the collection'.format(value))

                if not isinstance(value, BasisFn):
                    raise ValueError('You can only add a BasisFn to a TimefnCollection')
      
                ###Making a copy here so that original collection
                ###Can be modified independently if needed.
                self.data.append(copy.deepcopy(value))

        elif isinstance(coll, BasisFn):

            if coll in self.data:
                raise ValueError('{0} already exists in the collection'.format(coll))
            ###Making a copy here so that the original collection
            ###Can be modified independently if needed.
            self.data.append(copy.deepcopy(coll))

        else:
            raise ValueError('You can only add a TimefnCollection or a BasisFn to another TimefnCollection object')


    def __add__(self, coll):
        '''Add one collection to another and return a new collection.

        Returns a new collection by adding two collections.
        '''

        result = TimefnCollection()
        result += self
        result += coll

        return result

    def __setitem__(self, index, value):
        '''Add a basis function to a collection.

        Makes a collection behave like a list.
        '''

        if not isinstance(value, BasisFn):
            raise ValueError('Input value not of type BasisFn')

        self.data[index] = value

    def __getitem__(self, index):
        '''Return basisfn corresponding to a given index.

        Makes a collection behave like a list.
        '''
        return self.data[index]


    def __delitem__(self, index):
        '''Delete a basis function from a collection by index.

        Makes a collection behave like a list.
        '''
    
        del self.data[index]

    def append(self, coll):
        '''Overload the __iadd__ function.

        Makes a collection behave like a list.
        '''

        self.__iadd__(coll)

    def __call__(self, tinput):
        '''Return a 2D Timefn array built using the collection.

        Users are expected to interact with this function for building their temporal model matrices.
        '''

        tinput = np.atleast_1d(tinput)

        result = np.zeros((len(tinput), len(self)))
        index = 0
        for value in self.data:
            result[:,index] = value(tinput)
            index += 1

        if result.shape[0] == 1:
            return result.flatten()

        return result

    def design_matrix(self, t, observation="sample", t_end=None, time_units="years"):
        """Evaluate a fixed basis through a temporal observation operator.

        Parameters
        ----------
        t : array_like
            Epochs, or interval starts, as naive dates/datetimes, NumPy datetimes,
            or decimal years. Scalars produce one row.
        observation : {'sample', 'integral', 'average', 'difference'}, optional
            Sample the basis, integrate it, average it over an interval, or
            subtract its endpoint values. Default: 'sample'.
        t_end : array_like, optional
            Interval ends matching t exactly. Required except for samples.
        time_units : str, optional
            Elapsed-time units for integrals and durations. Default: 'years'
            (365.25 days), independent of each basis function's time units.

        Returns
        -------
        G : (N, P) numpy.ndarray
            Observation matrix, with unchanged basis column ordering. Averages
            use the full duration, including time outside active support.
        """
        dates, ends, durations = _normalize_observations(
            t, observation, t_end, time_units
        )
        G = np.zeros((len(dates), len(self)))
        seconds = _seconds_per_unit(time_units)
        for index, basis in enumerate(self.data):
            if observation == "sample":
                G[:, index] = basis(dates)
            elif observation == "difference":
                G[:, index] = basis(ends) - basis(dates)
            else:
                G[:, index] = basis._integrate(dates, ends, seconds)
        if observation == "average":
            G /= durations[:, None]
        return G

    def __repr__(self):
        '''Represent the collection as a string.

        Simple representation to help users debug and store for future reference.
        '''
        strlist = [repr(fn) for fn in self.data]
        outstr = '[' + ';'.join(strlist) + ']'
        return outstr


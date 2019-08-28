#-*- coding: utf-8 -*-

import numpy as np
import h5py

from .timeutils import tdec2datestr, datestr2tdec


def predict(stack_list, time_index, name='recon', islice=None, jslice=None):
    """
    Predict a time snapshot velocity field.
    """

    # Make sure input stacks are in a list
    if not isinstance(stack_list, (list, tuple)):
        stack_list = [stack_list]

    # Construct row slices
    Ny, Nx = stack_list[0].Ny, stack_list[0].Nx
    if islice is None:
        islice = slice(0, Ny)
    
    # Construct column slices
    if jslice is None:
        jslice = slice(0, Nx)

    # Compute reconstruction
    fit = 0.0
    for stack in stack_list:
        fit += stack[name][time_index, islice, jslice]

    return fit


def build_temporal_model(t, userfile, cov=False):

    from importlib.machinery import SourceFileLoader
    import pygeodesy as pg

    # If decimal times passed in, convert to dates
    if isinstance(t[0], (float, np.float32)):
        dates = tdec2datestr(t, returndate=True)
    elif isinstance(t[0], (datetime.datetime, datetime.date)):
        dates = t
    else:
        raise ValueError('Incompatible array of times.')
        
    # Load collection
    collfun = SourceFileLoader('build', userfile).load_module()
    collection = collfun.build(dates)

    # Create a model for handling the time function
    model = pg.model.Model(dates, collection=collection)

    # Also try to build a prior covariance matrix
    if cov:
        collfun = SourceFileLoader('computeCm', userfile).load_module()
        Cm = collfun.computeCm(collection)
        return model, Cm
    else:
        return model


def build_temporal_design_matrix(t, userfile):

    # Create a model for handling the time function
    model = build_temporal_model(dates, userfile)
    return model.G


def build_seasonal_matrix(t):

    import giant.utilities.timefn as timefn

    # If decimal times passed in, convert to dates
    if isinstance(t[0], (float, np.float32)):
        dates = tdec2datestr(t, returndate=True)
    elif isinstance(t[0], (datetime.datetime, datetime.date)):
        dates = t
    else:
        raise ValueError('Incompatible array of times.')

    # Get time bounds
    tstart, tend = dates[0], dates[-1]
    tdec_start = datestr2tdec(pydtime=tstart)
    tdec_end = datestr2tdec(pydtime=tend)

    # Initalize a collection and relevant basis functions
    collection = timefn.TimefnCollection()
    periodic = timefn.fnmap['periodic']
    poly = timefn.fnmap['poly']

    # Polynomial first
    collection.append(poly(tref=tstart, order=0, units='years'))

    # Seasonal terms
    collection.append(periodic(tref=tstart, units='years', period=0.5,
        tmin=tstart, tmax=tend))
    collection.append(periodic(tref=tstart, units='years', period=1.0,
        tmin=tstart, tmax=tend))

    # Evaluate collection
    G = collection(dates)

    return G


def get_model_component_indices(collection):

    import giant.utilities.timefn as timefn

    fnParts = timefn.getFunctionTypes(collection)
    index_dict = {}
    for key in ('seasonal', 'transient' ,'secular'):
        index_dict[key] = len(fnParts[key])

    return index_dict

# end of file

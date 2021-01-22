#-*- coding: utf-8 -*-

import numpy as np
import pickle
import time as pytime
from scipy import signal
import h5py
import copy
import sys
import os

from ..constants import *
from ..raster import get_chunks
from ..stack import Stack
from .. import pymp
from .LinearRegression import *
from .model import build_temporal_model, build_temporal_model_fromfile

def inversion(stack, userfile, outdir, cleaned_stack=None,
              solver_type='lsqr', dkey='data', nt_out=200, n_proc=8, regParam=1.0,
              rw_iter=1, robust=False, n_nonzero_coefs=10, n_min=20, n_iter=1,
              no_weights=False, prior_cov=True, mask_raster=None):

    # Create a time series model defined at the data points
    if prior_cov:
        data_model, Cm = build_temporal_model_fromfile(stack.tdec, userfile, cov=prior_cov)
        regMat = np.linalg.inv(Cm)
    else:
        data_model = build_temporal_model_fromfile(stack.tdec, userfile, cov=prior_cov)
        regMat = None
    # Cache the design matrix
    G = data_model.G

    # Create a time series model defined at equally spaced time points
    tfit = np.linspace(stack.tdec[0], stack.tdec[-1], nt_out)
    model = build_temporal_model_fromfile(tfit, userfile, cov=False)

    # Load a mask and resample to stack geometry
    if mask_raster is not None:
        from ..raster import Raster
        mrast = Raster(rasterfile=mask_raster)
        mrast.resample(stack.hdr, order=0)
        mask = mrast.data.astype(bool)
        del mrast
    else:
        mask = np.ones((stack.Ny, stack.Nx), dtype=bool)

    # Instantiate a solver
    solver = select_solver(solver_type, reg_indices=model.itransient, rw_iter=rw_iter,
                           regMat=regMat, robust=robust, penalty=regParam,
                           n_nonzero_coefs=n_nonzero_coefs, n_min=n_min)

    # Get list of chunks
    try:
        _, chunk_ny, chunk_nx = stack['chunk_shape'][()]
    except KeyError:
        # Fall back to default
        chunk_ny = chunk_nx = 128
    chunks = get_chunks((stack.Ny, stack.Nx), chunk_ny, chunk_nx)
    
    # Instantiate and initialize output stacks
    ostacks = {}
    for key in ('full', 'secular', 'seasonal', 'transient', 'sigma'):
        ostacks[key] = Stack(os.path.join(outdir, 'interp_output_%s.h5' % key), mode='w',
                             init_tdec=tfit, init_rasterinfo=stack.hdr)
        ostacks[key].init_default_datasets(chunks=(1, chunk_ny, chunk_nx))

    # If user wishes to output data stack with outliers removed
    if cleaned_stack is not None:
        clean_stack = Stack(os.path.join(outdir, cleaned_stack), mode='w', init_stack=stack)
        clean_stack.init_default_datasets(chunks=(1, chunk_ny, chunk_nx), weights=True)

    # Loop over chunks
    for islice, jslice in chunks:

        # Start timing
        t0 = pytime.time()

        # Get chunk of time series data and weights
        data2d = stack.get_chunk(islice, jslice, key=dkey)
        if no_weights:
            wgts2d = np.ones_like(data2d)
        else:
            wgts2d = stack.get_chunk(islice, jslice, key='weights')

        # Extract valid pixels in this chunk into 1d arrays
        if stack.fmt == 'NHW':
            _, chunk_ny, chunk_nx = data2d.shape
            chunk_npix = chunk_ny * chunk_nx
            chunk_mask = mask[islice, jslice]
            data1d = data2d[:, chunk_mask]
            wgts1d = wgts2d[:, chunk_mask]
        else:
            chunk_ny, chunk_nx, _ = data2d.shape
            chunk_npix = chunk_ny * chunk_nx
            chunk_mask = mask[islice, jslice]
            data1d = data2d[chunk_mask, :].T
            wgts1d = wgts2d[chunk_mask, :].T
        npix = data1d.shape[1]

        # Transfer to shared arrays
        manager = pymp.Manager()
        data = pymp.array(data1d.shape, dtype=np.float32)
        wgts = pymp.array(wgts1d.shape, dtype=np.float32)
        data[:, :] = data1d
        wgts[:, :] = wgts1d

        # Create shared arrays for results
        shape = (len(tfit), npix)
        results = {}
        for key in ('full', 'secular', 'seasonal', 'transient', 'sigma'):
            results[key] = pymp.array(shape, dtype=np.float32)

        # Loop over pixels in chunk in parallel
        with pymp.Parallel(n_proc, manager) as parallel:
            for index in parallel.range(npix):

                # Get pixel data
                d = data[:, index]
                w = wgts[:, index]
                
                # Perform inversion: iterative least squares with outlier detection
                # Outliers are set to NaN in-place
                status, m, Cm = iterate_lsqr(solver, data_model, G, d, w, n_iter=n_iter)
                # Check if least squares failed
                if status == FAIL:
                    continue

                # Compute prediction and store in arrays
                pred = model.predict(m, sigma=False)
                for key in ('full', 'secular', 'seasonal', 'transient'):
                    results[key][:,index] = pred[key]

                # Compute data prediction sigma manually
                sigmaval = np.sqrt(np.diag(np.dot(model.G, np.dot(Cm, model.G.T))))
                results['sigma'][:,index] = sigmaval

        # Save results in output stacks
        for key in ('full', 'secular', 'seasonal', 'transient', 'sigma'):
            rdata = np.zeros((len(tfit), chunk_ny, chunk_nx), dtype=np.float32)
            rdata[:, chunk_mask] = results[key]
            ostacks[key].set_chunk(islice, jslice, rdata)

        # Optional saving of cleaned stack
        if cleaned_stack is not None:
            # First transfer 1d arrays to original 2d chunks
            if stack.fmt == 'NHW':
                data2d[:, chunk_mask] = data
                wgts2d[:, chunk_mask] = wgts
            else:
                data2d[chunk_mask, :] = data.T
                wgts2d[chunk_mask, :] = wgts.T
            # Write to output stack
            clean_stack.set_chunk(islice, jslice, data2d, key='data')
            clean_stack.set_chunk(islice, jslice, wgts2d, key='weights')

        # Timing diagnostics
        print('Finished chunk', islice, jslice, 'in %f sec' % (pytime.time() - t0))

    # All done
    return 

def iterate_lsqr(solver, model, G, d, w, n_iter=5, n_std=3.0, n_min=20):
    """
    Iterative least squares to remove outliers.
    """
    for iternum in range(n_iter):
    
        # Fit
        status, m, Cm = solver.invert(G, d, wgt=w)
        # Check status
        if status == FAIL:
            return status, None, None
    
        # Predict
        pred = model.predict(m)

        # Compute outliers
        misfit = d - pred['full']
        std = np.nanstd(misfit)
        outliers = (np.abs(misfit) > (n_std * std)).nonzero()[0]
        if len(outliers) < 1:
            break
        d[outliers] = np.nan
        w[outliers] = np.nan

    # Done
    return SUCCESS, m, Cm

def inversion_points(stack, userfile, x, y, solver_type='lsqr',
                     nt_out=200, n_proc=8, regParam=1.0, rw_iter=1, robust=False,
                     n_nonzero_coefs=10, n_min=20):

    # Check consistency of input points
    n_pts = len(x)
    assert len(y) == n_pts, 'Mismatch in sizes of input points'

    # Create a time series model defined at the data points
    model, Cm = build_temporal_model(stack.tdec, userfile, cov=True)
    regMat = np.linalg.inv(Cm)
    # Cache the design matrix
    G = model.G.copy()

    # Create a time series model defined at equally spaced time points
    tfit = np.linspace(stack.tdec[0], stack.tdec[-1], nt_out)
    model = build_temporal_model(tfit, userfile, cov=False)

    # Instantiate a solver
    solver = select_solver(solver_type, reg_indices=model.itransient, rw_iter=rw_iter,
                           regMat=regMat, robust=robust, penalty=regParam,
                           n_nonzero_coefs=n_nonzero_coefs, n_min=n_min)
    
    # Create shared arrays for results
    manager = pymp.Manager()
    shape = (n_pts, len(tfit))
    results = {'tdec': tfit}
    for key in ('full', 'secular', 'seasonal', 'transient', 'sigma'):
        results[key] = pymp.array(shape, dtype=np.float32)

    # Loop over pixels in chunk in parallel
    with pymp.Parallel(n_proc, manager) as parallel:
        for index in parallel.range(n_pts):

            # Get time series
            d = stack.timeseries(xy=(x[index], y[index]))
            w = stack.timeseries(xy=(x[index], y[index]), key='weights')
            if d is None:
                continue

            # Perform inversion
            status, m, Cm = solver.invert(G, d, wgt=w)
            if status == FAIL:
                continue

            # Compute prediction and store in arrays
            pred = model.predict(m, sigma=False)
            for key in ('full', 'secular', 'seasonal', 'transient'):
                results[key][index,:] = pred[key]

            # Compute data prediction sigma manually
            sigmaval = np.sqrt(np.diag(np.dot(model.G, np.dot(Cm, model.G.T))))
            results['sigma'][index,:] = sigmaval

    # All done
    return results

def butterworth(stack, a, b, fname_long, fname_short, n_proc=1):

    # Instantiate and initialize output stacks
    shape = (stack.Nt, stack.Ny, stack.Nx)
    long_stack = Stack(fname_long, mode='w')
    short_stack = Stack(fname_short, mode='w')
    for ostack in (long_stack, short_stack):
        ostack.initialize(stack.tdec, stack.hdr, data=True, weights=False)

    # Get list of chunks
    chunks = get_chunks(stack, 128, 128)

    # Loop over chunks
    for islice, jslice in chunks:

        # Start timing
        t0 = pytime.time()

        # Get chunk of time series data
        data = stack.get_chunk(islice, jslice, key='data')
        _, chunk_ny, chunk_nx = data.shape
        npix = chunk_ny * chunk_nx

        # Create shared arrays for results
        manager = pymp.Manager()
        shape = (stack.Nt, chunk_ny, chunk_nx)
        results = {}
        for key in ('long_term', 'short_term'):
            results[key] = pymp.array(shape, dtype=np.float32)

        # Loop over pixels in chunk in parallel
        with pymp.Parallel(n_proc, manager) as parallel:
            for index in parallel.range(npix):

                # Get time series
                i, j = np.unravel_index(index, (chunk_ny, chunk_nx))
                d = data[:,i,j]

                # Perform Butterworth filtering
                d_filt = signal.filtfilt(b, a, d)

                # Save results
                results['long_term'][:,i,j] = d_filt
                results['short_term'][:,i,j] = d - d_filt

        # Save results in output stack
        for key, ostack in (('long_term', long_stack), ('short_term', short_stack)):
            ostack.set_chunk(islice, jslice, results[key], key='data')

        # Timing diagnostics
        print('Finished chunk', islice, jslice, 'in %f sec' % (pytime.time() - t0))

    # All done
    return 

def butterworth_coeffs(frequency=None, period=None, dt=1.0, order=3, btype='low'):
    """
    Compute butterworth coefficients for a given cutoff frequency or period in time coordinates
    determined by sampling interval dt.
    """
    # Compute sampling and Nyquist frequency
    Fs = 1.0 / dt
    Fn = 0.5 * Fs

    # Compute cutoff frequency
    if frequency is None and period is not None:
        frequency = 1.0 / period
    elif frequency is None:
        raise ValueError('Must supply cutoff frequency or period')

    # Compute normalized cutoff frequency
    w_low = frequency / Fn

    # Compute Butterworth filter coefficients
    b, a = signal.butter(order, w_low, btype=btype)
    return b,a

# end of file

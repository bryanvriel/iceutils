#-*- coding: utf-8 -*-

import numpy as np
import pickle
import time as pytime
from scipy import signal
import pymp
import h5py
import copy
import sys
import os

from ..stack import Stack
from .LinearRegression import *
from .model import build_temporal_model

def inversion(stack, userfile, outdir, cleaned_stack=None,
              solver_type='lsqr', dkey='data', nt_out=200, n_proc=8, regParam=1.0,
              rw_iter=1, robust=False, n_nonzero_coefs=10, n_min=20, n_iter=5,
              no_weights=False, mask_raster=None):

    # Create a time series model defined at the data points
    data_model, Cm = build_temporal_model(stack.tdec, userfile, cov=True)
    regMat = np.linalg.inv(Cm)
    # Cache the design matrix
    G = data_model.G

    # Create a time series model defined at equally spaced time points
    tfit = np.linspace(stack.tdec[0], stack.tdec[-1], nt_out)
    model = build_temporal_model(tfit, userfile, cov=False)

    # Load a mask
    if mask_raster is not None:
        from ..raster import Raster
        mrast = Raster(rasterfile=mask_raster)
        mask = mrast.data.astype(bool)
        del mrast
    else:
        mask = np.ones((stack.Ny, stack.Nx), dtype=bool)

    # Instantiate a solver
    solver = select_solver(solver_type, reg_indices=model.itransient, rw_iter=rw_iter,
                           regMat=regMat, robust=robust, penalty=regParam,
                           n_nonzero_coefs=n_nonzero_coefs)

    # Get list of chunks
    _, chunk_ny, chunk_nx = stack['chunk_shape'][()]
    chunks = get_chunks(stack, chunk_ny, chunk_nx)
    
    # Instantiate and initialize output stacks
    ostacks = {}
    for key in ('full', 'secular', 'seasonal', 'transient', 'sigma'):
        ostacks[key] = Stack(os.path.join(outdir, 'interp_output_%s.h5' % key), mode='w')
        ostacks[key].initialize(tfit, stack.hdr, chunks=(1, chunk_ny, chunk_nx))

    # If user wishes to output data stack with outliers removed
    if cleaned_stack is not None:
        clean_stack = Stack(os.path.join(outdir, cleaned_stack), mode='w')
        clean_stack.initialize(stack.tdec, stack.hdr, chunks=(1, chunk_ny, chunk_nx),
                               data=True, weights=True)

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
        _, chunk_ny, chunk_nx = data2d.shape
        chunk_npix = chunk_ny * chunk_nx

        # Extract valid pixels in this chunk into 1d arrays
        chunk_mask = mask[islice, jslice]
        data1d = data2d[:, chunk_mask]
        wgts1d = wgts2d[:, chunk_mask]
        npix = data1d.shape[1]

        # Transfer to shared arrays
        data = pymp.shared.array(data1d.shape, dtype=np.float32)
        wgts = pymp.shared.array(wgts1d.shape, dtype=np.float32)
        data[:, :] = data1d
        wgts[:, :] = wgts1d

        # Create shared arrays for results
        shape = (len(tfit), npix)
        results = {}
        for key in ('full', 'secular', 'seasonal', 'transient', 'sigma'):
            results[key] = pymp.shared.array(shape, dtype=np.float32)

        # Loop over pixels in chunk in parallel
        with pymp.Parallel(n_proc) as manager:
            for index in manager.range(npix):

                # Get time series
                d = data[:,index]
                w = wgts[:,index]
                valid_mask = np.isfinite(d)
                nfinite = len(valid_mask.nonzero()[0])
                if nfinite < n_min:
                    continue

                # Perform inversion: iterative least squares with outlier detection
                # Outliers are set to NaN in-place
                m, Cm = iterate_lsqr(solver, data_model, G, d, w, n_iter=n_iter, n_min=n_min)
                # Check if least squares failed
                if m is None:
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
            data2d[:, chunk_mask] = data
            wgts2d[:, chunk_mask] = wgts
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
        mask = np.isfinite(d)
        dsub = d[mask]
        if len(dsub) < n_min:
            return None, None
        m, Cm = solver.invert(G[mask], d[mask], wgt=w[mask])
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
    return m, Cm

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
                           n_nonzero_coefs=n_nonzero_coefs)
    
    # Create shared arrays for results
    shape = (n_pts, len(tfit))
    results = {'tdec': tfit}
    for key in ('full', 'secular', 'seasonal', 'transient', 'sigma'):
        results[key] = pymp.shared.array(shape, dtype=np.float32)

    # Loop over pixels in chunk in parallel
    with pymp.Parallel(n_proc) as manager:
        for index in manager.range(n_pts):

            # Get time series
            d = stack.timeseries(xy=(x[index], y[index]))
            w = stack.timeseries(xy=(x[index], y[index]), key='weights')
            if d is None:
                continue

            # Check number of valid data
            mask = np.isfinite(d)
            nfinite = len(mask.nonzero()[0])
            if nfinite < n_min:
                continue

            # Perform inversion
            m, Cm = solver.invert(G[mask,:], d[mask], wgt=w[mask])

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
        shape = (stack.Nt, chunk_ny, chunk_nx)
        results = {}
        for key in ('long_term', 'short_term'):
            results[key] = pymp.shared.array(shape, dtype=np.float32)

        # Loop over pixels in chunk in parallel
        with pymp.Parallel(n_proc) as manager:
            for index in manager.range(npix):

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

def butterworth_coeffs(frequency=None, period=None, dt=1.0, order=3):
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
    b, a = signal.butter(order, w_low)
    return b,a
    
def get_chunks(stack, chunk_y, chunk_x):
    """
    Utility function to get chunk bounds.

    Parameters
    ----------
    stack: Stack
        Stack instance.
    chunk_y: int
        Size of chunk in vertical dimension.
    chunk_x: int
        Size of chunk in horizontal dimension.

    Returns
    -------
    chunks: list
        List of all chunks in the image.
    """
    # First determine the number of chunks in each dimension
    Ny_chunk = int(stack.Ny // chunk_y)
    Nx_chunk = int(stack.Nx // chunk_x)
    if stack.Ny % chunk_y != 0:
        Ny_chunk += 1
    if stack.Nx % chunk_x != 0:
        Nx_chunk += 1

    # Now construct chunk bounds
    chunks = []
    for i in range(Ny_chunk):
        if i == Ny_chunk - 1:
            nrows = stack.Ny - chunk_y * i
        else:
            nrows = chunk_y
        istart = chunk_y * i
        iend = istart + nrows
        for j in range(Nx_chunk):
            if j == Nx_chunk - 1:
                ncols = stack.Nx - chunk_x * j
            else:
                ncols = chunk_x
            jstart = chunk_x * j
            jend = jstart + ncols
            chunks.append([slice(istart,iend), slice(jstart,jend)])

    return chunks 

# end of file

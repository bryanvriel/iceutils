#-*- coding: utf-8 -*-

import numpy as np
import pickle
import time as pytime
from scipy import signal
import h5py
import copy
import sys
import os
from joblib import Parallel, delayed

from ..constants import *
from ..raster import get_chunks
from ..stack import Stack
from ..timeutils import tdec2datestr
from .LinearRegression import *
from .model import Model, build_temporal_model, build_temporal_model_fromfile

_INVERSION_KEYS = ('full', 'secular', 'seasonal', 'transient', 'sigma')
_PREDICTION_KEYS = ('full', 'secular', 'seasonal', 'transient')


def _normalize_n_proc(n_proc):
    """
    Return a positive process count for user-facing n_proc values.
    """
    if n_proc is None:
        return 1
    return max(1, int(n_proc))


def _index_blocks(n_items, n_proc):
    """
    Split item indices into deterministic blocks for parallel workers.
    """
    n_items = int(n_items)
    if n_items < 1:
        return []

    n_jobs = min(_normalize_n_proc(n_proc), n_items)
    n_blocks = 1 if n_jobs == 1 else min(n_items, n_jobs * 4)
    edges = np.linspace(0, n_items, n_blocks + 1, dtype=int)
    return [
        (int(edges[index]), int(edges[index + 1]))
        for index in range(n_blocks)
        if edges[index] < edges[index + 1]
    ]


def _run_parallel_blocks(func, blocks, n_proc, *args):
    """
    Run block workers serially or with joblib depending on n_proc.
    """
    if len(blocks) < 1:
        return []

    n_jobs = min(_normalize_n_proc(n_proc), len(blocks))
    if n_jobs == 1:
        return [func(start, stop, *args) for start, stop in blocks]

    return Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(func)(start, stop, *args) for start, stop in blocks
    )


def _solver_config(solver_type, reg_indices=None, rw_iter=1, regMat=None,
                   robust=False, penalty=1.0, n_nonzero_coefs=10, n_min=20):
    """
    Return serializable solver configuration for joblib workers.
    """
    return {
        'solver_type': solver_type,
        'reg_indices': reg_indices,
        'rw_iter': rw_iter,
        'regMat': regMat,
        'robust': robust,
        'penalty': penalty,
        'n_nonzero_coefs': n_nonzero_coefs,
        'n_min': n_min,
    }


def _is_temporal_model(obj):
    """
    Return True for iceutils temporal model instances.
    """
    return isinstance(obj, Model)


def _model_at_tdec(model, tdec):
    """
    Re-evaluate a pre-built temporal model collection at new decimal years.
    """
    dates = tdec2datestr(tdec, returndate=True)
    return Model(dates, collection=model.collection)


def _regularization_matrix_from_model_prior(model, prior_cov):
    """
    Resolve a regularization matrix for a pre-built model.
    """
    if isinstance(prior_cov, np.ndarray):
        return np.linalg.inv(prior_cov)

    return None


def _regularization_indices(model):
    """
    Return the model-defined columns that should be regularized.
    """
    reg_indices = getattr(model, 'reg_indices', None)
    if reg_indices is not None and len(reg_indices) > 0:
        return reg_indices
    return model.itransient


def _resolve_temporal_models(model_or_userfile, stack_tdec, nt_out,
                             prior_cov=False):
    """
    Build data/output temporal models from either a model instance or userfile.
    """
    tfit = np.linspace(stack_tdec[0], stack_tdec[-1], nt_out)

    if _is_temporal_model(model_or_userfile):
        data_model = model_or_userfile
        if data_model.G.shape[0] != len(stack_tdec):
            raise ValueError(
                'Input model has %d epochs, but stack has %d time steps.' %
                (data_model.G.shape[0], len(stack_tdec))
            )
        output_model = _model_at_tdec(data_model, tfit)
        regMat = _regularization_matrix_from_model_prior(data_model, prior_cov)
        return data_model, output_model, regMat, tfit

    if prior_cov:
        data_model, Cm = build_temporal_model_fromfile(
            stack_tdec, model_or_userfile, cov=prior_cov
        )
        regMat = np.linalg.inv(Cm)
    else:
        data_model = build_temporal_model_fromfile(
            stack_tdec, model_or_userfile, cov=prior_cov
        )
        regMat = None

    output_model = build_temporal_model_fromfile(tfit, model_or_userfile, cov=False)
    return data_model, output_model, regMat, tfit


def _model_prediction_spec(model):
    """
    Return the array-only pieces needed for model prediction in workers.
    """
    return {
        'G': np.asarray(model.G),
        'secular': np.asarray(model.isecular, dtype=int),
        'seasonal': np.asarray(model.iseasonal, dtype=int),
        'transient': np.asarray(model.itransient, dtype=int),
        'step': np.asarray(model.istep, dtype=int),
    }


def _predict_model_parts(model_spec, m):
    """
    Predict model components from an array-only model spec.
    """
    G = model_spec['G']
    results = {}
    for key in ('secular', 'seasonal', 'transient', 'step'):
        indices = model_spec[key]
        if indices.size:
            results[key] = np.dot(G[:, indices], m[indices])
        else:
            results[key] = np.zeros((G.shape[0],), dtype=np.float64)

    results['full'] = (
        results['secular'] + results['seasonal'] +
        results['transient'] + results['step']
    )
    return results


def _prediction_sigma(model_spec, Cm):
    """
    Compute prediction uncertainty from an array-only model spec.
    """
    G = model_spec['G']
    return np.sqrt(np.diag(np.dot(G, np.dot(Cm, G.T))))


def _iterate_lsqr_arrays(solver, G, d, w, n_iter=5, n_std=3.0):
    """
    Iterative least squares using only arrays for worker-side execution.
    """
    for iternum in range(n_iter):

        # Fit
        status, m, Cm = solver.invert(G, d, wgt=w)
        if status == FAIL:
            return status, None, None

        # Compute outliers against the full prediction.
        misfit = d - np.dot(G, m)
        std = np.nanstd(misfit)
        outliers = (np.abs(misfit) > (n_std * std)).nonzero()[0]
        if len(outliers) < 1:
            break
        d[outliers] = np.nan
        w[outliers] = np.nan

    return SUCCESS, m, Cm


def _invert_pixel_block(start, stop, data, wgts, solver_config, G,
                        output_model_spec, n_iter, n_std, return_cleaned):
    """
    Invert a contiguous block of flattened grid pixels.
    """
    solver = select_solver(**solver_config)
    block_len = stop - start
    nt_out = output_model_spec['G'].shape[0]
    results = {
        key: np.full((nt_out, block_len), np.nan, dtype=np.float32)
        for key in _INVERSION_KEYS
    }
    clean_data = None
    clean_wgts = None
    if return_cleaned:
        clean_data = np.empty((data.shape[0], block_len), dtype=np.float32)
        clean_wgts = np.empty((wgts.shape[0], block_len), dtype=np.float32)

    for offset, index in enumerate(range(start, stop)):

        # Work on private copies because outlier removal mutates d and w.
        d = np.array(data[:, index], dtype=np.float64, copy=True)
        w = np.array(wgts[:, index], dtype=np.float64, copy=True)

        status, m, Cm = _iterate_lsqr_arrays(
            solver, G, d, w, n_iter=n_iter, n_std=n_std,
        )
        if return_cleaned:
            clean_data[:, offset] = d
            clean_wgts[:, offset] = w

        if status == FAIL:
            continue

        pred = _predict_model_parts(output_model_spec, m)
        for key in _PREDICTION_KEYS:
            results[key][:, offset] = pred[key]
        results['sigma'][:, offset] = _prediction_sigma(output_model_spec, Cm)

    return start, stop, results, clean_data, clean_wgts


def _invert_point_block(start, stop, data, wgts, solver_config, G, output_model_spec):
    """
    Invert a contiguous block of point time series.
    """
    solver = select_solver(**solver_config)
    block_len = stop - start
    nt_out = output_model_spec['G'].shape[0]
    results = {
        key: np.zeros((block_len, nt_out), dtype=np.float32)
        for key in _INVERSION_KEYS
    }

    for offset, index in enumerate(range(start, stop)):

        d = np.array(data[:, index], dtype=np.float64, copy=True)
        w = np.array(wgts[:, index], dtype=np.float64, copy=True)

        status, m, Cm = solver.invert(G, d, wgt=w)
        if status == FAIL:
            continue

        pred = _predict_model_parts(output_model_spec, m)
        for key in _PREDICTION_KEYS:
            results[key][offset, :] = pred[key]
        results['sigma'][offset, :] = _prediction_sigma(output_model_spec, Cm)

    return start, stop, results


def _filter_pixel_block(start, stop, data, a, b):
    """
    Butterworth-filter a contiguous block of flattened grid pixels.
    """
    block_len = stop - start
    long_term = np.empty((data.shape[0], block_len), dtype=np.float32)
    short_term = np.empty((data.shape[0], block_len), dtype=np.float32)

    for offset, index in enumerate(range(start, stop)):
        d = data[:, index]
        d_filt = signal.filtfilt(b, a, d)
        long_term[:, offset] = d_filt
        short_term[:, offset] = d - d_filt

    return start, stop, long_term, short_term


def _stack_chunk_array(stack, islice, jslice, key='data'):
    """
    Read a spatial chunk from an xarray-backed Stack in canonical time/y/x order.
    """
    data = stack[key].isel(y=islice, x=jslice).transpose('time', 'y', 'x')
    return np.asarray(data.values)

def _stack_chunk_to_timeseries(stack, islice, jslice, dkey='data',
                               no_weights=False, mask=None):
    """
    Return chunk arrays and time-by-pixel views for inversion.
    """
    data2d = _stack_chunk_array(stack, islice, jslice, key=dkey)
    if no_weights:
        wgts2d = np.ones_like(data2d)
    else:
        wgts2d = _stack_chunk_array(stack, islice, jslice, key='weights')

    _, chunk_ny, chunk_nx = data2d.shape
    if mask is None:
        chunk_mask = np.ones((chunk_ny, chunk_nx), dtype=bool)
    else:
        chunk_mask = np.asarray(mask[islice, jslice], dtype=bool)

    data1d = data2d[:, chunk_mask]
    wgts1d = wgts2d[:, chunk_mask]
    return data2d, wgts2d, data1d, wgts1d, chunk_mask

def inversion(stack, userfile, outdir, cleaned_stack=None,
              solver_type='lsqr', dkey='data', nt_out=200, n_proc=8, regParam=1.0,
              rw_iter=1, robust=False, n_nonzero_coefs=10, n_min=20, n_iter=1,
              n_std=3.0, no_weights=False, prior_cov=True, mask_raster=None):

    # Create temporal models defined at the data points and output points.
    data_model, model, regMat, tfit = _resolve_temporal_models(
        userfile, stack.tdec, nt_out, prior_cov=prior_cov
    )

    # Cache the design matrix
    G = data_model.G

    # Load a mask and resample to stack geometry
    if mask_raster is not None:
        from ..raster import Raster
        mrast = Raster(rasterfile=mask_raster)
        mrast.resample(stack.hdr, order=0)
        mask = mrast.data.astype(bool)
        del mrast
    else:
        mask = np.ones((stack.Ny, stack.Nx), dtype=bool)

    # Cache serializable worker configuration.
    solver_cfg = _solver_config(solver_type, reg_indices=_regularization_indices(model),
                                rw_iter=rw_iter, regMat=regMat, robust=robust,
                                penalty=regParam,
                                n_nonzero_coefs=n_nonzero_coefs, n_min=n_min)
    output_model_spec = _model_prediction_spec(model)

    # Get list of chunks
    try:
        _, chunk_ny, chunk_nx = stack['chunk_shape'].values
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

        # Get chunk data as canonical (time, y, x) arrays and flatten valid pixels.
        data2d, wgts2d, data1d, wgts1d, chunk_mask = _stack_chunk_to_timeseries(
            stack, islice, jslice, dkey=dkey, no_weights=no_weights, mask=mask
        )
        _, chunk_ny, chunk_nx = data2d.shape
        npix = data1d.shape[1]

        # Convert chunk data once for worker-side numerical work.
        data = np.asarray(data1d, dtype=np.float32)
        wgts = np.asarray(wgts1d, dtype=np.float32)

        # Create arrays for results
        shape = (len(tfit), npix)
        results = {}
        for key in _INVERSION_KEYS:
            results[key] = np.full(shape, np.nan, dtype=np.float32)

        # Loop over pixel blocks in parallel.
        blocks = _index_blocks(npix, n_proc)
        block_results = _run_parallel_blocks(
            _invert_pixel_block, blocks, n_proc, data, wgts, solver_cfg, G,
            output_model_spec, n_iter, n_std, cleaned_stack is not None
        )
        for start, stop, block, clean_data, clean_wgts in block_results:
            for key in _INVERSION_KEYS:
                results[key][:, start:stop] = block[key]
            if cleaned_stack is not None:
                data[:, start:stop] = clean_data
                wgts[:, start:stop] = clean_wgts

        # Save results in output stacks
        for key in _INVERSION_KEYS:
            rdata = np.zeros((len(tfit), chunk_ny, chunk_nx), dtype=np.float32)
            rdata[:, chunk_mask] = results[key]
            ostacks[key].set_chunk(islice, jslice, rdata)

        # Optional saving of cleaned stack
        if cleaned_stack is not None:
            # First transfer 1d arrays to original 2d chunks.
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

    # Create temporal models defined at the data points and output points.
    data_model, model, regMat, tfit = _resolve_temporal_models(
        userfile, stack.tdec, nt_out, prior_cov=True
    )

    # Cache the design matrix
    G = data_model.G.copy()

    # Cache serializable worker configuration.
    solver_cfg = _solver_config(solver_type, reg_indices=_regularization_indices(model),
                                rw_iter=rw_iter, regMat=regMat, robust=robust,
                                penalty=regParam,
                                n_nonzero_coefs=n_nonzero_coefs, n_min=n_min)
    output_model_spec = _model_prediction_spec(model)
    
    # Load point time series in the parent process before launching workers.
    data = np.full((stack.Nt, n_pts), np.nan, dtype=np.float32)
    wgts = np.full((stack.Nt, n_pts), np.nan, dtype=np.float32)
    for index in range(n_pts):
        d = stack.timeseries(xy=(x[index], y[index]))
        if d is None:
            continue
        data[:, index] = d
        wgts[:, index] = stack.timeseries(xy=(x[index], y[index]), key='weights')

    # Create arrays for results
    shape = (n_pts, len(tfit))
    results = {'tdec': tfit}
    for key in _INVERSION_KEYS:
        results[key] = np.zeros(shape, dtype=np.float32)

    # Loop over point blocks in parallel.
    blocks = _index_blocks(n_pts, n_proc)
    block_results = _run_parallel_blocks(
        _invert_point_block, blocks, n_proc, data, wgts, solver_cfg, G,
        output_model_spec
    )
    for start, stop, block in block_results:
        for key in _INVERSION_KEYS:
            results[key][start:stop, :] = block[key]

    # All done
    return results

def butterworth(stack, a, b, fname_long, fname_short, n_proc=1):

    # Instantiate and initialize output stacks
    long_stack = Stack(fname_long, mode='w')
    short_stack = Stack(fname_short, mode='w')
    for ostack in (long_stack, short_stack):
        ostack.initialize(stack.tdec, stack.hdr, data=True, weights=False)

    # Get list of chunks
    chunks = get_chunks((stack.Ny, stack.Nx), 128, 128)

    # Loop over chunks
    for islice, jslice in chunks:

        # Start timing
        t0 = pytime.time()

        # Get chunk of time series data
        data = _stack_chunk_array(stack, islice, jslice, key='data')
        _, chunk_ny, chunk_nx = data.shape
        npix = chunk_ny * chunk_nx

        # Loop over flattened pixel blocks in parallel.
        data1d = np.asarray(data.reshape(stack.Nt, npix), dtype=np.float32)
        long_term = np.empty((stack.Nt, npix), dtype=np.float32)
        short_term = np.empty((stack.Nt, npix), dtype=np.float32)
        blocks = _index_blocks(npix, n_proc)
        block_results = _run_parallel_blocks(
            _filter_pixel_block, blocks, n_proc, data1d, a, b
        )
        for start, stop, block_long, block_short in block_results:
            long_term[:, start:stop] = block_long
            short_term[:, start:stop] = block_short

        results = {
            'long_term': long_term.reshape(stack.Nt, chunk_ny, chunk_nx),
            'short_term': short_term.reshape(stack.Nt, chunk_ny, chunk_nx),
        }

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

#-*- coding: utf-8 -*-

import numpy as np
import pygeodesy as pg
import pickle
import time as pytime
import pymp
import h5py
import copy
import sys
import os

from ..stack import Stack
from .LinearRegression import *
from .model import build_temporal_model

def inversion(stack, userfile, outdir, solver_type='lsqr',
              nt_out=200, n_proc=8, regParam=1.0, rw_iter=1, robust=False,
              n_nonzero_coefs=10, n_min=20):

    # Create a time series model defined at the data points
    model, Cm = build_temporal_model(stack.tdec, userfile, cov=True)
    regMat = np.linalg.inv(Cm)
    # Cache the design matrix
    G = model.G.copy()

    # Create a time series model defined at equally spaced time points
    tfit = np.linspace(stack.tdec[0], stack.tdec[-1], nt_out)
    model = build_temporal_model(tfit, userfile, cov=False)

    # Instantiate a solver
    if solver_type == 'lasso':
        solver = LassoRegression(model.itransient, regParam, regMat=regMat, rw_iter=rw_iter)
    elif solver_type == 'ridge':
        solver = RidgeRegression(model.itransient, regParam, regMat=regMat)
    elif solver_type == 'omp':
        solver = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs, regMat=regMat)
    elif solver_type == 'lsqr':
        solver = LinearRegression(robust=robust)

    # Instantiate output stacks
    full = Stack(os.path.join(outdir, 'interp_output_full.h5'), mode='w')
    transient = Stack(os.path.join(outdir, 'interp_output_transient.h5'), mode='w')
    secular = Stack(os.path.join(outdir, 'interp_output_secular.h5'), mode='w')
    seasonal = Stack(os.path.join(outdir, 'interp_output_seasonal.h5'), mode='w')

    # Initialize the stack output Datasets
    for ostack in (full, transient, secular, seasonal):
        ostack.initialize(tfit, stack.hdr)

    # Get list of chunks
    chunks = get_chunks(stack, 128, 128)

    # Loop over chunks
    for islice, jslice in chunks:

        # Start timing
        t0 = pytime.time()

        # Get chunk of time series data and weights
        data = stack.get_chunk(islice, jslice, key='data')
        wgts = stack.get_chunk(islice, jslice, key='weights')
        _, chunk_ny, chunk_nx = data.shape
        npix = chunk_ny * chunk_nx

        # Create shared arrays for results
        shape = (len(tfit), chunk_ny, chunk_nx)
        results = {}
        for key in ('full', 'secular', 'seasonal', 'transient', 'sigma'):
            results[key] = pymp.shared.array(shape, dtype=np.float32)

        # Loop over pixels in chunk in parallel
        with pymp.Parallel(n_proc) as manager:
            for index in manager.range(npix):

                # Get time series
                i, j = np.unravel_index(index, (chunk_ny, chunk_nx))
                d = data[:,i,j]
                w = wgts[:,i,j]
                mask = np.isfinite(d)
                nfinite = len(mask.nonzero()[0])
                if nfinite < n_min:
                    continue

                # Perform inversion
                m, Cm = solver.invert(G[mask,:], d[mask], wgt=w[mask])

                # Compute prediction and store in arrays
                pred = model.predict(m, sigma=False)
                for key in ('full', 'secular', 'seasonal', 'transient'):
                    results[key][:,i,j] = pred[key]

                # Compute data prediction sigma manually
                sigma = np.sqrt(np.diag(np.dot(model.G, np.dot(Cm, model.G.T))))
                results['sigma'][:,i,j] = sigma

        # Save results in output stacks
        full.set_chunk(islice, jslice, results['full'])
        transient.set_chunk(islice, jslice, results['transient'])
        secular.set_chunk(islice, jslice, results['secular'])
        seasonal.set_chunk(islice, jslice, results['seasonal'])

        # Timing diagnostics
        print('Finished chunk', islice, jslice, 'in %f sec' % (pytime.time() - t0))

    # All done
    return 

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

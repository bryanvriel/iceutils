#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from .boundary import transform_coordinates
from .raster import get_utm_EPSG
from tqdm import tqdm

def variogram(raster, projWin=None, max_samples=1000, remove_ramp=True, measure='semi',
              dist_max=50.0e3, dist_spacing=1.0e3, epsg_in=None, seed=None):
    """
    Computes semivariogram for a given raster and optional subwindow. Optionally removes
    a ramp prior to computation of the variogram. Adapted from:
    https://github.com/jolivetr/csi/blob/master/imagecovariance.py

    Parameters
    ----------
    raster: Raster
        Input raster.
    projWin: list, optional
        List of [upper_left_x, upper_left_y, lower_right_x, lower_right_y] for geographic
        bounding box to subset.
    max_samples: int, optional
        Maximum number of samples to randomly sample from raster window. Default: 1000.
    remove_ramp: bool, optional
        Remove bilinear ramp prior to computing variogram. Default: True.
    measure: str, optional
        Variogram metric to use.
            - 'semi': (x_i - x_j)**2, default.
            - 'cov': abs(x_i * x_j)
    dist_max: float, optional
        Maximum distance bin in meters for constructing variogram. Default: 50e3.
    dist_spacing: float, optional
        Distance bin size in meters for variogram. Default: 1e3.
    epsg_in: int, optional
        Override EPSG code for raster. Otherwise, reads EPSG from raster.hdr.
    seed: int, optional
        Random seed.

    Returns
    -------
    distance: ndarray
        Variogram distance bins.
    semivariogram: ndarray
        Variogram values.
    std: ndarray
        Standard deviation of raster values per distance bin. 
    """
    # Create meshgrid
    X, Y = raster.hdr.meshgrid()

    # Crop the raster data if projWin is provided
    if projWin is not None:
        xmin, ymax, xmax, ymin = projWin
        i0, j0 = raster.hdr.xy_to_imagecoord(xmin, ymax)
        i1, j1 = raster.hdr.xy_to_imagecoord(xmax, ymin)
        data = raster.data[i0:i1, j0:j1].flatten()
        X, Y = [arr[i0:i1, j0:j1].ravel() for arr in (X, Y)]
    else:
        data = raster.data.flatten()
        X = X.flatten()
        Y = Y.flatten()

    # Crude conversion of coordinates to UTM if input EPSG = 4326
    if epsg_in is None:
        epsg_in = raster.hdr.epsg
    if epsg_in == 4326:
        epsg_utm = get_utm_EPSG(X[0], Y[0])
        X, Y = transform_coordinates(X, Y, epsg_in=epsg_in, epsg_out=epsg_utm)

    # Keep only finite points
    mask = np.isfinite(data)
    X, Y, data = [arr[mask] for arr in (X, Y, data)]

    # Randomly sample points
    rng = np.random.default_rng(seed)
    n_samples = min(max_samples, X.size)
    inds = rng.choice(X.size, size=n_samples, replace=False)
    x, y, z = [arr[inds] for arr in (X, Y, data)]

    # Remove ramp
    if remove_ramp:

        # Construct polynomial matrix
        G = np.zeros((n_samples, 6))
        G[:, 0] = x
        G[:, 1] = y
        G[:, 2] = 1.0
        G[:, 3] = x * y
        G[:, 4] = x * x
        G[:, 5] = y * y

        # Estimate the ramp and remove it
        m = np.linalg.lstsq(G, z, rcond=1.0e-12)[0]
        fit = np.dot(G, m)
        z -= fit

    # Build all the permutations
    ii, jj = [arr.flatten() for arr in np.meshgrid(range(n_samples), range(n_samples))]
    uu = np.flatnonzero(ii > jj)
    ii = ii[uu]
    jj = jj[uu]

    # Compute the distances
    dx = x[ii] - x[jj]
    dy = y[ii] - y[jj]
    dist = np.sqrt(dx**2 + dy**2)

    # Compute semivariogram
    if measure == 'semi':
        dv = (z[ii] - z[jj])**2
    elif measure == 'cov':
        dv = np.abs(z[ii] * z[jj])
    else:
        raise ValueError('Unsupported variogram measure.')

    # Digitize
    dist_bins = np.arange(0.0, dist_max, dist_spacing)
    inds = np.digitize(dist, dist_bins)

    # Average
    distance = []
    semivariogram = []
    std = []
    for i in tqdm(range(len(dist_bins) - 1)):
        uu = np.flatnonzero(inds == i)
        if len(uu) > 0:
            distance.append(dist_bins[i] + (dist_bins[i+1] - dist_bins[i]) / 2.0)
            semivariogram.append(0.5 * np.mean(dv[uu]))
            std.append(0.5 * np.std(dv[uu]))

    return np.array(distance), np.array(semivariogram), np.array(std)


# end of file


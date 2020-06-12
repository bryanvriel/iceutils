#-*- coding: utf-8 -*-

import numpy as np
from .boundary import Boundary, smoothe_line, transform_coordinates
import os

def load_glacier_boundaries(smooth=True, km=True, s=0.25, path=None):

    # Head path
    if path is None:
        path = '/home/briel/data/jakobshavn/velocity/giant2'

    # Scale factor
    if km:
        scale = 1.0e-3
    else:
        scale = 1.0

    # Load raw points for lower boundary
    dat = np.load(os.path.join(path, 'lower_boundary_points.npy'))
    lx, ly = dat[:,2], dat[:,3]
    lx *= scale
    ly *= scale

    # Optional smoothing
    if smooth:
        lx, ly = smoothe_line(lx, ly, n=200, s=s)
    lower = Boundary(x=lx, y=ly)

    # Load raw points for lower boundary
    dat = np.load(os.path.join(path, 'upper_boundary_points.npy'))
    ux, uy = dat[:,2], dat[:,3]
    ux *= scale
    uy *= scale

    # Optional smoothing
    if smooth:
        ux, uy = smoothe_line(ux, uy, n=200, s=s)
    upper = Boundary(x=ux, y=uy)

    return lower, upper

def load_front(smooth=True, km=True, s=0.25, path=None):

    # Head path
    if path is None:
        path = '/data0/briel/topo/jakobshavn_32m'
   
    # Scale factor
    if km:
        scale = 1.0e-3
    else:
        scale = 1.0

    # Load raw points for lower boundary
    dat = np.load(os.path.join(path, 'front_2017.npy'))
    x, y = dat[:,0], dat[:,1]
    x *= scale
    y *= scale

    # Optional smoothing
    if smooth:
        x, y = smoothe_line(x, y, n=200, s=s)
    front = Boundary(x=x, y=y)

    return front

def load_centerline(smooth=True, km=True, s=0.25, path=None, n=200):

    # Scale factor
    if km:
        scale = 1.0e-3
    else:
        scale = 1.0

    # Load transects
    if path is None:
        path = '/data0/briel/jakobshavn/velocity/giant2/long_term_analysis/along_flow_points.npy'
    tpts = np.load(path)
    cols, rows, x, y = [tpts[:,j] for j in range(4)]
    x *= scale
    y *= scale

    # Smoothing spline for points
    if smooth:
        x, y = smoothe_line(x, y, n=n, s=s)
    centerline = Boundary(x=x, y=y)

    return centerline

def load_extended_centerline(smooth=True, km=True, s=0.25, n=200, path=None, tpath=None,
                             txt=False):

    # Scale factor
    if km:
        scale = 1.0e-3
    else:
        scale = 1.0

    # Load transect along glacier
    if path is None:
        path = '/data0/briel/jakobshavn/velocity/giant2/long_term_analysis/along_flow_points.npy'
    if txt:
        x, y = np.loadtxt(path, unpack=True)
    else:
        tpts = np.load(path)
        cols, rows, x, y = [tpts[:,j] for j in range(4)]
    x *= scale
    y *= scale

    # Load transects along ice tongue
    if tpath is None:
        tpath = '/home/briel/python/iceutils/aux/jakobshavn_centerline.txt'
    lon, lat = np.loadtxt(tpath, unpack=True)

    # Convert to polar stereographic
    tx, ty = transform_coordinates(lon, lat, epsg_in=4326, epsg_out=3413)
    tx *= scale
    ty *= scale

    # Combine
    x = np.hstack((x, tx))
    y = np.hstack((y, ty))

    # Sort
    isort = np.argsort(x)
    x, y = x[isort], y[isort]

    # Smoothing spline for points
    if smooth:
        x, y = smoothe_line(x, y, n=n, s=s)
    centerline = Boundary(x=x, y=y)

    return centerline

def ocean_mask(hdr, path=None):
    """
    Create ocean mask for a given RasterInfo object.
    """
    from .raster import Raster

    # Load Arctic DEM
    if path is None:
        path = '/data0/briel/topo/jakobshavn_32m/jakobshavn_arcticdem_32m.tif'
    dem = Raster(rasterfile=path)

    # Crop to header
    dem.resample(hdr)

    # Use elevation as mask
    mask = dem.data < 100.0
    return mask

def sar_backscatter(hdr, smax=235.0, rgb=False, path=None, db=False):
    """
    Return array of SAR backscatter values for a given RasterInfo object. The source data 
    are from NSIDC GIMP:
    https://nsidc.org/data/nsidc-0723/versions/2
    """
    from .raster import Raster

    # Load the raster
    if path is None:
        path = '/data0/briel/jakobshavn/imagery/S1_2018-08-01_2018-08-06_northwest_v02.1.tif'
    slc = Raster(rasterfile=path)

    # Resample
    slc.resample(hdr)

    # Normalize
    sdata = slc.data
    sdata = sdata / smax

    # Return array or RGB
    if rgb:
        srgb = np.dstack((sdata, sdata, sdata, np.ones_like(sdata)))
        return srgb
    elif db:
        db = 10.0 * np.log10(sdata)
        mask = np.isfinite(db.ravel())
        low = np.percentile(db.ravel()[mask], 5)
        high = np.percentile(db.ravel()[mask], 99.9)
        return db, low, high
    else:
        return sdata

# end of file

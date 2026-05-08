#-*- coding: utf-8 -*-

import numpy as np
import numpy.lib.mixins
from numbers import Number

from scipy.ndimage import map_coordinates

try:
    from skimage.restoration.inpaint import inpaint_biharmonic
except ImportError:
    inpaint_biharmonic = None

import rasterio
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.errors import WindowError
from rasterio.transform import array_bounds
from rasterio.windows import Window
from rasterio.windows import crop as crop_window
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.windows import transform as window_transform
from rasterio.warp import calculate_default_transform, reproject

import warnings
import h5py

try:
    import cv2 as cv
except ImportError:
    cv = None

from .boundary import transform_coordinates

_SUPPORTED_READ_OPTIONS = {'out_dtype', 'masked', 'boundless', 'fill_value'}


def _as_crs(epsg=None, projstr=None, crs=None):
    """
    Normalize common CRS inputs to rasterio's CRS object.
    """
    if crs is not None:
        return CRS.from_user_input(crs)
    if epsg is not None:
        return CRS.from_epsg(int(epsg))
    if projstr is not None:
        return CRS.from_user_input(projstr)
    return None


def _as_dtype(dtype, data=None):
    """
    Normalize dtype inputs for rasterio profiles.
    """
    if dtype is None:
        if data is None:
            return None
        return np.dtype(data.dtype).name
    return np.dtype(dtype).name


def _validate_read_options(options):
    """
    Accept a small set of rasterio read options and reject legacy TranslateOptions.
    """
    unsupported = sorted(set(options) - _SUPPORTED_READ_OPTIONS)
    if unsupported:
        raise ValueError(
            'Unsupported raster read options for rasterio backend: %s' %
            ', '.join(unsupported)
        )
    return options


def _resampling_from_order(order):
    """
    Map scipy interpolation orders to rasterio resampling modes.
    """
    mapping = {
        0: Resampling.nearest,
        1: Resampling.bilinear,
        2: Resampling.cubic,
        3: Resampling.cubic,
    }
    return mapping.get(order, Resampling.cubic)


def _normalize_slice(s, size):
    """
    Convert a Python slice into positive start/stop indices.
    """
    if s is None:
        return 0, size
    start, stop, step = s.indices(size)
    if step != 1:
        raise ValueError('Raster windows do not support stepped slices.')
    return start, stop


def _window_from_inputs(transform, height, width, projWin=None, islice=None, jslice=None):
    """
    Convert projection windows or row/column slices to a rasterio Window.
    """
    if projWin is None and islice is None and jslice is None:
        return None

    if projWin is not None and islice is None and jslice is None:
        try:
            window = window_from_bounds(
                left=projWin[0], bottom=projWin[3],
                right=projWin[2], top=projWin[1],
                transform=transform
            )
            window = window.round_offsets().round_lengths()
            window = crop_window(window, height, width)
        except WindowError:
            warnings.warn('projWin outside of bounds; returning full extent.')
            return Window(0, 0, width, height)
        if window.width <= 0 or window.height <= 0:
            warnings.warn('projWin outside of bounds; returning full extent.')
            return Window(0, 0, width, height)
        return window

    row_start, row_stop = _normalize_slice(islice, height)
    col_start, col_stop = _normalize_slice(jslice, width)
    return Window(col_start, row_start, col_stop - col_start, row_stop - row_start)


def _slices_from_window(window):
    """
    Return row and column slices for a rasterio Window.
    """
    row_start = int(window.row_off)
    col_start = int(window.col_off)
    return (
        slice(row_start, row_start + int(window.height)),
        slice(col_start, col_start + int(window.width))
    )

# Dictionary for storing specific Numpy functions for operating on Raster objects
HANDLED_NP_FUNCTIONS = {}

class Raster(numpy.lib.mixins.NDArrayOperatorsMixin):
    """
    Class that encapsulates raster data and stores an instance of its header info.

    Parameters
    ----------
    args: list, optional
        Positional arguments for either raster file or stack file.
    data: ndarray, optional
        Array of raster data.
    hdr: RasterInfo, optional
        RasterInfo associated with data.
    rasterfile: str, optional
        Filename for rasterio-compatible raster to read.
    band: int, optional
        Band number to read from raster. Default: 1.
    stackfile: str, optional
        HDF5 file for Stack to read raster data from.
    h5path: str, optional
        H5 path from Stack corresponding to input dataset.
    islice: slice, optional
        Slice object specifying image rows to subset.
    jslice: slice, optional
        Slice object specifying image columns to subset.
    projWin: list, optional
        List of [upper_left_x, upper_left_y, lower_right_x, lower_right_y] for geographic
        bounding box to subset.
    gdalOpts: dict, optional
        Backward-compatible name for rasterio read options. Default: None.
    gdalMatch: bool, optional
        Find approximate projection info. Default: True.
    """

    def __init__(self,
                 *args,
                 data=None, hdr=None,
                 rasterfile=None, band=1,
                 stackfile=None, h5path=None,
                 islice=None, jslice=None,
                 projWin=None, gdalOpts=None, gdalMatch=True):

        # Default no data value
        self.nodataval = None
        if gdalOpts is None:
            gdalOpts = {}

        # If data and header are provided, save them and return
        if data is not None and hdr is not None:
            self.data = data
            self.hdr = hdr
            self.filename = None
            self.islice = islice
            self.jslice = jslice
            self.rasterfile = None
            return

        # Attempt to guess format of generic filename if provided
        if len(args) > 0:
            filename = args[0]
            if filename.endswith('.h5') or filename.endswith('.nc'):
                stackfile = filename
            else:
                rasterfile = filename

        # Load raster data and do any subsetting using rasterio directly
        if rasterfile is not None:
            self.data, self.hdr, self.nodataval = self.load_rasterio(
                rasterfile, band=band, projWin=projWin,
                islice=islice, jslice=jslice,
                gdalMatch=gdalMatch, **gdalOpts
            )
        elif stackfile is not None:
            assert h5path is not None
            # Load the header info manually
            self.hdr = RasterInfo(stackfile=stackfile)
            # Load subset/slicing information
            islice, jslice = self.hdr.subset_region(projWin=projWin, islice=islice, jslice=jslice)
            # Read data
            self.data = self.load_hdf5(stackfile, h5path, islice=islice, jslice=jslice)
        else:
            raise ValueError('Must provide rasterio-compatible raster or HDF5 stack.')

        # Cache the slices for provenance
        self.islice = islice
        self.jslice = jslice

        # Cache raster filename
        self.rasterfile = rasterfile

        return

    @staticmethod
    def load_rasterio(filename, band=1, projWin=None, islice=None, jslice=None,
                      gdalMatch=True, **gdalOpts):
        """
        Load raster data from file using rasterio.

        Parameters
        ----------
        filename: str
            Filename for rasterio-compatible raster to read.
        band: int, optional
            Band number to read from raster. Default: 1.
        projWin: list, optional
            Projection window for subsetting raster. Default: None.
        islice: slice, optional
            Slice object specifying image rows to subset.
        jslice: slice, optional
            Slice object specifying image columns to subset.
        gdalMatch: bool, optional
            Kept for API compatibility; ignored by rasterio backend.
        gdalOpts: **kwargs
            Extra rasterio read options. Supported: out_dtype, masked, boundless, fill_value.

        Returns
        -------
        d: ndarray
            Array for raster data.
        """
        read_opts = _validate_read_options(gdalOpts)
        with rasterio.open(filename) as src:
            window = _window_from_inputs(src.transform, src.height, src.width,
                                         projWin=projWin, islice=islice, jslice=jslice)
            d = src.read(band, window=window, **read_opts)
            transform = src.window_transform(window) if window is not None else src.transform
            height, width = d.shape[-2:]
            hdr = RasterInfo(transform=transform, crs=src.crs, shape=(height, width),
                             dtype=np.dtype(src.dtypes[band - 1]), nbands=src.count)
            nodataval = src.nodatavals[band - 1]

        return d, hdr, nodataval

    @staticmethod
    def load_gdal(filename, band=1, projWin=None, islice=None, jslice=None,
                  gdalMatch=True, **gdalOpts):
        """
        Compatibility alias for :meth:`load_rasterio`.
        """
        return Raster.load_rasterio(
            filename, band=band, projWin=projWin, islice=islice, jslice=jslice,
            gdalMatch=gdalMatch, **gdalOpts
        )

    @staticmethod
    def load_hdf5(filename, h5path, islice=None, jslice=None):
        """
        Load dataset from HDF5.

        Parameters
        ----------
        filename: str
            HDF5 file for Stack to read raster data from.
        h5path: str
            H5 path from Stack corresponding to input dataset.
        islice: slice, optional
            Slice object specifying image rows to subset.
        jslice: slice, optional
            Slice object specifying image columns to subset.

        Returns
        -------
        d: ndarray
            Array for raster data.
        """
        with h5py.File(filename, 'r') as fid:
            d = fid[h5path][()]
            if islice is not None:
                d = d[islice,:]
            if jslice is not None:
                d = d[:,jslice]
        return d

    def write_raster(self, filename, dtype=None, driver='ENVI',
                     epsg=None, nodataval=None, projstr=None):
        """
        Write data and header to a rasterio-supported raster.

        Parameters
        ----------
        filename: str
            Filename to write raster.
        dtype: dtype-like, optional
            Output dtype. Default: dtype of raster data.
        driver: str, optional
            Rasterio/GDAL-compatible raster driver for output raster file. Default: ENVI.
        epsg: int, optional
            EPSG code for output. Default: None.
        nodataval: int, float, optional
            No data value to write into the raster metadata. Default: None.
        projstr: str, optional
            PROJ string for output if no EPSG provided. Default: None.

        Returns
        -------
        None
        """
        dtype = _as_dtype(dtype, data=self.data)
        crs = _as_crs(epsg=epsg, projstr=projstr, crs=None)
        if crs is None:
            crs = self.hdr.crs

        profile = {
            'driver': driver,
            'height': int(self.hdr.ny),
            'width': int(self.hdr.nx),
            'count': 1,
            'dtype': dtype,
            'transform': self.hdr.transform,
        }
        if crs is not None:
            profile['crs'] = crs
        if nodataval is not None:
            profile['nodata'] = nodataval

        with rasterio.open(filename, 'w', **profile) as dst:
            dst.write(self.data.astype(dtype, copy=False), 1)
        return

    def write_gdal(self, filename, dtype=None, driver='ENVI',
                   epsg=None, nodataval=None, projstr=None):
        """
        Compatibility alias for :meth:`write_raster`.
        """
        return self.write_raster(
            filename, dtype=dtype, driver=driver, epsg=epsg,
            nodataval=nodataval, projstr=projstr
        )

    def resample(self, hdr, **kwargs):
        """
        Resample raster data in-place to another coordinate system provided by
        a RasterInfo object.

        Parameters
        ----------
        hdr: RasterInfo
            RasterInfo specifying output geometry to resample to.
        **kwargs:
            Extra parameters to pass to scipy.ndimage.map_coordinates.

        Returns
        -------
        None
        """
        # If RasterInfo objects are equivalent, do nothing
        if hdr == self.hdr:
            return

        # Prefer rasterio reprojection/resampling when full raster metadata is available.
        data = _reproject_array(self.data, self.hdr, hdr, nodataval=self.nodataval, **kwargs)
        if data is None:
            data = interpolate_raster(self, None, None, ref_hdr=hdr, time_index=None, **kwargs)

        # Update members
        self.data = data
        self.hdr = hdr

        return

    def downsample(self, factor=2, func=np.mean, cval=0.0):
        """
        Downsamples raster data in-place by an integer factor using local reduction
        defined by provided function. Calls skimage.measure.block_reduce.

        Parameters
        ----------
        factor: int or tuple of ints, optional
            Downsampling factor. Default: 2.
        func: callable, optional
            Function object which is used to calculate the return value for each local block.
            Default: numpy.mean.
        cval: float, optional
            Constant padding value if image is not perfectly divisible by the integer factors.

        Returns
        -------
        None
        """
        from skimage.measure import block_reduce

        # Create tuple if single integer provided
        if isinstance(factor, int):
            factor = (factor, factor)

        # Perform downscaling
        self.data = block_reduce(self.data, factor, func, cval)

        # Create new header
        X, Y = [arr[::factor[0], ::factor[1]] for arr in self.hdr.meshgrid()]
        self.hdr = RasterInfo(X=X, Y=Y, epsg=self.hdr.epsg)

        return

    def crop(self, xmin, xmax, ymin, ymax):
        """
        Crop a raster in-place using geographic bounds.

        Parameters
        ----------
        xmin: float
            Minimum X-coordinate.
        xmax: float
            Maximum X-coordinate.
        ymin: float
            Minimum Y-coordinate.
        ymax: float
            Maximum Y-coordinate.

        Returns
        -------
        None
        """
        # Crop header and get mask
        xmask, ymask = self.hdr.crop(xmin, xmax, ymin, ymax)

        # Crop data
        self.data = self.data[ymask,:][:,xmask]

        return

    def mask(self, mask, mask_value=np.nan):
        """
        Provide a mask and set values to mask_value in place.

        Parameters
        ----------
        mask: np.ndarray
            Boolean mask.
        mask_value: float, optional
            Mask value.

        Returns
        -------
        None
        """
        self.data[mask] = mask_value
        return

    def transect(self, point1, point2, n=200, order=3, return_location=False):
        """
        Extract a linear transect given two tuples of (X, Y) coordinates of the
        transect end points.

        Parameters
        ----------
        point1: list
            2-element list of [X, Y] for starting point of transect.
        point2: list
            2-element list of [X, Y] for ending point of transect.
        n: int, optional
            Number of points for transect. Default: 200.
        order: int, optional
            Order of interpolating spline. Default: 3.
        return_location: bool, optional
            Flag for returning transect coordinates in addition to data. Default: False.

        Returns
        -------
        z: ndarray
            Transect values.
        x: ndarray
            Transect X-coordinates (for return_location=True).
        y: ndarray
            Transect Y-coordinates (for return_location=True).
        """
        # Create the transect coordinates
        x = np.linspace(point1[0], point2[0], n)
        y = np.linspace(point1[1], point2[1], n)

        # Perform interpolation
        z = interpolate_raster(self, x, y, ref_hdr=None, time_index=None, order=order)

        # Return with or without coordinates
        if return_location:
            return z, x, y
        else:
            return z

    def __call__(self, x, y, order=3, mode='nearest', **kwargs):
        """
        Interpolates the raster at a set of coordinates.

        Parameters
        ----------
        x: ndarray
            X-coordinates for output interpolation grid.
        y: ndarray
            Y-coordinates for output interpolation grid.
        order: int, optional
            Order for interpolation. Default: 3.
        mode: str, optional
            Extrapolation flag for map_coordinates. Default: 'nearest'.
        **kwargs:
            Keyword arguments passed to scipy.ndimage.map_coordinates.

        Returns
        -------
        values: ndarray
            Interpolated values.
        """
        values = interpolate_raster(self, x, y, order=order, mode=mode, **kwargs)
        return np.squeeze(values)

    def __array__(self):
        """
        Return underlying raster data numpy array.
        """
        return self.data

    def __getitem__(self, coord):
        """
        Get raster data at given coordinates/slice.
        """
        i, j = coord
        return self.data[i, j]

    def __setitem__(self, coord, value):
        """
        Set raster data at given coordinates/slice.
        """
        i, j = coord
        self.data[i, j] = value

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Implements __array__ufunc__ for Raster objects following:
        https://numpy.org/doc/stable/user/basics.dispatch.html#basics-dispatch

        Arguments can be scalars, numpy arrays (w/ compatible shapes), and
        other Raster objects (w/ compatible hdr).
        """
        if method == '__call__':
            scalars = []
            for input in inputs:
                if isinstance(input, Number):
                    scalars.append(input)
                elif isinstance(input, np.ndarray):
                    assert input.shape == self.data.shape, 'Inconsistent array shapes.'
                    scalars.append(input)
                elif isinstance(input, self.__class__):
                    assert self.hdr == input.hdr, 'Inconsistent RasterInfo.'
                    scalars.append(input.data)
                else:
                    return NotImplemented
            return self.__class__(data=ufunc(*scalars, **kwargs), hdr=self.hdr)
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        """
        Implements __array__function__ for Raster objects following:
        https://numpy.org/doc/stable/user/basics.dispatch.html#basics-dispatch

        This allows for calling various numpy reduction functions on Raster
        data. Currently handled functions are in ice.HANDLED_NP_FUNCTIONS.
        """
        # Only allow implemented functions
        if func not in HANDLED_NP_FUNCTIONS:
            return NotImplemented

        # Note: this allows subclasses that don't override
        # __array_function__ to handle Raster objects.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented

        return HANDLED_NP_FUNCTIONS[func](*args, **kwargs)

    @property
    def size(self):
        return self.data.size
    @size.setter
    def size(self, value):
        raise ValueError('Cannot set size explicitly')

    def ravel(self):
        return self.data.ravel()

    def flatten(self):
        return self.data.flatten()


def implements(np_function):
   """
   Register an __array_function__ implementation for Raster objects.
   """
   def decorator(func):
       HANDLED_NP_FUNCTIONS[np_function] = func
       return func
   return decorator

@implements(np.any)
def any(r, **kwargs):
    """
    Implementation of np.any for Raster objects.
    """
    return np.any(r.data, **kwargs)

@implements(np.sum)
def sum(r, **kwargs):
    """
    Implementation of np.sum for Raster objects.
    """
    return np.sum(r.data, **kwargs)

@implements(np.mean)
def mean(r, **kwargs):
    """
    Implementation of np.mean for Raster objects.
    """
    return np.mean(r.data, **kwargs)

@implements(np.std)
def std(r, **kwargs):
    """
    Implementation of np.std for Raster objects.
    """
    return np.std(r.data, **kwargs)

@implements(np.median)
def median(r, **kwargs):
    """
    Implementation of np.median for Raster objects.
    """
    return np.median(r.data, **kwargs)

@implements(np.nansum)
def nansum(r, **kwargs):
    """
    Implementation of np.nansum for Raster objects.
    """
    return np.nansum(r.data, **kwargs)

@implements(np.nanmean)
def nanmean(r, **kwargs):
    """
    Implementation of np.nanmean for Raster objects.
    """
    return np.nanmean(r.data, **kwargs)

@implements(np.nanstd)
def nanstd(r, **kwargs):
    """
    Implementation of np.nanstd for Raster objects.
    """
    return np.nanstd(r.data, **kwargs)

@implements(np.nanmedian)
def nanmedian(r, **kwargs):
    """
    Implementation of np.nanmedian for Raster objects.
    """
    return np.nanmedian(r.data, **kwargs)

@implements(np.isfinite)
def isfinite(r, **kwargs):
    """
    Implementation of np.isfinite for Raster objects.
    """
    return np.isfinite(r.data, **kwargs)

@implements(np.flatnonzero)
def flatnonzero(r, **kwargs):
    """
    Implementation of np.flatnonzero for Raster objects.
    """
    return np.flatnonzero(r.data, **kwargs)

class RasterInfo:
    """
    Class that encapsulates raster size and geographic transform information.

    Parameters
    ----------
    rasterfile: str, optional
        Filename for rasterio-compatible raster to read.
    stackfile: str, optional
        HDF5 file for Stack to read raster data from.
    X: ndarray, optional
        Meshgrid of X-coordinates.
    Y: ndarray, optional
        Meshgrid of Y-coordinates.
    band: int, optional
        Band number to read from raster. Default: 1.
    epsg: int, optional
        EPSG code for input geographic data. Default: None.
    match: bool, optional
        Kept for API compatibility; ignored by rasterio backend.
    islice: slice, optional
        Slice object specifying image rows to subset.
    jslice: slice, optional
        Slice object specifying image columns to subset.
    nbands: int, optional
        Number of raster bands.
    """

    def __init__(self, rasterfile=None, stackfile=None, X=None, Y=None,
                 band=1, epsg=None, match=True, islice=None, jslice=None,
                 transform=None, crs=None, shape=None, dtype=None,
                 nbands=None, **kwargs):
        """
        Initialize attributes.
        """
        self.nbands = int(nbands) if nbands is not None else None

        if rasterfile is not None:
            self.load_rasterio_info(rasterfile, islice=islice, jslice=jslice,
                                    band=band, match=match)
            self.rasterfile = rasterfile
        elif stackfile is not None:
            self.load_stack_info(stackfile, islice=islice, jslice=jslice, **kwargs)
        elif X is not None and Y is not None:
            self.set_from_meshgrid(X, Y, epsg=epsg)
        elif transform is not None and shape is not None:
            self.ny, self.nx = int(shape[0]), int(shape[1])
            self.transform = Affine(*transform)
            self.crs = _as_crs(epsg=epsg, crs=crs)
            self._epsg = self.crs.to_epsg() if self.crs is not None else None
            self.units = 'm'
            self.dtype = np.dtype(dtype) if dtype is not None else None
            self._sync_from_transform()
        else:
            self.xstart = self.dx = self.ystart = self.dy = self.ny = self.nx = None
            self._epsg = None
            self.crs = None
            self.transform = Affine.identity()
            self.units = 'm'
            self.dtype = np.dtype(dtype) if dtype is not None else None

    def _sync_from_transform(self):
        """
        Keep legacy scalar transform attributes in sync with the affine transform.
        """
        self.xstart = self.transform.c
        self.dx = self.transform.a
        self.ystart = self.transform.f
        self.dy = self.transform.e

    def load_rasterio_info(self, rasterfile, projWin=None, islice=None, jslice=None,
                           band=1, match=False):
        """
        Read raster metadata from a rasterio dataset.

        Parameters
        ----------
        rasterfile: str
            Filename for rasterio-compatible raster to read.
        projWin: list, optional
            List of [upper_left_x, upper_left_y, lower_right_x, lower_right_y] for
            geographic bounding box to subset.
        islice: slice, optional
            Slice object specifying image rows to subset.
        jslice: slice, optional
            Slice object specifying image columns to subset.
        band: int, optional
            Band number to read from raster. Default: 1.
        match: bool, optional
            Kept for API compatibility; ignored by rasterio backend.

        Returns
        -------
        None
        """
        with rasterio.open(rasterfile) as src:
            window = _window_from_inputs(src.transform, src.height, src.width,
                                         projWin=projWin, islice=islice, jslice=jslice)
            self.ny = int(window.height) if window is not None else src.height
            self.nx = int(window.width) if window is not None else src.width
            self.transform = src.window_transform(window) if window is not None else src.transform
            self.crs = src.crs
            self._epsg = self.crs.to_epsg() if self.crs is not None else None
            self.units = 'm'
            self.dtype = np.dtype(src.dtypes[band - 1])
            self.nbands = int(src.count)
            self._sync_from_transform()

    def load_gdal_info(self, rasterfile, projWin=None, islice=None, jslice=None,
                       band=1, match=False):
        """
        Compatibility alias for :meth:`load_rasterio_info`.
        """
        return self.load_rasterio_info(
            rasterfile, projWin=projWin, islice=islice, jslice=jslice,
            band=band, match=match
        )

    def load_stack_info(self, stackfile, ds=None, islice=None, jslice=None):
        """
        Read header information from stack file.

        Parameters
        ----------
        stackfile: str
            HDF5 file for Stack to read raster data from.
        ds: str
            Name of dataset to read dimensions from.
        islice: slice, optional
            Slice object specifying image rows to subset.
        jslice: slice, optional
            Slice object specifying image columns to subset.

        Returns
        -------
        None
        """
        with h5py.File(stackfile, 'r') as fid:

            # Load coordinates
            try:
                X = fid['x'][()].squeeze()
                Y = fid['y'][()].squeeze()
            except KeyError:
                try:
                    X = fid['X'][()]
                    Y = fid['Y'][()]
                except KeyError:
                    Ny, Nx = fid[ds].shape[-2:]
                    X = np.arange(Nx)
                    Y = np.arange(Ny)

            # Extract 1D
            if X.ndim == 2:
                X = X[0,:]
            if Y.ndim == 2:
                Y = Y[:,0]

            # Incorporate row slicing
            if islice is not None:
                Y = Y[islice]
            # Incorporate column slicing
            if jslice is not None:
                X = X[jslice]

            # Try to read EPSG code
            try:
                epsg = int(fid.attrs['EPSG'])
            except KeyError:
                epsg = None

            # Set attributes using the same affine-aware path as rasters.
            self.set_from_meshgrid(*np.meshgrid(X, Y), epsg=epsg)

            # Set units
            self.units = 'm'

            # Set dtype when a reference dataset is available
            if ds is not None and ds in fid:
                self.dtype = fid[ds].dtype

    def subset_region(self, projWin=None, islice=None, jslice=None):
        """
        Subset the geographic metadata either by a projection window or image
        row and column slices.

        Parameters
        ----------
        projWin: list
            List of [upper_left_x, upper_left_y, lower_right_x, lower_right_y] for
            geographic bounding box to subset.
        islice: slice, optional
            Slice object specifying image rows to subset.
        jslice: slice, optional
            Slice object specifying image columns to subset.

        Returns
        -------
        islice: slice
            Slice object specifying image rows to subset.
        jslice: slice
            Slice object specifying image columns to subset.
        """
        window = _window_from_inputs(self.transform, self.ny, self.nx,
                                     projWin=projWin, islice=islice, jslice=jslice)
        if window is not None:
            islice, jslice = _slices_from_window(window)
            self.transform = window_transform(window, self.transform)
            self.ny, self.nx = int(window.height), int(window.width)
            self._sync_from_transform()

        return islice, jslice

    def set_from_meshgrid(self, X, Y, epsg=None, units='m'):
        """
        Set header information from meshgrid array.

        Parameters
        ----------
        X: ndarray, optional
            Meshgrid of X-coordinates.
        Y: ndarray, optional
            Meshgrid of Y-coordinates.
        epsg: int, optional
            EPSG code for input geographic data. Default: None.
        units: str, optional
            Units of coordinates.
        """
        X = np.asarray(X)
        Y = np.asarray(Y)
        self.ny, self.nx = X.shape
        a = X[0,1] - X[0,0] if self.nx > 1 else 1.0
        d = Y[0,1] - Y[0,0] if self.nx > 1 else 0.0
        b = X[1,0] - X[0,0] if self.ny > 1 else 0.0
        e = Y[1,0] - Y[0,0] if self.ny > 1 else 1.0
        self.transform = Affine(a, b, X[0,0], d, e, Y[0,0])
        self.crs = _as_crs(epsg=epsg)
        self._epsg = self.crs.to_epsg() if self.crs is not None else None
        self.units = units
        self.dtype = None
        self._sync_from_transform()

    def crop(self, xmin, xmax, ymin, ymax):
        """
        Crop a header by its geographic coordinates. Rounds to nearest pixel. Returns
        column/row masks.

        Parameters
        ----------
        xmin: float
            Minimum X-coordinate.
        xmax: float
            Maximum X-coordinate.
        ymin: float
            Minimum Y-coordinate.
        ymax: float
            Maximum Y-coordinate.

        Returns
        -------
        None
        """
        self._require_rectilinear()

        # Construct coordinates
        x = self.xcoords
        y = self.ycoords

        # Mask
        xmask = (x >= xmin) * (x <= xmax)
        ymask = (y >= ymin) * (y <= ymax)
        x = x[xmask]
        y = y[ymask]

        if len(x) == 0 or len(y) == 0:
            raise ValueError('Crop bounds do not overlap raster.')

        rows = np.flatnonzero(ymask)
        cols = np.flatnonzero(xmask)
        self.subset_region(islice=slice(rows[0], rows[-1] + 1),
                           jslice=slice(cols[0], cols[-1] + 1))

        return xmask, ymask

    def read_GCPs(self, rasterfile=None, gcp_epsg=None, epsg_out=None, k=3, s=5, scale=1.0):
        """
        Load ground control points (GCPs) from rasterio Dataset. Then, construct 2D
        interpolating splines that represent mapping from image to georeferenced
        coordinates with mapping determined from GCPs.

        Parameters
        ----------
        rasterfile: str, optional
            Raster file to read GCPs from. Use cached source raster by default.
        gcp_epsg: int, optional
            Override EPSG code for GCP coordinates. Default determined from GCP projection.
        epsg_out: int, optional
            EPSG code for output X-Y coordinates. It not provided, use GCP EPSG.
        k: int, optional
            Order of the splines. Default: 3.
        s: float, optional
            Smoothing factor for splines. See docs for SmoothBivariateSpline. Default: 5.
        scale: float, optional
            Scale factor for image and geographic coordinates. Default: 1.0.

        Returns
        -------
        None
        """
        from scipy.interpolate import SmoothBivariateSpline

        # Specify filename
        if rasterfile is None:
            rasterfile = self.rasterfile
        assert rasterfile is not None, 'No valid raster file specified.'

        with rasterio.open(rasterfile) as src:
            GCPs, gcp_crs = src.gcps
            if gcp_epsg is None and gcp_crs is not None:
                gcp_epsg = gcp_crs.to_epsg()

        # Unpack GCP information
        N_gcp = len(GCPs)
        pixel = np.zeros(N_gcp)
        line = np.zeros(N_gcp)
        x = np.zeros(N_gcp)
        y = np.zeros(N_gcp)
        for i, gcp in enumerate(GCPs):
            pixel[i], line[i], x[i], y[i] = gcp.col, gcp.row, gcp.x, gcp.y

        # Convert GCP coordinates to another projection if needed
        if epsg_out != gcp_epsg:
            x, y = transform_coordinates(x, y, gcp_epsg, epsg_out)

        # Scale the values
        self._gcp_scale = scale
        x, y, line, pixel = [scale * v for v in (x, y, line, pixel)]

        # Build splines
        self._gcp_spline_row = SmoothBivariateSpline(x, y, line, kx=k, ky=k, s=s)
        self._gcp_spline_col = SmoothBivariateSpline(x, y, pixel, kx=k, ky=k, s=s)
        self._gcp_epsg = epsg_out

    def __eq__(self, other):
        """
        Check for equivalence in headers.
        """
        if self.shape != other.shape: return False
        if not np.allclose(tuple(self.transform), tuple(other.transform), atol=1.0e-8):
            return False
        if self.units != other.units:
            return False
        if self.crs is not None and other.crs is not None and self.crs != other.crs:
            return False
        return True

    def contains_point(self, x, y):
        """
        Convenience function to check whether a coordinate is within the geographic
        extent of the raster. Coordinate must have same projection as raster.

        Parameters
        ----------
        x: float
            Input X-coordinate.
        y: float
            Input Y-coordinate.

        Returns
        -------
        flag: bool
            Boolean specifying whether point lies within bounds.
        """
        # Convert coordinate to image coordinate
        row, col = self.xy_to_imagecoord(x, y)

        # Check bounds
        if row < 0 or row > (self.ny - 1):
            return False
        if col < 0 or col > (self.nx - 1):
            return False
        return True

    def convert_units(self, out_units):
        """
        Convenience function to convert coordinate units.
        """
        # Get the scale factor
        if out_units == 'km':
            if self.units == 'm':
                scale = 1.0e-3
            elif self.units == 'km':
                scale = 1.0
        elif out_units == 'm':
            if self.units == 'km':
                scale = 1.0e3
            elif self.units == 'm':
                scale = 1.0
        else:
            raise ValueError('Unit %s not supported.' % out_units)

        # Apply scale to the full affine transform.
        self.transform = Affine(
            self.transform.a * scale, self.transform.b * scale, self.transform.c * scale,
            self.transform.d * scale, self.transform.e * scale, self.transform.f * scale
        )
        self._sync_from_transform()

        # Done
        return

    @property
    def is_rectilinear(self):
        """
        True when the affine transform has no rotation or shear terms.
        """
        return abs(self.transform.b) < 1.0e-12 and abs(self.transform.d) < 1.0e-12

    def _require_rectilinear(self):
        """
        Guard helpers that only make sense as one-dimensional x/y coordinates.
        """
        if not self.is_rectilinear:
            raise ValueError('Use meshgrid() for rasters with rotated or skewed affine transforms.')

    @property
    def bounds(self):
        """
        Return raster bounds as (west, south, east, north).
        """
        return array_bounds(self.ny, self.nx, self.transform)

    @property
    def shape(self):
        """
        Return raster shape.
        """
        return (self.ny, self.nx)

    @property
    def spacing(self):
        """
        Return pixel spacing.
        """
        return (self.transform.e, self.transform.a)

    @property
    def xstop(self):
        if self.is_rectilinear:
            return self.xstart + self.nx * self.dx
        return self.bounds[2]

    @property
    def ystop(self):
        if self.is_rectilinear:
            return self.ystart + self.ny * self.dy
        return self.bounds[1]

    @property
    def geotransform(self):
        """
        Return GDAL-compatible geo transform array.
        """
        return [
            self.transform.c,
            self.transform.a,
            self.transform.b,
            self.transform.f,
            self.transform.d,
            self.transform.e
        ]

    @property
    def epsg(self):
        """
        Return read-only EPSG code.
        """
        return self._epsg
    @epsg.setter
    def epsg(self, value):
        raise NotImplementedError('Cannot set EPSG value explicitly.')

    @property
    def extent(self):
        """
        Return matplotlib-compatible extent of (left, right, bottom, top).
        """
        west, south, east, north = self.bounds
        return np.array([west, east, south, north])

    @property
    def projWin(self):
        """
        Return GDAL-style projection window of:
        [upper_left_x, upper_left_y, lower_right_x, lower_right_y].
        """
        west, south, east, north = self.bounds
        return np.array([west, north, east, south])

    @property
    def xlim(self):
        """
        Return matplotlib-compatible x-limits (left, right).
        """
        west, _, east, _ = self.bounds
        return np.array([west, east])

    @property
    def ylim(self):
        """
        Return matplotlib-compatible y-limits (bottom, top).
        """
        _, south, _, north = self.bounds
        return np.array([south, north])

    @property
    def xspan(self):
        """
        Returns spatial span in X direction.
        """
        west, _, east, _ = self.bounds
        return abs(east - west)

    @property
    def yspan(self):
        """
        Returns spatial span in Y direction.
        """
        _, south, _, north = self.bounds
        return abs(north - south)

    @property
    def xcoords(self):
        """
        Returns array of X coordinates.
        """
        self._require_rectilinear()
        return self.xstart + self.dx * np.arange(self.nx)

    @property
    def ycoords(self):
        """
        Returns array of Y coordinates.
        """
        self._require_rectilinear()
        return self.ystart + self.dy * np.arange(self.ny)

    def meshgrid(self):
        """
        Construct meshgrids for geo coordinates.
        """
        col, row = self.coord_meshgrid()
        return self.imagecoord_to_xy(row, col)

    def coord_meshgrid(self):
        """
        Construct meshgrids for image coordinates.
        """
        row = np.arange(self.ny, dtype=int)
        col = np.arange(self.nx, dtype=int)
        return np.meshgrid(col, row)

    def xy_to_imagecoord(self, x, y, round_values=True):
        """
        Converts geographic XY point to row and column coordinate.
        """
        col, row = (~self.transform) * (x, y)
        if round_values:
            row = np.round(row).astype(int)
            col = np.round(col).astype(int)
        return row, col

    def imagecoord_to_xy(self, row, col):
        """
        Converts row and column coordinate to geographic XY.
        """
        x, y = self.transform * (col, row)
        return x, y

    def xy_to_imagecoord_gcp(self, x, y):
        """
        Converts geographic XY point to row and column coordinate using 2D splines
        formed from ground control points (GCPs). Must have called
        RasterInfo.read_GCPs first.
        """
        # Check for existence of GCP splines
        if not hasattr(self, '_gcp_spline_row') or not hasattr(self, '_gcp_spline_col'):
            raise ValueError('Must run RasterInfo.read_GCPs first.')

        # Evaluate splines
        col = self._gcp_spline_col(self._gcp_scale*x, self._gcp_scale*y, grid=False)
        row = self._gcp_spline_row(self._gcp_scale*x, self._gcp_scale*y, grid=False)

        # Return
        return row/self._gcp_scale, col/self._gcp_scale


# --------------------------------------------------------------------------------
# Global utility functions
# --------------------------------------------------------------------------------

def interpolate_raster(raster, x, y, ref_hdr=None, time_index=None, **kwargs):
    """
    Interpolate raster at arbitrary points.

    Parameters
    ----------
    raster: Raster
        Raster to interpolate.
    x: ndarray
        X-coordinates for output interpolation grid.
    y: ndarray
        Y-coordinates for output interpolation grid.
    ref_hdr: RasterInfo, optional
        RasterInfo to read output coordinates from. Default: None.
    time_index: int, optional
        Time index to extract time slice from raster stack. Default: None.
    **kwargs:
        Keyword arguments passed to scipy.ndimage.map_coordinates.

    Returns
    -------
    values: ndarray
        Interpolated values.
    """
    # Extract time slice if index provided
    if time_index is not None:
        r_data = raster.data[time_index,:,:]
    else:
        r_data = raster.data

    # Interpolate
    return interpolate_array(r_data, raster.hdr, x, y, ref_hdr=ref_hdr, **kwargs)

def interpolate_array(array, hdr, x, y, ref_hdr=None, **kwargs):
    """
    Interpolate 2D array at arbitrary points.

    Parameters
    ----------
    array: ndarray
        2D array to interpolate.
    hdr: RasterInfo
        RasterInfo object specifying geographic data for array.
    x: ndarray
        X-coordinates for output interpolation grid.
    y: ndarray
        Y-coordinates for output interpolation grid.
    ref_hdr: RasterInfo, optional
        RasterInfo to read output coordinates from. Default: None.
    **kwargs:
        Keyword arguments passed to scipy.ndimage.map_coordinates.

    Returns
    -------
    values: ndarray
        Interpolated values.
    """
    # If a RasterInfo object has been passed, generate output coordinates
    if ref_hdr is not None:
        x, y = ref_hdr.meshgrid()

    # Check if scalars are passed
    elif x is not None and not isinstance(x, np.ndarray):
        x = np.array([x])
        y = np.array([y])

    # Ravel points to 1D
    row, col = hdr.xy_to_imagecoord(x.ravel(), y.ravel(), round_values=False)
    coords = np.vstack((row, col))

    # Interpolate
    values = map_coordinates(array, coords, output=None, prefilter=False, **kwargs)

    # Recover original shape and return
    return values.reshape(x.shape)


def _reproject_array(array, src_hdr, dst_hdr, nodataval=None, order=3, **kwargs):
    """
    Reproject or resample an array with rasterio when complete spatial metadata exists.
    """
    if src_hdr.transform is None or dst_hdr.transform is None:
        return None
    if src_hdr.crs is None or dst_hdr.crs is None:
        return None

    dst_nodata = kwargs.get('dst_nodata', kwargs.get('cval', None))
    src_nodata = kwargs.get('src_nodata', nodataval)
    fill_value = 0 if dst_nodata is None else dst_nodata
    try:
        destination = np.full(dst_hdr.shape, fill_value, dtype=array.dtype)
    except ValueError:
        destination = np.zeros(dst_hdr.shape, dtype=array.dtype)

    reproject(
        source=array,
        destination=destination,
        src_transform=src_hdr.transform,
        src_crs=src_hdr.crs,
        src_nodata=src_nodata,
        dst_transform=dst_hdr.transform,
        dst_crs=dst_hdr.crs,
        dst_nodata=dst_nodata,
        resampling=_resampling_from_order(order),
        num_threads=kwargs.get('num_threads', 1)
    )
    return destination

def warp(raster, target_epsg=None, target_srs=None, source_srs=None,
         target_hdr=None, target_dims=None, target_res=None,
         n_proc=1, **kwargs):
    """
    Warp raster to another RasterInfo hdr object with a different projection system.
    Currently only supports EPSG projection representations.

    Parameters
    ----------
    raster: Raster
        Raster object to warp.
    target_epsg: int, optional
        Specific EPSG of output reference system. Default: None.
    target_srs: str, optional
        SRS for target projection if EPSG not provided. Default: None.
    source_srs: str, optional
        SRS for source projection if EPSG not in header. Default: None.
    target_hdr: RasterInfo, optional
        RasterInfo specifying output geographical grid and projection.
    target_dims: (list, tuple), optional
        Output warped image dimensions. Default: None.
    target_res: float, optional
        Output pixel spacing. Default: None.
    n_proc: int, optional
        Number of processors to run warping on. Default: 1.
    **kwargs:
        Keyword arguments passed to scipy.ndimage.map_coordinates.

    Returns
    -------
    warped_raster: Raster
        Output warped Raster object.
    """
    src_crs = raster.hdr.crs
    if src_crs is None and source_srs is not None:
        src_crs = CRS.from_user_input(source_srs)
    if src_crs is None:
        raise AssertionError('Must provide source_srs since no CRS found for input.')

    if target_epsg is not None:
        dst_crs = CRS.from_epsg(target_epsg)
    elif target_srs is not None:
        dst_crs = CRS.from_user_input(target_srs)
    elif target_hdr is not None and target_hdr.crs is not None:
        dst_crs = target_hdr.crs
    else:
        raise ValueError('Must provide RasterInfo, target_epsg, or target_srs.')

    if target_hdr is None:
        kwargs_transform = {}
        if target_dims is not None:
            kwargs_transform['dst_height'], kwargs_transform['dst_width'] = target_dims
        elif target_res is not None:
            kwargs_transform['resolution'] = target_res

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, raster.hdr.nx, raster.hdr.ny, *raster.hdr.bounds,
            **kwargs_transform
        )
        target_hdr = RasterInfo(transform=dst_transform, crs=dst_crs,
                                shape=(dst_height, dst_width), dtype=raster.data.dtype)
    else:
        if target_hdr.crs is None or target_hdr.crs != dst_crs:
            target_hdr = RasterInfo(transform=target_hdr.transform, crs=dst_crs,
                                    shape=target_hdr.shape, dtype=raster.data.dtype)

    data_warped = _reproject_array(
        raster.data, RasterInfo(transform=raster.hdr.transform, crs=src_crs,
                                shape=raster.hdr.shape, dtype=raster.data.dtype),
        target_hdr, nodataval=raster.nodataval, num_threads=n_proc, **kwargs
    )

    # Return new raster
    return Raster(data=data_warped, hdr=target_hdr)

def warp_with_gcp_splines(raster, gcp_hdr, x=None, y=None, out_hdr=None, order=3):
    """
    Warp a raster to output grid using pre-constructed 2D interpolation splines
    formed from GCPs. Must call Raster.read_GCPs first.

    Parameters
    ----------
    raster: Raster
        Input raster to warp.
    gcp_hdr: RasterInfo
        Input RasterInfo object with GCP spline attributes.
    x: ndarry, optional
        X-coordinates for output grid.
    y: ndarray, optional
        Y-coordinates for output grid.
    out_hdr: RasterInfo, optional
        RasterInfo for specifying output coordinates if not otherwise specified.
    order: int, optional
        Order of interpolation scheme. Default: 3.

    Returns
    -------
    out_raster: Raster
        Output warped raster object.
    """
    # Get output coordinates from ref_hdr if not specified
    if x is None and y is None and out_hdr is not None:
        assert out_hdr.epsg == gcp_hdr._gcp_epsg, 'EPSG mismatch with GCP splines.'
        x, y = out_hdr.meshgrid()
    else:
        out_hdr = RasterInfo(X=x, Y=y, epsg=gcp_hdr._gcp_epsg)

    # Evalute splines to get image coordinates for output grid
    grid_row, grid_col = gcp_hdr.xy_to_imagecoord_gcp(x.ravel(), y.ravel())

    # Adjust image coordinates for any offsets from subsetting
    if raster.islice is not None:
        grid_row -= raster.islice.start
    if raster.jslice is not None:
        grid_col -= raster.jslice.start

    # Interpolate
    points = np.vstack((grid_row, grid_col))
    out = map_coordinates(raster.data, points, prefilter=False, mode='constant', cval=np.nan,
                          order=order)

    # Create new Raster
    return Raster(data=out.reshape(x.shape), hdr=out_hdr)

def write_raster(arrays, filename, geotransform=None, epsg=None,
                 projstr=None, driver='ENVI', nodataval=None, dtype=None):
    """
    Global function for writing arrays to raster file with projection
    information.
    """
    # If only a single array is passed, make a tuple
    if not isinstance(arrays, tuple):
        arrays = (arrays,)
    n_bands = len(arrays)
    Ny, Nx = arrays[0].shape

    dtype = _as_dtype(dtype, data=arrays[0])
    transform = Affine.identity()
    if geotransform is not None:
        transform = Affine.from_gdal(*geotransform)
    crs = _as_crs(epsg=epsg, projstr=projstr)

    profile = {
        'driver': driver,
        'height': Ny,
        'width': Nx,
        'count': n_bands,
        'dtype': dtype,
        'transform': transform,
    }
    if crs is not None:
        profile['crs'] = crs
    if nodataval is not None:
        profile['nodata'] = nodataval

    with rasterio.open(filename, 'w', **profile) as dst:
        for bcnt, array in enumerate(arrays):
            dst.write(array.astype(dtype, copy=False), bcnt + 1)


def write_gdal(arrays, filename, geotransform=None, epsg=None,
               projstr=None, driver='ENVI', nodataval=None, dtype=None):
    """
    Compatibility alias for :func:`write_raster`.
    """
    return write_raster(
        arrays, filename, geotransform=geotransform, epsg=epsg,
        projstr=projstr, driver=driver, nodataval=nodataval, dtype=dtype
    )

def write_array_as_raster(array, hdr, filename, epsg=None, projstr=None,
                          dtype=None, driver='ENVI'):
    """
    Convenience function to write a NumPy array to a raster file with a given RasterInfo.

    Parameters
    ----------
    array: ndarray
        2D array to write values out.
    hdr: RasterInfo
        RasterInfo specifying geographical grid and projection for array.
    filename: str
        Output filename.
    epsg: int, optional
        Specific EPSG code for output projection. Default: None.
    projstr: str, optional
            PROJ string for output if no EPSG provided. Default: None.
    dtype: dtype-like, optional
        Output dtype. Default: dtype of input array.

    Returns
    -------
    None
    """
    # Check shapes
    assert array.shape == (hdr.ny, hdr.nx), 'Incompatible shapes'
    # Write raster
    raster = Raster(data=array, hdr=hdr)
    # Check if header has EPSG code
    if epsg is None and hdr.epsg is not None:
        epsg = hdr.epsg
    # Write
    raster.write_raster(filename, epsg=epsg, projstr=projstr, dtype=dtype, driver=driver)

def griddata(x, y, z, hdr=None, dx=100, dy=100, x_extent=None, y_extent=None,
             method='linear', epsg=None):
    """
    Utility function to create a 2D array for scattered data. Calls griddata from
    scipy.interpolate.

    Parameters
    ----------
    x: ndarray
        Array of x-coordinates.
    y: ndarray
        Array of y-coordinates.
    z: ndarray
        Array of data values.
    hdr: RasterInfo
        Output RasterInfo for specifying grid geometry.
    dx: float, optional
        Spacing of output grid in x-direction. Used if no hdr supplied.
    dy: float, optional
        Spacing of output grid in y-direction. Used if no hdr supplied.
    x_extent: list, optional
        [x_min, x_max] bounds of output grid. Default computed from data.
    y_extent: list, optional
        [y_min, y_max] bounds of output grid. Default computed from data.
    method: str, optional
        Interpolation method passed to scipy.interpolate.griddata. Default: 'linear'.
    epsg: int, optional
        EPSG of output raster. Default: None.

    Returns
    -------
    raster: Raster
        Output raster object.
    """
    from scipy.interpolate import griddata

    # Define the output grid
    if hdr is not None:
        Xg, Yg = hdr.meshgrid()
        out_hdr = hdr

    else:
        if x_extent is None:
            x_min, x_max = np.min(x), np.max(x)
        else:
            x_min, x_max = x_extent

        if y_extent is None:
            y_min, y_max = np.min(y), np.max(y)
        else:
            y_min, y_max = y_extent

        Nx = int((x_max - x_min) / dx) + 1
        Ny = int((y_max - y_min) / abs(dy)) + 1
        xg = x_min + dx * np.arange(Nx)
        yg = y_max + dy * np.arange(Ny)
        Xg, Yg = np.meshgrid(xg, yg)
        out_hdr = RasterInfo(X=Xg, Y=Yg, epsg=epsg)

    # Call griddata
    pts = np.column_stack((x, y))
    Zg = griddata(pts, z, (Xg, Yg), method=method)
    
    # Wrap in raster
    raster = Raster(data=Zg, hdr=out_hdr)

    return raster

def inpaint(raster, mask=None, method='spring', r=3.0):
    """
    Inpaint a raster at NaN values or with an input mask.

    Parameters
    ----------
    raster: Raster or ndarray
        Input raster or array object to inpaint.
    mask: None or ndarry, optional
        Mask with same shape as raster specifying pixels to inpaint. If None,
        mask computed from NaN values. Default: None.
    method: str, optional
        Inpainting method from ('telea', 'biharmonic'). Default: 'telea'.
    r: scalar, optional
        Radius in pixels of neighborhood for OpenCV inpainting. Default: 3.0.

    Returns
    -------
    out_raster: Raster
        Output raster object.
    """
    if isinstance(raster, Raster):
        rdata = raster.data
    else:
        rdata = raster

    # Create mask
    if mask is None:
        mask = np.isnan(rdata)
    else:
        assert mask.shape == rdata.shape, 'Mask and raster shape mismatch.'

    # Check suitability of inpainting method with available packages
    if method == 'telea' and cv is None:
        warnings.warn('OpenCV package cv2 not found; falling back to spring inpainting.')
        method = 'spring'

    # Call inpainting
    if method == 'spring':
        inpainted = _inpaint_spring(rdata, mask)
    elif method == 'telea': 
        umask = mask.astype(np.uint8)
        inpainted = cv.inpaint(rdata, umask, r, cv.INPAINT_TELEA)
    elif method == 'biharmonic' and inpaint_biharmonic is None:
        raise ImportError('scikit-image is required for biharmonic inpainting.')
    elif method == 'biharmonic': 
        inpainted = inpaint_biharmonic(rdata, mask, multichannel=False)
    else:
        raise ValueError('Unsupported inpainting method.')

    # Return new raster or array
    if isinstance(raster, Raster):
        return Raster(data=inpainted, hdr=raster.hdr)
    else:
        return inpainted

def pad_raster(raster, pad_width, mode='constant', **kwargs):
    """
    Pads a raster using np.pad and updates RasterInfo information.

    Parameters
    ----------
    raster: Raster
        Raster to pad.
    pad_width: {sequence, array_like, int}
        Number of values padded to the edges of each axis.
        ((before_1, after_1), ... (before_N, after_N)) unique pad widths
        for each axis.
        ((before, after),) yields same before and after pad for each axis.
        (pad,) or int is a shortcut for before = after = pad width for all
        axes.
    mode: str or function, optional
        Padding mode as defined by np.pad.
    **kwargs:
        Extra kwargs passed to np.pad

    Returns
    -------
    out_raster: Raster
        Output padded raster with adjusted geographic info.
    """
    def _adjust_coords(coords, pad_before, pad_after):
        x0 = coords[0]
        x1 = coords[-1]
        dx = coords[1] - coords[0]
        x0_adj = x0 - dx * pad_before
        x_before = x0_adj + dx * np.arange(pad_before)
        x_after = x1 + dx + dx * np.arange(pad_after)
        return np.hstack((x_before, coords, x_after))

    # Unpack pad width in each dimension
    if isinstance(pad_width, int):
        xpad_before = xpad_after = ypad_before = ypad_after = pad_width
    else:
        (ypad_before, ypad_after), (xpad_before, xpad_after) = pad_width

    # Adjust the coordinates
    x = _adjust_coords(raster.hdr.xcoords, xpad_before, xpad_after)
    y = _adjust_coords(raster.hdr.ycoords, ypad_before, ypad_after)
    X, Y = np.meshgrid(x, y)
    out_hdr = RasterInfo(X=X, Y=Y, epsg=raster.hdr.epsg)

    # Do the padding
    data = np.pad(raster.data, pad_width, mode=mode, **kwargs)

    # Return new raster
    return Raster(data=data, hdr=out_hdr)
        
def render_kml(raster, filename, dpi=300, cmap='viridis', clim=None, colorbar=False, n_proc=1):
    """
    Renders Raster data to an image and creates a KML for viewing in Google Earth.

    Parameters
    ----------
    raster: Raster
        Raster to render to KML.
    filename: str
        Name of output KML file.
    dpi: int
        DPI of saved PNG file. Default: 300.
    cmap: {str, matplotlib.colors.ListedColormap}, optional
        Colormap for plotting data. Default: 'viridis'.
    clim: {tuple, None}, optional
        Color limit for plotting data. Default: None.
    colorbar: bool, optional
        Put colorbar overlay on image. Default: False.
    n_proc: int, optional
        Number of processors for warping raster if not in EPSG:4326. Default: 1.
    kmz: bool, optional
        Save as KMZ instead of KML. Default: False.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt
    import simplekml

    # First warp raster if not provided in EPSG:4326 projection
    if raster.hdr.epsg != 4326:
        print('warping')
        raster = warp(raster, target_epsg=4326, order=1, n_proc=n_proc)

    # Make an image
    fig, ax = plt.subplots(figsize=(11,7))
    im = ax.imshow(raster.data, cmap=cmap, clim=clim)
    ax.axis('off')

    # Save to PNG
    froot = filename.split('.')[0]
    pngfile = froot + '.png'
    fig.savefig(pngfile, dpi=dpi, bbox_inches='tight', pad_inches=0.0, transparent=True)

    # Get coordinate bounds
    west, east, south, north = raster.hdr.extent

    # Make KML
    kml = simplekml.Kml()
    ground = kml.newgroundoverlay(name='GroundOverlay')
    ground.icon.href = pngfile
    ground.latlonbox.north = north
    ground.latlonbox.south = south
    ground.latlonbox.east = east
    ground.latlonbox.west = west

    # Colorbar
    if colorbar:

        # Make the colorbar image
        fig_cbar = plt.figure(figsize=(1.0, 2.0))
        cax = fig_cbar.add_axes([0.0, 0.05, 0.2, 0.9])
        cbar = fig_cbar.colorbar(im, cax=cax)
        cbarfile = froot + '_colorbar.png'
        fig_cbar.savefig(cbarfile, dpi=200, transparent=False)

        # Add to KML
        screen = kml.newscreenoverlay(name='ScreenOverlay')
        screen.icon.href = cbarfile
        screen.overlayxy = simplekml.OverlayXY(x=0, y=0,
                                               xunits=simplekml.Units.fraction,
                                               yunits=simplekml.Units.fraction)
        screen.screenxy = simplekml.ScreenXY(x=0.015, y=0.075,
                                             xunits=simplekml.Units.fraction,
                                             yunits=simplekml.Units.fraction)
        screen.rotationXY = simplekml.RotationXY(x=0.5, y=0.5,
                                                 xunits=simplekml.Units.fraction,
                                                 yunits=simplekml.Units.fraction)
        screen.size.x = 0
        screen.size.y = 0
        screen.size.xunits = simplekml.Units.fraction
        screen.size.yunits = simplekml.Units.fraction
        screen.visibility = 1

    if filename.endswith('.kmz'):
        kml.savekmz(filename)
    else:
        kml.save(filename)

    return

def wkt_to_epsg(wkt, match=False):
    """
    Convenience function to convert a projection formatted as a WKT (Well Known Transformation)
    to an EPSG code. The match argument is kept for compatibility and is ignored by
    the rasterio backend.
    """
    if wkt is None or wkt == '':
        return None
    crs = CRS.from_user_input(wkt)
    return crs.to_epsg()

def get_chunks(dims, chunk_y, chunk_x):
    """
    Utility function to get chunk bounds.

    Parameters
    ----------
    dims: tuples for dimensions
        (Ny, Nx) dimensions.
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
    Ny, Nx = dims
    Ny_chunk = int(Ny // chunk_y)
    Nx_chunk = int(Nx // chunk_x)
    if Ny % chunk_y != 0:
        Ny_chunk += 1
    if Nx % chunk_x != 0:
        Nx_chunk += 1

    # Now construct chunk bounds
    chunks = []
    for i in range(Ny_chunk):
        if i == Ny_chunk - 1:
            nrows = Ny - chunk_y * i
        else:
            nrows = chunk_y
        istart = chunk_y * i
        iend = istart + nrows
        for j in range(Nx_chunk):
            if j == Nx_chunk - 1:
                ncols = Nx - chunk_x * j
            else:
                ncols = chunk_x
            jstart = chunk_x * j
            jend = jstart + ncols
            chunks.append([slice(istart,iend), slice(jstart,jend)])

    return chunks

def load_ann(filename, comment=';'):
    """
    Load UAVSAR annotation file values into dictionary.

    Parameters
    ----------
    filename: str
        Filename for UAVSAR annotation file.
    comment: str, optional
        Comment string. Default: ';'

    Returns
    -------
    ann: dict
        Dictionary of metadata values.
    """
    ann = {}
    with open(filename, 'r') as fid:
        for input_line in fid:

            # Skip empty lines
            line = input_line.strip()
            if len(line) < 1:
                continue

            # Skip lines that start with a comment
            if line.startswith(comment):
                continue

            # Split the line
            items = line.split(' = ')
            if len(items) < 2:
                continue

            # Parse first item for key
            key = items[0].split('(')[0].strip()

            # Strip second item of any trailing comments
            value_str = items[1].strip()
            ind_comment = value_str.find(comment)
            if ind_comment > -1:
                value_str = value_str[:ind_comment].strip()

            # Store in dictionary
            ann[key] = value_str

    return ann

def get_utm_zone(lon):
    """
    Computes UTM zone from longitude (in degrees).
    """
    z = lon + 180.0
    z /= 6.0
    zone = int(np.ceil(z))
    return zone

def get_utm_EPSG(lon, lat):
    """
    Automatically constructs UTM EPSG code from a lon/lat coordinate (degrees).
    """
    zone = get_utm_zone(lon)
    if lat >= 0.0:
        epsg = '326%02d' % zone
    else:
        epsg = '327%02d' % zone
    return int(epsg)


def _inpaint_spring(ain, mask):
    '''Returns the inpainted matrix using the spring metaphor.
       All NaN values in the matrix are filled in.
       
       Based on the original inpaintnans package by John D'Errico.
       http://www.mathworks.com/matlabcentral/fileexchange/4551-inpaintnans'''
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla

    dims = ain.shape
    bout = ain.copy()
    nnn = dims[0]
    mmm = dims[1]
    nnmm = nnn * mmm

    [iii, jjj] = np.where(mask == False)
    [iin, jjn] = np.where(mask == True)
    nnan = len(iin)    #Number of nan.

    if nnan == 0:
        return bout

    hv_springs = np.zeros((4 * nnan, 2), dtype=int)
    cnt = 0
    for kkk in range(nnan):
        ypos = iin[kkk]
        xpos = jjn[kkk]
        indc = ypos * mmm + xpos
        if(ypos > 0):
            hv_springs[cnt, :]   = [indc - mmm, indc]   #Top
            cnt = cnt + 1

        if(ypos < (nnn - 1)):
            hv_springs[cnt, :] = [indc, indc + mmm]  #Bottom
            cnt = cnt + 1

        if(xpos>0):
            hv_springs[cnt, :] = [indc - 1, indc]  #Left
            cnt = cnt + 1

        if(xpos < (mmm - 1)):
            hv_springs[cnt, :] = [indc, indc + 1]  #Right
            cnt = cnt + 1

    hv_springs = hv_springs[0:cnt, :]

    tempb = _unique_rows(hv_springs)
    cnt = tempb.shape[0]

    alarge = sp.csc_matrix((np.ones(cnt), (np.arange(cnt), tempb[:, 0])),
            shape=(cnt, nnmm))
    alarge = alarge + sp.csc_matrix((-np.ones(cnt), (np.arange(cnt)
        , tempb[:, 1])), shape=(cnt, nnmm))

    indk = iii * mmm + jjj
    indu = iin * mmm + jjn
    dkk  = -ain[iii, jjj]
    del iii
    del jjj

    aknown = sp.csc_matrix(alarge[:, indk])
    rhs = sp.csc_matrix.dot(aknown, dkk)
    del aknown
    del dkk
    anan = sp.csc_matrix(alarge[:, indu])
    dku = sla.lsqr(anan, rhs)
    bout[iin, jjn] = dku[0]
    return bout


def _unique_rows(scenes):
    '''Unique rows utility similar to matlab.'''
    uscenes = np.unique(scenes.view([('',scenes.dtype)]*scenes.shape[1])).view(scenes.dtype).reshape(-1,scenes.shape[1])
    return uscenes


# end of file

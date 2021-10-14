#-*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from skimage.restoration.inpaint import inpaint_biharmonic
import warnings
import pyproj
import h5py
from osgeo import gdal, osr
import sys

try:
    import cv2 as cv
except ImportError:
    cv = None

from .boundary import transform_coordinates

# Map from GDAL data type to numpy
gdal_type_to_numpy = {
    gdal.GDT_Byte: np.uint8,
    gdal.GDT_Int16: np.int16,
    gdal.GDT_Int32: np.int32,
    gdal.GDT_UInt16: np.uint16,
    gdal.GDT_UInt32: np.uint32,
    gdal.GDT_Float32: np.float32,
    gdal.GDT_Float64: np.float64,
    gdal.GDT_CFloat32: np.complex64,
    gdal.GDT_CFloat64: np.complex128
}

# Map from GDAL data type to Python format string
gdal_type_to_str = {
    gdal.GDT_Byte: 'B',
    gdal.GDT_Int16: 'h',
    gdal.GDT_Int32: 'i',
    gdal.GDT_UInt16: 'H',
    gdal.GDT_UInt32: 'I',
    gdal.GDT_Float32: 'f',
    gdal.GDT_Float64: 'd',
    gdal.GDT_CFloat32: 'ff',
    gdal.GDT_CFloat64: 'dd'
}

# Map from numpy dtype to GDAL data type
numpy_to_gdal_type = {
    '|i1': gdal.GDT_Byte,
    '<i2': gdal.GDT_Int16,
    '<i4': gdal.GDT_Int32,
    '|u1': gdal.GDT_Byte,
    '<u2': gdal.GDT_UInt16,
    '<u4': gdal.GDT_UInt32,
    '<f4': gdal.GDT_Float32,
    '<f8': gdal.GDT_Float64,
    '<c8': gdal.GDT_CFloat32,
    '<c16': gdal.GDT_CFloat64
}

class Raster:
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
        Filename for GDAL-compatible raster to read.
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
        Dictionary of extra gdal.TranslateOptions kwargs. Default: None.
    """

    def __init__(self,
                 *args,
                 data=None, hdr=None,
                 rasterfile=None, band=1,
                 stackfile=None, h5path=None,
                 islice=None, jslice=None,
                 projWin=None, gdalOpts={}):

        # If data and header are provided, save them and return
        if data is not None and hdr is not None:
            self.data = data
            self.hdr = hdr
            self.filename = None
            return

        # Attempt to guess format of generic filename if provided
        if len(args) > 0:
            filename = args[0]
            if filename.endswith('.h5'):
                stackfile = filename
            else:
                rasterfile = filename

        # Load raster data and do any subsetting using GDAL directly
        if rasterfile is not None:
            self.data, self.hdr = self.load_gdal(rasterfile, band=band, projWin=projWin,
                                                 islice=islice, jslice=jslice, **gdalOpts)
        elif stackfile is not None:
            assert h5path is not None
            # Load the header info manually
            self.hdr = RasterInfo(stackfile=stackfile)
            # Load subset/slicing information
            islice, jslice = self.hdr.subset_region(projWin=projWin, islice=islice, jslice=jslice)
            # Read data
            self.data = self.load_hdf5(stackfile, h5path, islice=islice, jslice=jslice)
        else:
            raise ValueError('Must provide GDAL raster of HDF5 stack.')

        # Cache the slices for provenance
        self.islice = islice
        self.jslice = jslice

        # Cache raster filename
        self.rasterfile = rasterfile

        return

    @staticmethod
    def load_gdal(filename, band=1, projWin=None, islice=None, jslice=None, **gdalOpts):
        """
        Load GDAL raster data from file.

        Parameters
        ----------
        filename: str
            Filename for GDAL-compatible raster to read.
        band: int, optional
            Band number to read from raster. Default: 1.
        projWin: list, optional
            GDAL-compatible projWin for subsetting raster. Default: None.
        islice: slice, optional
            Slice object specifying image rows to subset.
        jslice: slice, optional
            Slice object specifying image columns to subset.
        gdalOpts: **kwargs
            Extra kwargs for gdal.TranslateOptions (see https://gdal.org/python/osgeo.gdal-module.html#TranslateOptions) for complete list of options.

        Returns
        -------
        d: ndarray
            Array for raster data.
        """
        # Open dataset
        dset = gdal.Open(filename, gdal.GA_ReadOnly)

        # Compute srcWin if no projWin given and islice/jslice given
        srcWin = None
        if projWin is None and islice is not None and jslice is not None:
            # Unpack the slice bounds
            y0, y1 = int(islice.start), int(islice.stop)
            x0, x1 = int(jslice.start), int(jslice.stop)
            # Construct srcWin
            srcWin = [x0, y0, x1 - x0, y1 - y0]

        # Use translate to convert dataset to in-memory data
        opts = gdal.TranslateOptions(
            bandList=[band,], projWin=projWin, srcWin=srcWin, **gdalOpts
        )
        mem_ds = gdal.Translate('/vsimem/temp.tif', dset, options=opts)

        # Load RasterInfo
        hdr = RasterInfo('/vsimem/temp.tif')

        # Convert to Numy array
        d = mem_ds.ReadAsArray()

        # Close temporary datasets
        gdal.Unlink('/vsimem/temp.tif')
        dset = None
        mem_ds = None

        # Return array
        return d, hdr

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

    def write_gdal(self, filename, dtype=gdal.GDT_Float32, driver='ENVI',
                   epsg=None, projstr=None):
        """
        Write data and header to a GDAL raster.

        Parameters
        ----------
        filename: str
            Filename to write raster.
        dtype: int, optional
            Enum for GDAL datatype. Default: gdal.GDT_Float32
        driver: str, optional
            GDAL-compatible raster driver for output raster file. Default: ENVI.
        epsg: int, optional
            EPSG code for output. Default: None.
        projstr: str, optional
            PROJ string for output if no EPSG provided. Default: None.

        Returns
        -------
        None
        """
        # Create driver
        driver = gdal.GetDriverByName(driver)

        # Create dataset
        ds = driver.Create(filename, xsize=int(self.hdr.nx), ysize=int(self.hdr.ny),
                           bands=1, eType=dtype)

        # Create geotransform and projection
        if epsg is None and self.hdr._epsg is not None:
            epsg = self.hdr._epsg
        if epsg is not None or projstr is not None:
            from osgeo import osr
            ds.SetGeoTransform(self.hdr.geotransform)
            srs = osr.SpatialReference()
            if epsg is not None:
                srs.SetFromUserInput('EPSG:%d' % epsg)
            else:
                srs.SetFromUserInput(projstr)
            ds.SetProjection(srs.ExportToWkt())

        # Write data
        ds.GetRasterBand(1).WriteArray(self.data)
        ds = None

        return

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

        # Interpolate
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
        factor: int, optional
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

        # Perform downscaling
        self.data = block_reduce(self.data, (factor, factor), func, cval)

        # Create new header
        X, Y = [arr[::factor, ::factor] for arr in self.hdr.meshgrid()]
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

    def __getitem__(self, coord):
        """
        Access data at given coordinates.
        """
        i, j = coord
        return self.data[i,j]

    def __add__(self, other):
        """
        Addition between two rasters.
        """
        # Check RasterInfo consistency
        assert self.hdr == other.hdr, 'RasterInfo objects not equal.'
        # Perform addition and return a new Raster
        data = self.data + other.data
        return Raster(data=data, hdr=self.hdr)

    def __sub__(self, other):
        """
        Subtraction between two rasters.
        """
        # Check RasterInfo consistency
        assert self.hdr == other.hdr, 'RasterInfo objects not equal.'
        # Perform subtraction and return a new Raster
        data = self.data - other.data
        return Raster(data=data, hdr=self.hdr)

    def __mul__(self, other):
        """
        Multiplication between two rasters.
        """
        # Check RasterInfo consistency
        assert self.hdr == other.hdr, 'RasterInfo objects not equal.'
        # Perform multiplication and return a new Raster
        data = self.data * other.data
        return Raster(data=data, hdr=self.hdr)

    def __truediv__(self, other):
        """
        Floating point division between two rasters.
        """
        # Check RasterInfo consistency
        assert self.hdr == other.hdr, 'RasterInfo objects not equal.'
        # Perform division and return a new Raster
        data = self.data / other.data
        return Raster(data=data, hdr=self.hdr)

    def __pow__(self, exponent):
        """
        Raise Raster data to a power.
        """
        # Raise to power and return a new Raster
        data = self.data**exponent
        return Raster(data=data, hdr=self.hdr)

    def sqrt(self):
        """
        Return square root of data. Used for NumPy compatibility.
        """
        return np.sqrt(self.data)


class RasterInfo:
    """
    Class that encapsulates raster size and geographic transform information.

    Parameters
    ----------
    rasterfile: str, optional
        Filename for GDAL-compatible raster to read.
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
        Find an approximate match using FindMatches. Default: False.
    islice: slice, optional
        Slice object specifying image rows to subset.
    jslice: slice, optional
        Slice object specifying image columns to subset.
    """

    def __init__(self, rasterfile=None, stackfile=None, X=None, Y=None,
                 band=1, epsg=None, match=False, islice=None, jslice=None):
        """
        Initialize attributes.
        """
        if rasterfile is not None:
            self.load_gdal_info(rasterfile, islice=islice, jslice=jslice, band=band, match=match)
            self.rasterfile = rasterfile
        elif stackfile is not None:
            self.load_stack_info(stackfile, islice=islice, jslice=jslice)
        elif X is not None and Y is not None:
            self.set_from_meshgrid(X, Y, epsg=epsg)
        else:
            self.xstart = self.dx = self.ystart = self.dy = self.ny = self.nx = None
            self._epsg = None

    def load_gdal_info(self, rasterfile, projWin=None, islice=None, jslice=None,
                       band=1, match=False):
        """
        Read raster and geotransform information from GDAL dataset.

        Parameters
        ----------
        rasterfile: str
            Filename for GDAL-compatible raster to read.
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
            Find an approximate match using FindMatches. Default: False.

        Returns
        -------
        None
        """
        # Open GDAL dataset
        dset = gdal.Open(rasterfile, gdal.GA_ReadOnly)

        # Unpack raster sizes
        self.ny = dset.RasterYSize
        self.nx = dset.RasterXSize

        # Attempt to read geo transform
        try:
            self.xstart, self.dx, _, self.ystart, _, self.dy = dset.GetGeoTransform()
        except AttributeError:
            self.ystart = self.xstart = 0.0
            self.dx = self.dy = 1.0

        # Attempt to extract projection information as an EPSG code
        try:
            self._epsg = wkt_to_epsg(dset.GetProjection(), match=match)
        except TypeError:
            self._epsg = None
            pass

        # Optional subset
        self.subset_region(projWin=projWin, islice=islice, jslice=jslice)

        # Set units (not yet used)
        self.units = 'm'

        # Get data type from band
        b = dset.GetRasterBand(band)
        self.dtype = gdal_type_to_numpy[b.DataType]

        # Close dataset
        dset = None

    def load_stack_info(self, stackfile, islice=None, jslice=None):
        """
        Read header information from stack file.

        Parameters
        ----------
        stackfile: str
            HDF5 file for Stack to read raster data from.
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
                X = fid['X'][()]
                Y = fid['Y'][()]

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

            # Set attributes
            self.xstart = X[0]
            self.ystart = Y[0]
            self.dx = X[1] - X[0]
            self.dy = Y[1] - Y[0]
            self.ny, self.nx = Y.size, X.size

            # Try to read EPSG code
            try:
                self._epsg = fid.attrs['EPSG']
            except KeyError:
                self._epsg = None

            # Set units
            self.units = 'm'

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
        # Convert any projection window into image coordinates
        if projWin is not None and islice is None and jslice is None:

            # First check the points
            in_bounds = self.contains_point(projWin[0], projWin[1])
            in_bounds *= self.contains_point(projWin[2], projWin[3])

            # If not in bounds, return slices for full image
            if not in_bounds:
                warnings.warn('projWin outside of bounds; returning full extent.')
                islice = slice(0, self.ny)
                jslice = slice(0, self.nx)

            # Otherwise, compute slices
            else:
                # Convert coordinates
                i0, j0 = self.xy_to_imagecoord(projWin[0], projWin[1])
                i1, j1 = self.xy_to_imagecoord(projWin[2], projWin[3])
                # Construct slices
                islice = slice(i0, i1)
                jslice = slice(j0, j1)

        # Apply row slicing
        if islice is not None:
            self.ystart += islice.start * self.dy
            self.ny = islice.stop - islice.start

        # Apply column slicing
        if jslice is not None:
            self.xstart += jslice.start * self.dx
            self.nx = jslice.stop - jslice.start

        # Return slices
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
        self.xstart = X[0,0]
        self.ystart = Y[0,0]
        self.dx = X[0,1] - X[0,0]
        self.dy = Y[1,0] - Y[0,0]
        self.ny, self.nx = X.shape
        self._epsg = epsg
        self.units = units

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
        # Construct coordinates
        x = self.xcoords
        y = self.ycoords

        # Mask
        xmask = (x >= xmin) * (x <= xmax)
        ymask = (y >= ymin) * (y <= ymax)
        x = x[xmask]
        y = y[ymask]

        # Save new starting coordinates and sizes
        self.xstart = x[0]
        self.ystart = y[0]
        self.nx = len(x)
        self.ny = len(y)

        return xmask, ymask

    def read_GCPs(self, rasterfile=None, gcp_epsg=None, epsg_out=None, k=3, s=5, scale=1.0):
        """
        Load ground control points (GCPs) from gdal Dataset. Then, construct 2D
        interpolating splines that represent mapping from image to georeferenced
        coordinates with mapping determined from GCPs.

        Parameters
        ----------
        rasterfile: str, optional
            GDAL raster file to read GCPs from. Use cached source raster by default.
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

        # Read GCP coordinates
        ds = gdal.Open(rasterfile, gdal.GA_ReadOnly)
        GCPs = ds.GetGCPs()

        # Read GCP projection info if not specified
        if gcp_epsg is None:
            gcp_proj = ds.GetGCPProjection()
            gcp_epsg = wkt_to_epsg(gcp_proj)

        # Close the dataset
        ds = None

        # Unpack GCP information
        N_gcp = len(GCPs)
        pixel = np.zeros(N_gcp)
        line = np.zeros(N_gcp)
        x = np.zeros(N_gcp)
        y = np.zeros(N_gcp)
        for i, gcp in enumerate(GCPs):
            pixel[i], line[i], x[i], y[i] = gcp.GCPPixel, gcp.GCPLine, gcp.GCPX, gcp.GCPY

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
        for attr in ('xstart', 'ystart', 'dx', 'dy'):
            if abs(getattr(self, attr) - getattr(other, attr)) > 1.0e-8:
                return False
            if self.units != other.units:
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

        # Apply scale
        for attr in ('xstart', 'ystart', 'dx', 'dy'):
            value = getattr(self, attr)
            setattr(self, attr, value * scale)

        # Done
        return

    @property
    def shape(self):
        """
        Return raster shape.
        """
        return (self.ny, self.nx)

    @property
    def xstop(self):
        return self.xstart + (self.nx - 1) * self.dx

    @property
    def ystop(self):
        return self.ystart + (self.ny - 1) * self.dy

    @property
    def geotransform(self):
        """
        Return GDAL-compatible geo transform array.
        """
        return [self.xstart, self.dx, 0.0, self.ystart, 0.0, self.dy]

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
        return np.array([self.xstart, self.xstop, self.ystop, self.ystart])

    @property
    def projWin(self):
        """
        Return GDAL-style projection window of:
        [upper_left_x, upper_left_y, lower_right_x, lower_right_y].
        """
        return np.array([self.xstart, self.ystart, self.xstop, self.ystop])

    @property
    def xlim(self):
        """
        Return matplotlib-compatible x-limits (left, right).
        """
        return np.array([self.xstart, self.xstop])

    @property
    def ylim(self):
        """
        Return matplotlib-compatible y-limits (bottom, top).
        """
        return np.array([self.ystop, self.ystart])

    @property
    def xcoords(self):
        """
        Returns array of X coordinates.
        """
        return self.xstart + self.dx * np.arange(self.nx)

    @property
    def ycoords(self):
        """
        Returns array of Y coordinates.
        """
        return self.ystart + self.dy * np.arange(self.ny)

    def meshgrid(self):
        """
        Construct meshgrids for geo coordinates.
        """
        return np.meshgrid(self.xcoords, self.ycoords)

    def xy_to_imagecoord(self, x, y):
        """
        Converts geographic XY point to row and column coordinate.
        """
        row = (np.round((y - self.ystart) / self.dy)).astype(int)
        col = (np.round((x - self.xstart) / self.dx)).astype(int)
        return row, col

    def imagecoord_to_xy(self, row, col):
        """
        Converts row and column coordinate to geographic XY.
        """
        y = self.ystart + row * self.dy
        x = self.xstart + col * self.dx
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
    row = (y.ravel() - hdr.ystart) / hdr.dy
    col = (x.ravel() - hdr.xstart) / hdr.dx
    coords = np.vstack((row, col))

    # Interpolate
    values = map_coordinates(array, coords, output=None, prefilter=False, **kwargs)

    # Recover original shape and return
    return values.reshape(x.shape)

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
    # Check source RasterInfo has EPSG value set or source_srs is provided
    if raster.hdr.epsg is None:
        assert source_srs is not None, 'Must provide source_srs since no EPSG found for input.'
        src_proj = pyproj.Proj(source_srs)
    else:
        src_proj = pyproj.Proj('EPSG:%d' % raster.hdr.epsg)

    # Create target projection
    if target_epsg is None:
        if target_hdr is not None and target_hdr.epsg is not None:
            trg_proj = pyproj.Proj('EPSG:%d' % target_hdr.epsg)
        elif target_srs is not None:
            trg_proj = pyproj.Proj(target_srs)
        else:
            raise ValueError('Must provide RasterInfo or target_srs.')
    elif target_epsg is not None:
        trg_proj = pyproj.Proj('EPSG:%d' % target_epsg)

    # If only EPSG code is provided, compute target grid
    if target_hdr is None:

        # Convert bounding coordinates from source to target projection
        src_xmin, src_xmax = raster.hdr.xlim
        src_ymin, src_ymax = raster.hdr.ylim
        x0, y0 = pyproj.transform(src_proj, trg_proj, src_xmin, src_ymax, always_xy=True)
        x1, y1 = pyproj.transform(src_proj, trg_proj, src_xmax, src_ymax, always_xy=True)
        x2, y2 = pyproj.transform(src_proj, trg_proj, src_xmax, src_ymin, always_xy=True)
        x3, y3 = pyproj.transform(src_proj, trg_proj, src_xmin, src_ymin, always_xy=True)
        xvals = np.array([x0, x1, x2, x3])
        yvals = np.array([y0, y1, y2, y3])
        trg_xmin, trg_xmax = np.min(xvals), np.max(xvals)
        trg_ymin, trg_ymax = np.min(yvals), np.max(yvals)

        # Get target dimensions from user input or source raster
        if target_dims is not None:
            out_ny, out_nx = target_dims
        elif target_res is not None:
            out_ny = int(np.floor((trg_ymax - trg_ymin) / target_res)) + 1
            out_nx = int(np.floor((trg_xmax - trg_xmin) / target_res)) + 1
        else:
            out_ny, out_nx = raster.hdr.ny, raster.hdr.nx

        # Construct meshgrid with same dimensions (may be a bad idea in polar regions)
        xarr = np.linspace(trg_xmin, trg_xmax, out_nx)
        yarr = np.linspace(trg_ymax, trg_ymin, out_ny)
        trg_x, trg_y = np.meshgrid(xarr, yarr)

        # Create a RasterInfo object for target
        target_hdr = RasterInfo(X=trg_x, Y=trg_y, epsg=target_epsg)

    # Otherwise, get meshgrid straight from target_hdr
    else:
        trg_x, trg_y = target_hdr.meshgrid()

    # Chunk geometry
    chunks = get_chunks(trg_x.shape, 128, 128)
    n_chunks = len(chunks)

    # Perform transformation on chunks in parallel
    if n_proc > 1:
        from . import pymp

        manager = pymp.Manager()
        data_warped = pymp.array(trg_y.shape, dtype=raster.data.dtype)

        # Loop over chunks
        with pymp.Parallel(n_proc, manager) as parallel:
            for k in parallel.range(n_chunks):

                # Convert target coordinates to source coordinates
                islice, jslice = chunks[k]
                src_x, src_y = pyproj.transform(trg_proj, src_proj,
                                                trg_x[islice, jslice],
                                                trg_y[islice, jslice],
                                                always_xy=True)

                # Interpolate source raster
                data_warped[islice, jslice] = interpolate_raster(raster, src_x, src_y,
                                                                 ref_hdr=None, time_index=None,
                                                                 **kwargs)

    else:
        data_warped = np.zeros(trg_y.shape, dtype=raster.data.dtype)
        for k in range(n_chunks):
            # Convert target coordinates to source coordinates
            islice, jslice = chunks[k]
            src_x, src_y = pyproj.transform(trg_proj, src_proj,
                                            trg_x[islice, jslice],
                                            trg_y[islice, jslice],
                                            always_xy=True)

            print(src_x.min(), src_x.max())

            # Interpolate source raster
            data_warped[islice, jslice] = interpolate_raster(raster, src_x, src_y,
                                                             ref_hdr=None, time_index=None, 
                                                             **kwargs)

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
    dtype: int, optional
        Enum for GDAL datatype for output raster. Default: None.

    Returns
    -------
    None
    """
    # Check shapes
    assert array.shape == (hdr.ny, hdr.nx), 'Incompatible shapes'
    # Write raster
    raster = Raster(data=array, hdr=hdr)
    # Try to determine dtype if not passed
    if dtype is None:
        dtype = np.dtype(array.dtype)
        dtype = numpy_to_gdal_type[dtype.str]
    # Check if header has EPSG code
    if epsg is None and hdr.epsg is not None:
        epsg = hdr.epsg
    # Write
    raster.write_gdal(filename, epsg=epsg, projstr=projstr, dtype=dtype, driver=driver)

def griddata(x, y, z, dx, dy, x_extent=None, y_extent=None, method='linear', epsg=None):
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
    dx: float
        Spacing of output grid in x-direction.
    dy: float
        Spacing of output grid in y-direction.
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

    # Call griddata
    pts = np.column_stack((x, y))
    Zg = griddata(pts, z, (Xg, Yg), method=method)
    
    # Wrap in raster
    hdr = RasterInfo(X=Xg, Y=Yg, epsg=epsg)
    raster = Raster(data=Zg, hdr=hdr)

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
    elif method == 'biharmonic': 
        inpainted = inpaint_biharmonic(rdata, mask, multichannel=False)
    else:
        raise ValueError('Unsupported inpainting method.')

    # Return new raster or array
    if isinstance(raster, Raster):
        return Raster(data=inpainted, hdr=raster.hdr)
    else:
        return inpainted
        
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
    to an EPSG code. Sometimes AutoIdentifyEPSG cannot determine the correct projection,
    and we need to find an approximate match using FindMatches (specify match=True).
    """
    proj = osr.SpatialReference(wkt=wkt)
    if match:
        match_proj = proj.FindMatches()[0][0]
        match_proj.AutoIdentifyEPSG()
        epsg = int(match_proj.GetAttrValue('AUTHORITY', 1))
    else: 
        proj.AutoIdentifyEPSG()
        epsg = int(proj.GetAttrValue('AUTHORITY', 1))
    return epsg

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

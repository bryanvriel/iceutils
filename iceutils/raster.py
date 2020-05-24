#-*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
import pyproj
import struct
import h5py
import gdal
import osr
import sys

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
    """

    def __init__(self,
                 *args,
                 data=None, hdr=None,
                 rasterfile=None, band=1,
                 stackfile=None, h5path=None,
                 islice=None, jslice=None,
                 projWin=None):

        # If data and header are provided, save them and return
        if data is not None and hdr is not None:
            self.data = data
            self.hdr = hdr
            return

        # Attempt to guess format of generic filename if provided
        if len(args) > 0:
            filename = args[0]
            if filename.endswith('.h5'):
                stackfile = filename
            else:
                rasterfile = filename

        # Load the header info
        self.hdr = RasterInfo(rasterfile=rasterfile, stackfile=stackfile)

        # Load subset/slicing information
        islice, jslice = self.hdr.subset_region(projWin=projWin, islice=islice, jslice=jslice)

        # Load the raster data
        if rasterfile is not None:
            self.data = self.load_gdal(rasterfile, band=band, islice=islice, jslice=jslice)
        elif stackfile is not None:
            assert h5path is not None
            self.data = self.load_hdf5(stackfile, h5path, islice=islice, jslice=jslice)
        else:
            raise ValueError('Must provide GDAL raster of HDF5 stack.')

        return

    @staticmethod
    def load_gdal(filename, band=1, islice=None, jslice=None):
        """
        Load GDAL raster data.
        """
        # Open dataset
        dset = gdal.Open(filename, gdal.GA_ReadOnly)

        # Get band
        b = dset.GetRasterBand(band)

        # Read whole dataset
        if islice is None and jslice is None:
            d = b.ReadAsArray()

        # Or subset using gdal raster functionality
        else:
    
            # Unpack the slice bounds
            y0, y1 = int(islice.start), int(islice.stop)
            x0, x1 = int(jslice.start), int(jslice.stop)
            # Compute buffer size
            xsize = x1 - x0
            ysize = y1 - y0

            # Read raster portion
            scanline = b.ReadRaster(xoff=x0, yoff=y0, xsize=xsize, ysize=ysize,
                                    buf_xsize=xsize, buf_ysize=ysize, buf_type=b.DataType)

            # Convert to Python values
            fmt = gdal_type_to_str[b.DataType]
            values = struct.unpack(fmt * (xsize * ysize), scanline)

            # Convert to NumPy array
            d = np.array(values, dtype=fmt[0])
            # Handle complex values separately
            if fmt in ('ff', 'dd'):
                d = arr[0::2] + 1j * arr[1::2]
            # Reshape
            d = d.reshape(ysize, xsize)
        
        # Return array                    
        return d

    @staticmethod
    def load_hdf5(filename, h5path, islice=None, jslice=None):
        """
        Load dataset from HDF5.
        """
        with h5py.File(filename, 'r') as fid:
            d = fid[h5path][()]
            if islice is not None:
                d = d[islice,:]
            if jslice is not None:
                d = d[:,jslice]
        return d

    def write_gdal(self, filename, dtype=gdal.GDT_Float32, driver='ENVI', epsg=None):
        """
        Write data and header to a GDAL raster.
        """
        # Create driver
        driver = gdal.GetDriverByName(driver)

        # Create dataset
        ds = driver.Create(filename, xsize=self.hdr.nx, ysize=self.hdr.ny, bands=1, eType=dtype)

        # Create geotransform and projection
        if epsg is None and self.hdr._epsg is not None:
            epsg = self.hdr._epsg
        if epsg is not None:
            from osgeo import osr
            ds.SetGeoTransform(self.hdr.geotransform)
            srs = osr.SpatialReference()
            srs.SetFromUserInput('EPSG:%d' % epsg)
            ds.SetProjection(srs.ExportToWkt())

        # Write data
        ds.GetRasterBand(1).WriteArray(self.data)
        ds = None

        return

    def resample(self, hdr, order=3):
        """
        Resample raster data to another coordinate system provided by a RasterInfo object.
        """
        # If RasterInfo objects are equivalent, do nothing
        if hdr == self.hdr:
            return

        # Interpolate
        data = interpolate_raster(self, None, None, ref_hdr=hdr, order=order)

        # Update members
        self.data = data
        self.hdr = hdr
        
        return

    def downsample(self, factor=2):
        """
        Downsamples by an integer factor.
        """
        from skimage.transform import downscale_local_mean

        # Perform downscaling via local mean 
        self.data = downscale_local_mean(self.data, (factor, factor))

        # Create new header
        X, Y = [arr[::factor, ::factor] for arr in self.hdr.meshgrid()]
        self.hdr = RasterInfo(X=X, Y=Y)

        return

    def crop(self, xmin, xmax, ymin, ymax):
        """
        Crop a raster by its coordinates.
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
        """
        # Create the transect coordinates
        x = np.linspace(point1[0], point2[0], n)
        y = np.linspace(point1[1], point2[1], n)

        # Perform interpolation
        z = interpolate_raster(self, x, y, order=order)

        # Return with or without coordinates
        if return_location:
            return x, y, z
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
    """

    def __init__(self, rasterfile=None, stackfile=None, X=None, Y=None,
                 band=1, epsg=None, islice=None, jslice=None):
        """
        Initialize attributes.
        """
        if rasterfile is not None:
            self.load_gdal_info(rasterfile, islice=islice, jslice=jslice, band=band)
        elif stackfile is not None:
            self.load_stack_info(stackfile, islice=islice, jslice=jslice)
        elif X is not None and Y is not None:
            self.set_from_meshgrid(X, Y, epsg=epsg)
        else:
            self.xstart = self.dx = self.ystart = self.dy = self.ny = self.nx = None
            self._epsg = None

    def load_gdal_info(self, rasterfile, projWin=None, islice=None, jslice=None, band=1):
        """
        Read raster and geotransform information from GDAL dataset.
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

        # Extract projection information as an EPSG code
        proj = osr.SpatialReference(wkt=dset.GetProjection())
        proj.AutoIdentifyEPSG()
        self._epsg = int(proj.GetAttrValue('AUTHORITY', 1))

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
        """
        with h5py.File(stackfile, 'r') as fid:
            
            # Load coordinates
            X = fid['x'][()]
            Y = fid['y'][()]

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
        """
        # Convert any projection window into image coordinates
        if projWin is not None and islice is None and jslice is None:

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
        Set header information from meshgrid array.s
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


def load_ann(filename, comment=';'):
    """
    Load UAVSAR annotation file values into dictionary.
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

def interpolate_raster(raster, x, y, ref_hdr=None, order=3, time_index=None):
    """
    Interpolate raster at arbitrary points.
    """
    # Extract time slice if index provided
    if time_index is not None:
        r_data = raster.data[time_index,:,:]
    else:
        r_data = raster.data

    # Interpolate
    return interpolate_array(r_data, raster.hdr, x, y, ref_hdr=ref_hdr, order=order)

def interpolate_array(array, hdr, x, y, ref_hdr=None, order=3):
    """
    Interpolate 2D array at arbitrary points.
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
    values = map_coordinates(array, coords, order=order, prefilter=False,
                             mode='constant', cval=np.nan)

    # Recover original shape and return
    return values.reshape(x.shape) 

def warp(raster, target_epsg=None, target_hdr=None, target_dims=None, order=3, n_proc=1):
    """
    Warp raster to another RasterInfo hdr object with a different projection system.
    Currently only supports EPSG projection representations.
    """
    # Check source RasterInfo has EPSG value set
    assert raster.hdr.epsg is not None, 'No EPSG information found for source raster.'

    # Create projection objects
    src_proj = pyproj.Proj('EPSG:%d' % raster.hdr.epsg)
    if target_epsg is None and target_hdr is not None:
        assert target_hdr.epsg is not None, 'No EPSG information found for target raster.'
        trg_proj = pyproj.Proj('EPSG:%d' % target_hdr.epsg)
    elif target_epsg is not None:
        trg_proj = pyproj.Proj('EPSG:%d' % target_epsg)
    else:
        raise ValueError('Must supply EPSG or RasterInfo to specify target projection.')

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

    # Perform transformation on chunks in parallel
    import pymp
    data_warped = pymp.shared.array(trg_y.shape, dtype=raster.data.dtype)
    chunks = get_chunks(trg_x.shape, 128, 128)
    n_chunks = len(chunks)

    # Loop over chunks
    with pymp.Parallel(n_proc) as manager:
        for k in manager.range(n_chunks):

            # Convert target coordinates to source coordinates
            islice, jslice = chunks[k]
            src_x, src_y = pyproj.transform(trg_proj, src_proj,
                                            trg_x[islice, jslice],
                                            trg_y[islice, jslice],
                                            always_xy=True)

            # Interpolate source raster
            data_warped[islice, jslice] = interpolate_raster(raster, src_x, src_y, order=order)

    # Return new raster
    return Raster(data=data_warped, hdr=target_hdr)

def write_array_as_raster(array, hdr, filename, epsg=None, dtype=None):
    """
    Convenience function to write a NumPy array to a raster file with a given RasterInfo.
    """
    # Check shapes
    assert array.shape == (hdr.ny, hdr.nx), 'Incompatible shapes'
    # Write raster
    raster = Raster(data=array, hdr=hdr)
    # Try to determine dtype if not passed
    if dtype is None:
        dtype = np.dtype(array.dtype)
        dtype = numpy_to_gdal_type[dtype.str]
    # Write
    raster.write_gdal(filename, epsg=epsg, dtype=dtype)

def render_kml(raster, filename, dpi=300, cmap='viridis', clim=None, n_proc=1):
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
    n_proc: int, optional
        Number of processors for warping raster if not in EPSG:4326. Default: 1.

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
    kml.save(filename)

    return

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

# end of file

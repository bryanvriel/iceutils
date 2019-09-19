#-*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
import h5py
import gdal
import sys

class Raster:
    """
    Class that encapsulates raster data and stores an instance of its header info.
    """

    def __init__(self,
                 *args,
                 data=None, hdr=None,
                 rasterfile=None, band=1,
                 stackfile=None, h5path=None,
                 islice=None, jslice=None):

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
        self.hdr = RasterInfo(rasterfile=rasterfile, stackfile=stackfile,
                              islice=islice, jslice=jslice)

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
        dset = gdal.Open(filename, gdal.GA_ReadOnly)
        d = dset.GetRasterBand(band).ReadAsArray()
        if islice is not None:
            d = d[islice,:]
        if jslice is not None:
            d = d[:,jslice]
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

    def resample(self, hdr):
        """
        Resample raster data to another coordinate system provided by a RasterInfo object.
        """
        # Interpolate
        data = interpolate_raster(self, None, None, ref_hdr=hdr)

        # Update members
        self.data = data
        self.hdr = hdr
        
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


class RasterInfo:
    """
    Class that encapsulates raster size and geographic transform information.
    """

    def __init__(self, rasterfile=None, stackfile=None, X=None, Y=None,
                 islice=None, jslice=None):
        """
        Initialize attributes.
        """
        if rasterfile is not None:
            self.load_gdal_info(rasterfile, islice=islice, jslice=jslice)
        elif stackfile is not None:
            self.load_stack_info(stackfile, islice=islice, jslice=jslice)
        elif X is not None and Y is not None:
            self.set_from_meshgrid(X, Y)
        else:
            self.xstart = self.dx = self.ystart = self.dy = self.ny = self.nx = None

    def load_gdal_info(self, rasterfile, islice=None, jslice=None):
        """
        Read raster and geotransform information from GDAL dataset.
        """
        # Open GDAL dataset
        dset = gdal.Open(rasterfile, gdal.GA_ReadOnly)

        # Unpack geo transform and raster sizes
        self.xstart, self.dx, _, self.ystart, _, self.dy = dset.GetGeoTransform()
        self.ny = dset.RasterYSize
        self.nx = dset.RasterXSize

        # Incorporate row slicing
        if islice is not None:
            self.ystart += islice.start * self.dy
            self.ny = islice.stop - islice.start
        # Incorporate column slicing
        if jslice is not None:
            self.xstart += jslice.start * self.dx
            self.nx = jslice.stop - jslice.start

        # Set units
        self.units = 'm'

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

            # Set units
            self.units = 'm'

    def set_from_meshgrid(self, X, Y, units='m'):
        """
        Set header information from meshgrid array.s
        """
        self.xstart = X[0,0]
        self.ystart = Y[0,0]
        self.dx = X[0,1] - X[0,0]
        self.dy = Y[1,0] - Y[0,0]
        self.ny, self.nx = X.shape
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
    def extent(self):
        """
        Return matplotlib-compatible extent of (left, right, bottom, top).
        """
        return (self.xstart, self.xstop, self.ystop, self.ystart)

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
    from scipy.ndimage.interpolation import map_coordinates

    # If a RasterInfo object has been passed, generate output coordinates
    if ref_hdr is not None:
        x, y = ref_hdr.meshgrid()

    # Ravel points to 1D
    row = (y.ravel() - raster.hdr.ystart) / raster.hdr.dy
    col = (x.ravel() - raster.hdr.xstart) / raster.hdr.dx
    coords = np.vstack((row, col))

    # Extract time slice if index provided
    if time_index is not None:
        r_data = raster.data[time_index,:,:]
    else:
        r_data = raster.data

    # Interpolate
    values = map_coordinates(r_data, coords, order=order, prefilter=False,
                             mode='constant', cval=np.nan)

    # Recover original shape and return
    return values.reshape(x.shape)

# end of file

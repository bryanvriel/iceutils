## Installation and package structure

Prior to installing `iceutils`, we'll need to install a number of dependencies.

For `iceutils`, first, clone the repository:
```
git clone git@github.com:BryanRiel/iceutils.git
```
Then, simply place the `iceutils` directory in your `PYTHONPATH`.

In the cloned directory, you'll find several Python source files, each containing various functions and classes. While the naming of the source files gives a hint about what they contain, all functions are classes are imported into a common namespace. For example, the file `timeutils.py` contains a function `generateRegularTimeArray()`. This function would be called as follows:
```python
import iceutils as ice

t = ice.generateRegularTimeArray(tmin, tmax)
```

## Raster interface

In the file `raster.py` are two classes, `Raster` and `RasterInfo`. The former encapsulates basic raster-type data (i.e., 2D imagery) and provides some convenience functions to interface with the GDAL Python API. Therefore, any raster format that is compatible with GDAL can be read in with the `Raster` class, e.g.:
```python
raster = ice.Raster(rasterfile='velocity.tif')
```
Any instance of the `Raster` class will have as an attribute an instance of the `RasterInfo` class. This class is a separate class that encapsulates all relevant metadata associated with the raster: 1) upper left pixel coordinates; 2) pixel spacing; and 3) coordinate projection system. The `RasterInfo` instance and its data can be accessed via the `hdr` class variable, e.g.:
```python
raster = ice.Raster(rasterfile='velocity.tif')
print(raster.hdr.xstart)  # Upper left X-coordinate
print(raster.hdr.ystart)  # Upper left Y-coordinate
print(raster.hdr.dx)      # X-spacing
print(raster.hdr.dy)      # Y-spacing
```

### Loading rasters

As shown before, one can load raster files by providing the path to the GDAL-compatible file via the keyword argument `rasterfile=` (later, we'll see how to interface with certain HDF5 datasets using a different keyword argument):
```python
raster = ice.Raster(rasterfile='velocity.tif')
```
If you would like to load only a subset of the raster, then you can provide Python `slice` objects to the `Raster` constructor. These slices correspond to min-max indices of the image in row and column coordinates:
```python

# Row bounds
islice = slice(100, 300)

# Column bounds
jslice = slice(400, 800)

# Read a raster subset
raster = ice.Raster(rasterfile='velocity.tif',
                    islice=islice,
                    jslice=jslice)
```
Of course, the `RasterInfo` instance contained in the raster will have its data automatically adjusted for the subset bounds.

### Writing rasters to file

To dump raster data to a file, we use the `Raster.write_gdal` function:
```python
raster.write_gdal('output.dat', driver='ENVI', epsg=3413)
```
Any GDAL-compatible driver can be passed to the `driver` keyword argument. Additionally, one can pass in an EPSG code to specify the coordinate system of the data in order for GDAL to write relevant projection data.

### Resampling a raster to the region of another raster

Let's say we have a raster covering a geographic area, and we wish to resample the data to the geographic area of another raster. This can be done using the `Raster.resample` function:
```python
# First raster
raster = ice.Raster(rasterfile='velocity.tif')

# Another raster with a different geographic area
ref_raster = ice.Raster(rasterfile='velocity2.tif')

# Resample first raster to area of second
raster.resample(ref_raster.hdr)
```

### Converting between projection and image coordinates

To convert between physical XY-coordinates of the raster (in its projection system) to image coordinates:
```python
# XY -> image
row, col = raster.hdr.xy_to_imagecoord(x, y)

# image -> XY
x, y = raster.hdr.imagecoord_to_xy(row, col)
```
Note that row and column indices for a given XY-coordinate are rounded to the nearest integer.

### Extracting linear transect from raster

We often extract 1D transects from rasters in our analysis. At the moment, this functionality exists by providing the XY-coordinates of the endpoints of the transect of interest:
```python
# The starting point coordinate (X, Y) tuple
start = (1000.0, 1000.0)

# The ending point coordinate
end = (1200.0, 800.0)

# Extract transect
transect = raster.transect(start, end, n=200)
```
Note that the keyword argument `n` specifies the number of equally-spaced points you would like the transect to have.

## Stack analysis

`iceutils` also provides an implementation for raster time series, which we call a `Stack`. The key attribute of a stack is that all rasters will have the same projection system, geographic area, and image size, which will allow us to store stacks as 3D numpy arrays in memory. To store stacks on disk, the current implementation uses HDF5. The dataset layout of the HDF5 file is:
```
x           Dataset {N}
y           Dataset {M}
tdec        Dataset {K}
igram       Dataset {K, M, N}
weights     Dataset {K, M, N}
```
The first two datasets are 1D datasets corresponding to the coordinates of the stack. The third dataset, `tdec` is a 1D dataset corresponding to the decimal years for each raster in the stack. The 3D dataset `igram` contains the actual stack (note: the name `igram` is InSAR based, so I'll be changing this to something else very soon). Finally, the 3D dataset `weights` correspond to the weights associated with each raster. Generally, I set these to be the inverse of error/uncertainty maps associated with my data (e.g., `*ex.tif` and `*ey.tif` files for the Measures GIMP data), but ostensibly these can be set to anything sensible.

### The `Stack` class

The `Stack` class implemented in `iceutils` is for the most part a convenience interface to the underlying HDF5 file via the `h5py` Python package. For example, let's read in a stack and get a numpy array for the nearest raster image for a certain time:
```python
import numpy as np
import iceutils as ice

# Load the stack
stack = ice.Stack('velocity_stack.h5')

# Find the nearest time index for a decimal year of interest
t = 2014.43
index = np.argmin(np.abs(stack['tdec'] - t))

# Get numpy array for that index
raster_array = stack['igram'][index, :, :]
```
In the above example, we access the HDF5 datasets in the `Stack` via a key (similar to a Python dictionary). Also, when we "load" a stack, we don't actually load it into memory. Similarly, accessing an HDF5 dataset does not automatically load that dataset into memory (`h5py` behavior). However, indexing an HDF5 dataset will load the data into memory (e.g., `raster_array` is in memory).

### Extract time series at a given point

A common action performed on stacks is extracting the 1D time series at a given geographic coordinate:
```python
# The XY-coordinate of interest
xy = (1000.0, 1000.0)

# Extract time series
data = stack.timeseries(xy=xy)

# Alternatively, use row/column coordinates
coord = (300, 400)
data = stack.timeseries(coord=coord)

# Extract a 3x3 window centered around coordinate
# of interest, and get mean time series
data = stack.timeseries(xy=xy, win_size=3)
```

## Miscellaneous Utilities
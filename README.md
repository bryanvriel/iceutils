## Installation and package structure

Prior to installing `iceutils`, we'll need to install a number of Python dependencies.
```
numpy
scipy
matplotlib
gdal
h5py
pyproj
scikit-learn
scikit-image
tqdm
pint
opencv
cvxopt (optional)
```
All of the packages can be installed via Anaconda using the `conda-forge` channel. Also note that the package `cvxopt` is optional and only used for the `iceutils.tseries` module. The installation process can be streamlined by copying those packages into a `requirements.txt` file and running:
```
conda install -c conda-forge --file requirements.txt
```
If you are using the main Anaconda channel, you'll likely have to install `pint` with pip:
```
pip install pint
```

To install `iceutils`, you may clone a read-only version of the repository:
```
git clone https://github.com/bryanvriel/iceutils.git
```
Or, if you are developer, you may clone with SSH:
```
git clone git@github.com:bryanvriel/iceutils.git
```
Then, simply run `python setup.py install` in the main repository directory to install.

In the cloned directory, you'll find several Python source files, each containing various functions and classes. While the naming of the source files gives a hint about what they contain, all functions are classes are imported into a common namespace. For example, the file `timeutils.py` contains a function `generateRegularTimeArray()`. This function would be called as follows:
```python
import iceutils as ice

t = ice.generateRegularTimeArray(tmin, tmax)
```
### Known installation issues
1. Out-of-date `pyproj`: for some of the routines in `ice.Raster`, you need to have a version of `pyproj` > 2.0. However, some conda environments will only allow you to install versions 1.9.x. A temporary fix is to uninstall `pyproj` from conda and install it with `pip`, e.g.:
```
conda uninstall pyproj
pip install pyproj
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
If you would like to load only a subset of the raster, you may provide a GDAL-compatible projection window (e.g., `projWin`) defined by the standard `[ulx, uly, lrx, lry]`:
```python
# Projection window (same coordinate system/SRS as raster)
projWin = [-120, 30, -118, 28]

# Load subset raster
raster = ice.Raster('large_mosaic_raster.vrt', projWin=projWin)
```
Alternatively, you may provide Python `slice` objects to the `Raster` constructor. These slices correspond to min-max indices of the image in row and column coordinates:
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
Of course, for both the `projWin` and `islice/jslice` interfaces, the `RasterInfo` instance contained in the raster will have its data automatically adjusted for the subset bounds.

### Writing rasters to file

To dump raster data to a file, we use the `Raster.write_gdal` function:
```python
raster.write_gdal('output.dat', driver='ENVI', epsg=3413)
```
Any GDAL-compatible driver can be passed to the `driver` keyword argument. Additionally, one can pass in an EPSG code to specify the coordinate system of the data in order for GDAL to write relevant projection data.

### Resampling a raster to the region of another raster (with same projection)

Let's say we have a raster covering a geographic area, and we wish to resample the data to the geographic area of another raster with _the same projection_. This can be done using the `Raster.resample` function:
```python
# First raster
raster = ice.Raster(rasterfile='velocity.tif')

# Another raster with a different geographic area
ref_raster = ice.Raster(rasterfile='velocity2.tif')

# Resample first raster to area of second
raster.resample(ref_raster.hdr)
```

### Resampling a raster to the region of another raster (with a different projection)

In some cases, we would like to resample a raster to a geographic area with a _different_ projection (warping). In this case, the target geographic region can be specified by an EPSG code or by a separate `RasterInfo` object, either from an existing raster or created on-the-fly. For example, to warp a latitude-longitude raster to Polar Stereographic defined by a separate raster:
```python
# Source raster in latitude-longitude (EPSG: 4326)
src_raster = ice.Raster(rasterfile='velocity_latlon.tif')

# Another raster in polar stereographic north (EPSG: 3413)
trg_raster = ice.Raster(rasterfile='velocity_polar.tif')

# Warp source raster
ice.warp(src_raster, target_hdr=trg_raster.hdr)
```
We can also pre-construct the target coordinates in memory and create a `RasterInfo` object on the fly:
```python
# Meshgrid of target coordinates in polar stereographic
x = np.linspace(1000, 3000, 100)
y = np.linspace(-2200, -2400, 100)
X, Y = np.meshgrid(x, y)

# Create RasterInfo
trg_hdr = ice.RasterInfo(X=X, Y=Y, epsg=3413)

# Warp source raster
ice.warp(src_raster, target_hdr=trg_hdr)
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

### The `MultiStack` class

More often than not, we would like to compute quantities of multiple stacks, e.g. velocity magnitude given stacks of x- and y- velocity components. Instead of creating new stack objects that compute those quantities and then write the data to disk, we can use a `MultiStack` class that acts as a "virtual" Stack object with a limited set of arithmetic operations evaluated on multiple Stacks when queried. Different child classes inherit from `MultiStack` to implement the different arithmetic operations. Currently, we only have `MagStack` for vector magnitudes and `SumStack` for summing multiple stacks. As an example, let's create a velocity magnitude Stack from two independent Stacks `vx.h5` and `vy.h5` which represent velocities in the X- and Y-directions, respectively:
```python
stack = ice.MagStack(files=['vx.h5', 'vy.h5'])
```
This object can then be queried in the same way a `Stack` can be queried:
```python
# Get time series
d = stack.timeseries(xy=(1000.0, 1000.0))

# Get velocity slice at a given time index
index = 100
v = stack.slice(index)
```

## Time series decomposition

See the IPython notebook `doc/time_series_inversion.ipynb` for a full example.

## Miscellaneous Utilities

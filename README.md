# iceutils

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

`iceutils` is a collection of Python classes and functions for interfacing with geospatial rasters. Support includes common operations on rasters, such as warping, cropping, masking, gradient computaton, basic arithmetic, etc. The motivation for `iceutils` is the simplification of I/O of rasters and associated metadata while streamlining the interface between raster data and various NumPy, SciPy, and Matplotlib functions. In short, by treating rasters as metadata-decorated NumPy arrays, we can easily assimilate rasters of different projections into our analysis and visualization code using familiar interfaces.

Time-dependent rasters can be stored in a stack object in order to perform time series analysis and visualization of raster changes. Time series methods are based on the regularized least squares approaches outlined in "Riel, B., Minchew, B., & Joughin, I. (2021). Observing traveling waves in glaciers with remote sensing: new flexible time series methods and application to Sermeq Kujalleq (Jakobshavn Isbræ), Greenland. _The Cryosphere_."

## Installation and package structure

Prior to installing `iceutils`, we'll need to install a number of Python dependencies.

```
numpy
scipy
matplotlib
rasterio
h5py
h5netcdf
xarray
pyproj
scikit-image
tqdm
opencv (optional)
scikit-learn (optional)
pint (optional)
cvxopt (optional)
```
All of the packages can be installed via Anaconda using the `conda-forge` channel. Also note that optional packages `scikit-learn`, `pint`, and `cvxopt` are only necessary for using the `iceutils.tseries` module. Additionally, `opencv` is only needed for the `iceutils.correlate` module and some isolated inpainting routines. The installation process can be streamlined using the files `requirements_base.txt` and `requirements_extra.txt`:

```
# Install only base requirements
conda install -c conda-forge --file=requirements_base.txt

# Or install all requirements
conda install -c conda-forge --file=requirements_base.txt --file=requirements_extra.txt
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
Then, simply run `pip install .` in the main repository directory to install.

In the cloned directory, you'll find several Python source files, each containing various functions and classes. While the naming of the source files gives a hint about what they contain, all functions are classes are imported into a common namespace. For example, the file `timeutils.py` contains a function `generateRegularTimeArray()`. This function would be called as follows:

```python
import iceutils as ice

t = ice.generateRegularTimeArray(tmin, tmax)
```

## Raster interface

In the file `raster.py` are two classes, `Raster` and `RasterInfo`. The former encapsulates basic raster-type data (i.e., 2D imagery) and provides some convenience functions for rasterio/GDAL-backed raster I/O. Therefore, any raster format that is compatible with rasterio can be read in with the `Raster` class, e.g.:

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

As shown before, one can load raster files by providing the path to the rasterio-compatible file via the keyword argument `rasterfile=` (later, we'll see how to interface with certain HDF5 datasets using a different keyword argument):

```python
raster = ice.Raster(rasterfile='velocity.tif')
```
If you would like to load only a subset of the raster, you may provide a projection window (e.g., `projWin`) defined by the standard `[ulx, uly, lrx, lry]`:

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

To dump raster data to a file, use the `Raster.write_raster` function:

```python
raster.write_raster('output.dat', driver='ENVI', epsg=3413)
```
Any rasterio/GDAL-compatible driver can be passed to the `driver` keyword argument (e.g., `'GTiff'`, `'ISCE'`, `'GMT'`, etc.). Additionally, one can pass in an EPSG code to specify the coordinate system of the data in order for the backend to write relevant projection data (as shown in the example above; if `epsg=None`, the CRS is retrieved from the `hdr` attribute of the raster). The previous `Raster.write_gdal` name is retained as a compatibility alias.

Alternatively, we can write a NumPy array directly to a raster if we have an associated `RasterInfo` object.
```
ice.write_array_as_raster(array, hdr, 'output.dat', driver='ENVI')
```

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

`iceutils` also provides an implementation for raster time series, which we call a `Stack`. The key attribute of a stack is that all rasters will have the same projection system, geographic area, and image size. `Stack` uses `xarray` as its primary in-memory representation and stores new stacks as NetCDF-compatible HDF5 files. The standard dimension layout is:

```
x           Dataset {N}
y           Dataset {M}
time        Dataset {K}
data        Dataset {K, M, N}
weights     Dataset {K, M, N}
```
The first two datasets are 1D datasets corresponding to the coordinates of the stack. The `time` coordinate is stored using CF-style units, `seconds since 1970-01-01 00:00:00`, and is decoded by `xarray` as datetimes. The `Stack.tdec` property computes decimal years on demand for compatibility with older `iceutils` workflows. The 3D dataset `data` contains the actual stack, and `weights` corresponds to optional weights associated with each raster. Legacy HDF5 stacks with `tdec`, `igram`, `data`, `weights`, `NHW`, or `HWN` layouts are still readable and are normalized to `("time", "y", "x")` in memory.

### The `Stack` class

The `Stack` class implemented in `iceutils` is a light wrapper around an `xarray.Dataset`. For example, let's read in a stack and get a numpy array for the nearest raster image for a certain time:

```python
import numpy as np
import iceutils as ice

# Load the stack
stack = ice.Stack('velocity_stack.h5')

# Find the nearest time index for a decimal year of interest
t = 2014.43
index = stack.time_to_index(t)

# Get numpy array for that index
raster_array = stack['data'].isel(time=index).values

# Or select by decoded time coordinate
raster_array = stack['data'].sel(time='2014-06-01', method='nearest').values
```
In the above example, we access stack variables through xarray `DataArray` objects. This exposes named-axis indexing, reductions, coordinate-aware selection, and methods such as `differentiate`.

### Loading NetCDF stacks

The current `Stack` strategy is to keep stack files roughly NetCDF-compliant and to use `xarray` as the backbone for labeled dimensions, coordinates, and array operations. This keeps the `Stack` wrapper small while making common time-series operations available through the broader xarray ecosystem.

NetCDF-compatible stack files can be opened directly:

```python
import iceutils as ice

stack = ice.Stack('velocity_stack.nc')

# Access the underlying xarray Dataset
ds = stack.ds

# Access a data variable as an xarray DataArray
velocity = stack['data']

# Select by decoded time coordinate
frame = velocity.sel(time='2020-01-01', method='nearest').values
```

For external NetCDF files, `Stack` expects coordinates/dimensions named `time`, `y`, and `x`. If a file uses names like `lat`/`lon` or a different data variable name, rename with `xarray` first or access the matching variable key from `stack.ds`.

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

## License

iceutils is MIT licensed, as found in the [LICENSE file](LICENSE.md)

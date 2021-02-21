#-*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import UnivariateSpline
from matplotlib.path import Path
import pyproj
import sys

class Boundary:
    """
    Convenience class for non-mutable container for points representing a line/boundary.
    """

    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._init_path()

    def truncate(self, x_min=-1.0e16, x_max=1.0e16, y_min=-1.0e16, y_max=1.0e16):
        mask  = (self._x > x_min) * (self._x < x_max)
        mask *= (self._y > y_min) * (self._y < y_max)
        self.mask(mask)

    def mask(self, mask):
        self._x = self._x[mask]
        self._y = self._y[mask]
        self._init_path()

    def scale(self, value):
        self._x *= value
        self._y *= value
        self._init_path()

    def reverse(self):
        self._x = self._x[::-1]
        self._y = self._y[::-1]
        self._init_path()

    def contains_points(self, x, y):
        """
        Check if point resides within boundary.
        """
        # Optional cast to 1d
        if x.ndim > 1:
            points = np.column_stack((x.ravel(), y.ravel()))
        else:
            points = np.column_stack((x, y))

        # Check if within boundary
        flags = self.path.contains_points(points)

        # Return mask reshaped to original shapes
        return flags.reshape(x.shape)

    def _init_path(self):
        """
        Private function to initialize new Path instance for coordinates.
        """
        self.path = Path(np.column_stack((self._x, self._y)))
    
    @property
    def x(self):
        return self._x
    @x.setter
    def x(self, value):
        raise ValueError('Cannot set x-points')

    @property
    def y(self):
        return self._y
    @y.setter
    def y(self, value):
        raise ValueError('Cannot set y-points')


def smoothe_line(x, y, path_smooth=False, n=200, s=1.0, scale=1.0):
    """
    Smoothe a horizontal line with a smoothing spline.

    Parameters
    ----------
    x: (N,) ndarray
        Array of X-coordinates.
    y: (N,) ndarray
        Array of Y-coordinates.
    path_smooth: bool, optional
        Compute path length and use as independent coordinate for smoothing. Useful
        for irregular lines with non-monotonic X-coordinates. Default: False.
    n: int, optional
        Number of evenly spaced points for output line. Default: 200.
    s: float, optional
        Spline smoothing factor. Default: 1.0.
    scale: float, optional
        Pre-scaling factor. Useful for reducing data dynamic range. Default: 1.0.

    Returns
    -------
    xs: (n,) ndarray
        Smoothed X-coordinates.
    ys: (n,) ndarray
        Smoothed Y-coordinates.
    """
    if path_smooth:

        # Path coordinates
        sp = compute_path_length(scale*x, scale*y)
        sgrid = np.linspace(sp[0], sp[-1], n)

        # Smooth X
        spline = UnivariateSpline(sp, scale*x, s=s)
        xgrid = spline(sgrid)

        # Smooth Y
        spline = UnivariateSpline(sp, scale*y, s=s)
        ygrid = spline(sgrid)

    else:
        xgrid = scale*np.linspace(x[0], x[-1], n)
        spline = UnivariateSpline(scale*x, scale*y, s=s)
        ygrid = spline(xgrid)

    # Done
    return xgrid/scale, ygrid/scale

def compute_path_length(x, y):
    """
    Computes path length along a path specified by X and Y points. Simply accumulates
    Euclidean distance between points.
    """
    # Compute distances
    dist = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)

    # Cumulative sum
    s = np.zeros_like(x)
    s[1:] = np.cumsum(dist)

    return s

def load_termini_gmt(files, name='Jakobshavn'):

    import datetime

    # Loop over file
    termini = {}
    for filename in files:

        # Read points from file
        with open(filename, 'r') as fid:
            xpts = []; ypts = []
            started = False
            for line in fid:

                # Skip if glaicer name not in header
                if name not in line and not started:
                    continue

                # Start parsing segment of points
                if started and not line.startswith('>'):
                    x, y = [float(val) for val in line.strip().split()]
                    xpts.append(x)
                    ypts.append(y)

                # Exit if reached the end of segment
                elif started and line.startswith('>'):
                    break

                # Signal that we've reached segment of interest
                # Parse dates to get middle date
                else:
                    #fields = line[1:].strip().split('|')
                    #datestr = fields[3]
                    #datestr1, datestr2 = datestr.split('-')
                    #date1 = datetime.datetime.strptime(datestr1, '%d%b%Y')
                    #date2 = datetime.datetime.strptime(datestr2, '%d%b%Y')
                    #delta = 0.5 * (date2 - date1).total_seconds()
                    #date_mid = date1 + datetime.timedelta(seconds=delta)
                    started = True

        # Store points
        termini[filename] = np.column_stack((xpts, ypts))

    # Done
    return termini

def load_kml(kmlfile, out_epsg=4326):
    """
    Convenience function to parse a KML file to get the coordinates.
    """
    import xml.etree.ElementTree as ET

    # Parse the file
    tree = ET.parse(kmlfile)
    root = tree.getroot()

    # Iterate over main KML document
    for child in root.iter():
        tag = child.tag.split('}')[-1]
        if tag == 'coordinates':

            # Get raw data
            s = child.text.strip()
            lines = s.split()

            # Loop over coordinates
            N = len(lines)
            lon = np.zeros(N)
            lat = np.zeros(N)
            for i, line in enumerate(lines):
                values = [float(x) for x in line.split(',')]
                lon[i] = values[0]
                lat[i] = values[1]

    # Perform transformation if another EPSG is specified
    if out_epsg != 4326:
        proj = pyproj.Proj('EPSG:%d' % out_epsg)
        wgs84 = pyproj.Proj('EPSG:4326')
        x, y = pyproj.transform(wgs84, proj, lon, lat, always_xy=True)
        return x, y
    else:
        return lon, lat

def transform_coordinates(x_in, y_in, epsg_in=None, epsg_out=None, proj_in=None, proj_out=None):
    """
    Transforms coordinates from one projection to another specified by EPSG codes.

    Parameters
    ----------
    x_in: ndarray
        Input X-coordinates.
    y_in: ndarray
        Input Y-coordinates.
    epsg_in: int, optional
        Input EPSG projection.
    epsg_out: int, optional
        Output EPSG projection.
    proj_in: pyproj.Proj, optional
        Input Proj object.
    proj_out: pyproj.Proj, optional
        Output Proj object.

    Returns
    -------
    x_out: ndarray
        Output X-coordinates.
    y_out: ndarray
        Output Y-coordinates.
    """
    # Create projection objects
    if proj_in is None or proj_out is None:
        assert epsg_in is not None and epsg_out is not None, 'Must specify EPSG codes.'
        proj_in = pyproj.Proj('EPSG:%d' % epsg_in)
        proj_out = pyproj.Proj('EPSG:%d' % epsg_out)

    # Perform transformation
    return pyproj.transform(proj_in, proj_out, x_in, y_in, always_xy=True)

def transform_projwin(projWin, epsg_in=None, epsg_out=None, proj_in=None, proj_out=None):
    """
    Transforms projection window from one projection to another specified by EPSG codes
    or pyproj.Proj objects. The output projection window is guaranteed to have
    ordering compatible for use with ice.Raster.

    Parameters
    ----------
    projWin: list or array_like
        Projection window of [upper_left_x, upper_left_y, lower_right_x, lower_right_y].
    epsg_in: int, optional
        Input EPSG projection.
    epsg_out: int, optional
        Output EPSG projection.
    proj_in: pyproj.Proj, optional
        Input Proj object.
    proj_out: pyproj.Proj, optional
        Output Proj object.

    Returns
    -------
    out_projWin: list
        Transformed projection window.
    """
    # Unpack input projection window (we use the names lon/lat here)
    lon_min, lat_max, lon_max, lat_min = projWin

    # Traverse four corners of input projection window
    x = []; y = []
    for lon, lat in ((lon_min, lat_max), (lon_max, lat_max),
                     (lon_max, lat_min), (lon_min, lat_min)):
        xval, yval = transform_coordinates(
            lon, lat, epsg_in=epsg_in, epsg_out=epsg_out, proj_in=proj_in, proj_out=proj_out
        )
        x.append(xval)
        y.append(yval)
    
    # Compute and return transformed projection window
    return [np.min(x), np.max(y), np.max(x), np.min(y)]

def extract_perpendicular_transects(x, y, raster, W=15.0e3, N=100, N_perp=100,
                                    return_coords=False, return_theta=False):
    """
    Traverse a line and extract multiple perpendicular transects from a raster.

    Parameters
    ----------
    x: ndarray
        Array of X-coordinates for center line.
    y: ndarray
        Array of Y-coordinates for center line.
    raster: Raster
        Input raster to extract transects from.
    W: float, optional
        Half-width of perpendicular transects. Default: 15.0e3.
    N: int, optional
        Number of perpendicular transects to extract. Default: 100.
    N_per: int, optional
        Number of points in each perpendicular transect. Default: 100.
    return_coords: bool, optional
        Return coordinates of transect. Default: False.
    return_theta: bool, optional
        Return along-transect rotation angle. Default: FAlse

    Returns
    -------
    out: (N, N_per) ndarray
        Output array of transects.
    coords: (N, N_per, 2) ndarray
        Coordinates of transects (if return_coords = True).
    angle: (N,) ndarray
        Rotation angle along transect (if return_theta = True).
    """
    # If number of transects is greater than the number of points in centerline, interpolate
    s = compute_path_length(x, y)
    if N > x.size:
        s_new = np.linspace(s[0], s[-1], N)
        x_new = np.interp(s_new, s, x)
        y_new = np.interp(s_new, s, y)
        s, x, y = s_new, x_new, y_new

    # Pre-compute centerline spacing
    dx = np.gradient(x, edge_order=2)
    dy = np.gradient(y, edge_order=2)

    # Interpolate centerline coordinates
    s_out = np.linspace(s[0], s[-1], N)
    x, y, dx, dy = [np.interp(s_out, s, arr) for arr in (x, y, dx, dy)]

    # Allocate array for transects
    out = np.zeros((N, N_perp))
    if return_coords:
        coords = np.zeros((N, N_perp, 2))
    for k in range(N):

        # Get local flowline coordinate and gradients
        fx = x[k]
        fy = y[k]
        dfx = dx[k]
        dfy = dy[k]

        # Angles
        theta = np.arctan2(dfy, dfx)
        theta_rot = theta + 0.5 * np.pi
        u = np.cos(theta_rot)
        v = np.sin(theta_rot)

        # Generate endpoints of a perpendicular line
        xmin = fx - u * W
        xmax = fx + u * W
        ymin = fy - v * W
        ymax = fy + v * W
        point1 = (xmax, ymax)
        point2 = (xmin, ymin)

        # Get transects of bed, thickness, and velocity
        out[k, :], px, py = raster.transect(point1, point2, n=N_perp, return_location=True)
        if return_coords:
            coords[k, :, 0] = px
            coords[k, :, 1] = py

    # Done
    if return_coords:
        if return_theta:
            return out, coords, np.arctan2(dy, dx)
        else:
            return out, coords
        
    elif return_theta:
        return out, np.arctan2(dy, dx)

    else:
        return out


# end of file    

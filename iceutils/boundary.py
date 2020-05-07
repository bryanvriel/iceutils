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
        points = np.column_stack((x, y))
        flags = self.path.contains_points(points)
        return flags

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


def smoothe_line(x, y, n=200, s=1.0):
    """
    Smoothe a horizontal line with a smoothing spline.
    """
    # Uniform grid for x coordinates
    xgrid = np.linspace(x[0], x[-1], n)
    # Spline smoothing
    spline = UnivariateSpline(x, y, s=s)
    # Evaluate spline
    ygrid = spline(xgrid)
    # Done
    return xgrid, ygrid

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
                x, y, z = [float(x) for x in line.split(',')]
                lon[i] = x
                lat[i] = y

    # Perform transformation if another EPSG is specified
    if out_epsg != 4326:
        proj = pyproj.Proj('EPSG:%d' % out_epsg)
        wgs84 = pyproj.Proj('EPSG:4326')
        x, y = pyproj.transform(wgs84, proj, lon, lat, always_xy=True)
        return x, y
    else:
        return lon, lat

# end of file    

#-*- coding: utf-8 -*-

import warnings
import numpy as np
import h5py
import sys

from .raster import RasterInfo
from .timeutils import datestr2tdec

class Stack:
    """
    Class that encapsulates standard HDF5 stack file.
    """

    def __init__(self, filename, mode='r', init=None):

        self.fid = None

        # Store the mode
        assert mode in ('r', 'r+', 'w', 'a'), 'Unsupported HDF5 file open mode'
        self.mode = mode

        # Save RasterInfo if file opened in read mode
        if self.mode in ('r', 'r+', 'a'):
            self.hdr = RasterInfo(stackfile=filename)

        # Open HDF5 file 
        self.fid = h5py.File(filename, self.mode)

        # Read time information
        if self.mode in ('r', 'r+', 'a'):
            self.tdec = self.fid['tdec'][()]

        # If another stack is provided, copy metadata and save into Datasets
        if isinstance(init, Stack):
            self.hdr = init.hdr
            self.tdec = init.tdec
            if self.mode in ('r+', 'w', 'a'):
                self.fid['x'] = self.hdr.xcoords
                self.fid['y'] = self.hdr.ycoords
                self.fid['tdec'] = self.tdec

        # Initialize datasets dictionary
        self._datasets = {}
        if self.mode in ('r', 'r+', 'a'):
            for key in self.fid.keys():
                self._datasets[key] = self.fid[key]

        return


    def initialize(self, tdec, raster_info, data=True, weights=False, chunks=(1, 128, 128)):
        """
        For a stack in write mode, initialize metadata Datasets.
        """
        # The time array
        self.tdec = tdec
        self.create_dataset('tdec', tdec.shape, dtype='d', data=tdec)

        # Spatial information
        self.hdr = raster_info
        self.set_spatial_metadata(raster_info)
        self.create_dataset('x', self.x.shape, dtype='f', data=self.x)
        self.create_dataset('y', self.y.shape, dtype='f', data=self.y)

        # Create datasets for stack data
        if data:
            shape = (self.Nt, self.Ny, self.Nx)
            self.create_dataset('data', shape, dtype='f', chunks=chunks)

        # Optional weights dataset
        if weights:
            self.create_dataset('weights', shape, dtype='f', chunks=chunks)

    def set_spatial_metadata(self, raster_info):
        """
        Set spatial arrays from a RasterInfo object.
        """
        X, Y = raster_info.meshgrid()
        self.x = X[0,:]
        self.y = Y[:,0]
        return

    def create_dataset(self, name, shape, dtype='f', chunks=None, **kwargs):
        """
        Create an HDF5 dataset for data.
        """
        # Create the dataset
        self._datasets[name] = self.fid.create_dataset(
            name, shape, dtype, chunks=chunks, **kwargs
        )
        # Check if we need to save chunk shape
        if chunks is not None and 'chunk_shape' not in self.fid.keys() and len(chunks) == 3:
            self.fid['chunk_shape'] = list(chunks)

        # Return reference to dataset
        return self._datasets[name]

    def __getitem__(self, name):
        """
        Provides access to underlying HDF5 dataset.
        """
        return self._datasets[name]

    def __setitem__(self, name, value):
        """
        Creates a new dataset.
        """
        # Make sure dataset doesn't already exist
        if name in self._datasets.keys():
            raise ValueError('Dataset %s already exists' % name)

        # Create dataset automatically
        assert isinstance(value, np.ndarray), 'Must input NumPy array to set data.'
        self.create_dataset(name, value.shape, dtype=value.dtype, data=value)

        return

    def slice(self, index, key='data'):
        """
        Extract Stack 2d slice at given time index.
        """
        return self._datasets[key][index, :, :]

    def set_slice(self, index, data, key='data'):
        """
        Set Stack 2d slice at given time index.
        """
        self._datasets[key][index, :, :] = data

    def get_chunk(self, slice_y, slice_x, key='data'):
        """
        Get a 3d chunk of data defined by 2d slice objects.
        """
        return self._datasets[key][:, slice_y, slice_x]

    def set_chunk(self, slice_y, slice_x, data, key='data'):
        """
        Set a 3d chunk of data defined by 2d slice objects.
        """
        self._datasets[key][:, slice_y, slice_x] = data

    def timeseries(self, xy=None, coord=None, key='data', win_size=1):
        """
        Extract time series at a given spatial coordinate. Optionally extract a window of
        time series and average spatially. If requested coordinate is outside of stack
        bounds, NaN array is returned.
        """
        # Get the image coordinate if not provided
        if xy is not None and coord is None:
            x, y = xy
            row, col = self.hdr.xy_to_imagecoord(x, y)
        elif coord is not None:
            row, col = coord
        else:
            raise ValueError('Must pass in geographic or image coordinate.')

        # Check bounds. Warn user and return NaN if outside of bounds
        half_win = win_size // 2
        if row >= (self.Ny - half_win) or row < half_win:
            warnings.warn('Requested point outside of stack bounds. Returning NaN.',
                          category=UserWarning)
            return np.nan
        if col >= (self.Nx - half_win) or col < half_win:
            warnings.warn('Requested point outside of stack bounds. Returning NaN.',
                          category=UserWarning)
            return np.nan

        # Spatial slice
        if win_size > 1:
            islice = slice(row - half_win, row + half_win + 1)
            jslice = slice(col - half_win, col + half_win + 1)
        else:
            islice, jslice = row, col

        # Extract the data
        data = self._datasets[key][:, islice, jslice]

        # Average
        if win_size > 1:
            data = np.nanmean(data, axis=(1, 2))

        # Done
        return data

    def resample(self, ref_hdr, output, key='data', dtype='f', chunks=None):
        """
        Resample dataset from one coordinate system to another provided by a
        RasterInfo object.
        """
        from tqdm import tqdm
        from .raster import interpolate_array

        # Check if dataset exists to clean it
        if not key in self._datasets.keys():
            print('Warning: dataset %s not in stack' % key)
            return
        
        # Initialize dataset in output stack
        Ny, Nx = ref_hdr.shape
        shape = (self.Nt, Ny, Nx)
        output.create_dataset(key, shape, dtype=dtype, chunks=chunks)

        # Loop over slices and interpolate
        for k in tqdm(range(self.Nt)):
            d = self.slice(k, key=key)
            output[key][k,:,:] = interpolate_array(d, self.hdr, None, None, ref_hdr=ref_hdr)

        # Done
        return

    def time_to_index(self, t, date=None):
        """
        Convenience function to convert decimal year (or datetime) to a time index using
        nearest neighbor.
        """
        if t is None and date is not None:
            t = datestr2tdec(pydtime=date)
        return np.argmin(np.abs(self.tdec - t))

    @property
    def dt(self):
        """
        Return mean sampling interval.
        """
        return np.mean(np.diff(self.tdec))

    @property
    def Nt(self):
        if self.tdec is not None:
            return self.tdec.size
        else:
            return None

    @property
    def Ny(self):
        if self.hdr is not None:
            return self.hdr.shape[0]
        else:
            return None

    @property
    def Nx(self):
        if self.hdr is not None:
            return self.hdr.shape[1]
        else:
            return None


class MultiStack:
    """
    Stack object that represents some arithmetic manipulation of multiple Stacks. Child
    classes should inherit from this class and implement the self.slice,
    self.timeseries, and self.get_chunk methods.
    """

    def __init__(self, stacks=None, files=None):
        """
        In the constructor, either store a list of Stack objects or create a list
        of Stack objects from a list of filenames.
        """
        # Store or create list of stacks
        if stacks is not None:
            self.stacks = stacks
        elif files is not None:
            self.stacks = [Stack(fname) for fname in files]
        else:
            raise ValueError('Must pass in stacks or filenames.')

        # Cache time and header objects
        self.tdec = self.stacks[0].tdec
        self.hdr = self.stacks[0].hdr

    def slice(self, index, key='data'):
        raise NotImplementedError('Child classes must implement slice function')

    def timeseries(self, xy=None, coord=None, key='data', win_size=1):
        raise NotImplementedError('Child classes must implement timeseries function')

    def get_chunk(self, *args, **kwargs):
        raise NotImplementedError('Child classes must implement get_chunk function')

    def __getitem__(self, key):
        raise NotImplementedError('Child classes must implement __getitem__ function')

    def time_to_index(self, t, date=None):
        return self.stacks[0].time_to_index(t, date=date)

    @property
    def dt(self):
        return self.stacks[0].dt

    @property
    def Nt(self):
        if self.tdec is not None:
            return self.tdec.size
        else:
            return None

    @property
    def Ny(self):
        if self.hdr is not None:
            return self.hdr.shape[0]
        else:
            return None

    @property
    def Nx(self):
        if self.hdr is not None:
            return self.hdr.shape[1]
        else:
            return None


class MagStack(MultiStack):
    """
    MultiStack class that computes magnitude of stack objects.
    """

    def slice(self, index, key='data'):
        dsum = 0.0
        for stack in self.stacks:
            dsum += (stack[key][index, :, :])**2
        return np.sqrt(dsum)

    def timeseries(self, xy=None, coord=None, key='data', win_size=1):
        dsum = 0.0
        for stack in self.stacks:
            dsum += (stack.timeseries(xy=xy, coord=coord, key=key, win_size=win_size))**2
        return np.sqrt(dsum)

    def get_chunk(self, slice_y, slice_x, key='data'):
        dsum = 0.0
        for stack in self.stacks:
            dsum += (stack.get_chunk(slice_y, slice_x, key=key))**2
        return np.sqrt(dsum)


class SumStack(MultiStack):
    """
    MultiStack class that performs a sum on the stack objects.
    """

    def slice(self, index, key='data'):
        dsum = 0.0
        for stack in self.stacks:
            dsum += stack[key][index, :, :]
        return dsum

    def timeseries(self, xy=None, coord=None, key='data', win_size=1):
        dsum = 0.0
        for stack in self.stacks:
            dsum += stack.timeseries(xy=xy, coord=coord, key=key, win_size=win_size)
        return dsum

    def get_chunk(self, slice_y, slice_x, key='data'):
        dsum = 0.0
        for stack in self.stacks:
            dsum += stack.get_chunk(slice_y, slice_x, key=key)
        return dsum


# end of file

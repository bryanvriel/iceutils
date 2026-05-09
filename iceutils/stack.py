#-*- coding: utf-8 -*-

from typing import List, Union
import copy
import datetime
import os
import warnings

import h5netcdf
import h5py
import numpy as np
import xarray as xr

from .raster import Raster, RasterInfo


TIME_UNITS = 'seconds since 1970-01-01 00:00:00'
TIME_CALENDAR = 'proleptic_gregorian'
EPOCH = np.datetime64('1970-01-01T00:00:00', 'ns')


def _is_datetime_like(value):
    return isinstance(value, (datetime.datetime, datetime.date, np.datetime64, str))


def _as_datetime64_scalar(value):
    if isinstance(value, np.datetime64):
        return value.astype('datetime64[ns]')
    if isinstance(value, datetime.datetime):
        return np.datetime64(value.replace(tzinfo=None), 'ns')
    if isinstance(value, datetime.date):
        return np.datetime64(datetime.datetime.combine(value, datetime.time()), 'ns')
    if isinstance(value, str):
        return np.datetime64(value, 'ns')
    raise TypeError('Input is not datetime-like.')


def _tdec_to_datetime64(tdec):
    """
    Convert decimal years to timezone-naive datetime64 values.
    """
    arr = np.asarray(tdec, dtype=float)
    flat = arr.ravel()
    out = np.empty(flat.size, dtype='datetime64[ns]')
    for index, value in enumerate(flat):
        year = int(np.floor(value))
        start = np.datetime64('%04d-01-01T00:00:00' % year, 'ns')
        stop = np.datetime64('%04d-01-01T00:00:00' % (year + 1), 'ns')
        span = (stop - start) / np.timedelta64(1, 'ns')
        delta = np.timedelta64(int(round((value - year) * span)), 'ns')
        out[index] = start + delta
    return out.reshape(arr.shape)


def _datetime64_to_tdec(time):
    """
    Convert datetime64 values to decimal years.
    """
    arr = np.asarray(time)
    if not np.issubdtype(arr.dtype, np.datetime64):
        arr = EPOCH + np.rint(arr.astype(float) * 1.0e9).astype('timedelta64[ns]')
    arr = arr.astype('datetime64[ns]')
    year_coord = arr.astype('datetime64[Y]')
    years = year_coord.astype(int) + 1970
    starts = year_coord.astype('datetime64[ns]')
    stops = (year_coord + 1).astype('datetime64[ns]')
    elapsed = (arr - starts) / np.timedelta64(1, 's')
    span = (stops - starts) / np.timedelta64(1, 's')
    return years.astype(float) + elapsed / span


def _time_values_to_datetime64(values, units=None):
    """
    Normalize supported time encodings to datetime64.
    """
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype('datetime64[ns]')
    if units is not None and units.startswith('seconds since 1970-01-01'):
        return EPOCH + np.rint(arr.astype(float) * 1.0e9).astype('timedelta64[ns]')
    return _tdec_to_datetime64(arr)


def _needs_float_time_encoding(times):
    seconds = (np.asarray(times).astype('datetime64[ns]') - EPOCH) / np.timedelta64(1, 's')
    return not np.allclose(seconds, np.rint(seconds))


def _time_encoding(times):
    dtype = 'float64' if _needs_float_time_encoding(times) else 'int64'
    return {'units': TIME_UNITS, 'calendar': TIME_CALENDAR, 'dtype': dtype}


def _dataset_values(obj):
    return obj.values if hasattr(obj, 'values') else np.asarray(obj)


class Stack:
    """
    Xarray-backed stack for raster time series.
    """

    def __init__(self, filename, mode='r', fmt='NHW',
                 init_stack=None, init_tdec=None, init_rasterinfo=None,
                 init_names=None, init_data=False, ds_hdr=None, time_key='tdec'):
        """Reads Stack from an existing file or creates a new Stack.

        New files are NetCDF-compatible HDF5 files with xarray dimensions
        ``time``, ``y``, and ``x``. Legacy HDF5 stacks are normalized into the
        same xarray dimension order.
        """
        assert mode in ('r', 'r+', 'w', 'a', 'x'), 'Unsupported HDF5 file open mode'
        self.filename = filename
        self.mode = mode
        self.original_fmt = fmt
        self.fmt = 'NHW'
        self.hdr = None
        self.ds = xr.Dataset()
        self.fid = None
        self._datasets = {}
        self._dirty = False
        self._legacy = False
        self._legacy_source_ds = None
        self._time_key = time_key
        self._ds_hdr = ds_hdr

        if mode in ('r', 'r+') or (mode == 'a' and os.path.exists(filename)):
            self._open_existing(ds_hdr=ds_hdr, time_key=time_key)
            if mode in ('r+', 'a') and not self._legacy:
                self.fid = h5netcdf.File(filename, 'a')
        else:
            self._create_empty_file(mode)
            if isinstance(init_stack, Stack):
                self.initialize(init_stack.tdec, init_stack.hdr, names=init_names,
                                data=init_data, fmt=fmt)
                if init_names is None and 'names' in init_stack.ds:
                    self.names = init_stack.names
            elif init_tdec is not None and init_rasterinfo is not None:
                self.initialize(init_tdec, init_rasterinfo, names=init_names,
                                data=init_data, fmt=fmt)
            elif any(value is not None for value in (init_stack, init_tdec, init_rasterinfo)):
                raise ValueError('Must supply init_stack or init_tdec+init_rasterinfo.')

        if self.Nt is not None:
            self._nan_tseries = np.full((self.Nt,), np.nan, dtype='f')
        else:
            self._nan_tseries = None

    def _create_empty_file(self, mode):
        h5_mode = 'w' if mode == 'w' else mode
        self.fid = h5netcdf.File(self.filename, h5_mode)

    def _open_existing(self, ds_hdr=None, time_key='tdec'):
        try:
            ds = xr.open_dataset(self.filename, engine='h5netcdf', decode_times=True,
                                 phony_dims='sort')
            if {'time', 'y', 'x'}.issubset(ds.sizes):
                self.ds = ds
                self.original_fmt = ds.attrs.get('original_format', ds.attrs.get('format', 'NHW'))
                self.fmt = 'NHW'
                self.hdr = self._rasterinfo_from_dataset(ds)
                self._update_dataset_cache()
                return
            ds.close()
        except Exception:
            pass

        self._legacy = True
        self._load_legacy_hdf5(ds_hdr=ds_hdr, time_key=time_key)

    def _load_legacy_hdf5(self, ds_hdr=None, time_key='tdec'):
        self._close_legacy_source()
        data_vars = {}
        raw_ds = None
        raw_ds_used = False
        try:
            raw_ds = xr.open_dataset(self.filename, engine='h5netcdf',
                                     decode_times=False, phony_dims='sort')
        except Exception:
            raw_ds = None

        with h5py.File(self.filename, 'r') as fid:
            self.original_fmt = fid.attrs.get('format', 'NHW')
            if isinstance(self.original_fmt, bytes):
                self.original_fmt = self.original_fmt.decode('utf-8')

            x, y = self._read_legacy_xy(fid, ds_hdr)
            time = self._read_legacy_time(fid, time_key)
            coords = {'time': ('time', time), 'y': ('y', y), 'x': ('x', x)}

            for key, value in self._iter_hdf5_datasets(fid):
                if key in ('x', 'X', 'y', 'Y', time_key, 'tdec', 't', 'time'):
                    continue
                if key == 'names':
                    data_vars[key] = ('time', self._decode_names(value[()]))
                    continue
                if key == 'chunk_shape':
                    data_vars[key] = ('chunk_dim', value[()])
                    continue

                data_array = self._legacy_data_array(raw_ds, key, value, x, y, time)
                if data_array is not None:
                    data_vars[key] = data_array
                    raw_ds_used = True
                    continue

                arr = value[()]
                if arr.ndim == 3:
                    if self.original_fmt == 'HWN' or arr.shape == (y.size, x.size, time.size):
                        arr = np.moveaxis(arr, -1, 0)
                    data_vars[key] = (('time', 'y', 'x'), arr)
                elif arr.ndim == 2 and arr.shape == (y.size, x.size):
                    data_vars[key] = (('y', 'x'), arr)
                elif arr.ndim == 1 and arr.shape[0] == time.size:
                    data_vars[key] = ('time', arr)
                elif arr.ndim == 1:
                    dim = '%s_dim' % key.replace('/', '_')
                    data_vars[key] = (dim, arr)
                else:
                    dims = tuple('%s_dim_%d' % (key.replace('/', '_'), i) for i in range(arr.ndim))
                    data_vars[key] = (dims, arr)

            attrs = dict(fid.attrs)
            attrs['original_format'] = self.original_fmt
            attrs['format'] = 'xarray'
            attrs.setdefault('time_units', TIME_UNITS)
            if 'EPSG' in fid.attrs:
                attrs['EPSG'] = int(fid.attrs['EPSG'])

        if raw_ds is not None:
            if raw_ds_used:
                self._legacy_source_ds = raw_ds
            else:
                raw_ds.close()

        self.ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        self.hdr = self._rasterinfo_from_dataset(self.ds)
        self.fmt = 'NHW'
        self._update_dataset_cache()

    def _legacy_data_array(self, raw_ds, key, value, x, y, time):
        if raw_ds is None or '/' in key or key not in raw_ds:
            return None

        data = raw_ds[key]
        shape = value.shape
        if value.ndim == 3:
            if self.original_fmt == 'HWN' or shape == (y.size, x.size, time.size):
                data = data.rename(dict(zip(data.dims, ('y', 'x', 'time'))))
                return data.transpose('time', 'y', 'x')
            data = data.rename(dict(zip(data.dims, ('time', 'y', 'x'))))
            return data.transpose('time', 'y', 'x')
        if value.ndim == 2 and shape == (y.size, x.size):
            return data.rename(dict(zip(data.dims, ('y', 'x'))))
        return None

    def _close_legacy_source(self):
        if self._legacy_source_ds is not None:
            self._legacy_source_ds.close()
            self._legacy_source_ds = None

    @staticmethod
    def _iter_hdf5_datasets(group, prefix=''):
        for key, value in group.items():
            name = '%s/%s' % (prefix, key) if prefix else key
            if isinstance(value, h5py.Dataset):
                yield name, value
            elif isinstance(value, h5py.Group):
                yield from Stack._iter_hdf5_datasets(value, name)

    @staticmethod
    def _decode_names(values):
        arr = np.asarray(values)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        names = []
        for value in arr:
            if isinstance(value, bytes):
                names.append(value.decode('utf-8'))
            else:
                names.append(str(value))
        return np.asarray(names)

    @staticmethod
    def _read_legacy_xy(fid, ds_hdr=None):
        try:
            x = fid['x'][()].squeeze()
            y = fid['y'][()].squeeze()
        except KeyError:
            try:
                x = fid['X'][()]
                y = fid['Y'][()]
            except KeyError:
                if ds_hdr is None:
                    ds_hdr = Stack._guess_legacy_data_key(fid)
                shape = fid[ds_hdr].shape
                fmt = fid.attrs.get('format', 'NHW')
                if isinstance(fmt, bytes):
                    fmt = fmt.decode('utf-8')
                if fmt == 'HWN' and len(shape) >= 3:
                    ny, nx = shape[0], shape[1]
                else:
                    ny, nx = shape[-2:]
                x = np.arange(nx)
                y = np.arange(ny)

        if x.ndim == 2:
            x = x[0, :]
        if y.ndim == 2:
            y = y[:, 0]
        return np.asarray(x), np.asarray(y)

    @staticmethod
    def _guess_legacy_data_key(fid):
        for key in ('data', 'igram', 'weights'):
            if key in fid and fid[key].ndim >= 2:
                return key
        for key, value in Stack._iter_hdf5_datasets(fid):
            if value.ndim >= 2:
                return key
        raise ValueError('Could not infer raster dimensions from stack file.')

    def _read_legacy_time(self, fid, time_key):
        for key in (time_key, 'time', 'tdec', 't'):
            if key in fid:
                units = fid[key].attrs.get('units')
                if isinstance(units, bytes):
                    units = units.decode('utf-8')
                if key == 'time':
                    return _time_values_to_datetime64(fid[key][()], units=units)
                if units is not None and units.startswith('seconds since 1970-01-01'):
                    return _time_values_to_datetime64(fid[key][()], units=units)
                return _tdec_to_datetime64(fid[key][()])

        data_key = self._ds_hdr or Stack._guess_legacy_data_key(fid)
        nt = fid[data_key].shape[0] if self.original_fmt != 'HWN' else fid[data_key].shape[-1]
        warnings.warn('No time vector found. Using Unix epoch seconds.', category=UserWarning)
        return EPOCH + np.arange(nt).astype('timedelta64[s]')

    @staticmethod
    def _rasterinfo_from_dataset(ds):
        x = np.asarray(ds['x'].values)
        y = np.asarray(ds['y'].values)
        epsg = ds.attrs.get('EPSG')
        if epsg is not None:
            epsg = int(epsg)
        return RasterInfo(X=np.meshgrid(x, y)[0], Y=np.meshgrid(x, y)[1], epsg=epsg)

    def _update_dataset_cache(self):
        self._datasets = {key: self.ds[key] for key in self.ds.variables}

    def _refresh_dataset_if_dirty(self):
        if not self._dirty:
            return
        if self._legacy:
            self._load_legacy_hdf5(ds_hdr=self._ds_hdr, time_key=self._time_key)
        else:
            self.ds.close()
            self.ds = xr.open_dataset(self.filename, engine='h5netcdf', decode_times=True)
            self._update_dataset_cache()
        self._dirty = False

    def _require_initialized(self):
        if self.hdr is None or 'time' not in self.ds.coords:
            raise ValueError('Stack has not been initialized.')

    def _require_writable(self):
        if self.mode == 'r':
            raise OSError('Stack opened read-only.')
        if self._legacy:
            raise OSError('Legacy HDF5 stacks are read-only; write a new xarray-format stack.')
        if self.fid is None:
            self.fid = h5netcdf.File(self.filename, 'a')

    def _ensure_dimension(self, name, size):
        if name not in self.fid.dimensions:
            self.fid.dimensions[name] = int(size)

    def initialize(self, tdec, hdr, data=False, weights=False,
                   chunks=(1, 128, 128), names=None, fmt='NHW'):
        """
        Initialize a new xarray/NetCDF-compatible stack file.
        """
        if self.fid is not None:
            self.fid.close()

        times = _tdec_to_datetime64(tdec)
        attrs = {'format': 'xarray', 'original_format': fmt, 'time_units': TIME_UNITS}
        if hdr.epsg is not None:
            attrs['EPSG'] = int(hdr.epsg)
        ds = xr.Dataset(
            coords={
                'time': ('time', times),
                'y': ('y', hdr.ycoords),
                'x': ('x', hdr.xcoords),
            },
            attrs=attrs,
        )
        ds.to_netcdf(
            self.filename,
            engine='h5netcdf',
            encoding={'time': _time_encoding(times)},
        )
        ds.close()
        with h5py.File(self.filename, 'r+') as fid:
            fid['time'].attrs['units'] = TIME_UNITS
            fid['time'].attrs['calendar'] = TIME_CALENDAR

        self.fid = h5netcdf.File(self.filename, 'a')
        self.ds = xr.open_dataset(self.filename, engine='h5netcdf', decode_times=True)
        self.hdr = hdr
        self.original_fmt = fmt
        self.fmt = 'NHW'
        self._legacy = False
        self._dirty = False
        self._update_dataset_cache()

        if names is not None:
            self.names = names
        if data:
            self.init_default_datasets(weights=weights, chunks=chunks)

    def init_default_datasets(self, weights=False, chunks=(1, 128, 128)):
        """
        Initialize default datasets 'data' and optionally 'weights'.
        """
        self._require_initialized()
        shape = (self.Nt, self.Ny, self.Nx)
        self.create_dataset('data', shape, dtype='f', chunks=chunks)
        if weights:
            self.create_dataset('weights', shape, dtype='f', chunks=chunks)

    def create_dataset(self, name, shape, dtype='f', chunks=None, **kwargs):
        """
        Create a NetCDF-compatible dataset.
        """
        self._require_initialized()
        self._require_writable()
        if name in self.ds.variables or name in self.fid.variables:
            raise ValueError('Dataset %s already exists' % name)

        data = kwargs.pop('data', None)
        fillvalue = kwargs.pop('fillvalue', None)
        dims = self._dims_for_shape(shape, name)
        chunks = self._normalize_chunks(chunks, shape)
        variable = self.fid.create_variable(
            name, dims, dtype=np.dtype(dtype), data=data, fillvalue=fillvalue,
            chunks=chunks, **kwargs
        )

        if chunks is not None and len(chunks) == 3 and 'chunk_shape' not in self.fid.variables:
            self._ensure_dimension('chunk_dim', 3)
            self.fid.create_variable('chunk_shape', ('chunk_dim',), dtype='i8', data=np.asarray(chunks))

        self.fid.flush()
        self._dirty = True
        self._refresh_dataset_if_dirty()
        return variable

    def _dims_for_shape(self, shape, name):
        shape = tuple(shape)
        if shape == (self.Nt, self.Ny, self.Nx):
            return ('time', 'y', 'x')
        if shape == (self.Ny, self.Nx):
            return ('y', 'x')
        if shape == (self.Nt,):
            return ('time',)
        if shape == (self.Ny,):
            return ('y',)
        if shape == (self.Nx,):
            return ('x',)
        dims = []
        for index, size in enumerate(shape):
            dim = '%s_dim_%d' % (name.replace('/', '_'), index)
            self._ensure_dimension(dim, size)
            dims.append(dim)
        return tuple(dims)

    @staticmethod
    def _normalize_chunks(chunks, shape):
        if chunks is None:
            return None
        if len(chunks) != len(shape):
            return chunks
        return tuple(min(int(chunk), int(size)) for chunk, size in zip(chunks, shape))

    def __getitem__(self, name):
        """
        Return an xarray DataArray.
        """
        self._refresh_dataset_if_dirty()
        return self.ds[name]

    def __setitem__(self, name, value):
        """
        Creates a new dataset.
        """
        assert isinstance(value, np.ndarray), 'Must input NumPy array to set data.'
        self.create_dataset(name, value.shape, dtype=value.dtype, data=value)

    @property
    def names(self):
        """
        Get the names of the rasters in the Stack.
        """
        if 'names' not in self.ds:
            raise KeyError('names')
        return np.asarray(self.ds['names'].values).astype(str)

    @names.setter
    def names(self, names: Union[np.ndarray, List[str]]):
        """
        Set the name of each Raster in a Stack for reference.
        """
        self._require_writable()
        self._require_initialized()
        values = np.asarray(names, dtype=str)
        if values.size != self.Nt:
            raise ValueError('names must have one value per time step.')
        if 'names' in self.fid.variables:
            del self.fid.variables['names']
        self.fid.create_variable('names', ('time',), dtype=str, data=values)
        self.fid.flush()
        self._dirty = True

    def slice(self, index, key='data', as_raster=False, as_xarray=False):
        """
        Extract Stack 2D slice at given time index.
        """
        data = self[key]
        if 'time' in data.dims:
            data = data.isel(time=index)
        if as_xarray:
            return data
        values = np.asarray(data.values).squeeze()
        if as_raster:
            return Raster(data=values, hdr=copy.deepcopy(self.hdr))
        return values

    def set_slice(self, index, data, key='data'):
        """
        Set Stack 2D slice at given time index.
        """
        self._require_writable()
        values = _dataset_values(data)
        self.fid.variables[key][index, :, :] = values
        self.fid.flush()
        self._dirty = True

    def get_chunk(self, slice_y, slice_x, key='data', as_xarray=False):
        """
        Get a 3D chunk of data defined by 2D slice objects.
        """
        data = self[key]
        indexers = {}
        if 'y' in data.dims:
            indexers['y'] = slice_y
        if 'x' in data.dims:
            indexers['x'] = slice_x
        data = data.isel(**indexers) if indexers else data
        if as_xarray:
            return data
        return np.asarray(data.values)

    def set_chunk(self, slice_y, slice_x, data, key='data'):
        """
        Set a 3D chunk of data defined by 2D slice objects.
        """
        self._require_writable()
        values = _dataset_values(data)
        self.fid.variables[key][:, slice_y, slice_x] = values
        self.fid.flush()
        self._dirty = True

    def mean(self, key='data', as_xarray=False):
        """
        Compute mean along time dimension.
        """
        data = self[key].mean(dim='time', skipna=True)
        return data if as_xarray else np.asarray(data.values)

    def median(self, key='data', as_xarray=False):
        """
        Compute median along time dimension.
        """
        data = self[key].median(dim='time', skipna=True)
        return data if as_xarray else np.asarray(data.values)

    def std(self, key='data', as_xarray=False):
        """
        Compute standard deviation along time dimension.
        """
        data = self[key].std(dim='time', skipna=True)
        return data if as_xarray else np.asarray(data.values)

    def timeseries(self, xy=None, coord=None, key='data', win_size=1, as_xarray=False):
        """
        Extract time series at a coordinate, optionally averaging a spatial window.
        """
        if xy is not None and coord is None:
            x, y = xy
            row, col = self.hdr.xy_to_imagecoord(x, y)
        elif coord is not None:
            row, col = coord
        else:
            raise ValueError('Must pass in geographic or image coordinate.')

        half_win = win_size // 2
        if row >= (self.Ny - half_win) or row < half_win:
            warnings.warn('Requested point outside of stack bounds. Returning NaN.',
                          category=UserWarning)
            return self._nan_tseries.copy()
        if col >= (self.Nx - half_win) or col < half_win:
            warnings.warn('Requested point outside of stack bounds. Returning NaN.',
                          category=UserWarning)
            return self._nan_tseries.copy()

        if win_size > 1:
            data = self[key].isel(
                y=slice(row - half_win, row + half_win + 1),
                x=slice(col - half_win, col + half_win + 1),
            ).mean(dim=('y', 'x'), skipna=True)
        else:
            data = self[key].isel(y=row, x=col)

        return data if as_xarray else np.asarray(data.values)

    def resample(self, ref_hdr, output, key='data', dtype='f', order=3, chunks=None):
        """
        Resample dataset from one coordinate system to another RasterInfo object.
        """
        from tqdm import tqdm
        from .raster import interpolate_array

        if key not in self.ds:
            print('Warning: dataset %s not in stack' % key)
            return

        output.create_dataset(key, (self.Nt, ref_hdr.shape[0], ref_hdr.shape[1]),
                              dtype=dtype, chunks=chunks)
        for k in tqdm(range(self.Nt)):
            d = self.slice(k, key=key)
            d_interp = interpolate_array(d, self.hdr, None, None,
                                         order=order, ref_hdr=ref_hdr)
            output.set_slice(k, d_interp, key=key)

    def time_to_index(self, t=None, date=None):
        """
        Convert decimal year or datetime-like value to nearest time index.
        """
        if t is None and date is None:
            raise ValueError('Must provide t or date.')
        value = date if date is not None else t
        if _is_datetime_like(value):
            target = _as_datetime64_scalar(value)
            times = np.asarray(self.ds['time'].values).astype('datetime64[ns]')
            delta = np.abs((times - target) / np.timedelta64(1, 'ns'))
            return int(np.argmin(delta))
        return int(np.argmin(np.abs(self.tdec - value)))

    def close(self):
        """
        Close open file handles.
        """
        if self.ds is not None:
            self.ds.close()
        self._close_legacy_source()
        if self.fid is not None:
            self.fid.close()
            self.fid = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_args):
        self.close()

    @property
    def tdec(self):
        """
        Decimal years computed from the xarray time coordinate.
        """
        if 'time' not in self.ds.coords:
            return None
        return _datetime64_to_tdec(self.ds['time'].values)

    @property
    def dt(self):
        """
        Return mean sampling interval in decimal years.
        """
        return np.mean(np.diff(self.tdec))

    @property
    def Nt(self):
        return self.ds.sizes.get('time') if self.ds is not None else None

    @property
    def Ny(self):
        return self.hdr.shape[0] if self.hdr is not None else None

    @property
    def Nx(self):
        return self.hdr.shape[1] if self.hdr is not None else None

    @property
    def shape(self):
        if self.Nt is None or self.Ny is None or self.Nx is None:
            return None
        return (self.Nt, self.Ny, self.Nx)


class MultiStack:
    """
    Virtual stack representing arithmetic manipulation of multiple Stacks.
    """

    def __init__(self, stacks=None, files=None):
        if stacks is not None:
            self.stacks = stacks
        elif files is not None:
            self.stacks = [Stack(fname) for fname in files]
        else:
            raise ValueError('Must pass in stacks or filenames.')

        self.hdr = self.stacks[0].hdr
        self.fmt = 'NHW'

    def _combine(self, key):
        raise NotImplementedError('Child classes must implement _combine')

    def __getitem__(self, key):
        return self._combine(key)

    def slice(self, index, key='data'):
        return np.asarray(self[key].isel(time=index).values)

    def timeseries(self, xy=None, coord=None, key='data', win_size=1):
        if xy is not None and coord is None:
            row, col = self.hdr.xy_to_imagecoord(*xy)
        elif coord is not None:
            row, col = coord
        else:
            raise ValueError('Must pass in geographic or image coordinate.')

        half_win = win_size // 2
        if row >= (self.Ny - half_win) or row < half_win:
            return np.full((self.Nt,), np.nan, dtype='f')
        if col >= (self.Nx - half_win) or col < half_win:
            return np.full((self.Nt,), np.nan, dtype='f')

        if win_size > 1:
            data = self[key].isel(
                y=slice(row - half_win, row + half_win + 1),
                x=slice(col - half_win, col + half_win + 1),
            ).mean(dim=('y', 'x'), skipna=True)
        else:
            data = self[key].isel(y=row, x=col)
        return np.asarray(data.values)

    def get_chunk(self, slice_y, slice_x, key='data'):
        return np.asarray(self[key].isel(y=slice_y, x=slice_x).values)

    def time_to_index(self, t=None, date=None):
        return self.stacks[0].time_to_index(t=t, date=date)

    @property
    def tdec(self):
        return self.stacks[0].tdec

    @property
    def dt(self):
        return self.stacks[0].dt

    @property
    def Nt(self):
        return self.stacks[0].Nt

    @property
    def Ny(self):
        return self.stacks[0].Ny

    @property
    def Nx(self):
        return self.stacks[0].Nx


class MagStack(MultiStack):
    """
    MultiStack class that computes magnitude of stack objects.
    """

    def _combine(self, key):
        dsum = None
        for stack in self.stacks:
            term = stack[key] ** 2
            dsum = term if dsum is None else dsum + term
        return np.sqrt(dsum)


class SumStack(MultiStack):
    """
    MultiStack class that performs a sum on the stack objects.
    """

    def _combine(self, key):
        dsum = None
        for stack in self.stacks:
            term = stack[key]
            dsum = term if dsum is None else dsum + term
        return dsum


# --------------------------------------------------------------------------------
# Global utility functions
# --------------------------------------------------------------------------------


def h5read(filename, dataset):
    """
    Mimics the MATLAB function h5read for reading into memory.
    """
    if isinstance(dataset, str):
        with h5py.File(filename, 'r') as fid:
            data = fid[dataset][()]
        return data
    elif isinstance(dataset, (list, tuple)):
        data = []
        with h5py.File(filename, 'r') as fid:
            for key in dataset:
                data.append(fid[key][()])
        return data
    else:
        raise ValueError('Must provide dataset as str or list of str')


# end of file

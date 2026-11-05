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
DECIMAL_YEAR_UNITS = 'decimal_year'


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return value


def _normalize_time_units(units):
    units = _decode_attr(units)
    if units is None:
        return None
    units = str(units).strip()
    aliases = {
        'decimal years': DECIMAL_YEAR_UNITS,
        'decimal year': DECIMAL_YEAR_UNITS,
        'decimal_years': DECIMAL_YEAR_UNITS,
        'decimal_year': DECIMAL_YEAR_UNITS,
        'tdec': DECIMAL_YEAR_UNITS,
    }
    return aliases.get(units.lower(), units)


def _is_calendar_units(units):
    return units is not None and ' since ' in units.lower()


def _is_plain_numeric_units(units):
    return units is not None and units != DECIMAL_YEAR_UNITS and not _is_calendar_units(units)


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
    arr = arr.astype('datetime64[ns]')
    year_coord = arr.astype('datetime64[Y]')
    years = year_coord.astype(int) + 1970
    starts = year_coord.astype('datetime64[ns]')
    stops = (year_coord + 1).astype('datetime64[ns]')
    elapsed = (arr - starts) / np.timedelta64(1, 's')
    span = (stops - starts) / np.timedelta64(1, 's')
    return years.astype(float) + elapsed / span


def _calendar_units_to_datetime64(values, units):
    """
    Convert simple CF-style numeric time values to datetime64.
    """
    if units is not None and units.startswith('seconds since 1970-01-01'):
        return _seconds_since_epoch_to_datetime64(values)

    unit, origin = units.split(' since ', 1)
    unit = unit.strip().lower()
    origin = origin.strip()
    if origin.endswith(' UTC'):
        origin = origin[:-4]

    scale = {
        'second': 1.0e9,
        'seconds': 1.0e9,
        'sec': 1.0e9,
        'secs': 1.0e9,
        's': 1.0e9,
        'minute': 60.0e9,
        'minutes': 60.0e9,
        'min': 60.0e9,
        'mins': 60.0e9,
        'hour': 3600.0e9,
        'hours': 3600.0e9,
        'hr': 3600.0e9,
        'hrs': 3600.0e9,
        'day': 86400.0e9,
        'days': 86400.0e9,
        'd': 86400.0e9,
    }.get(unit)
    if scale is None:
        raise ValueError('Unsupported calendar time unit: %s' % units)

    arr = np.asarray(values)
    if np.any(~np.isfinite(arr.astype(float))):
        raise ValueError('Time coordinate contains non-finite values.')

    nanoseconds = np.rint(arr.astype(float) * scale).astype(np.int64)
    return np.datetime64(origin, 'ns') + nanoseconds.astype('timedelta64[ns]')


def _time_values_to_coordinate(values, units=None):
    """
    Normalize file time values to either datetime64 or native numeric values.
    """
    units = _normalize_time_units(units)
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype('datetime64[ns]'), units or TIME_UNITS
    if _is_calendar_units(units):
        return _calendar_units_to_datetime64(arr, units), units
    if _is_plain_numeric_units(units):
        return arr, units
    return _tdec_to_datetime64(arr), DECIMAL_YEAR_UNITS


def _seconds_since_epoch_to_datetime64(values):
    """
    Convert Unix seconds to datetime64 values.

    Older Stack writes could store integer nanoseconds while still labelling
    the coordinate as Unix seconds. Treat implausibly large second values as
    nanoseconds so those files remain readable.
    """
    arr = np.asarray(values)
    if np.any(~np.isfinite(arr.astype(float))):
        raise ValueError('Time coordinate contains non-finite values.')

    if arr.size and np.nanmax(np.abs(arr.astype(float))) > 1.0e12:
        nanoseconds = np.rint(arr.astype(float)).astype(np.int64)
    else:
        nanoseconds = np.rint(arr.astype(float) * 1.0e9).astype(np.int64)
    return EPOCH + nanoseconds.astype('timedelta64[ns]')


def _needs_float_time_encoding(times):
    nanoseconds = (
        np.asarray(times).astype('datetime64[ns]') - EPOCH
    ).astype('timedelta64[ns]').astype(np.int64)
    return np.any(nanoseconds % 1000000000 != 0)


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
                 init_names=None, init_data=False, ds_hdr=None, time_key='tdec',
                 indexers=None, time_units=None):
        """Reads Stack from an existing file or creates a new Stack.

        New files are NetCDF-compatible HDF5 files with xarray dimensions
        ``time``, ``y``, and ``x``. Legacy HDF5 stacks are normalized into the
        same xarray dimension order.
        """
        # Store file access and format settings.
        assert mode in ('r', 'r+', 'w', 'a', 'x'), 'Unsupported HDF5 file open mode'
        self.filename = filename
        self.mode = mode
        self.original_fmt = fmt
        self.fmt = 'NHW'
        self.hdr = None
        self.ds = xr.Dataset()
        self.fid = None
        self._dirty = False
        self._legacy = False
        self._legacy_source_ds = None
        self._time_key = time_key
        self._ds_hdr = ds_hdr
        self._indexers = self._normalize_indexers(indexers)
        self._time_units_override = _normalize_time_units(time_units)

        # Open an existing stack or initialize a new file.
        if mode in ('r', 'r+') or (mode == 'a' and os.path.exists(filename)):
            self._open_existing(ds_hdr=ds_hdr, time_key=time_key)
            if mode in ('r+', 'a') and not self._legacy:
                self.fid = h5netcdf.File(filename, 'a')
        else:
            self.fid = h5netcdf.File(self.filename, mode)
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

        # Cache the time-series shape used for out-of-bounds requests.
        if self.Nt is not None:
            self._nan_tseries = np.full((self.Nt,), np.nan, dtype='f')
        else:
            self._nan_tseries = None

    def _normalize_dataset_time(self, ds):
        if 'time' not in ds.coords:
            return ds

        # Resolve units with an explicit caller override taking precedence.
        units = self._time_units_override
        if units is None:
            units = self._time_units_from_dataarray(ds['time'])
            units = units or _normalize_time_units(ds.attrs.get('time_units'))
        if units is None:
            units = TIME_UNITS if np.issubdtype(ds['time'].dtype, np.datetime64) else None

        # Reinterpret the coordinate only when the caller supplied units.
        if self._time_units_override is not None:
            time, units = _time_values_to_coordinate(ds['time'].values, units=units)
            ds = ds.assign_coords(time=('time', time))

        # Datetime units belong in xarray's encoding, not its attributes.
        if units is not None:
            if np.issubdtype(ds['time'].dtype, np.datetime64):
                ds['time'].attrs.pop('units', None)
                if _is_calendar_units(units):
                    ds['time'].encoding['units'] = units
            else:
                ds['time'].attrs['units'] = units
            ds.attrs['time_units'] = units
        return ds

    def _open_existing(self, ds_hdr=None, time_key='tdec'):
        decode_times = self._time_units_override is None

        # Open a dataset with xarray when it has a supported representation.
        try:
            ds = xr.open_dataset(
                self.filename, engine='h5netcdf', decode_times=decode_times,
                phony_dims='sort'
            )
        except ValueError:
            ds = None

        # Initialize canonical datasets after opening succeeds.
        if ds is not None and {'time', 'y', 'x'}.issubset(ds.sizes):
            opened_ds = ds
            try:
                ds = self._normalize_dataset_time(ds)
                original_fmt = ds.attrs.get('original_format', ds.attrs.get('format', 'NHW'))
                hdr = self._rasterinfo_from_dataset(ds)
            except Exception:
                ds.close()
                if ds is not opened_ds:
                    opened_ds.close()
                raise

            self.ds = ds
            self.original_fmt = original_fmt
            self.fmt = 'NHW'
            self.hdr = hdr
            return

        # Reconstruct datasets that lack canonical xarray dimensions.
        if ds is not None:
            ds.close()

        self._legacy = True
        self._load_legacy_hdf5(ds_hdr=ds_hdr, time_key=time_key)

    def _load_legacy_hdf5(self, ds_hdr=None, time_key='tdec'):
        """
        Read legacy HDF5 datasets into canonical xarray dimensions.
        """
        # Discard a prior lazy source before rebuilding legacy variables.
        self._close_legacy_source()
        data_vars = {}
        raw_ds = None
        raw_ds_used = False
        ds = None

        # Use xarray lazily where its representation matches legacy datasets.
        try:
            raw_ds = xr.open_dataset(
                self.filename, engine='h5netcdf', decode_times=False,
                phony_dims='sort'
            )
        except ValueError:
            raw_ds = None

        # Read legacy coordinates, attributes, and data variables from HDF5.
        try:
            with h5py.File(self.filename, 'r') as fid:
                self.original_fmt = fid.attrs.get('format', 'NHW')
                if isinstance(self.original_fmt, bytes):
                    self.original_fmt = self.original_fmt.decode('utf-8')

                x, y = self._read_legacy_xy(fid, ds_hdr)
                time, time_units = self._read_legacy_time(fid, time_key)
                coords = {
                    'time': ('time', time, {'units': time_units}),
                    'y': ('y', y),
                    'x': ('x', x),
                }

                # Convert recognized raster datasets to canonical time/y/x order.
                for key, value in self._iter_hdf5_datasets(fid):
                    if key in ('x', 'X', 'y', 'Y', time_key, 'tdec', 't', 'time'):
                        continue
                    if key == 'names':
                        data_vars[key] = ('time', self._decode_names(value[()]))
                        continue
                    if key == 'chunk_shape':
                        data_vars[key] = ('chunk_dim', value[()])
                        continue

                    data_array = self._legacy_data_array(
                        raw_ds, key, value, x, y, time
                    )
                    if data_array is not None:
                        data_vars[key] = data_array
                        raw_ds_used = True
                        continue

                    arr = value[()]
                    stack_dims = self._legacy_stack_dims(arr.shape, x, y, time)
                    if stack_dims is not None:
                        if stack_dims == ('y', 'x', 'time'):
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
                        dims = tuple(
                            '%s_dim_%d' % (key.replace('/', '_'), i)
                            for i in range(arr.ndim)
                        )
                        data_vars[key] = (dims, arr)

                attrs = dict(fid.attrs)
                attrs['original_format'] = self.original_fmt
                attrs['format'] = 'xarray'
                attrs['time_units'] = time_units
                if 'EPSG' in fid.attrs:
                    attrs['EPSG'] = int(fid.attrs['EPSG'])

            # Construct canonical metadata once all source datasets are known.
            ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
            ds = self._normalize_dataset_time(ds)
            hdr = self._rasterinfo_from_dataset(ds)
        except Exception:
            if raw_ds is not None:
                raw_ds.close()
            if ds is not None:
                ds.close()
            raise

        # Keep the lazy xarray source alive only when the new dataset uses it.
        if raw_ds is not None and raw_ds_used:
            self._legacy_source_ds = raw_ds
        elif raw_ds is not None:
            raw_ds.close()

        self.ds = ds
        self.hdr = hdr
        self.fmt = 'NHW'

    def _legacy_data_array(self, raw_ds, key, value, x, y, time):
        if raw_ds is None or '/' in key or key not in raw_ds:
            return None

        data = raw_ds[key]
        shape = value.shape
        stack_dims = self._legacy_stack_dims(shape, x, y, time)
        if stack_dims is not None:
            data = data.rename(dict(zip(data.dims, stack_dims)))
            return data.transpose('time', 'y', 'x')
        if value.ndim == 2 and shape == (y.size, x.size):
            return data.rename(dict(zip(data.dims, ('y', 'x'))))
        return None

    def _legacy_stack_dims(self, shape, x, y, time):
        if len(shape) != 3:
            return None
        nhw_shape = (time.size, y.size, x.size)
        hwn_shape = (y.size, x.size, time.size)
        if shape == hwn_shape and (self.original_fmt == 'HWN' or shape != nhw_shape):
            return ('y', 'x', 'time')
        if shape == nhw_shape:
            return ('time', 'y', 'x')
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
        # Read explicit coordinate vectors from common legacy dataset names.
        try:
            x = fid['x'][()].squeeze()
            y = fid['y'][()].squeeze()
        except KeyError:
            try:
                x = fid['X'][()]
                y = fid['Y'][()]
            except KeyError:
                # Infer pixel coordinates when the legacy file has none.
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

        # Reduce stored coordinate grids to their one-dimensional axes.
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
        # Read the first recognized time vector with its associated units.
        for key in (time_key, 'time', 'tdec', 't'):
            if key in fid:
                units = self._time_units_override
                if units is None:
                    units = _normalize_time_units(fid[key].attrs.get('units'))
                return _time_values_to_coordinate(fid[key][()], units=units)

        # Build a minimal epoch-based coordinate when legacy metadata is absent.
        data_key = self._ds_hdr or Stack._guess_legacy_data_key(fid)
        nt = fid[data_key].shape[0] if self.original_fmt != 'HWN' else fid[data_key].shape[-1]
        warnings.warn('No time vector found. Using Unix epoch seconds.', category=UserWarning)
        return EPOCH + np.arange(nt).astype('timedelta64[s]'), TIME_UNITS

    @staticmethod
    def _rasterinfo_from_dataset(ds):
        # Construct raster coordinates from the canonical xarray axes.
        x = np.asarray(ds['x'].values)
        y = np.asarray(ds['y'].values)
        epsg = ds.attrs.get('EPSG')
        if epsg is not None:
            epsg = int(epsg)
        X, Y = np.meshgrid(x, y)
        return RasterInfo(X=X, Y=Y, epsg=epsg)

    def _refresh_dataset_if_dirty(self):
        if not self._dirty:
            return

        # Rebuild legacy datasets or reopen a canonical file after direct writes.
        if self._legacy:
            self._load_legacy_hdf5(ds_hdr=self._ds_hdr, time_key=self._time_key)
        else:
            self.ds.close()
            decode_times = self._time_units_override is None
            ds = xr.open_dataset(
                self.filename, engine='h5netcdf', decode_times=decode_times
            )
            try:
                ds = self._normalize_dataset_time(ds)
            except Exception:
                ds.close()
                raise
            self.ds = ds

        # Subsequent reads now see the latest on-disk variables and values.
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
        # Release the empty writable file before creating NetCDF coordinates.
        if self.fid is not None:
            self.fid.close()

        # Write canonical time and spatial coordinates with stack metadata.
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

        # Preserve the time encoding expected by existing Stack files.
        with h5py.File(self.filename, 'r+') as fid:
            fid['time'].attrs['units'] = TIME_UNITS
            fid['time'].attrs['calendar'] = TIME_CALENDAR

        # Reopen the file for direct writes and xarray-backed access.
        self.fid = h5netcdf.File(self.filename, 'a')
        self.ds = xr.open_dataset(self.filename, engine='h5netcdf', decode_times=True)
        self.ds = self._normalize_dataset_time(self.ds)
        self.hdr = hdr
        self.original_fmt = fmt
        self.fmt = 'NHW'
        self._legacy = False
        self._dirty = False

        # Add optional metadata and default raster variables.
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
        # Validate that a writable, initialized stack has a new variable name.
        self._require_initialized()
        self._require_writable()
        if name in self.ds.variables or name in self.fid.variables:
            raise ValueError('Dataset %s already exists' % name)

        # Map the requested shape to canonical or private NetCDF dimensions.
        data = kwargs.pop('data', None)
        fillvalue = kwargs.pop('fillvalue', None)
        dims = self._dims_for_shape(shape, name)
        chunks = self._normalize_chunks(chunks, shape)
        variable = self.fid.create_variable(
            name, dims, dtype=np.dtype(dtype), data=data, fillvalue=fillvalue,
            chunks=chunks, **kwargs
        )

        # Record spatial chunking once for callers that use it for tiling.
        if chunks is not None and len(chunks) == 3 and 'chunk_shape' not in self.fid.variables:
            self._ensure_dimension('chunk_dim', 3)
            self.fid.create_variable('chunk_shape', ('chunk_dim',), dtype='i8', data=np.asarray(chunks))

        # Flush and refresh xarray so the new variable is immediately visible.
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

    @staticmethod
    def _normalize_indexers(indexers):
        if indexers is None:
            return {}
        return {str(key): int(value) for key, value in dict(indexers).items()}

    @staticmethod
    def _time_units_from_dataarray(data):
        units = data.attrs.get('units')
        if units is None:
            units = data.encoding.get('units')
        return _normalize_time_units(units)

    def __getitem__(self, name):
        """
        Return an xarray DataArray.
        """
        # Reload variables created or written through the HDF5 handle.
        self._refresh_dataset_if_dirty()
        data = self.ds[name]

        # Apply configured indexers only to dimensions present in this variable.
        indexers = {}
        for key, value in self._indexers.items():
            if key in data.dims:
                indexers[key] = value
        if indexers:
            return data.isel(**indexers)
        return data

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
        # Reload variables created or written through the HDF5 handle.
        self._refresh_dataset_if_dirty()
        if 'names' not in self.ds:
            raise KeyError('Stack has no names variable. Assign to Stack.names to create one.')
        return np.asarray(self.ds['names'].values).astype(str)

    @names.setter
    def names(self, names: Union[np.ndarray, List[str]]):
        """
        Set the name of each Raster in a Stack for reference.
        """
        # Validate names against the initialized time coordinate.
        self._require_writable()
        self._require_initialized()

        values = np.asarray(names, dtype=str)
        if values.size != self.Nt:
            raise ValueError('names must have one value per time step.')

        # Write the names as variable-length strings. HDF5 has no fixed-width unicode
        # type, so the data must be an object array rather than a NumPy unicode array.
        # h5netcdf cannot delete a variable, so an existing one is overwritten in place;
        # its length always matches because it is dimensioned on the time coordinate.
        values = values.astype(object)
        if 'names' in self.fid.variables:
            self.fid['names'][:] = values
        else:
            self.fid.create_variable('names', ('time',), dtype=h5py.string_dtype(),
                                     data=values)

        # Mark the xarray view stale so subsequent reads pick up the new values.
        self.fid.flush()
        self._dirty = True

    def slice(self, index, key='data', as_raster=False, as_xarray=False):
        """
        Extract Stack 2D slice at given time index.
        """
        # Select the requested time plane when the variable has a time dimension.
        data = self[key]
        if 'time' in data.dims:
            data = data.isel(time=index)

        # Return xarray data directly or materialize values for legacy callers.
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
        # Build spatial indexers only for dimensions available in this variable.
        data = self[key]
        indexers = {}
        if 'y' in data.dims:
            indexers['y'] = slice_y
        if 'x' in data.dims:
            indexers['x'] = slice_x
        data = data.isel(**indexers) if indexers else data

        # Return xarray data directly or materialize the selected chunk.
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
        # Map geographic coordinates to a pixel or use the supplied pixel index.
        if xy is not None and coord is None:
            x, y = xy
            row, col = self.hdr.xy_to_imagecoord(x, y)
        elif coord is not None:
            row, col = coord
        else:
            raise ValueError('Must pass in geographic or image coordinate.')

        # Return a NaN series for points outside the requested spatial window.
        half_win = win_size // 2
        if row >= (self.Ny - half_win) or row < half_win:
            warnings.warn('Requested point outside of stack bounds. Returning NaN.',
                          category=UserWarning)
            return self._nan_tseries.copy()
        if col >= (self.Nx - half_win) or col < half_win:
            warnings.warn('Requested point outside of stack bounds. Returning NaN.',
                          category=UserWarning)
            return self._nan_tseries.copy()

        # Select one pixel or average the requested spatial window.
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

        # Confirm that the requested variable can be resampled.
        if key not in self.ds:
            print('Warning: dataset %s not in stack' % key)
            return

        # Create output storage and interpolate every time slice.
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
        # Validate the requested coordinate and select the supplied value.
        if t is None and date is None:
            raise ValueError('Must provide t or date.')
        value = date if date is not None else t
        if _is_datetime_like(value):
            # Compare calendar dates only against calendar-valued coordinates.
            if not self.time_is_calendar:
                raise ValueError('Datetime selection requires calendar time coordinates.')
            target = _as_datetime64_scalar(value)
            times = np.asarray(self.ds['time'].values).astype('datetime64[ns]')
            delta = np.abs((times - target) / np.timedelta64(1, 'ns'))
            return int(np.argmin(delta))

        # Compare numeric values in their native time coordinate units.
        return int(np.argmin(np.abs(self.tdec - value)))

    def close(self):
        """
        Close open file handles.
        """
        # Release xarray datasets and their optional lazy legacy source.
        if self.ds is not None:
            self.ds.close()
        self._close_legacy_source()

        # Close the writable HDF5 handle when this Stack owns one.
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
        Decimal years for calendar time, or native numeric time values.
        """
        if 'time' not in self.ds.coords:
            return None
        values = self.ds['time'].values
        if np.issubdtype(np.asarray(values).dtype, np.datetime64):
            return _datetime64_to_tdec(values)
        return np.asarray(values, dtype=float)

    @property
    def dt(self):
        """
        Return mean sampling interval in the units represented by tdec.
        """
        return np.mean(np.diff(self.tdec))

    @property
    def time_units(self):
        """
        Return the active time coordinate units.
        """
        if 'time' not in self.ds.coords:
            return None
        units = self._time_units_from_dataarray(self.ds['time'])
        units = units or _normalize_time_units(self.ds.attrs.get('time_units'))
        if units is not None:
            return units
        return TIME_UNITS if self.time_is_calendar else None

    @property
    def time_is_calendar(self):
        """
        True when the time coordinate represents calendar datetimes.
        """
        if 'time' not in self.ds.coords:
            return False
        return np.issubdtype(np.asarray(self.ds['time'].values).dtype, np.datetime64)

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

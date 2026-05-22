import os
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import pytest
import xarray as xr
import rasterio
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "iceutils-mpl"))

from iceutils.raster import Raster, RasterInfo, warp, write_gdal
from iceutils.stack import Stack, TIME_UNITS


def _write_geotiff(path, data, transform=None, crs="EPSG:3413", nodata=None):
    if transform is None:
        transform = from_origin(100.0, 200.0, 10.0, 20.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)
    return transform


def _write_multiband_geotiff(path, data, transform=None, crs="EPSG:3413"):
    if transform is None:
        transform = from_origin(100.0, 200.0, 10.0, 20.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)
    return transform


def _stack_hdr(shape=(3, 4)):
    return RasterInfo(
        transform=Affine(2.0, 0.0, 10.0, 0.0, -3.0, 20.0),
        crs=CRS.from_epsg(3413),
        shape=shape,
    )


def _write_stack(path, data, key="data", weights=None):
    tdec = np.arange(data.shape[0], dtype=float) + 2020.0
    with Stack(str(path), mode="w") as stack:
        stack.initialize(tdec, _stack_hdr(data.shape[1:]), data=False)
        stack.create_dataset(key, data.shape, dtype=data.dtype)
        stack.set_chunk(slice(None), slice(None), data, key=key)
        if weights is not None:
            stack.create_dataset("weights", weights.shape, dtype=weights.dtype)
            stack.set_chunk(slice(None), slice(None), weights, key="weights")


def _load_bin_script(name):
    path = Path(__file__).resolve().parents[1] / "bin" / name
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _solver_module():
    try:
        from iceutils.tseries import solver
    except ImportError as err:
        pytest.skip("optional time-series solver dependencies unavailable: %s" % err)
    return solver


def _write_linear_user_model(path):
    path.write_text(
        "\n".join(
            [
                "import numpy as np",
                "from iceutils.tseries import timefn",
                "",
                "def build(dates, **kwargs):",
                "    collection = timefn.TimefnCollection()",
                "    collection.append(timefn.fnmap['poly'](tref=dates[0], order=1, units='years'))",
                "    return collection",
                "",
                "def computeCm(collection, **kwargs):",
                "    return np.eye(len(collection), dtype=float)",
                "",
            ]
        )
    )
    return str(path)


def _read_stack_variable(path, key="data"):
    with Stack(str(path)) as stack:
        return np.asarray(stack[key].values).copy()


def _xarray_storage_type_names(data_array):
    names = []
    obj = data_array.variable._data
    for _ in range(10):
        names.append(type(obj).__name__)
        if isinstance(obj, np.ndarray):
            break
        next_obj = None
        for attr in ("array", "_array"):
            if hasattr(obj, attr):
                candidate = getattr(obj, attr)
                if candidate is not obj:
                    next_obj = candidate
                    break
        if next_obj is None:
            break
        obj = next_obj
    return names


def _assert_h5netcdf_lazy(data_array):
    storage_types = _xarray_storage_type_names(data_array)
    assert "H5NetCDFArrayWrapper" in storage_types
    assert "NumpyIndexingAdapter" not in storage_types
    assert "ndarray" not in storage_types


def test_read_write_round_trip_preserves_data_and_metadata(tmp_path):
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    src = tmp_path / "src.tif"
    transform = _write_geotiff(src, data, nodata=-9999.0)

    raster = Raster(str(src))

    assert np.array_equal(raster.data, data)
    assert raster.nodataval == -9999.0
    assert raster.hdr.shape == data.shape
    assert raster.hdr.epsg == 3413
    assert raster.hdr.nbands == 1
    assert tuple(raster.hdr.transform) == tuple(transform)
    assert raster.hdr.dtype == np.dtype("float32")

    out = tmp_path / "out.tif"
    raster.write_raster(str(out), driver="GTiff", nodataval=-9999.0)

    with rasterio.open(out) as ds:
        assert np.array_equal(ds.read(1), data)
        assert ds.nodata == -9999.0
        assert ds.crs.to_epsg() == 3413
        assert tuple(ds.transform) == tuple(transform)


def test_rasterinfo_and_ice_info_report_raster_band_count(tmp_path):
    data = np.stack([
        np.full((2, 3), 1, dtype=np.float32),
        np.full((2, 3), 2, dtype=np.float32),
        np.full((2, 3), 3, dtype=np.float32),
    ])
    src = tmp_path / "multiband.tif"
    _write_multiband_geotiff(src, data)

    hdr = RasterInfo(str(src))
    raster = Raster(str(src), band=2)

    assert hdr.nbands == 3
    assert raster.hdr.nbands == 3

    script = Path(__file__).resolve().parents[1] / "bin" / "ice_info.py"
    result = subprocess.run(
        [sys.executable, str(script), str(src)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Number of bands: 3" in result.stdout


def test_projwin_and_slice_reads_update_array_and_transform(tmp_path):
    data = np.arange(20, dtype=np.float32).reshape(4, 5)
    src = tmp_path / "src.tif"
    _write_geotiff(src, data, transform=from_origin(0.0, 40.0, 10.0, 10.0), crs="EPSG:4326")

    by_bounds = Raster(str(src), projWin=[10.0, 40.0, 40.0, 20.0])
    assert np.array_equal(by_bounds.data, data[0:2, 1:4])
    assert by_bounds.hdr.shape == (2, 3)
    assert tuple(by_bounds.hdr.transform) == tuple(from_origin(10.0, 40.0, 10.0, 10.0))

    by_slice = Raster(str(src), islice=slice(1, 3), jslice=slice(2, 5))
    assert np.array_equal(by_slice.data, data[1:3, 2:5])
    assert by_slice.hdr.shape == (2, 3)
    assert tuple(by_slice.hdr.transform) == tuple(from_origin(20.0, 30.0, 10.0, 10.0))


def test_affine_coordinate_helpers_support_rotated_grids():
    transform = Affine(2.0, 0.5, 100.0, 0.25, -3.0, 200.0)
    hdr = RasterInfo(transform=transform, crs=CRS.from_epsg(3413), shape=(4, 5))

    rows = np.array([0, 2, 3])
    cols = np.array([0, 3, 4])
    x, y = hdr.imagecoord_to_xy(rows, cols)
    out_rows, out_cols = hdr.xy_to_imagecoord(x, y)

    assert np.array_equal(out_rows, rows)
    assert np.array_equal(out_cols, cols)

    X, Y = hdr.meshgrid()
    expected_x, expected_y = transform * (3, 2)
    assert X[2, 3] == expected_x
    assert Y[2, 3] == expected_y

    with pytest.raises(ValueError):
        _ = hdr.xcoords


def test_write_gdal_alias_writes_rasterio_compatible_multiband_file(tmp_path):
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = tmp_path / "alias.tif"
    geotransform = [100.0, 5.0, 0.0, 200.0, 0.0, -5.0]

    write_gdal((data, data + 1), str(out), geotransform=geotransform,
               epsg=3413, driver="GTiff", nodataval=-9999.0)

    with rasterio.open(out) as ds:
        assert ds.count == 2
        assert np.array_equal(ds.read(1), data)
        assert np.array_equal(ds.read(2), data + 1)
        assert ds.crs.to_epsg() == 3413
        assert ds.nodata == -9999.0
        assert tuple(ds.transform) == tuple(Affine.from_gdal(*geotransform))


def test_warp_matches_rasterio_reproject_for_epsg_change(tmp_path):
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    src = tmp_path / "src.tif"
    transform = _write_geotiff(
        src, data, transform=from_origin(-45.0, 75.0, 1.0, 1.0), crs="EPSG:4326"
    )
    raster = Raster(str(src))

    warped = warp(raster, target_epsg=3413, target_dims=(6, 6), order=1)

    expected_transform, expected_width, expected_height = calculate_default_transform(
        CRS.from_epsg(4326), CRS.from_epsg(3413), 4, 4, *raster.hdr.bounds,
        dst_height=6, dst_width=6
    )
    expected = np.zeros((expected_height, expected_width), dtype=np.float32)
    reproject(
        data,
        expected,
        src_transform=transform,
        src_crs=CRS.from_epsg(4326),
        dst_transform=expected_transform,
        dst_crs=CRS.from_epsg(3413),
        resampling=Resampling.bilinear,
    )

    assert warped.hdr.shape == expected.shape
    assert warped.hdr.epsg == 3413
    assert tuple(warped.hdr.transform) == tuple(expected_transform)
    assert np.allclose(warped.data, expected)


def test_resample_matches_rasterio_reproject_for_same_crs_target_grid():
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    src_hdr = RasterInfo(
        transform=from_origin(0.0, 4.0, 1.0, 1.0),
        crs=CRS.from_epsg(3857),
        shape=data.shape,
    )
    dst_hdr = RasterInfo(
        transform=from_origin(0.0, 4.0, 2.0, 2.0),
        crs=CRS.from_epsg(3857),
        shape=(2, 2),
    )
    raster = Raster(data=data.copy(), hdr=src_hdr)

    expected = np.zeros(dst_hdr.shape, dtype=np.float32)
    reproject(
        data,
        expected,
        src_transform=src_hdr.transform,
        src_crs=src_hdr.crs,
        dst_transform=dst_hdr.transform,
        dst_crs=dst_hdr.crs,
        resampling=Resampling.bilinear,
    )

    raster.resample(dst_hdr, order=1)

    assert raster.hdr == dst_hdr
    assert np.allclose(raster.data, expected)


def test_stack_rasterinfo_keeps_existing_hdf5_layout(tmp_path):
    stack = tmp_path / "stack.h5"
    with h5py.File(stack, "w") as fid:
        fid["x"] = np.array([10.0, 12.0, 14.0])
        fid["y"] = np.array([20.0, 17.0])
        fid["tdec"] = np.array([2020.0, 2021.0])
        fid["data"] = np.zeros((2, 2, 3), dtype=np.float32)
        fid.attrs["EPSG"] = 3413
        fid.attrs["format"] = "NHW"

    hdr = RasterInfo(stackfile=str(stack), ds="data")

    assert hdr.shape == (2, 3)
    assert hdr.epsg == 3413
    assert tuple(hdr.transform) == tuple(Affine(2.0, 0.0, 10.0, 0.0, -3.0, 20.0))
    assert np.array_equal(hdr.xcoords, np.array([10.0, 12.0, 14.0]))
    assert np.array_equal(hdr.ycoords, np.array([20.0, 17.0]))


def test_stack_writes_xarray_format_with_unix_second_time_encoding(tmp_path):
    path = tmp_path / "stack_xarray.h5"
    hdr = RasterInfo(
        transform=Affine(2.0, 0.0, 10.0, 0.0, -3.0, 20.0),
        crs=CRS.from_epsg(3413),
        shape=(2, 3),
    )
    tdec = np.array([2020.0, 2021.0])
    data = np.arange(12, dtype=np.float32).reshape(2, 2, 3)

    with Stack(str(path), mode="w") as stack:
        stack.initialize(tdec, hdr, data=True)
        stack.set_chunk(slice(None), slice(None), data)
        assert isinstance(stack["data"], xr.DataArray)
        assert stack["data"].dims == ("time", "y", "x")
        assert stack.time_to_index(date="2021-01-01") == 1

    with h5py.File(path, "r") as fid:
        units = fid["time"].attrs["units"]
        if isinstance(units, bytes):
            units = units.decode("utf-8")
        assert units == TIME_UNITS
        assert fid["time"].dtype.kind in ("i", "u")
        assert fid.attrs["format"] == "xarray"

    with Stack(str(path)) as stack:
        assert np.allclose(stack.tdec, tdec)
        assert stack["data"].dims == ("time", "y", "x")
        assert np.array_equal(stack.slice(1), data[1])
        assert np.array_equal(stack.get_chunk(slice(0, 2), slice(1, 3)), data[:, :, 1:3])
        assert np.array_equal(stack.timeseries(coord=(1, 2)), data[:, 1, 2])
        assert np.array_equal(stack.mean(), data.mean(axis=0))
        selected = stack["data"].sel(time=np.datetime64("2021-01-01"))
        assert np.array_equal(selected.values, data[1])
        deriv = stack["data"].differentiate("time")
        assert deriv.dims == ("time", "y", "x")


def test_stack_writes_fractional_second_times_as_float_seconds(tmp_path):
    path = tmp_path / "stack_fractional_time.h5"
    hdr = RasterInfo(
        transform=Affine(2.0, 0.0, 10.0, 0.0, -3.0, 20.0),
        crs=CRS.from_epsg(3413),
        shape=(2, 3),
    )
    tdec = np.array([2020.0409836065573, 2024.872950819672])

    with Stack(str(path), mode="w") as stack:
        stack.initialize(tdec, hdr, data=False)
        assert stack.ds["time"].sizes["time"] == 2

    with h5py.File(path, "r") as fid:
        units = fid["time"].attrs["units"]
        if isinstance(units, bytes):
            units = units.decode("utf-8")
        assert units == TIME_UNITS
        assert fid["time"].dtype.kind == "f"

    with Stack(str(path)) as stack:
        assert np.allclose(stack.tdec, tdec)


def test_stack_reads_legacy_nanosecond_time_values_with_second_units(tmp_path):
    path = tmp_path / "stack_legacy_nanosecond_time.h5"
    times = np.array(
        [
            "2020-01-15T23:59:59.999998976",
            "2020-02-14T23:59:59.999997184",
            "2020-03-16T00:00:00.000002560",
        ],
        dtype="datetime64[ns]",
    )
    epoch = np.datetime64("1970-01-01T00:00:00", "ns")
    nanoseconds = (times - epoch).astype("timedelta64[ns]").astype(np.int64)

    with h5py.File(path, "w") as fid:
        fid.attrs["format"] = "xarray"
        fid.attrs["EPSG"] = 3413
        time = fid.create_dataset("time", data=nanoseconds)
        time.attrs["units"] = TIME_UNITS
        time.attrs["calendar"] = "proleptic_gregorian"
        fid.create_dataset("y", data=np.array([20.0, 17.0]))
        fid.create_dataset("x", data=np.array([10.0, 12.0, 14.0]))
        fid.create_dataset("data", data=np.zeros((times.size, 2, 3), dtype=np.float32))

    with Stack(str(path)) as stack:
        assert np.array_equal(stack.ds["time"].values.astype("datetime64[ns]"), times)
        assert np.ptp(stack.tdec) > 0.0


def test_stack_indexers_select_extra_netcdf_dimensions(tmp_path):
    path = tmp_path / "multiband_stack.nc"
    data = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    time1 = np.array([10.0, 20.0], dtype=np.float32)
    ds = xr.Dataset(
        data_vars={
            "VelocityMap": (("time", "band", "y", "x"), data),
            "time1": ("time", time1),
        },
        coords={
            "time": ("time", np.array(["2020-01-01", "2021-01-01"], dtype="datetime64[ns]")),
            "band": ("band", np.arange(3)),
            "y": ("y", np.array([20.0, 17.0, 14.0, 11.0])),
            "x": ("x", np.array([10.0, 12.0, 14.0, 16.0, 18.0])),
        },
        attrs={"EPSG": 3413},
    )
    ds.to_netcdf(path, engine="h5netcdf")
    ds.close()

    with Stack(str(path), indexers={"band": 1}) as stack:
        assert stack["VelocityMap"].dims == ("time", "y", "x")
        assert np.array_equal(stack["VelocityMap"].values, data[:, 1])
        assert np.array_equal(stack["time1"].values, time1)
        assert np.array_equal(stack.slice(1, key="VelocityMap"), data[1, 1])
        assert np.array_equal(
            stack.get_chunk(slice(1, 3), slice(2, 5), key="VelocityMap"),
            data[:, 1, 1:3, 2:5],
        )
        assert np.array_equal(
            stack.timeseries(coord=(2, 3), key="VelocityMap"),
            data[:, 1, 2, 3],
        )
        assert np.array_equal(stack.mean(key="VelocityMap"), data[:, 1].mean(axis=0))


def test_stack_reads_legacy_nhw_hdf5_as_canonical_xarray(tmp_path):
    path = tmp_path / "legacy_nhw.h5"
    data = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    with h5py.File(path, "w") as fid:
        fid["x"] = np.array([10.0, 12.0, 14.0])
        fid["y"] = np.array([20.0, 17.0])
        fid["tdec"] = np.array([2020.0, 2021.0])
        fid["data"] = data
        fid["weights"] = data + 1
        fid["quality"] = np.array([4, 5, 6, 7], dtype=np.int16)
        fid.attrs["EPSG"] = 3413
        fid.attrs["format"] = "NHW"

    with Stack(str(path)) as stack:
        assert stack.fmt == "NHW"
        assert stack.original_fmt == "NHW"
        assert stack._legacy_source_ds is not None
        assert stack["data"].dims == ("time", "y", "x")
        _assert_h5netcdf_lazy(stack["data"])
        _assert_h5netcdf_lazy(stack["weights"])
        assert np.array_equal(stack["data"].isel(time=1, y=slice(None), x=slice(1, 3)).values,
                              data[1, :, 1:3])
        assert np.array_equal(stack["data"].values, data)
        assert np.array_equal(stack["quality"].values, np.array([4, 5, 6, 7], dtype=np.int16))
        assert np.array_equal(stack.slice(0), data[0])
        assert np.array_equal(stack.get_chunk(slice(0, 2), slice(1, 3)), data[:, :, 1:3])
        assert np.array_equal(stack.timeseries(coord=(1, 2)), data[:, 1, 2])


def test_stack_reads_legacy_hwn_hdf5_as_canonical_xarray(tmp_path):
    path = tmp_path / "legacy_hwn.h5"
    canonical = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    hwn = np.moveaxis(canonical, 0, -1)
    with h5py.File(path, "w") as fid:
        fid["x"] = np.array([10.0, 12.0, 14.0])
        fid["y"] = np.array([20.0, 17.0])
        fid["tdec"] = np.array([2020.0, 2021.0])
        fid["data"] = hwn
        fid.attrs["EPSG"] = 3413
        fid.attrs["format"] = "HWN"

    with Stack(str(path)) as stack:
        assert stack.fmt == "NHW"
        assert stack.original_fmt == "HWN"
        assert stack.shape == canonical.shape
        assert stack._legacy_source_ds is not None
        assert stack["data"].dims == ("time", "y", "x")
        _assert_h5netcdf_lazy(stack["data"])
        assert np.array_equal(stack["data"].isel(time=1, y=slice(None), x=slice(1, 3)).values,
                              canonical[1, :, 1:3])
        assert np.array_equal(stack["data"].values, canonical)
        assert np.array_equal(stack.get_chunk(slice(None), slice(None)), canonical)


def test_solver_chunk_flattening_uses_canonical_xarray_order(tmp_path):
    solver = _solver_module()
    path = tmp_path / "solver_stack.h5"
    data = np.arange(36, dtype=np.float32).reshape(3, 3, 4)
    weights = data + 100.0
    mask = np.array([
        [True, False, True, True],
        [False, True, True, False],
        [True, True, False, True],
    ])
    _write_stack(path, data, weights=weights)

    with Stack(str(path)) as stack:
        data2d, wgts2d, data1d, wgts1d, chunk_mask = solver._stack_chunk_to_timeseries(
            stack, slice(0, 2), slice(1, 4), mask=mask
        )

    expected_data = data[:, 0:2, 1:4]
    expected_weights = weights[:, 0:2, 1:4]
    expected_mask = mask[0:2, 1:4]
    assert np.array_equal(data2d, expected_data)
    assert np.array_equal(wgts2d, expected_weights)
    assert np.array_equal(chunk_mask, expected_mask)
    assert np.array_equal(data1d, expected_data[:, expected_mask])
    assert np.array_equal(wgts1d, expected_weights[:, expected_mask])


def test_solver_chunk_flattening_reads_legacy_hwn_as_canonical(tmp_path):
    solver = _solver_module()
    path = tmp_path / "solver_legacy_hwn.h5"
    canonical = np.arange(36, dtype=np.float32).reshape(3, 3, 4)
    weights = canonical + 10.0
    mask = np.array([
        [True, False, True, True],
        [False, True, True, False],
        [True, True, False, True],
    ])
    with h5py.File(path, "w") as fid:
        fid["x"] = np.array([10.0, 12.0, 14.0, 16.0])
        fid["y"] = np.array([20.0, 17.0, 14.0])
        fid["tdec"] = np.array([2020.0, 2021.0, 2022.0])
        fid["data"] = np.moveaxis(canonical, 0, -1)
        fid["weights"] = np.moveaxis(weights, 0, -1)
        fid.attrs["EPSG"] = 3413
        fid.attrs["format"] = "HWN"

    with Stack(str(path)) as stack:
        data2d, wgts2d, data1d, wgts1d, chunk_mask = solver._stack_chunk_to_timeseries(
            stack, slice(1, 3), slice(0, 3), mask=mask
        )

    expected_data = canonical[:, 1:3, 0:3]
    expected_weights = weights[:, 1:3, 0:3]
    expected_mask = mask[1:3, 0:3]
    assert np.array_equal(data2d, expected_data)
    assert np.array_equal(wgts2d, expected_weights)
    assert np.array_equal(chunk_mask, expected_mask)
    assert np.array_equal(data1d, expected_data[:, expected_mask])
    assert np.array_equal(wgts1d, expected_weights[:, expected_mask])


def test_solver_inversion_joblib_matches_serial_with_cleaned_stack(tmp_path):
    solver = _solver_module()
    userfile = _write_linear_user_model(tmp_path / "linear_model.py")
    stack_path = tmp_path / "invert_stack.h5"
    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"
    serial_dir.mkdir()
    parallel_dir.mkdir()

    time = np.arange(6, dtype=np.float32)[:, None, None]
    rows, cols = np.indices((2, 3), dtype=np.float32)
    data = 1.0 + 0.25 * time + 0.5 * rows + 0.1 * cols
    data = data.astype(np.float32)
    data[3, 1, 2] += 5.0
    _write_stack(stack_path, data)

    with Stack(str(stack_path)) as stack:
        solver.inversion(
            stack,
            userfile,
            str(serial_dir),
            cleaned_stack="cleaned.h5",
            solver_type="lsqr",
            nt_out=5,
            n_proc=1,
            n_min=2,
            n_iter=2,
            n_std=1.5,
            no_weights=True,
            prior_cov=False,
        )

    with Stack(str(stack_path)) as stack:
        solver.inversion(
            stack,
            userfile,
            str(parallel_dir),
            cleaned_stack="cleaned.h5",
            solver_type="lsqr",
            nt_out=5,
            n_proc=2,
            n_min=2,
            n_iter=2,
            n_std=1.5,
            no_weights=True,
            prior_cov=False,
        )

    for key in ("full", "secular", "seasonal", "transient", "sigma"):
        serial = _read_stack_variable(serial_dir / ("interp_output_%s.h5" % key))
        parallel = _read_stack_variable(parallel_dir / ("interp_output_%s.h5" % key))
        np.testing.assert_allclose(parallel, serial, equal_nan=True)

    for key in ("data", "weights"):
        serial = _read_stack_variable(serial_dir / "cleaned.h5", key=key)
        parallel = _read_stack_variable(parallel_dir / "cleaned.h5", key=key)
        np.testing.assert_allclose(parallel, serial, equal_nan=True)


def test_solver_inversion_accepts_model_instance(tmp_path):
    solver = _solver_module()
    userfile = _write_linear_user_model(tmp_path / "linear_model.py")
    stack_path = tmp_path / "model_stack.h5"
    file_dir = tmp_path / "file_model"
    instance_dir = tmp_path / "instance_model"
    file_dir.mkdir()
    instance_dir.mkdir()

    time = np.arange(6, dtype=np.float32)[:, None, None]
    rows, cols = np.indices((2, 3), dtype=np.float32)
    data = 1.5 + 0.4 * time + 0.25 * rows + 0.05 * cols
    _write_stack(stack_path, data.astype(np.float32))

    with Stack(str(stack_path)) as stack:
        solver.inversion(
            stack,
            userfile,
            str(file_dir),
            solver_type="lsqr",
            nt_out=5,
            n_proc=1,
            n_min=2,
            no_weights=True,
        )

    with Stack(str(stack_path)) as stack:
        model = solver.build_temporal_model(
            stack.tdec, poly=1, periods=[], isplines=[], bsplines=[]
        )
        solver.inversion(
            stack,
            model,
            str(instance_dir),
            solver_type="lsqr",
            nt_out=5,
            n_proc=1,
            n_min=2,
            no_weights=True,
        )

    for key in ("full", "secular", "seasonal", "transient", "sigma"):
        from_file = _read_stack_variable(file_dir / ("interp_output_%s.h5" % key))
        from_instance = _read_stack_variable(instance_dir / ("interp_output_%s.h5" % key))
        np.testing.assert_allclose(from_instance, from_file, equal_nan=True)


def test_solver_uses_model_regularization_indices_for_bsplines():
    solver = _solver_module()
    tdec = np.linspace(2020.0, 2021.0, 8)
    model = solver.build_temporal_model(
        tdec, poly=1, bsplines=[4], isplines=[4], periods=[]
    )

    reg_indices = solver._regularization_indices(model)

    assert len(model.reg_indices) > len(model.itransient)
    assert np.array_equal(reg_indices, model.reg_indices)


def test_solver_inversion_points_joblib_matches_serial(tmp_path):
    solver = _solver_module()
    userfile = _write_linear_user_model(tmp_path / "linear_model.py")
    stack_path = tmp_path / "points_stack.h5"

    time = np.arange(6, dtype=np.float32)[:, None, None]
    rows, cols = np.indices((2, 3), dtype=np.float32)
    data = 2.0 + 0.5 * time + rows + 0.2 * cols
    weights = np.ones_like(data, dtype=np.float32)
    _write_stack(stack_path, data.astype(np.float32), weights=weights)

    hdr = _stack_hdr((2, 3))
    x0, y0 = hdr.imagecoord_to_xy(0, 0)
    x1, y1 = hdr.imagecoord_to_xy(1, 2)

    with Stack(str(stack_path)) as stack:
        serial = solver.inversion_points(
            stack, userfile, [x0, x1], [y0, y1],
            solver_type="lsqr", nt_out=5, n_proc=1, n_min=2,
        )

    with Stack(str(stack_path)) as stack:
        parallel = solver.inversion_points(
            stack, userfile, [x0, x1], [y0, y1],
            solver_type="lsqr", nt_out=5, n_proc=2, n_min=2,
        )

    for key in ("tdec", "full", "secular", "seasonal", "transient", "sigma"):
        np.testing.assert_allclose(parallel[key], serial[key], equal_nan=True)


def test_solver_butterworth_joblib_matches_serial(tmp_path):
    solver = _solver_module()
    stack_path = tmp_path / "butter_stack.h5"
    serial_long = tmp_path / "serial_long.h5"
    serial_short = tmp_path / "serial_short.h5"
    parallel_long = tmp_path / "parallel_long.h5"
    parallel_short = tmp_path / "parallel_short.h5"

    time = np.linspace(0.0, 4.0 * np.pi, 24, dtype=np.float32)[:, None, None]
    rows, cols = np.indices((2, 3), dtype=np.float32)
    data = np.sin(time) + 0.1 * np.cos(4.0 * time) + 0.25 * rows + 0.1 * cols
    _write_stack(stack_path, data.astype(np.float32))

    b, a = solver.butterworth_coeffs(frequency=0.2, dt=1.0, order=1)

    with Stack(str(stack_path)) as stack:
        solver.butterworth(stack, a, b, str(serial_long), str(serial_short), n_proc=1)

    with Stack(str(stack_path)) as stack:
        solver.butterworth(stack, a, b, str(parallel_long), str(parallel_short), n_proc=2)

    np.testing.assert_allclose(
        _read_stack_variable(parallel_long),
        _read_stack_variable(serial_long),
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        _read_stack_variable(parallel_short),
        _read_stack_variable(serial_short),
        rtol=1.0e-6,
    )


@pytest.mark.parametrize(
    "source_key, include_weights",
    [
        ("data", True),
        ("data", False),
        ("igram", False),
    ],
)
def test_ice_crop_stack_uses_xarray_dims_and_optional_weights(
    tmp_path, source_key, include_weights
):
    src = tmp_path / "crop_input.h5"
    out = tmp_path / "crop_output.h5"
    data = np.arange(36, dtype=np.float32).reshape(3, 3, 4)
    weights = data + 50.0 if include_weights else None
    _write_stack(src, data, key=source_key, weights=weights)
    script = Path(__file__).resolve().parents[1] / "bin" / "ice_crop_stack.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            str(src),
            str(out),
            "-srcWin",
            "1",
            "0",
            "2",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    with Stack(str(out)) as stack:
        assert stack["data"].dims == ("time", "y", "x")
        assert np.array_equal(stack["data"].values, data[:, 0:2, 1:3])
        assert ("weights" in stack.ds) == include_weights
        if include_weights:
            assert np.array_equal(stack["weights"].values, weights[:, 0:2, 1:3])


def test_ice_resample_detects_netcdf_stack_inputs():
    module = _load_bin_script("ice_resample.py")

    assert module._is_stack_file("velocity_stack.h5")
    assert module._is_stack_file("velocity_stack.nc")
    assert not module._is_stack_file("velocity_stack.tif")


def test_ice_explore_stack_parses_extra_dimension_indexers():
    module = _load_bin_script("ice_explore_stack.py")

    assert module.parse_indexer("band=0") == ("band", 0)
    assert module.parse_indexer("component=12") == ("component", 12)
    assert module.indexer_dict([("band", 1), ("component", 2)]) == {
        "band": 1,
        "component": 2,
    }

    with pytest.raises(argparse.ArgumentTypeError):
        module.parse_indexer("band")
    with pytest.raises(argparse.ArgumentTypeError):
        module.parse_indexer("=0")
    with pytest.raises(argparse.ArgumentTypeError):
        module.parse_indexer("band=first")


def test_ice_view_stack_mean_parses_extra_dimension_indexers():
    module = _load_bin_script("ice_view_stack_mean.py")

    assert module.parse_indexer("band=0") == ("band", 0)
    assert module.parse_indexer("component=12") == ("component", 12)
    assert module.indexer_dict([("band", 1), ("component", 2)]) == {
        "band": 1,
        "component": 2,
    }

    with pytest.raises(argparse.ArgumentTypeError):
        module.parse_indexer("band")
    with pytest.raises(argparse.ArgumentTypeError):
        module.parse_indexer("=0")
    with pytest.raises(argparse.ArgumentTypeError):
        module.parse_indexer("band=first")

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

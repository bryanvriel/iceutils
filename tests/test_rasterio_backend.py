import os
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


def test_read_write_round_trip_preserves_data_and_metadata(tmp_path):
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    src = tmp_path / "src.tif"
    transform = _write_geotiff(src, data, nodata=-9999.0)

    raster = Raster(str(src))

    assert np.array_equal(raster.data, data)
    assert raster.nodataval == -9999.0
    assert raster.hdr.shape == data.shape
    assert raster.hdr.epsg == 3413
    assert tuple(raster.hdr.transform) == tuple(transform)
    assert raster.hdr.dtype == np.dtype("float32")

    out = tmp_path / "out.tif"
    raster.write_raster(str(out), driver="GTiff", nodataval=-9999.0)

    with rasterio.open(out) as ds:
        assert np.array_equal(ds.read(1), data)
        assert ds.nodata == -9999.0
        assert ds.crs.to_epsg() == 3413
        assert tuple(ds.transform) == tuple(transform)


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


def test_stack_reads_legacy_nhw_hdf5_as_canonical_xarray(tmp_path):
    path = tmp_path / "legacy_nhw.h5"
    data = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    with h5py.File(path, "w") as fid:
        fid["x"] = np.array([10.0, 12.0, 14.0])
        fid["y"] = np.array([20.0, 17.0])
        fid["tdec"] = np.array([2020.0, 2021.0])
        fid["data"] = data
        fid["weights"] = data + 1
        fid.attrs["EPSG"] = 3413
        fid.attrs["format"] = "NHW"

    with Stack(str(path)) as stack:
        assert stack.fmt == "NHW"
        assert stack.original_fmt == "NHW"
        assert stack["data"].dims == ("time", "y", "x")
        assert np.array_equal(stack["data"].values, data)
        assert np.array_equal(stack.slice(0), data[0])
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
        assert stack["data"].dims == ("time", "y", "x")
        assert np.array_equal(stack["data"].values, canonical)
        assert np.array_equal(stack.get_chunk(slice(None), slice(None)), canonical)

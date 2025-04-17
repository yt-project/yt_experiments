import numpy as np
import pytest
import unyt
from numpy.testing import assert_equal
from yt.testing import fake_amr_ds, requires_module

from yt_experiments.tiled_grid import (
    YTArbitraryGridOctPyramid,
    YTArbitraryGridPyramid,
    YTTiledArbitraryGrid,
)


def test_arbitrary_grid():
    ds = fake_amr_ds()
    tag = YTTiledArbitraryGrid(
        ds.domain_left_edge,
        ds.domain_right_edge,
        (20, 20, 20),
        5,
        ds=ds,
    )
    assert tag._ngrids == (20 // 5) ** 3
    _ = tag.__repr__()

    fld = ("stream", "Density")
    den = tag.to_array(fld)
    assert isinstance(den, unyt.unyt_array)
    den2 = np.empty(tag.dims)
    _ = tag.to_array(fld, output_array=den2)
    assert not isinstance(den2, unyt.unyt_array)
    assert_equal(den, den2)

    den3 = tag[fld]
    assert_equal(den3, den)

    assert np.min(den) > 0.0
    assert np.all(np.isfinite(den))

    den = tag.to_array(fld, dtype=np.float32)
    assert den.dtype == np.float32


def test_arbitray_grid_pyramid():
    ds = fake_amr_ds()
    levels = [(16, 16, 16), (10, 10, 10)]
    pyr = YTArbitraryGridPyramid(
        ds.domain_left_edge, ds.domain_right_edge, levels, 2, ds=ds
    )

    assert pyr.n_levels == 2
    fld = ("stream", "Density")
    for ilev in range(pyr.n_levels):
        vals = pyr[ilev][fld]
        assert vals.shape == levels[ilev]
        assert np.all(np.isfinite(vals))

    level_arrays = pyr.to_arrays(fld)
    for ilev in range(pyr.n_levels):
        assert level_arrays[ilev].shape == levels[ilev]


def test_arbitrary_grid_oct():
    ds = fake_amr_ds()
    expected_levels = [(16, 16, 16), (8, 8, 8)]
    oct = YTArbitraryGridOctPyramid(
        ds.domain_left_edge, ds.domain_right_edge, (16, 16, 16), 2, 2, ds=ds
    )

    assert oct.n_levels == 2
    fld = ("stream", "Density")
    for ilev in range(oct.n_levels):
        vals = oct[ilev][fld]
        assert vals.shape == expected_levels[ilev]
        assert np.all(np.isfinite(vals))

    level_arrays = oct.to_arrays(fld)
    for ilev in range(oct.n_levels):
        assert level_arrays[ilev].shape == expected_levels[ilev]


def test_missing_ds():
    with pytest.raises(ValueError, match="Please provide a dataset"):
        _ = YTTiledArbitraryGrid(
            unyt.unyt_array([0, 0, 0], "m"),
            unyt.unyt_array([1, 1, 1], "m"),
            (20, 20, 20),
            5,
        )


@requires_module("xarray")
def test_arbitrary_grid_to_xarray():
    import xarray as xr

    ds = fake_amr_ds()
    tag = YTTiledArbitraryGrid(
        ds.domain_left_edge,
        ds.domain_right_edge,
        (20, 20, 20),
        5,
        ds=ds,
    )

    vals = tag.to_xarray(("stream", "Density"))
    assert isinstance(vals, xr.DataArray)
    assert hasattr(vals, "coords")

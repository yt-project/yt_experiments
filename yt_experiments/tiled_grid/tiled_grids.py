from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import unyt
from numpy import typing as npt
from yt._typing import FieldKey
from yt.data_objects.construction_data_containers import YTArbitraryGrid
from yt.data_objects.static_output import Dataset

_GridInfo = tuple[
    npt.NDArray, npt.NDArray, unyt.unyt_array, unyt.unyt_array, Any, npt.NDArray
]


def _validate_edge(edge: npt.ArrayLike, ds: Dataset):
    if not isinstance(edge, unyt.unyt_array):
        return ds.arr(edge, "code_length")
    return edge


def _validate_nd_int(nd: int, x: int | npt.ArrayLike) -> npt.NDArray:
    if isinstance(x, int):
        x = (x,) * nd
    x = np.array(x).astype(int)
    if len(x) != 3:
        raise ValueError("Variable must have a length of 3")
    return x


class YTTiledArbitraryGrid:

    _ndim = 3

    def __init__(
        self,
        left_edge: npt.ArrayLike,
        right_edge: npt.ArrayLike,
        dims: int | npt.ArrayLike,
        chunks: int | npt.ArrayLike,
        *,
        ds: Dataset | None = None,
        field_parameters: Any | None = None,
        data_source: Any | None = None,
    ):
        """
        An assembly of adjacent YTArbitaryGrid objects, representing a chunked regular
        grid.

        Parameters
        ----------
        left_edge: ArrayLike
            The left edge of the bounding box
        right_edge: ArrayLike
            The right edge of the bounding box
        dims: int | ArrayLike
            The size of the whole grid
        chunks: int | ArrayLike
            chunk size (or sizes in each dimension), not number of chunks. The number
            of chunks will be given by dims / chunks.
        ds: Dataset
            the yt dataset to operate on
        field_parameters:
            field parameters passed to YTArbitraryGrid when constructing individual grid
            objects
        data_source:
            a data source to operate on.


        Notes
        -----

        With yt < 4.4.0, YTTiledArbitraryGrid may contain artifacts at chunk boundaries
        along the z-axis. If observed, try upgrading yt (you may need to install from
        source).

        """

        if ds is None:
            raise ValueError("Please provide a dataset via the ds keyword argument")

        self.ds = ds
        self.left_edge = _validate_edge(left_edge, ds)
        self.right_edge = _validate_edge(right_edge, ds)
        self.data_source = data_source
        self.field_parameters = field_parameters
        self.dims = _validate_nd_int(self._ndim, dims)
        self.chunks = _validate_nd_int(self._ndim, chunks)

        nchunks = self.dims / self.chunks
        if np.any(np.mod(nchunks, nchunks.astype(int)) != 0):
            msg = (
                "The dimensions and chunks provide result in partially filled "
                f"chunks, which is not supported at this time: {nchunks}"
            )
            raise NotImplementedError(msg)
        self.nchunks = nchunks.astype(int)

        self.dds = (self.right_edge - self.left_edge) / self.dims
        self._grids: list[YTArbitraryGrid] = []
        self._grid_slc: list[tuple[slice, slice, slice]] = []
        self._ngrids = np.prod(self.nchunks)
        self._left_cell_center = self.left_edge + self.dds / 2.0
        self._right_cell_center = self.right_edge - self.dds / 2.0

    def __repr__(self) -> str:
        nm = self.__class__.__name__
        shape = tuple(self.dims)
        n_chunks = tuple(self.nchunks)
        n_tot = self._ngrids
        msg = (
            f"{nm} with total shape of {shape} divided into {n_tot} grids: "
            f"{n_chunks} grids in each dimension."
        )
        return msg

    def _get_grid_by_ijk(self, ijk_grid: npt.NDArray[int]) -> _GridInfo:
        chunksizes = self.chunks

        le_index = []
        re_index = []
        le_val: unyt.unyt_array = self.ds.domain_left_edge.copy()
        re_val: unyt.unyt_array = self.ds.domain_right_edge.copy()

        for idim in range(self._ndim):
            chunk_i = ijk_grid[idim]
            lei = chunk_i * chunksizes[idim]
            rei = lei + chunksizes[idim]
            lei_val = self.left_edge[idim] + self.dds[idim] * lei
            rei_val = self.left_edge[idim] + self.dds[idim] * (chunksizes[idim] + lei)
            le_index.append(lei)
            re_index.append(rei)
            le_val[idim] = lei_val
            re_val[idim] = rei_val

        slc = np.s_[
            le_index[0] : re_index[0],
            le_index[1] : re_index[1],
            le_index[2] : re_index[2],
        ]

        le_index_ = np.array(le_index, dtype=int)
        re_index_ = np.array(re_index, dtype=int)
        shape = chunksizes

        return le_index_, re_index_, le_val, re_val, slc, shape

    def _get_grid(self, igrid: int) -> _GridInfo:
        # get grid extent of a **single** grid
        ijk_grid = np.unravel_index(igrid, self.nchunks)
        return self._get_grid_by_ijk(ijk_grid)

    def _coord_array(self, idim: int) -> npt.NDArray:
        LE = self._left_cell_center[idim]
        RE = self._right_cell_center[idim]
        N = self.dims[idim]
        return np.mgrid[LE : RE : N * 1j]

    def to_xarray(
        self, field: tuple[str, str], *, output_array: npt.ArrayLike | None = None
    ) -> Any:

        import xarray as xr

        vals = self.to_array(field, output_array=output_array)

        dims = self.ds.coordinates.axis_order
        dim_list = list(dims)
        coords = {dim: self._coord_array(idim) for idim, dim in enumerate(dims)}

        xrname = field[0] + "_" + field[1]

        xr_ds = xr.DataArray(
            data=vals,
            dims=dim_list,
            coords=coords,
            attrs={"ngrids": self._ngrids, "fieldname": field},
            name=xrname,
        )
        return xr_ds

    def single_grid_values(
        self,
        igrid: int,
        field: tuple[str, str],
        *,
        ops: list[Callable[[npt.NDArray], npt.NDArray]] | None = None,
    ) -> tuple[npt.NDArray, Any]:
        """
        Get the values for a field for a single grid chunk as in-memory array.

        Parameters
        ----------
        igrid
        field
        ops

        Returns
        -------
        tuple
            (vals, slcs) where vals is a np array for the specified chunk
            and slcs are the index-slices for each dimension for the global
            array.


        """
        if ops is None:
            ops = []
        _, _, le, re, slc, shp = self._get_grid(igrid)
        vals = _get_filled_grid(le, re, shp, field, self.ds, self.field_parameters)
        for op in ops:
            vals = op(vals)
        return vals, slc

    def to_array(
        self,
        field: FieldKey,
        *,
        output_array: npt.ArrayLike | None = None,
        ops: list[Callable[[npt.NDArray], npt.NDArray]] | None = None,
        dtype: str | np.dtype | None = None,
    ):
        """
        Sample the field for each grid in the tiled grid set.

        Parameters
        ----------
        field: tuple(str, str)
            the field to sample
        output_array: ArrayLike | None
            Optional array to fill. if not provided, defaults to an empty
            np array. Can provide any array type (np, zarr) that supports
            np-like indexing.
        ops: list[Callable[[npt.NDArray], npt.NDArray]]] | None
            an optional list of callback functions to apply to the
            sampled field. Must accept a single parameter, the values
            of the sampled field for each grid, and return the modified
            values of the grid. Operations are by-grid. For example:

                def my_func(values):
                    modified_values = np.abs(values)
                    return modified_values

                ops = [my_func, ]

        dtype: str | np.dtype | None
            Optional, a dtype to cast to (default np.float64). Note that
            if using this option, arrays in output_array should be initialized
            to this dtype.


        Returns
        -------
        array
            a filled array

        """

        if dtype is None:
            dtype = np.float64

        if output_array is None:
            output_units = self.ds._get_field_info(field).units
            output_array = self.ds.arr(np.empty(self.dims, dtype=dtype), output_units)

        for igrid in range(self._ngrids):
            vals, slc = self.single_grid_values(igrid, field, ops=ops)
            output_array[slc] = vals.astype(dtype)
        return output_array

    def __getitem__(self, item: FieldKey):
        return self.to_array(item)


class YTArbitraryGridPyramid:
    _ndim = 3

    def __init__(
        self,
        left_edge: npt.ArrayLike,
        right_edge: npt.ArrayLike,
        level_dims: Sequence[int | tuple[int, int, int] | npt.ArrayLike],
        level_chunks: int | npt.ArrayLike,
        *,
        ds: Dataset = None,
        field_parameters=None,
        data_source: Any | None = None,
    ):
        """

        An image pyramid built from YTTiledArbitraryGrid objects.

        Following conventions of image pyramids, level 0 of a YTArbitraryGridPyramid is the
        pyramid base (the level with the highest resolution).

        Parameters
        ----------
        left_edge: ArrayLike
            The left edge of the bounding box
        right_edge: ArrayLike
            The right edge of the bounding box
        level_dims: Sequence of dimensions
            The global dimensions of each level. Number of levels given by len(level_dims)
        level_chunks: Sequence of chunk sizes
            The chunk size of each level. Number of chunks in each level is given by
            level_dims / level_chunks.
        ds: Dataset
            the yt dataset to operate on
        field_parameters:
            field parameters passed to YTArbitraryGrid when constructing individual grid
            objects
        data_source:
            a data source to operate on.

        """

        levels = []

        n_levels = len(level_dims)
        self.n_levels = n_levels

        level_chunks = _validate_nd_int(self._ndim, level_chunks)

        if isinstance(level_chunks, np.ndarray):
            level_chunks = [level_chunks for _ in range(n_levels)]

        if len(level_chunks) != n_levels:
            msg = (
                "length of level_chunks must match the total number of levels."
                f" Found {len(level_chunks)}, expected {n_levels}"
            )
            raise RuntimeError(msg)

        for ilev in range(n_levels):
            if isinstance(level_chunks[ilev], int):
                level_chunks[ilev] = (level_chunks[ilev],) * self._ndim

        # should be ready by this point
        self._validate_levels(level_dims)

        for ilev in range(n_levels):
            chunksizes = np.asarray(level_chunks[ilev], dtype=int)
            current_dims = np.asarray(level_dims[ilev], dtype=int)
            tag = YTTiledArbitraryGrid(
                left_edge,
                right_edge,
                current_dims,
                chunksizes,
                ds=ds,
                field_parameters=field_parameters,
                data_source=data_source,
            )
            levels.append(tag)

        self.levels: list[YTTiledArbitraryGrid] = levels

    def _validate_levels(
        self, levels: Sequence[int | tuple[int, int, int] | npt.ArrayLike]
    ):

        for ilev in range(1, self.n_levels):
            res = np.prod(levels[ilev])
            res_higher = np.prod(levels[ilev - 1])
            if res > res_higher:
                msg = (
                    "Image pyramid initialization failed: expected highest resolution "
                    "at level 0, with decreasing resolution with increasing level but found "
                    f" that level {ilev} resolution is higher than {ilev-1}."
                )
                raise ValueError(msg)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__} with {self.n_levels} levels and base resolution "
            f"{self.base_resolution}"
        )

    def base_resolution(self) -> tuple[int, int, int]:
        return tuple(self[0].dims)

    def to_arrays(
        self,
        field: FieldKey,
        *,
        output_arrays: list[npt.ArrayLike | None] | None = None,
        ops: list[Callable[[npt.NDArray], npt.NDArray]] | None = None,
        dtype: str | np.dtype | None = None,
    ) -> list[npt.ArrayLike]:
        """Generate arrays for each level of the image pyramid

        Parameters
        ----------
        field: tuple(str, str)
            the field to process (required).
        output_arrays: list(ArrayLike) | None
            optional list of array-like objects to store output in.
        ops: list[Callable[[npt.NDArray], npt.NDArray]]] | None
            an optional list of callback functions to apply to the
            sampled field. Must accept a single parameter, the values
            of the sampled field for each grid, and return the modified
            values of the grid. Operations are by-grid. For example:

                def my_func(values):
                    modified_values = np.abs(values)
                    return modified_values

                ops = [my_func, ]
        dtype: str | np.dtype | None
            Optional, a dtype to cast to (default np.float64). Note that
            if using this option, arrays in output_array should be initialized
            to this dtype.


        Returns
        -------
        list[npt.ArrayLike]
            a list of filled arrays
        """
        if output_arrays is None:
            output_arrays = [None for _ in range(len(self.levels))]

        for ilev, yttag in enumerate(self.levels):
            output_arrays[ilev] = yttag.to_array(
                field,
                output_array=output_arrays[ilev],
                ops=ops,
                dtype=dtype,
            )

        return output_arrays

    def __getitem__(self, item: int) -> YTTiledArbitraryGrid:
        return self.levels[item]


class YTArbitraryGridOctPyramid(YTArbitraryGridPyramid):
    def __init__(
        self,
        left_edge: npt.ArrayLike,
        right_edge: npt.ArrayLike,
        dims: int | npt.ArrayLike,
        chunks: int | npt.ArrayLike,
        n_levels: int,
        *,
        factor: int | tuple[int, int, int] = 2,
        ds: Dataset = None,
        field_parameters=None,
        data_source: Any | None = None,
    ):
        """
        An octree image pyramid.

        Parameters
        ----------
        left_edge: ArrayLike
            The left edge of the bounding box
        right_edge: ArrayLike
            The right edge of the bounding box
        dims: int | ArrayLike
            The dimensions of the image pyramid base level
        chunks: int | ArrayLike
            The chunk size of the base level, needs to be divisible by factor
        n_levels: int
            Number of levels for the pyramid
        factor: int | tuple[int, int, int]
            The refinement factor between level, default 2 in each dimension.
        ds: Dataset
            the yt dataset to operate on
        field_parameters:
            field parameters passed to YTArbitraryGrid when constructing individual grid
            objects
        data_source:
            a data source to operate on.

        """

        dims_valid = _validate_nd_int(self._ndim, dims)

        if isinstance(chunks, int):
            chunks = (chunks,) * self._ndim

        factor_ = self._validate_factor(factor)

        level_dims = []
        for lev in range(n_levels):
            current_dims = dims_valid / factor_**lev
            level_dims.append(current_dims)

        super().__init__(
            left_edge,
            right_edge,
            level_dims,
            chunks,
            ds=ds,
            field_parameters=field_parameters,
            data_source=data_source,
        )

    def _validate_factor(
        self, input_factor: int | tuple[int, int, int]
    ) -> npt.NDArray[int]:
        if isinstance(input_factor, int):
            temp_factor = (input_factor,) * self._ndim
            return np.asarray(temp_factor, dtype=int)
        return np.asarray(input_factor, dtype=int)


def _get_filled_grid(
    le: npt.NDArray,
    re: npt.NDArray,
    shp: npt.NDArray,
    field: tuple[str, str],
    ds: Dataset,
    field_parameters: Any,
) -> npt.NDArray:
    grid = YTArbitraryGrid(le, re, shp, ds=ds, field_parameters=field_parameters)
    vals = grid[field]
    return vals

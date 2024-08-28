from typing import Any, Optional

import numpy as np
import xarray as xr
from yt.data_objects.construction_data_containers import YTArbitraryGrid
from yt.data_objects.static_output import Dataset


class YTTiledArbitraryGrid:

    _ndim = 3

    def __init__(
        self,
        left_edge,
        right_edge,
        dims: tuple[int, int, int],
        chunks: int | tuple[int, int, int],
        *,
        ds: Dataset = None,
        field_parameters=None,
        parallel_method: Optional[str] = None,
        data_source: Optional[Any] = None,
    ):
        """

        Parameters
        ----------
        left_edge
        right_edge
        dims
        chunks
            chunk size (or sizes in each dimension), not number of chunks
        ds
        field_parameters
        parallel_method
        data_source

        Notes
        -----

        With yt < 4.4.0, YTTiledArbitraryGrid may contain artifacts at chunk boundaries
        along the z-axis. If observed, try upgrading yt (you may need to install from
        source).

        """

        self.left_edge = left_edge
        self.right_edge = right_edge
        self.ds = ds
        self.data_source = data_source
        self.field_parameters = field_parameters
        self.parallel_method = parallel_method
        self.dims = dims
        if isinstance(chunks, int):
            chunks = (chunks,) * self._ndim
        self.chunks = chunks

        nchunks = self._dims / self._chunks
        if np.any(np.mod(nchunks, nchunks.astype(int)) != 0):
            msg = (
                "The dimensions and chunks provide result in partially filled "
                f"chunks, which is not supported at this time: {nchunks}"
            )
            raise NotImplementedError(msg)
        self.nchunks = nchunks.astype(int)

        self.dds = (self.right_edge - self.left_edge) / self._dims

        self._grids: list[YTArbitraryGrid] = []
        self._grid_slc: list[tuple[slice, slice, slice]] = []
        self._ngrids = np.prod(self.nchunks)
        self._left_cell_center = self.left_edge + self.dds / 2.0
        self._right_cell_center = self.right_edge - self.dds / 2.0

    @property
    def _chunks(self):
        return np.array(self.chunks, dtype=int)

    @property
    def _dims(self):
        return np.array(self.dims, dtype=int)

    def _get_grid_by_ijk(self, ijk_grid):
        chunksizes = self._chunks

        le_index = []
        re_index = []
        le_val = self.ds.domain_left_edge.copy()
        re_val = self.ds.domain_right_edge.copy()

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

        le_index = np.array(le_index, dtype=int)
        re_index = np.array(re_index, dtype=int)
        shape = chunksizes

        return le_index, re_index, le_val, re_val, slc, shape

    def _get_grid(self, igrid: int):
        # get grid extent of a **single** grid
        ijk_grid = np.unravel_index(igrid, self.nchunks)
        return self._get_grid_by_ijk(ijk_grid)

    def to_dask(self, field, chunks=None):
        from dask import array as da, delayed

        if chunks is None:
            chunks = self.chunks

        full_domain = da.empty(self.dims, chunks=chunks, dtype="float64")
        for igrid in range(self._ngrids):
            _, _, le, re, slc, shp = self._get_grid(igrid)
            vals = delayed(_get_filled_grid)(
                le, re, shp, field, self.ds, self.field_parameters
            )
            vals = da.from_delayed(vals, shp, dtype="float64")
            slc = self._grid_slc[igrid]
            full_domain[slc] = vals
        return full_domain

    def _coord_array(self, idim):
        LE = self._left_cell_center[idim]
        RE = self._right_cell_center[idim]
        N = self.dims[idim]
        return np.mgrid[LE : RE : N * 1j]

    def to_xarray(self, field, chunks=None, backend: str = "dask") -> xr.DataArray:

        if backend == "dask":
            da = self.to_dask(field, chunks=chunks)
        elif backend == "numpy":
            da = self.to_array(field)
        else:
            raise NotImplementedError()

        dims = self.ds.coordinates.axis_order
        dim_list = list(dims)
        coords = {dim: self._coord_array(idim) for idim, dim in enumerate(dims)}

        xrname = field[0] + "_" + field[1]

        xr_ds = xr.DataArray(
            data=da,
            dims=dim_list,
            coords=coords,
            attrs={"ngrids": self._ngrids, "fieldname": field},
            name=xrname,
        )
        return xr_ds

    def single_grid_values(self, igrid, field, *, ops=None):
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
        field,
        *,
        full_domain=None,
        ops=None,
        dtype=None,
    ):
        """
        Sample the field for each grid in the tiled grid set.

        Parameters
        ----------
        field
            the field to sample
        full_domain
            the array to fill. if not provided, defaults to an empty
            np array. Can provide a numpy or zarr array.
        ops
            an optional list of callback functions to apply to the
            sampled field. Must accept a single parameter, the values
            of the sampled field for each grid, and return the modified
            values of the grid. Operations are by-grid. For example:

                def my_func(values):
                    modified_values = np.abs(values)
                    return modified_values


        Returns
        -------
        array
            a filled array

        """
        if full_domain is None:
            full_domain = np.empty(self.dims, dtype="float64")

        if dtype is None:
            dtype = np.float64

        for igrid in range(self._ngrids):
            vals, slc = self.single_grid_values(igrid, field, ops=ops)
            full_domain[slc] = vals.astype(dtype)
        return full_domain

    def to_zarr(
        self,
        field,
        zarr_store,
        *,
        zarr_name: str | None = None,
        ops=None,
        dtype=None,
        **kwargs,
    ):
        """
        write to a zarr Store or Group

        Parameters
        ----------
        field
        zarr_store
        zarr_name
        ops
        kwargs
            passed to the empty zarr array creation

        Returns
        -------

        """

        import zarr

        _allowed_types = (zarr.storage.Store, zarr.hierarchy.Group)
        if not isinstance(zarr_store, _allowed_types):
            raise TypeError(
                "zarr_store must be a zarr `Store` or `Group` but has "
                f"type of {type(zarr_store)}."
            )

        if dtype is None:
            dtype = np.float64

        if ops is None:
            ops = []

        if zarr_name is None:
            zarr_name = "_".join(field)

        full_domain = zarr_store.create(
            zarr_name, shape=self.dims, chunks=self.chunks, dtype=dtype, **kwargs
        )
        full_domain = self.to_array(
            field, full_domain=full_domain, ops=ops, dtype=dtype
        )
        return full_domain


class YTPyramid:
    _ndim = 3

    def __init__(
        self,
        left_edge,
        right_edge,
        level_dims: [tuple[int, int, int]],
        level_chunks,
        ds: Dataset = None,
        field_parameters=None,
        data_source: Optional[Any] = None,
    ):
        """

        Parameters
        ----------
        left_edge
        right_edge
        level_dims
        level_chunks
        ds
        field_parameters
        data_source
        """

        levels = []

        n_levels = len(level_dims)

        if isinstance(level_chunks, int):
            level_chunks = (level_chunks,) * self._ndim

        if isinstance(level_chunks, tuple):
            level_chunks = [level_chunks for _ in range(n_levels)]

        if len(level_chunks) != n_levels:
            msg = (
                "length of level_chunks must match the total number of levels."
                f" Found {len(level_chunks)}, expected {n_levels}"
            )
            raise ValueError(msg)

        for ilev in range(n_levels):
            if isinstance(level_chunks[ilev], int):
                level_chunks[ilev] = (level_chunks[ilev],) * self._ndim

        # should be ready by this point
        self._validate_levels(levels)

        for ilev in range(n_levels):
            chunksizes = np.array(level_chunks[ilev], dtype=int)
            current_dims = np.asarray(level_dims[ilev], dtype=int)
            n_chunks_lev = int(np.prod(current_dims / chunksizes))
            print(
                f"Decomposing {current_dims} into {n_chunks_lev} chunks for level {ilev}"
            )
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

        self.levels: [YTTiledArbitraryGrid] = levels

    def _validate_levels(self, levels):
        for ilev in range(2, len(levels)):
            res = np.prod(levels[ilev])
            res_higher = np.prod(levels[ilev - 1])
            if res > res_higher:
                msg = (
                    "Image pyramid initialization failed: expected highest resolution "
                    "at level 0, with decreasing resolution with increasing level but found "
                    f" that level {ilev} resolution is higher than {ilev-1}."
                )
                raise ValueError(msg)

    def to_zarr(
        self,
        field,
        zarr_store,
        zarr_name: str | None = None,
        ops=None,
        dtype=None,
        **kwargs,
    ):
        import zarr

        _allowed_types = (zarr.storage.Store, zarr.hierarchy.Group)
        if not isinstance(zarr_store, _allowed_types):
            raise TypeError(
                "zarr_store must be a zarr `Store` or `Group` but has "
                f"type of {type(zarr_store)}."
            )

        if zarr_name is None:
            zarr_name = "_".join(field)

        if zarr_name not in zarr_store:
            zarr_store.create_group(zarr_name)

        field_store = zarr_store[zarr_name]

        for lev, tag in enumerate(self.levels):
            print(f"writing level {lev}")
            tag.to_zarr(
                field,
                field_store,
                zarr_name=str(lev),
                ops=ops,
                dtype=dtype,
                **kwargs,
            )


class YTOctPyramid(YTPyramid):
    def __init__(
        self,
        left_edge,
        right_edge,
        dims: tuple[int, int, int],
        chunks: int | tuple[int, int, int],
        n_levels: int,
        factor: int = 2,
        ds: Dataset = None,
        field_parameters=None,
        data_source: Optional[Any] = None,
    ):

        dims_ = np.array(dims, dtype=int)
        if isinstance(chunks, int):
            chunks = (chunks,) * self._ndim

        level_dims = []
        for lev in range(n_levels):
            current_dims = dims_ / factor**lev
            level_dims.append(current_dims)

        super().__init__(
            left_edge,
            right_edge,
            dims,
            chunks,
            ds=ds,
            field_parameters=field_parameters,
            data_source=data_source,
        )


def _get_filled_grid(le, re, shp, field, ds, field_parameters):
    grid = YTArbitraryGrid(le, re, shp, ds=ds, field_parameters=field_parameters)
    vals = grid[field]
    return vals

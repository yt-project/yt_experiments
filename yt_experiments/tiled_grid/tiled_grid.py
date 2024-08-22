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

    def _get_grid(self, igrid: int):
        # get grid extent of a **single** grid

        chunksizes = self._chunks

        # get the left/right index and value of this grid
        ijk_grid = np.unravel_index(igrid, self.nchunks)
        le_index = []
        re_index = []
        le_val = []
        re_val = []
        for idim in range(self._ndim):
            chunk_i = ijk_grid[idim]
            lei = chunk_i * chunksizes[idim]
            rei = lei + chunksizes[idim]
            lei_val = self.left_edge[idim] + self.dds[idim] * lei
            rei_val = lei_val + self.dds[idim] * chunksizes[idim]
            le_index.append(lei)
            re_index.append(rei)
            le_val.append(lei_val)
            re_val.append(rei_val)

        slc = np.s_[
            le_index[0] : re_index[0],
            le_index[1] : re_index[1],
            le_index[2] : re_index[2],
        ]

        le_index = np.array(le_index, dtype=int)
        re_index = np.array(re_index, dtype=int)
        le_val = np.array(le_val)
        re_val = np.array(re_val)
        shape = chunksizes

        return le_index, re_index, le_val, re_val, slc, shape

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
        if ops is None:
            ops = []

        for igrid in range(self._ngrids):
            vals, slc = self.single_grid_values(igrid, field, ops=ops)
            full_domain[slc] = vals
        return full_domain

    def to_zarr(
        self,
        field,
        zarr_store,
        *,
        zarr_name: str | None = None,
        ops=None,
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

        dtype = np.float64
        if ops is None:
            ops = []

        if zarr_name is None:
            zarr_name = "_".join(field)

        full_domain = zarr_store.empty(
            zarr_name, shape=self.dims, chunks=self.chunks, dtype=dtype, **kwargs
        )
        full_domain = self.to_array(field, full_domain=full_domain, ops=ops)
        return full_domain


class YTPyramid:
    _ndim = 3

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

        levels = []
        dims_ = np.array(dims, dtype=int)
        if isinstance(chunks, int):
            chunks = (chunks,) * self._ndim
        chunksizes = np.array(chunks, dtype=int)

        for lev in range(n_levels):
            current_dims = dims_ / factor**lev
            n_chunks_lev = int(np.prod(current_dims / chunksizes))
            print(
                f"Decomposing {current_dims} into {n_chunks_lev} chunks for level {lev}"
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

    def to_zarr(
        self,
        field,
        zarr_store,
        zarr_name: str | None = None,
        ops=None,
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
            )


def _get_filled_grid(le, re, shp, field, ds, field_parameters):
    grid = YTArbitraryGrid(le, re, shp, ds=ds, field_parameters=field_parameters)
    vals = grid[field]
    return vals

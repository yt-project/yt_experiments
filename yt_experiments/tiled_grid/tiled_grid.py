from typing import Any, Optional

import numpy as np
import xarray as xr
from yt.data_objects.construction_data_containers import YTArbitraryGrid
from yt.data_objects.static_output import Dataset
from yt.utilities.decompose import get_psize, split_array


class YTTiledArbitraryGrid:
    def __init__(
        self,
        left_edge,
        right_edge,
        dims: tuple[int, int, int],
        nchunks: int,
        ds: Dataset = None,
        field_parameters=None,
        parallel_method: Optional[str] = None,
        data_source: Optional[Any] = None,
        cache: Optional[bool] = False,
    ):
        self.left_edge = left_edge
        self.right_edge = right_edge
        self.ds = ds
        self.data_source = data_source
        self.field_parameters = field_parameters
        self.parallel_method = parallel_method
        self.dims = dims
        self.nchunks = nchunks
        self._psize = get_psize(np.array(dims), nchunks)
        self._grids: list[YTArbitraryGrid] = []
        self._grid_slc: list[tuple[slice, slice, slice]] = []
        self.chunks = None
        self._get_grids()
        self._ngrids = len(self._grids)
        self.dds = (self.right_edge - self.left_edge) / self.dims

        self.cache = cache
        self._left_cell_center = self.left_edge + self.dds / 2.0
        self._right_cell_center = self.right_edge - self.dds / 2.0

    def _get_grids(self):
        # initialize the arbitrary grid args. not optimal cause we need to
        # do two passes: first to get the grids then to parse in the expected
        # form for dask chunks.
        grid_list = split_array(self.left_edge, self.right_edge, self.dims, self._psize)
        n_grids = len(grid_list[0])

        # also record the chunks in the way dask expects them
        le_chunks = [[] for _ in range(3)]
        re_chunks = [[] for _ in range(3)]

        for igrid in range(n_grids):
            le = grid_list[0][igrid]
            re = grid_list[1][igrid]
            shp = grid_list[2][igrid]
            slc = grid_list[3][igrid]
            self._grids.append((le, re, shp))
            self._grid_slc.append(slc)
            for idim in range(3):
                le_chunks[idim].append(slc[idim].start)
                re_chunks[idim].append(slc[idim].stop)

        le_chunks = np.array(le_chunks, dtype=int)
        re_chunks = np.array(re_chunks, dtype=int)

        chunks = []
        for idim in range(3):
            le_idim = np.unique(le_chunks[idim, :])
            re_idim = np.unique(re_chunks[idim, :])
            chunks.append(tuple(re_idim - le_idim))
        self.chunks = tuple(chunks)

    def to_dask(self, field, chunks=None):
        from dask import array as da, delayed

        if chunks is None:
            chunks = self.chunks

        full_domain = da.empty(self.dims, chunks=chunks, dtype="float64")
        for igrid in range(self._ngrids):
            le, re, shp = self._grids[igrid]
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
            da = self.to_numpy(field)
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

    def to_numpy(self, field):
        # get a full, in-mem np array. if you can use this
        # why not just use a YTArbitraryGrid though...
        full_domain = np.empty(self.dims, dtype="float64")
        for igrid in range(self._ngrids):
            le, re, shp = self._grids[igrid]
            vals = _get_filled_grid(le, re, shp, field, self.ds, self.field_parameters)
            full_domain[self._grid_slc[igrid]] = vals
        return full_domain

    def to_zarr(self,
                field,
                base_path: str,
                ops = None
                ):

        shape = self.dims
        chunks = self.nchunks
        dtype = np.float64
        if ops is None:
            ops = []

        import os
        import zarr

        fld_str = "_".join(field)
        fpath = os.path.join(base_path, fld_str + '.zarr')
        full_domain = zarr.open(fpath, mode='w', shape=shape,
                       chunks=chunks, dtype=dtype)

        for igrid in range(self._ngrids):
            le, re, shp = self._grids[igrid]
            vals = _get_filled_grid(
                le, re, shp, field, self.ds, self.field_parameters
            )
            slc = self._grid_slc[igrid]
            for op in ops:
                vals = op(vals)
            full_domain[slc] = vals

        return full_domain

class YTPyramid:
    def __init__(
            self,
            left_edge,
            right_edge,
            dims: tuple[int, int, int],
            nchunks: int,
            n_levels: int,
            factor: int = 2,
            ds: Dataset = None,
            field_parameters=None,
            parallel_method: Optional[str] = None,
            data_source: Optional[Any] = None,
            cache: Optional[bool] = False,
    ):

        levels = []
        dims_ = np.array(dims,dtype=int)

        for lev in range(n_levels):
            current_dims = dims_ / factor**lev
            print(current_dims)
            tag = YTTiledArbitraryGrid(
                left_edge,
                right_edge,
                current_dims,
                int(nchunks/(lev+1)),
                ds=ds,
                field_parameters =field_parameters,
                data_source=data_source,
            )
            levels.append(tag)

        self._levels = levels

    def to_zarr(self,
                field,
                base_path: str,
                ops=None
                ):

        import os
        vals = []
        for lev, tag in enumerate(self._levels):
            print(f"writing level {lev}")
            lev_path = os.path.join(base_path, str(lev))
            vals.append(tag.to_zarr(field,
                        lev_path,
                        ops=ops,
                        ))
        return vals




def _get_filled_grid(le, re, shp, field, ds, field_parameters):
    grid = YTArbitraryGrid(le, re, shp, ds=ds, field_parameters=field_parameters)
    vals = grid[field]
    return vals

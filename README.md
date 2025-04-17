# yt_experiments

The `yt_experiments` package is a home for rapid prototyping of ideas that don't quite fit into the existing [yt-Project](https://yt-project.org/) ecosystem or would benefit from some community testing before being merged to `yt`.

## Current experimental features:

### OctTree conversion tools

`OctTree`, imported with

```python
from yt_experiments.octree.converter import OctTree
```

can be used to create an Octree of *cells* then convert it into a format yt can ingest.

### Tiled Arbitrary Grids and Image Pyramids

The `YTTiledArbitraryGrid`, `YTArbitraryGridOctPyramid` and `YTArbitraryGridPyramid` data objects allow creation of larger-than-memory fixed resolution arrays for fields from yt datasets by tiling together `YTArbitraryGrid` objects.

Import with

```python
from yt_experiments.tiled_grid import (
    YTArbitraryGridOctPyramid,
    YTArbitraryGridPyramid,
    YTTiledArbitraryGrid,
)
```

and see the [example notebook here](https://github.com/yt-project/yt_experiments/blob/main/examples/tiled_grids_intro.ipynb) for usage demonstrating how to convert yt field data to on-disk zarr arrays, including generation of a zarr image pyramid.

## Contributing

Contributions are welcome and generally follow the same principles as [main yt](https://yt-project.org/docs/dev/developing/index.html), including the [Community Code of Conduct](https://yt-project.org/docs/dev/developing/developing.html#yt-community-code-of-conduct).

We'll try to review contributions quickly and cut new releases as new features are merged to make them accesible to the community as soon as possible.

Not sure your idea fits here or in one of the other yt Project packages? Reach out on slack (instructions for joining are [here](https://yt-project.org/community.html)) or [open an issue](https://github.com/yt-project/yt_experiments/issues/new).

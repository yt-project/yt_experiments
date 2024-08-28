# requires https://github.com/napari/napari/pull/6043
from napari.experimental._progressive_loading import (
    add_progressive_loading_image,
)

if __name__ == "__main__":
    import napari
    import zarr

    viewer = napari.Viewer(ndisplay=3)

    dens_log10 = zarr.group("deeply_nested_pyramid.zarr")["density_log10"]
    multiscale_img = [vals for _, vals in dens_log10.arrays()]
    print(multiscale_img)
    import numpy as np

    print(np.max(multiscale_img[-1]))
    print(np.min(multiscale_img[-1]))
    add_progressive_loading_image(
        multiscale_img,
        viewer=viewer,
        contrast_limits=(-26, -24),
        colormap="viridis",
        ndisplay=3,
        rendering="mip",
    )

    napari.run()

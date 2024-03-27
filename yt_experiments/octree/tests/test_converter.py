import numpy as np

from yt_experiments.octree.converter import OctTree


def test_conversion():
    xyz = np.array(
        [
            [0.375, 0.125, 0.125],
            [0.125, 0.125, 0.125],
            [0.375, 0.375, 0.375],
            [0.75, 0.75, 0.75],
        ]
    )
    levels = np.array([2, 2, 2, 1], dtype=np.int32)

    # This should not raise an exception
    octree = OctTree.from_list(xyz, levels, check=True)

    ref_mask, leaf_order = octree.get_refmask()

    np.testing.assert_equal(
        ref_mask,
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    )
    np.testing.assert_equal(
        leaf_order, [1, -1, -1, -1, 0, -1, -1, 2, -1, -1, -1, -1, -1, -1, 3]
    )

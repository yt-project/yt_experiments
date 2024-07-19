import numpy as np
import pytest

from yt_experiments.octree.converter import OctTree

test_data = (
    {
        "xyz": np.array(
            [
                [0.375, 0.125, 0.125],
                [0.125, 0.125, 0.125],
                [0.375, 0.375, 0.375],
                [0.75, 0.75, 0.75],
            ],
        ),
        "levels": np.array([2, 2, 2, 1], dtype=np.int32),
        "ref_mask": np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        "leaf_order": np.array(
            [1, -1, -1, -1, 0, -1, -1, 2, -1, -1, -1, -1, -1, -1, 3]
        ),
    },
    {
        "xyz": np.array(
            [
                [0.015625, 0.015625, 0.015625],
                [0.046875, 0.015625, 0.015625],
                [0.078125, 0.015625, 0.015625],
                [0.109375, 0.015625, 0.015625],
                [0.015625, 0.046875, 0.015625],
                [0.046875, 0.046875, 0.015625],
                [0.078125, 0.046875, 0.015625],
                [0.109375, 0.046875, 0.015625],
                [0.015625, 0.078125, 0.015625],
                [0.046875, 0.078125, 0.015625],
            ]
        ),
        "levels": np.array([5] * 10, dtype=np.int32),
        "ref_mask": np.asarray(
            [
                int(_)
                for _ in ("111110000000001000000000100000000000000000000000000000000")
            ]
        ),
        "leaf_order": np.array(
            [
                int(_)
                for _ in (
                    "0 -1 4 -1 1 -1 5 -1 -1 8 -1 -1 -1 9 -1 -1 -1 -1 2 -1 6 -1 3 -1 7 "
                    "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 "
                    "-1 -1 -1 -1"
                ).split(" ")
            ]
        ),
    },
)


@pytest.mark.parametrize(
    "xyz, levels, ref_mask_exp, leaf_order_exp",
    [(t["xyz"], t["levels"], t["ref_mask"], t["leaf_order"]) for t in test_data],
)
def test_conversion(xyz, levels, ref_mask_exp, leaf_order_exp):
    # This should not raise an exception
    octree = OctTree.from_list(xyz, levels, check=True)

    ref_mask, leaf_order = octree.get_refmask()

    np.testing.assert_equal(
        ref_mask,
        ref_mask_exp,
    )
    np.testing.assert_equal(
        leaf_order,
        leaf_order_exp,
    )

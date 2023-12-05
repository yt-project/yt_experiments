# distutils: language = c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cimport numpy as np
from libc.stdlib cimport malloc
from libcpp.vector cimport vector

import cython
import numpy as np


cdef struct Oct:
    Oct** children
    double x
    double y
    double z
    np.int32_t ind


cdef class OctTree:
    """Octree of *cells* with utilities to convert it into a format yt can ingest."""
    cdef Oct* root
    cdef int count

    def __init__(self):
        self.root = <Oct*>malloc(sizeof(Oct))
        self.root.x = 0.5
        self.root.y = 0.5
        self.root.z = 0.5
        self.root.children = NULL
        self.root.ind = -1
        self.count = 1

    @cython.boundscheck(False)
    cdef Oct* add(self, const double[::1] x, const int level, const int unique_index, bint check = False) except NULL:
        """Add a cell to the octree.

        Parameters
        ----------
        x : double array (3,)
            The position of the cell in unitary units (within [0, 1])
        level : int
            The level of the cell. It should be such that x * 2**level is an integer
        unique_index : int
            A unique index for the cell.
        check : bool, optional
            If True, make sure that the position obtained by going down the octree matches
            the cell position.

        Returns
        -------
        Oct*
            The node in the tree at the cell location.
        """
        cdef float dx = 0.5

        cdef size_t ilvl, i, ind
        cdef np.int8_t ix, iy, iz

        cdef Oct* node = self.root
        cdef Oct* child = self.root

        for ilvl in range(level):
            ix = x[0] > node.x
            iy = x[1] > node.y
            iz = x[2] > node.z
            ind = 4 * ix + 2 * iy + iz

            if node.children == NULL:
                node.children = <Oct **> malloc(8 * sizeof(Oct*))
                for i in range(8):
                    node.children[i] = NULL

            if node.children[ind] == NULL:
                # Create a new node
                child = <Oct*>malloc(sizeof(Oct))
                child.children = NULL
                child.x = node.x + 0.5 * dx * (2 * ix - 1)
                child.y = node.y + 0.5 * dx * (2 * iy - 1)
                child.z = node.z + 0.5 * dx * (2 * iz - 1)
                child.ind = -1

                self.count += 1

                node.children[ind] = child

            dx *= 0.5

            node = node.children[ind]

        if check:
            if (
                not np.isclose(node.x, x[0]) or
                not np.isclose(node.y, x[1]) or
                not np.isclose(node.z, x[2])
            ):
                raise ValueError(
                    "Node xc does not match. Expected "
                    f"{x[0]}, {x[1]}, {x[2]}, got {node.x}, {node.y}, {node.z} @ level {level}"
                )

        if node.ind != -1:
            raise ValueError(
                "Node ind already set. Make sure that the same cell is not added twice."
            )

        node.ind = unique_index
        return node

    @classmethod
    @cython.boundscheck(False)
    def from_list(cls, const double[:, ::1] Xc, const int[::1] levels):
        """Create an octree from a list of cells.

        Parameters
        ----------
        Xc : double array (N, 3)
            The positions of the cells in unitary units (within [0, 1])
        levels : int array (N,)
            The levels of the cells. It should be such that Xc * 2**levels is an integer

        Returns
        -------
        OctTree
            The octree.
        """
        cdef int i

        cdef Oct* node

        cdef OctTree tree = cls()
        for i in range(Xc.shape[0]):
            node = tree.add(Xc[i], levels[i], i)

        return tree

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void depth_first_refmask(self, Oct* node, vector[np.uint8_t]& ref_mask, vector[np.int32_t]& leaf_order) noexcept nogil:
        """Depth-first traversal of the octree to get the refmask and leaf order.

        Parameters
        ----------
        node : Oct*
            The node to start the traversal from.
        ref_mask : vector[np.uint8_t]&
            The reference mask. Will be modified in place.
        leaf_order : vector[np.int32_t]&
            The leaf order. Will be modified in place.
        """
        cdef int i
        cdef Oct* child
        ret = 0
        for i in range(8):
            child = node.children[i]
            if child is NULL:
                leaf_order.push_back(-1)
                ref_mask.push_back(False)
                continue

            if child.children:   # Child has children
                ref_mask.push_back(True)
                self.depth_first_refmask(child, ref_mask, leaf_order)
            else:                # Child is a leaf
                leaf_order.push_back(child.ind)
                ref_mask.push_back(False)

    def get_refmask(self):
        """Get the reference mask and leaf order of the octree.

        Returns
        -------
        ref_mask : np.uint8 array (N,)
            The reference mask (see notes).
        leaf_order : np.int32 array (N,)
            The leaf order (see notes).

        Notes
        -----
        The reference mask is an array that is computed as follows.
        For each node in the tree, we store False if the node is a leaf cell.
        Otherwise, we store True and recursively iterate over its children in a
        depth-first manner.

        For each leaf cell, we append to the leaf order the index of the cell.
        The cell can either have an index if it corresponds to a cell added with
        :meth:`add` or -1 if it was added as part of the tree construction.

        The leaf_order array can then be used to index the input data such that
        they appear in the same order as in the refmask.
        """
        cdef vector[np.uint8_t] ref_mask
        cdef vector[np.int32_t] leaf_order

        # Preallocate memory
        ref_mask.reserve(self.count)
        leaf_order.reserve(self.count)

        self.depth_first_refmask(self.root, ref_mask, leaf_order)

        # Copy back data
        cdef np.uint8_t[:] ref_mask_view = np.zeros(1 + ref_mask.size(), dtype=np.uint8)
        cdef np.int32_t[:] leaf_order_view = np.zeros(leaf_order.size(), dtype=np.int32)

        cdef size_t i
        ref_mask_view[0] = 8
        for i in range(ref_mask.size()):
            ref_mask_view[i + 1] = ref_mask[i] * 8
        for i in range(leaf_order.size()):
            leaf_order_view[i] = leaf_order[i]
        return np.asarray(ref_mask_view), np.asarray(leaf_order_view)

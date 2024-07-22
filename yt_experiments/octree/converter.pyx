# distutils: language = c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cimport numpy as np
from libc.stdlib cimport malloc, free

import cython
import numpy as np


# Minimal requirements, adapted from cython's libcpp/vector.pxd
cdef extern from "<vector>" namespace "std" nogil:
    cdef cppclass vector[T,ALLOCATOR=*]:
        ctypedef size_t size_type

        vector() except +
        T& operator[](size_type)
        void reserve(size_type) except +
        void push_back(T&)
        size_type size()


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

    def __del__(self):
        self.deallocate(self.root)

    cdef void deallocate(self, Oct* node):
        """Deallocate the memory of the octree."""
        cdef int i
        if node.children != NULL:
            for i in range(8):
                if node.children[i] != NULL:
                    self.deallocate(node.children[i])
            free(node.children)
        free(node)

    @cython.boundscheck(False)
    cdef Oct* add_check(self, const double[3] x, const int level, const int unique_index) except NULL:
        """Add a cell to the octree and verify that the position obtained by going down the
        octree matches the cell position.

        Parameters
        ----------
        x : double[3]
            The position of the cell in unitary units (within [0, 1])
        level : int
            The level of the cell. It should be such that x * 2**level is a half-integer
        unique_index : int
            A unique index for the cell.

        Returns
        -------
        Oct*
            The node in the tree at the cell location.
        """
        cdef Oct* node = self.add(x, level, unique_index)

        if (
            not np.isclose(node.x, x[0]) or
            not np.isclose(node.y, x[1]) or
            not np.isclose(node.z, x[2])
        ):
            raise ValueError(
                "Node xc does not match. Expected "
                f"{x[0]}, {x[1]}, {x[2]}, got {node.x}, {node.y}, {node.z} @ level {level}"
            )

        if node.ind != unique_index:
            raise ValueError(
                "Node ind already set. Make sure that the same cell is not added twice."
            )

        return node

    @cython.boundscheck(False)
    cdef Oct* add(self, const double[3] x, const int level, const int unique_index) noexcept:
        """Add a cell to the octree.

        Parameters
        ----------
        x : double[3]
            The position of the cell in unitary units (within [0, 1])
        level : int
            The level of the cell. It should be such that x * 2**level is a half-integer
        unique_index : int
            A unique index for the cell.

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

        if node.ind == -1:
            node.ind = unique_index
        return node

    @classmethod
    @cython.boundscheck(False)
    def from_list(cls, const cython.floating[:, ::1] Xc, const cython.integral[::1] levels, bint check = False):
        """Create an octree from a list of cells.

        Parameters
        ----------
        Xc : double array (N, 3)
            The positions of the cells in unitary units (within [0, 1])
        levels : int array (N,)
            The levels of the cells. It should be such that Xc * 2**levels are half-integers.
        check : bool, optional
            If True, make sure that the position obtained by going down the octree matches
            the cell position.

        Returns
        -------
        OctTree
            The octree.

        Examples
        --------
        >>> import numpy as np
        ... import yt
        ... from yt_experiments.octree.converter import OctTree
        ... xyz = np.array([
        ...     [0.375, 0.125, 0.125],
        ...     [0.125, 0.125, 0.125],
        ...     [0.375, 0.375, 0.375],
        ...     [0.75, 0.75, 0.75],
        ... ])
        ... levels = np.array([2, 2, 2, 1], dtype=np.int32)
        ... data = {
        ...     ("gas", "density"): np.random.rand(4)
        ... }
        ...
        ... oct = OctTree.from_list(xyz, levels, check=True)
        ... ref_mask, leaf_order = oct.get_refmask()
        ... ref_mask, leaf_order
        (array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=uint8),
         array([ 1, -1, -1, -1,  0, -1, -1,  2, -1, -1, -1, -1, -1, -1,  3],
             dtype=int32))
        >>> for k, v in data.items():
        ...     # Make it 2D so that yt doesn't think those are particles
        ...     data[k] = np.where(leaf_order >= 0, v[leaf_order], np.nan)[:, None]
        ...
        ... ds = yt.load_octree(ref_mask, data)
        """
        cdef int i

        cdef Oct* node

        cdef OctTree tree = cls()
        cdef double[3] x
        for i in range(Xc.shape[0]):
            x[0] = Xc[i, 0]
            x[1] = Xc[i, 1]
            x[2] = Xc[i, 2]
            if check:
                node = tree.add_check(x, levels[i], i)
            else:
                node = tree.add(x, levels[i], i)

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
        ref_mask_view[0] = 1
        for i in range(ref_mask.size()):
            ref_mask_view[i + 1] = ref_mask[i]
        for i in range(leaf_order.size()):
            leaf_order_view[i] = leaf_order[i]
        return np.asarray(ref_mask_view), np.asarray(leaf_order_view)

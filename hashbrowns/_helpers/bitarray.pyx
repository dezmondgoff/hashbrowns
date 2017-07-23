# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = False
from Cython.Compiler.Naming import vtabptr_prefix
from numpy.core.numeric import outer
"""
Created on Sat Apr 29 01:25:09 2017

@author: root
"""
import numpy as np
from numpy.linalg import LinAlgError

cdef void c_bitdot(const npy_intp dim, const npy_intp n, const double * x,
                   const npy_intp m, const unsigned long * y,
                   unsigned long * out) nogil:
    cdef npy_intp i, j, k
    cdef double d
    cdef npy_intp size = sizeof(long)
    cdef npy_intp L = <npy_intp> ceil(<double> dim / size)

    for i in range(n):
        for j in range(m):
            d = 0
            for k in range(dim):
                if y[j * L + k // size] & (1 << (k % size)):
                    d += x[i * dim + k]
                else:
                    d -= x[i * dim + k]
            if d > 0:
                out[i * m + j] = 1
            else:
                out[i * m + j] = 0


def bitdot(a, b, unsigned long[:, ::1] out=None):
    """
    Returns an array with the sign of each element of the dot product of double
    vectors in *a* and bit vectors in *b*.

    Args:
        a (array-like): Floating point vectors
        b (array-like): Bit vectors as integers
        out (memoryview of array-like): if large enough, the output is packed
        contiguously into the beginning of `out`.

    Raises:
        LinAlgError: If `a` and `b` have incompatible dimensions
        ValueError: If `out` array has wrong dimension
    """
    cdef npy_intp n, m, size
    cdef npy_intp dim = a.shape[1]
    cdef double[::1] a_1d
    cdef double[:, ::1] a_2d
    cdef unsigned long[::1] b_1d
    cdef unsigned long[:, ::1] b_2d
    cdef double * aptr
    cdef unsigned long * bptr

    if b.shape[0] < ceil(dim / 32):
        raise LinAlgError('Incompatible dimensions')

    if a.ndim == 1:
        n = 1
        a_1d = a
        aptr = &a_1d[0]
    else:
        n = a.shape[1]
        a_2d = a
        aptr = &a_2d[0, 0]

    if b.ndim == 1:
        m = 1
        b_1d = b
        bptr = &b_1d[0]
    else:
        m = b.shape[1]
        b_2d = b
        bptr = &b_2d[0, 0]

    if out is None:
        out = np.empty((n, m), dtype=np.uint)
    else:
        if out.shape[0] != n or out.shape[1] != m:
            raise ValueError("Output array has wrong dimension.")

    with nogil:
        c_bitdot(dim, n, aptr, m, bptr, &out[0, 0])

    return np.asarray(out)

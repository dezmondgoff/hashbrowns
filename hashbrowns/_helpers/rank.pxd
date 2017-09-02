cdef extern from "numpy/npy_no_deprecated_api.h": pass
cimport numpy as np
from numpy cimport npy_intp
from libc.stdlib cimport malloc, free, const_void, qsort

cdef struct double_sorter:
    int index
    double value

cdef int c_double_compare(const_void *a, const_void *b) nogil

cdef void c_double_argsort(const long start, const long end, const double * a, 
                           npy_intp * out) nogil

cdef void c_reorder_inplace(npy_intp * a, const npy_intp start, 
                            const npy_intp end, npy_intp * order) nogil

cdef void c_rank(npy_intp m, npy_intp n, const double * a, npy_intp * indices, 
                 const npy_intp * indptr, npy_intp * sort, 
                 npy_intp * out) nogil
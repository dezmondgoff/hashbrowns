cimport numpy as np
from libc.stdlib cimport malloc, free, const_void, qsort
from cpython.array cimport array, clone
from cython cimport view

cdef struct long_sorter:
    long index
    long value

cdef struct double_sorter:
    int index
    double value

cdef int _long_compare(const_void *a, const_void *b) nogil

cdef int _double_compare(const_void *a, const_void *b) nogil

cdef void long_argsort(long start, long end, long *data, long *out) nogil

cdef void double_argsort(long start, long end, double *data, long *out) nogil

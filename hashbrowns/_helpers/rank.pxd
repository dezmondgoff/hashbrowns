cimport numpy as np
from argsort cimport double_argsort

cdef void reorder_inplace(long * a, long start, long end, long * order) nogil

cdef void c_rank(long k, long n, double * a, long * indices, long * indptr,
               long * sort, long * out) nogil
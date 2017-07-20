cimport numpy as np
from numpy cimport npy_intp
from libc.math cimport INFINITY, ceil

cdef void c_hash_argmin(const double * d, const long n, const long m, 
                        const npy_intp * idx, const long el, const long size, 
                        long * out) nogil
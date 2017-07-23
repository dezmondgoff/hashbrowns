cimport numpy as np
from numpy cimport npy_intp
from libc.math cimport ceil

cdef void c_bitdot(const npy_intp dim, const npy_intp n, const double * x, 
                   const npy_intp m, const unsigned long * y, 
                   unsigned long * out) nogil
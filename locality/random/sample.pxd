"""
index arrays for sampling from ndarray objects
"""
cimport numpy as np
from cpython.array cimport array, clone
from locality.random.random cimport randint
from locality.fused.numeric cimport numeric, numeric_int_flag

cdef array cython_sample(int numpoints, int num_samples, int sample_size, 
                         bint replace)

cdef array cython_sample_intervals(int numpoints, int interval_size, 
                                   int num_samples, int sample_size, bint replace)

cdef void cython_shuffle_inplace(numeric[::1] A) nogil
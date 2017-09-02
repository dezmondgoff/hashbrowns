"""
index arrays for sampling from ndarray objects
"""

cdef extern from "numpy/npy_no_deprecated_api.h": pass
from numpy cimport npy_intp, npy_uint64, npy_uint32
from randomkit_wrap cimport RandomStateInterface, rk_state, rk_interval
from randomkit_wrap cimport rk_random_uint64, rk_random_uint32
from libc.stdlib cimport malloc, free

cdef void c_sample_no_replace(npy_intp n, npy_intp m, npy_intp s, 
                              npy_intp * out, rk_state * state) nogil

cdef void c_sample_intervals_no_replace(npy_intp n, npy_intp k, npy_intp el, 
                                        npy_intp m, npy_intp s, npy_intp * out, 
                                        rk_state * state) nogil

cdef void c_sample_intervals_replace_64(npy_intp n, npy_intp k, npy_intp el,
                                        npy_intp m, npy_intp s, npy_intp * out, 
                                        rk_state * state) nogil
                                         
cdef void c_sample_intervals_replace_32(npy_intp n, npy_intp k, npy_intp el,
                                        npy_intp m, npy_intp s, npy_intp * out, 
                                        rk_state * state) nogil
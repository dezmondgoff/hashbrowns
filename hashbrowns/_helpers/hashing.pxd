cdef extern from "numpy/npy_no_deprecated_api.h": pass
cimport numpy as np
from numpy cimport npy_intp
from libc.math cimport INFINITY

cdef void c_hash_argmin(const double * d, const long n, const long m, 
                        const npy_intp * idx, const long el, const long size, 
                        long * out) nogil

cdef void c_encode_by_bits(const npy_intp n, const npy_intp m, const long * a, 
                           const long bitshift, unsigned long * out) nogil
            
cdef void c_decode_by_bits(const npy_intp n, const npy_intp m, 
                           const unsigned long * a, const long bitshift, 
                           long * out) nogil

cdef void c_encode_by_place(const npy_intp n, const npy_intp m, const long * a, 
                            const long width, unsigned long * out) nogil
            
cdef void c_decode_by_place(const npy_intp n, const npy_intp m, 
                            const unsigned long * a, const long width, 
                            long * out) nogil

cdef char c_count_set_bits(unsigned long n) nogil

cdef void c_hash_dist(npy_intp n, const unsigned long * h, 
                      const unsigned long ref, const long b, char * out) nogil
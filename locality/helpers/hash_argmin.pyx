# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = True

import numpy as np

cdef void c_hash_argmin(const double * d, const long n, const long m, 
                        const npy_intp * idx, const long el, const long size, 
                        long * out) nogil:
    
    cdef npy_intp i, j, k, r = 0
    cdef long argmin
    cdef double best, current
    
    for i in range(n):
        j = 0
        while j < el:
            best = INFINITY
            argmin = 0
            for k in range(size):
                current = d[i * m + idx[j]]
                if current < best:
                    best = current
                    argmin = k
                j += 1
            out[r] = argmin
            r += 1
            
def hash_argmin(const double[:,::1] d, const npy_intp[::1] idx, const long size, 
                long[::1] out):
    with nogil:
        c_hash_argmin(&d[0,0], d.shape[0], d.shape[1], &idx[0], idx.shape[0], 
                      size, &out[0])
    
    return np.asarray(out)
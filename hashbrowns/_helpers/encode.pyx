# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = False
"""
Created on Sat Apr 29 01:22:29 2017

@author: root
"""
import numpy as np

cdef void c_encode_by_bits(const npy_intp n, const npy_intp m, const long * a, 
                           const long bitshift, unsigned long * out) nogil:
    cdef npy_intp i, j 
    cdef long shift
    cdef unsigned long value
    
    for i in range(n):
        value = 0
        shift = 0
        for j in range(m):
            value += a[i * m + j] << shift
            shift += bitshift
        out[i] = value
            
cdef void c_decode_by_bits(const npy_intp n, const npy_intp m, 
                           const unsigned long * a, const long bitshift, 
                           long * out) nogil:
    cdef npy_intp i, j, k
    cdef long shift, mask = (1 << bitshift) - 1
    
    for i in range(n):
        shift = 0
        for j in range(m):
            out[i * m + j] = (a[i] >> shift) & mask 
            shift += bitshift

cdef void c_encode_by_place(const npy_intp n, const npy_intp m, const long * a, 
                            const long width, unsigned long * out) nogil:
    cdef npy_intp i, j 
    cdef long place, value, x, k
    
    for i in range(n):
        value = 0
        place = 1
        for j in range(m):
            x = a[i * m + j]
            x = 2 * x if x > 0 else -2 * x - 1
            k = width
            while x > 0 and k > 0:
                value += (x % 10) * place
                x //= 10
                place *= 10
                k -= 1
            while k > 0:
                place *= 10
                k -= 1
        out[i] = value
            
cdef void c_decode_by_place(const npy_intp n, const npy_intp m, 
                            const unsigned long * a, const long width, 
                            long * out) nogil:
    cdef npy_intp i, j
    cdef long place, value, x, k 
        
    for i in range(n):
        shift = 0
        x = a[i]
        while x > 0:
            j = m - 1
            k = 0
            place = 1
            value = 0
            while k < width:
                value += (x % 10) * place
                x //= 10
                place *= 10
                k += 1
            value = value // 2 if value % 2 == 0 else -(value + 1) // 2
            out[i * m + j] = value 
            j -= 1

def encode_by_bits(const long[:,::1] a, const long bitshift, 
                   unsigned long[::1] out):
    cdef npy_intp n = a.shape[0]
    cdef npy_intp m = a.shape[1]

    with nogil:    
        c_encode_by_bits(n, m, &a[0,0], bitshift, &out[0])
     
    return np.asarray(out)
 
def encode_by_place(const long[:,::1] a, const long width, 
                    unsigned long[::1] out):
    cdef npy_intp n = a.shape[0]
    cdef npy_intp m = a.shape[1]
    
    with nogil:
        c_encode_by_place(n, m, &a[0,0], width, &out[0])
     
    return np.asarray(out)

def decode_by_bits(const unsigned long[::1] a, const npy_intp m, 
                   const long bitshift, long[:,::1] out):
    cdef npy_intp n = a.shape[0]
    
    with nogil:
        c_decode_by_bits(n, m, &a[0], bitshift, &out[0,0])
    
    return np.asarray(out)

def decode_by_place(const unsigned long[::1] a, const npy_intp m, 
                    const long width, long[:,::1] out):
    cdef npy_intp n = a.shape[0]

    with nogil:
        c_decode_by_place(n, m, &a[0], width, &out[0,0])
     
    return np.asarray(out)
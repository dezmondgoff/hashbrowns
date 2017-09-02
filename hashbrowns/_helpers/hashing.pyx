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

cdef char c_count_set_bits(unsigned long n) nogil:
    cdef char count
    
    while n > 0:
        n &= n - 1
        count += 1
        
    return count

cdef void c_hash_dist(npy_intp n, const unsigned long * h, 
                    const unsigned long ref, const long b, char * out) nogil:
    cdef npy_intp i
    cdef unsigned long tmp
    cdef unsigned long mask
    cdef char x
    
    if b == 1:
        for i in range(n):
            out[i] = c_count_set_bits(h[i] ^ ref)
    else:
        mask = (2 << b) - 1
        for i in range(n):
            tmp = h[i] ^ ref
            x = 0
            while tmp > 0:
                if tmp & mask != 0:
                    x += 1
                tmp >>= b
            out[i] = x

def hash_argmin(const double[:,::1] d, const npy_intp[::1] idx, const long size, 
                long[::1] out):
    with nogil:
        c_hash_argmin(&d[0,0], d.shape[0], d.shape[1], &idx[0], idx.shape[0], 
                      size, &out[0])
    
    return np.asarray(out)

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

def get_hash_distances(const unsigned long hashkey, 
                       const unsigned long[::1] hashlist, 
                       const long bitshift_or_width, char[::1] out=None):
    if out is None:
        out = np.empty(hashlist.shape[0], dtype=np.int8)
    elif out.shape[0] != hashlist.shape[0]:
        raise ValueError("Output array has incorrect shape.")
    
    with nogil:
        c_hash_dist(hashlist.shape[0], &hashlist[0], hashkey, 
                    bitshift_or_width, &out[0])
    
    return np.asarray(out)
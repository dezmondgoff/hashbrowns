# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = False
"""
Created on Sat Apr 29 20:53:31 2017

@author: root
"""
import numpy as np
    
cdef int c_double_compare(const_void *a, const_void *b) nogil:
    cdef double v = ((<double_sorter*>a)).value-((<double_sorter*>b)).value
    if v < 0: return -1
    if v >= 0: return 1

cdef void c_double_argsort(const long start, const long end, const double * a, 
                           npy_intp * out) nogil:
    cdef npy_intp i, j = start, n = end - start
    cdef double_sorter * order = <double_sorter *> malloc(n * sizeof(double_sorter))
    for i in range(n):
        order[i].index = i
        order[i].value = a[j]
        j += 1
    qsort(<void *> order, n, sizeof(double_sorter), c_double_compare)
    for i in range(n):
        out[i] = <npy_intp> order[i].index
    free(order)

cdef void c_reorder_inplace(npy_intp * a, const npy_intp start, 
                            const npy_intp end, npy_intp * order) nogil:    
    cdef npy_intp i, j, t
    
    a += start
    n = end - start

    for i in range(n):
        if order[i] >= 0:
            if order[i] != i:
                t = a[i]
                j = i
                while True:
                    if order[j] != i:
                        a[j] = a[order[j]]
                    else:
                        a[j] = t
                        order[j] = -1
                        break
                    order[j], j = -1, order[j]

cdef void c_rank(npy_intp m, npy_intp n, const double * a, npy_intp * indices, 
                 const npy_intp * indptr, npy_intp * sort, 
                 npy_intp * out) nogil:
    cdef npy_intp i, j, k=0, start, end
    
    for i in range(n):
        start, end = indptr[i], indptr[i+1]
        c_double_argsort(start, end, a, sort)
        c_reorder_inplace(indices, start, end, sort)
        
        if end - start >= m:
            for j in range(m):
                out[k + j] = indices[start + j]
        else:
            for j in range(end - start):
                out[k + j] = indices[start + j]
            for j in range(m - end + start):
                out[k + end - start + j] = -1
        k += m

def rank(long q, long n, double[::1] dist, npy_intp[::1] indices, 
         npy_intp [::1] indptr, sort=None, out=None):
    cdef npy_intp[::1] sort_1d
    cdef npy_intp[::1] out_1d
    
    if sort is None:
        sort = np.empty(np.max(np.diff(indptr)), dtype=np.int)
    if out is None:
        out = np.empty(q * n, dtype=np.int)
    
    sort_1d = sort
    out_1d = out
    
    with nogil:
        c_rank(q, n, &dist[0], &indices[0], &indptr[0], &sort_1d[0], 
               &out_1d[0])
    
    return out
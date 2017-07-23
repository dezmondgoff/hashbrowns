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

cdef void reorder_inplace(long * a, long start, long end, long * order) nogil:
    
    cdef long i, j, t
    
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

cdef void c_rank(long m, long n, double * a, long * indices, long * indptr,
               long * sort, long * out) nogil:
    
    cdef long i, j, k=0, start, end
    
    for i in range(n):
        start, end = indptr[i], indptr[i+1]
        double_argsort(start, end, a, sort)
        reorder_inplace(indices, start, end, sort)
        
        if end - start >= m:
            for j in range(m):
                out[k + j] = indices[start + j]
        else:
            for j in range(end - start):
                out[k + j] = indices[start + j]
            for j in range(m - end + start):
                out[k + end - start + j] = -1
        k += m

def rank(long query_num, long num_points, double[::1] dist, long[::1] indices, 
         long[::1] indptr, sort=None, out=None):
    
    cdef long[::1] sort_1d
    cdef long[::1] out_1d
    
    if sort is None:
        sort = np.empty(np.max(np.diff(indptr)), dtype=np.int)
    if out is None:
        out = np.empty(query_num * num_points, dtype=np.int)
    
    sort_1d = sort
    out_1d = out
    
    c_rank(query_num, num_points, &dist[0], &indices[0], &indptr[0], &sort_1d[0], 
           &out_1d[0])
    
    return out
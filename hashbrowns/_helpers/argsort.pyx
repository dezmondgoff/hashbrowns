# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = True

import numpy as np

cdef int _long_compare(const_void *a, const_void *b) nogil:
    cdef int v = ((<long_sorter*>a)).value-((<long_sorter*>b)).value
    if v < 0: return -1
    if v >= 0: return 1
    
cdef int _double_compare(const_void *a, const_void *b) nogil:
    cdef double v = ((<double_sorter*>a)).value-((<double_sorter*>b)).value
    if v < 0: return -1
    if v >= 0: return 1

cdef void long_argsort(long start, long end, long * data, long * out) nogil:
    cdef long i, j = start
    cdef long n = end - start
    cdef long_sorter * order = <long_sorter *> malloc(n * sizeof(long_sorter))
    for i in range(n):
        order[i].index = i
        order[i].value = data[j]
        j += 1
    qsort(<void *> order, n, sizeof(long_sorter), _long_compare)
    for i in range(n):
        out[i] = order[i].index
    free(order)

cdef void double_argsort(long start, long end, double * data, long * out) nogil:
    cdef long i, j = start
    cdef long n = end - start
    cdef double_sorter *order = <double_sorter *> malloc(n * sizeof(double_sorter))
    for i in range(n):
        order[i].index = i
        order[i].value = data[j]
        j += 1
    qsort(<void *> order, n, sizeof(double_sorter), _double_compare)
    for i in range(n):
        out[i] = order[i].index
    free(order)



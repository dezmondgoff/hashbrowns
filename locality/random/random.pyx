# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = False
"""
Created on Tue Apr 18 09:12:05 2017

@author: root
"""

srand(time(NULL))

cdef int randint(int a, int b) nogil:
    
    cdef int n = b - a
    cdef int x
    cdef long end
    
    if n - 1 == RAND_MAX:
        return a + rand()
    else:
        end = RAND_MAX // n
        end *= n
        x = rand()
        while x >= end:
            x = rand()
        return a + x % n

cdef double random() nogil:
    
    cdef int x
    
    x = rand()
    while x == RAND_MAX:
        x = rand()
    return (<double> x)/RAND_MAX
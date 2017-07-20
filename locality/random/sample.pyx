# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = False

"""
Index arrays for sampling from array-like objects (ndarrays, typed memoryviews,
etc.)
"""

import numpy as np
from locality.fused.numeric import _strings as dtype_strings

cdef array cython_sample(int numpoints, int num_samples, int sample_size, 
                    bint replace):
    """
    Take multiple samples from the range of indices for a 1D array-like,

    Parameters
    ----------
    numpoints: int
        Number of points in set
    num_samples: int
        Number of samples to be taken
    sample_size: int
        Number of indices in each sample
    replace: bint
        Indicate sampling with or without replacement
    
    Returns
    -------
    array
        Sampled indices in flat array
    """
    cdef int i, j, r, curr, start
    cdef array out, A, template = array('l',[])
    cdef long tmp
    
    out = clone(template,num_samples * sample_size, False)
    
    if not replace:
        A = clone(template,numpoints, False)
        for i in range(numpoints):
            A.data.as_longs[i] = i
        start = 0
        for i in range(num_samples):
            curr = numpoints-1
            for j in range(sample_size):
                r = randint(0, curr + 1)
                out.data.as_longs[start + j] = A.data.as_longs[r]
                tmp = A.data.as_longs[curr]
                A.data.as_longs[curr] = A.data.as_longs[r]
                A.data.as_longs[r] = tmp
                curr -= 1
            start += sample_size
    else:
        for i in range(num_samples*sample_size):
            r = randint(0, numpoints)
            out.data.as_longs[i] = r    
    return out

cdef array cython_sample_intervals(int numpoints, int interval_size, 
                                   int num_samples, int sample_size, 
                                   bint replace):
    """
    Take multiple samples from non-overlapping intervals along a 1D array-like

    Parameters
    ----------
    numpoints: int
        Number of points in set
    interval_size: int
        Size of intervals
    num_samples: int
        Number of samples to be taken from each interval
    sample_size: int
        Number of indices in each sample
    replace: bint
        Indicate sampling with or without replacement
    
    Returns
    -------
    array
        Sampled indices in flat array
    """
    cdef int _, i, j, curr, start
    cdef int r = 0
    cdef long tmp
    cdef int offset, block_size
    cdef array out, A, template = array('l',[])
    cdef bint flag = False    
    if numpoints % interval_size == 0:
        num_intervals = numpoints//interval_size 
    else:
        num_intervals = numpoints//interval_size + 1
        flag = True 
    out = clone(template, num_intervals*num_samples*sample_size, False)
    if not replace:
        A = clone(template, interval_size, False)
        start = 0
        offset = 0
        for _ in range(num_intervals):
            for i in range(interval_size):
                A.data.as_longs[i] = i + offset
            for i in range(num_samples):
                curr = interval_size - 1
                for j in range(sample_size):
                    r = randint(0, curr + 1)
                    out.data.as_longs[start + j] = A.data.as_longs[r]
                    tmp = A.data.as_longs[curr]
                    A.data.as_longs[curr] = A.data.as_longs[r]
                    A.data.as_longs[r] = tmp
                    curr -= 1
                start += sample_size
            offset += interval_size
    else:
        offset = 0
        start = 0
        block_size = num_samples*sample_size
        for _ in range(num_intervals):
            for i in range(block_size):
                r = randint(0, interval_size) + offset
                while r >= numpoints:
                    r = randint(0, interval_size) + offset
                out.data.as_longs[start + i] = r
            offset += interval_size
            start += block_size
    return out

cdef void cython_shuffle_inplace(numeric[::1] A) nogil:
    """
    Randomly shuffles array values
    
    Parameters
    ----------
    A: memoryview of 1D array-like
    
    Returns
    -------
    void
    """
    cdef int i, curr, r, n = A.shape[0]
    curr = n-1
    for i in range(n):
        r = randint(0, curr + 1)
        A[curr], A[r] = A[r], A[curr]
        curr -= 1

def sample(int numpoints, int num_samples, int sample_size, 
           bint replace=False):
    """Python wrapper for cysample."""
    return np.asarray(cython_sample(numpoints, num_samples, sample_size, replace))

def sample_intervals(int numpoints, int interval_size, int num_samples, 
                     int sample_size, int replace=False):
    """Python wrapper for cysample_intervals."""        
    return np.asarray(cython_sample_intervals(numpoints, interval_size, num_samples, sample_size, replace))

def shuffle(numeric[::1] A, bint inplace=True):
    """Python wrapper for cyshuffle."""
    cdef numeric[::1] out
    cdef int flag
    
    if inplace:
        cython_shuffle_inplace(A)
        return 
    flag = numeric_int_flag(A[0])
    out = np.empty(A.shape[0], dtype=dtype_strings[flag])
    out[...] = A
    cython_shuffle_inplace(out)
    return np.asarray(out)
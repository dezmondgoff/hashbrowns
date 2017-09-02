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
import randomkit_wrap as randomkit

cdef void c_sample_no_replace(npy_intp n, npy_intp m, npy_intp s, 
                              npy_intp * out, rk_state * state) nogil:
    cdef npy_intp i, j, r, curr, start, tmp
    cdef npy_intp * a = <npy_intp *> malloc(n * sizeof(npy_intp))
    
    for i in range(n):
        a[i] = i
        
    start = 0
    for i in range(m):
        curr = n - 1
        for j in range(s):
            r = <npy_intp> rk_interval(curr, state)
            out[start + j] = a[r]
            tmp = a[curr]
            a[curr] = a[r]
            a[r] = tmp
            curr -= 1
        start += s
    free(a)

cdef void c_sample_intervals_no_replace(npy_intp n, npy_intp k, npy_intp el,
                                        npy_intp m, npy_intp s, npy_intp * out, 
                                        rk_state * state) nogil:
    
    cdef npy_intp _, i, j, r, rng = k
    cdef npy_intp curr, pos = 0, offset = 0, block_size, tmp
    cdef bint flag = False   
    cdef npy_intp * a = <npy_intp *> malloc(k * sizeof(npy_intp)) 
    
    for _ in range(el):
        if offset + k > n:
            rng = n - offset
        for i in range(rng):
            a[i] = offset
            offset += 1
        for i in range(m):
            curr = rng - 1
            for j in range(s):
                r = <npy_intp> rk_interval(curr, state)
                out[pos] = a[r]
                tmp = a[curr]
                a[curr] = a[r]
                a[r] = tmp
                curr -= 1
                pos += 1
    free(a)
    
cdef void c_sample_intervals_replace_64(npy_intp n, npy_intp k, npy_intp el, 
                                        npy_intp m, npy_intp s, npy_intp * out, 
                                        rk_state * state) nogil:
    cdef npy_intp _, off = 0, rng = k - 1, cnt = m * s
    cdef npy_uint64 * ptr = <npy_uint64 * > out 
    
    for _ in range(el):
        rk_random_uint64(off, rng, cnt, ptr, state) 
        off += k
        if off + k > n:
            rng = n - 1 - off
        ptr += cnt
            
cdef void c_sample_intervals_replace_32(npy_intp n, npy_intp k, npy_intp el, 
                                        npy_intp m, npy_intp s, npy_intp * out, 
                                        rk_state * state) nogil:
    cdef npy_intp _, off = 0, rng = k - 1, cnt = m * s
    cdef npy_uint32 * ptr = <npy_uint32 * > out 
    
    for _ in range(el):
        rk_random_uint32(off, rng, cnt, ptr, state)
        off += k
        if off + k > n:
            rng = n - 1 - off
        ptr += cnt

def sample_indices(npy_intp n, npy_intp m, npy_intp s, bint replace=False,
           npy_intp[::1] out=None, RandomStateInterface rsi=None):
    """
    Take `m` samples of size `s` from the range of indices of a 1D array with 
    length `n`.
    
    Parameters
    ----------
    n : int
        The size of the array to be indexed
    m : int
        The number of samples
    s : int
        The number of indices in each sample
    out : ndarray, optional
        The output array
        If not None, output is stored in this array
    replace : boolean
        Sample with or without replacement. Defaults: False
    rsi : optional, RandomStateInterface
        A Cython interface to a Numpy random number generator 
        Defaults: and interface to RandomState generated on import of np.random
    
    Returns
    -------
    out : ndarray
        An array of indices
    
    Raises
    ------
    ValueError
        If provided output array has incorrect shape.
    StandardError
        If system npy_intp is not defined as 64 or 32-bit int. 
    """
    cdef rk_state * state
    
    if rsi is None:
        rsi = randomkit._rand_interface
    state = rsi.state_copy
    
    if out is None:
        out = np.empty(m * s, dtype=np.intp)
    elif out.shape[0] != m * s:
        raise ValueError("Output array has incorrect shape.")
    
    with rsi.lock:
        rsi.retreive_state()
        
        if not replace:
            c_sample_no_replace(n, m, s, &out[0], state)
        elif np.intp is np.int64:
            rk_random_uint64(0, n, m * s, <npy_uint64 *> &out[0], state)
        elif np.intp is np.int32:
            rk_random_uint32(0, n, m * s, <npy_uint32 *> &out[0], state)
        else:
            raise Exception("This should not happen. Report as bug.")

        rsi.return_state()
        
    return np.asarray(out)

def sample_intervals(npy_intp n, npy_intp k, npy_intp m, npy_intp s, 
                     npy_intp[::1] out=None, bint replace=False, 
                     RandomStateInterface rsi=None):
    """
    Take `m` samples of size 's' from non-overlapping intervals of size `k` 
    along a 1D array of length `n`.
    
    Parameters
    ----------
    n : int
        The size of the array to be indexed
    k : int
        The size of the intervals along the array
    m : int
        The number of samples per interval
    s : int
        The number of indices in each sample
    out : ndarray, optional
        The output array
        If not None, output is stored in this array
    replace : boolean
        Sample with or without replacement. Defaults: False
    rsi : optional, RandomStateInterface
        A Cython interface to a Numpy random number generator 
        Defaults: and interface to RandomState generated on import of np.random
    
    Returns
    -------
    out : ndarray
        An array of indices
    
    Raises
    ------
    ValueError
        If provided output array has incorrect shape.
    StandardError
        If system npy_intp is not defined as 64 or 32-bit int. 
    """
    cdef rk_state * state
    cdef npy_intp el = n // k
    
    if rsi is None:
        rsi = randomkit._rand_interface
    state = rsi.state_copy
    
    if n % k != 0:
        el += 1
    if out is None:
        out = np.empty(el * m * s, dtype=np.intp)
    elif out.shape[0] != el * m * s:
        raise ValueError("Output array has incorrect shape.")
    
    with rsi.lock:
        rsi.retreive_state()
    
        if not replace:
            c_sample_intervals_no_replace(n, k, el, m, s, &out[0], state)
        elif np.intp is np.int64:
            c_sample_intervals_replace_64(n, k, el, m, s, &out[0], state)
        elif np.intp is np.int32:
            c_sample_intervals_replace_32(n, k, el, m, s, &out[0], state)
        else:
            raise Exception("This should not happen. Report as bug.")
        
        rsi.return_state()
        
    return np.asarray(out)

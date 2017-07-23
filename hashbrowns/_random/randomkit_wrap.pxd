cdef extern from "numpy/npy_no_deprecated_api.h":
    pass
cimport numpy as np
from numpy cimport npy_intp
from numpy cimport npy_uint64, npy_uint32, npy_uint16, npy_uint8, npy_bool
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

cdef extern from "randomkit.h":

    ctypedef struct rk_state:
        unsigned long key[624]
        int pos
        int has_gauss
        double gauss

    ctypedef enum rk_error:
        RK_NOERR = 0
        RK_ENODEV = 1
        RK_ERR_MAX = 2

    char *rk_strerror[2]

    # 0xFFFFFFFFUL
    unsigned long RK_MAX

    void rk_seed(unsigned long seed, rk_state *state)
    rk_error rk_randomseed(rk_state *state)
    unsigned long rk_random(rk_state *state)
    long rk_long(rk_state *state) nogil
    unsigned long rk_ulong(rk_state *state) nogil
    unsigned long rk_interval(unsigned long max, rk_state *state) nogil
    double rk_double(rk_state *state) nogil
    void rk_fill(void *buffer, size_t size, rk_state *state) nogil
    rk_error rk_devfill(void *buffer, size_t size, int strong)
    rk_error rk_altfill(void *buffer, size_t size, int strong,
            rk_state *state) nogil
    double rk_gauss(rk_state *state) nogil
    void rk_random_uint64(npy_uint64 off, npy_uint64 rng, npy_intp cnt,
                          npy_uint64 *out, rk_state *state) nogil
    void rk_random_uint32(npy_uint32 off, npy_uint32 rng, npy_intp cnt,
                          npy_uint32 *out, rk_state *state) nogil
    void rk_random_uint16(npy_uint16 off, npy_uint16 rng, npy_intp cnt,
                          npy_uint16 *out, rk_state *state) nogil
    void rk_random_uint8(npy_uint8 off, npy_uint8 rng, npy_intp cnt,
                         npy_uint8 *out, rk_state *state) nogil
    void rk_random_bool(npy_bool off, npy_bool rng, npy_intp cnt,
                        npy_bool *out, rk_state *state) nogil  
    

cdef class RandomStateInterface:
    cdef object rstate
    cdef rk_state * state_copy
    cdef object lock
    
    cdef rk_state * retreive_state(self)
  
    cdef void return_state(self)
    
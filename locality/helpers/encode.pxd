#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 01:22:29 2017

@author: root
"""
cimport numpy as np
from numpy cimport npy_intp

cdef void c_encode_by_bits(const npy_intp n, const npy_intp m, const long * a, 
                            const long bitshift, unsigned long * out) nogil
            
cdef void c_decode_by_bits(const npy_intp n, const npy_intp m, 
                           const unsigned long * a, const long bitshift, 
                           long * out) nogil

cdef void c_encode_by_place(const npy_intp n, const npy_intp m, const long * a, 
                            const long width, unsigned long * out) nogil
            
cdef void c_decode_by_place(const npy_intp n, const npy_intp m, 
                            const unsigned long * a, const long width, 
                            long * out) nogil
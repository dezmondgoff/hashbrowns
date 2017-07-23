#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:11:53 2017

@author: dezmond
"""

cdef extern from "numpy/npy_no_deprecated_api.h": pass
from numpy cimport npy_intp
from randomkit_wrap cimport RandomStateInterface, rk_state, rk_double
from libc.math cimport sin, cos, tan, atan, log, floor, M_PI, M_PI_2, M_2_PI

cdef void c_stable_non_gauss(npy_intp n, double alpha, double beta, double c, 
                             double mu, double * out, rk_state * state) nogil
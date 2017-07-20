#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:11:53 2017

@author: dezmond
"""

cimport numpy as np
from libc.math cimport sin, cos, tan, atan, log, floor, M_PI, M_PI_2
from locality.random.random cimport random

cdef double stable_non_gauss_scalar(double alpha, double beta, double c, 
                                    double mu) nogil

cdef void stable_non_gauss_vector(int n, double alpha, double beta, double c, 
                                  double mu, double[::1] out) nogil

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:12:26 2017

@author: root
"""
from libc.stdlib cimport RAND_MAX, rand, srand
from libc.time cimport time_t, time

cdef int randint(int a, int b) nogil

cdef double random() nogil
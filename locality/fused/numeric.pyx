#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = False
"""
Created on Mon May  1 16:36:14 2017

@author: root
"""

_strings  = ["int8", "uint8" "int16", "uint16" "int32", "uint32" "int64", 
             "uint64", "float32","float64","float128", "complex64", 
             "complex128", "complex256"] 
      
cdef int numeric_int_flag(numeric x):
    
    cdef int i

    if numeric is char: 
        i = 0
    elif numeric is uchar:
        i = 1
    elif numeric is short: 
        i = 2
    elif numeric is ushort:
        i = 3
    elif numeric is int: 
        i = 4
    elif numeric is uint:
        i = 5
    elif numeric is long: 
        i = 6
    elif numeric is ulong:
        i = 7
    elif numeric is float: 
        i = 8
    elif numeric is double: 
        i = 9
    elif numeric is longdouble: 
        i = 10
    elif numeric is floatcomplex: 
        i = 11
    elif numeric is doublecomplex: 
        i = 12
    else: 
        i = 13
  
    return i
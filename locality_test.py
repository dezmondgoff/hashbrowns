#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 21:01:07 2017

@author: root
"""

import numpy as np
from locality.locality import VoroniLSH, CosineLSH

dim = 1000
num_points = 100000
x = np.random.random((num_points, dim))
y = np.random.random((10,dim))
L = 1
k = 4
m = 20
w = 5
lsh0 = VoroniLSH(x, L=L, k=k, m=m, w=w, metric="cosine")
lsh1 = CosineLSH(x, L=L, k=32)
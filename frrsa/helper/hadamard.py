#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 16:38:33 2020

@author: kaniuth
"""
import numpy as np
from numba import prange, njit

def hadamard_products(inputs_z):
    m, n = inputs_z.shape
    hadamard_prod = np.empty((m, n * (n - 1) // 2), dtype = inputs_z.dtype)
    ind1 = (np.kron(np.arange(0, n).reshape((n, 1)), np.ones((n, 1)))).squeeze().astype(int)
    ind2 = (np.kron(np.ones((n, 1)), np.arange(0, n).reshape((n, 1)))).squeeze().astype(int)
    first_pair_members = ind1[ind1 < ind2]
    second_pair_members = ind2[ind1 < ind2]
    return numba_func_parallel_trans(inputs_z, hadamard_prod, m, n), first_pair_members, second_pair_members
@njit(parallel=True)
def numba_func_parallel_trans(inputs_z, hadamard_prod, m, n):
    for p in prange(m):
        I = 0
        for i in range(n):
            for j in range(i+1,n):
                hadamard_prod[p, I] = inputs_z[p,i] * inputs_z[p, j]
                I += 1
    return hadamard_prod
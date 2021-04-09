#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 16:38:33 2020

@author: kaniuth
"""
import numpy as np
from numba import prange, njit

def hadamard(inputs_z):
    '''Returns the element-wise products of all pairs of columns'''
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

def euclidian(inputs_z, squared):
    '''
    Returns element-wise euclidian distance of all pairs of columns, either
    squared or absolute.
    '''
    #TODO: needs to return first_pair_members, second_pair_members as well, until then not functional!
    #TODO: in higher-order package "crossvalidation" in line 224, adapt scaling if needed.
    #TODO: in higher-order package "crossvalidation" add parameters to choose one of the predictor_distance funcs.
    n_conditions = inputs_z.shape[1]
    n_pairs = n_conditions*(n_conditions-1)//2
    idx = np.concatenate(([0], np.arange(n_conditions-1,0,-1).cumsum()))
    start, stop = idx[:-1], idx[1:]
    X = np.empty((inputs_z.shape[0], n_pairs), dtype=inputs_z.dtype)
    for j,i in enumerate(range(n_conditions-1)):
        X[:, start[j]:stop[j]] = inputs_z[:,i,None] - inputs_z[:,i+1:]
    if squared:
        np.square(X, out=X)
    else:
        np.absolute(X, out=X)
    return X
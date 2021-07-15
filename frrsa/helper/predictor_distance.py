#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spyder 4.1.4, Python 3.7.6 64-bit | Qt 5.9.6 | PyQt5 5.9.2 | Darwin 18.7.0
"""
@author: Philipp Kaniuth (kaniuth@cbs.mpg.de)
"""

import numpy as np
from numba import prange, njit

def hadamard(predictor):
    '''Compute the Hadamard products for all column pairs.
    
    Parameters
    ----------
    predictor : ndarray
        A 2d data array
    
    Returns
    -------
    hadamard_prod: ndarray
        Hadamard products for all column-pairs of `predictor`.
    '''
    m, n = predictor.shape
    hadamard_prod = np.empty((m, n * (n - 1) // 2), dtype = predictor.dtype)
    ind1 = (np.kron(np.arange(0, n).reshape((n, 1)), np.ones((n, 1)))).squeeze().astype(int)
    ind2 = (np.kron(np.ones((n, 1)), np.arange(0, n).reshape((n, 1)))).squeeze().astype(int)
    first_pair_members = ind1[ind1 < ind2]
    second_pair_members = ind2[ind1 < ind2]
    return numba_func_parallel_trans(predictor, hadamard_prod, m, n), first_pair_members, second_pair_members
@njit(parallel=True)
def numba_func_parallel_trans(predictor, hadamard_prod, m, n):
    for p in prange(m):
        I = 0
        for i in range(n):
            for j in range(i+1,n):
                hadamard_prod[p, I] = predictor[p,i] * predictor[p, j]
                I += 1
    return hadamard_prod

def euclidian(predictor, squared):
    '''Compute element-wise euclidian distance of all pairs of columns
    
    Compute the euclidian distance of each column-pair, separately for each 
    row. Either use squared or standard euclidian distance.
    
    Parameters
    ----------
    predictor : ndarray
        A 2d data array
    squared : bool
        Indicates whether to use standard or squared Euclidian distance.
    
    Returns
    -------
    X : ndarray
        Euclidian distance between all column-pairs for each row.
    '''
    #TODO: needs to return first_pair_members, second_pair_members as well, until then not functional!
    #TODO: in higher-order package "crossvalidation" in line 224, adapt scaling if needed.
    #TODO: in higher-order package "crossvalidation" add parameters to choose one of the predictor_distance funcs.
    n_conditions = predictor.shape[1]
    n_pairs = n_conditions*(n_conditions-1)//2
    idx = np.concatenate(([0], np.arange(n_conditions-1,0,-1).cumsum()))
    start, stop = idx[:-1], idx[1:]
    X = np.empty((predictor.shape[0], n_pairs), dtype=predictor.dtype)
    for j,i in enumerate(range(n_conditions-1)):
        X[:, start[j]:stop[j]] = predictor[:,i,None] - predictor[:,i+1:]
    if squared:
        np.square(X, out=X)
    else:
        np.absolute(X, out=X)
    return X
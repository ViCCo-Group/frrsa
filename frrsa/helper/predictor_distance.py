#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spyder 4.1.4, Python 3.7.6 64-bit | Qt 5.9.6 | PyQt5 5.9.2 | Darwin 18.7.0
"""
@author: Philipp Kaniuth (kaniuth@cbs.mpg.de)
"""
import itertools

import numpy as np

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
    r, c = np.triu_indices(predictor.shape[1], 1)
    hadamard_prod = np.einsum('ij,ik->ijk', predictor, predictor)[:, r, c]
    n_columns = predictor.shape[1]
    pairs = np.array(list((itertools.combinations(range(n_columns), 2))))
    first_pair_members = pairs[:, 0]
    second_pair_members = pairs[:, 1]
    return hadamard_prod, first_pair_members, second_pair_members

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
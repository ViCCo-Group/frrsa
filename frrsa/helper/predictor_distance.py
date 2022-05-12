#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains functions that compute feature-specific distances.

@author: Philipp Kaniuth (kaniuth@cbs.mpg.de)
"""

import itertools
import numpy as np


def check_predictor_data(predictor):
    """Check the predictor shape.

    Parameters
    ----------
    predictor : ndarray
        A 2d data array.
    """
    if len(predictor.shape) < 2:
        raise Exception('Predictor has to be two-dimensional.')
    if predictor.shape[1] < 2:
        raise Exception('Predictor needs at least 2 columns.')


def calculate_pair_indices(n_conditions):
    """Calculate the indices in the original predictor data of the first and second member of a pair.

    Parameters
    ----------
    n_conditions : int
        Amount of conditions in the original predictor.
        
    Returns
    -------
    first_pair_members : ndarray
        Indices of the first elements of all possible pairs of for `n_conditions` conditions.
    second_pair_members : ndarray
        Indices of the second elements of all possible pairs of for `n_conditions` conditions.
    """
    pairs = np.array(list((itertools.combinations(range(n_conditions), 2))))
    first_pair_members = pairs[:, 0]
    second_pair_members = pairs[:, 1]
    return first_pair_members, second_pair_members


def hadamard(predictor):
    """Compute the Hadamard products for all column pairs.

    Parameters
    ----------
    predictor : ndarray
        A 2d data array

    Returns
    -------
    hadamard_prod: ndarray
        Hadamard products for all column-pairs of `predictor`.
    first_pair_members : ndarray
        Indices of the first elements of all possible column-pairs of `predictor`.
    second_pair_members : ndarray
        Indices of the first elements of all possible column-pairs of `predictor`.
    """
    check_predictor_data(predictor)
    r, c = np.triu_indices(predictor.shape[1], 1)
    hadamard_prod = np.einsum('ij,ik->ijk', predictor, predictor)[:, r, c]
    n_conditions = predictor.shape[1]
    first_pair_members, second_pair_members = calculate_pair_indices(n_conditions)
    return hadamard_prod, first_pair_members, second_pair_members


def sqeuclidean(predictor):
    """Compute element-wise squared euclidean distance of all pairs of columns.

    Parameters
    ----------
    predictor : ndarray
        A 2d data array

    Returns
    -------
    X : ndarray
        Squared euclidean distance between all column-pairs for each row.
    first_pair_members : ndarray
        Indices of the first elements of all possible column-pairs of `predictor`.
    second_pair_members : ndarray
        Indices of the first elements of all possible column-pairs of `predictor`.
    """
    check_predictor_data(predictor)
    n_conditions = predictor.shape[1]
    n_pairs = n_conditions * (n_conditions - 1) // 2
    idx = np.concatenate(([0], np.arange(n_conditions - 1, 0, -1).cumsum()))
    start, stop = idx[:-1], idx[1:]
    X = np.empty((predictor.shape[0], n_pairs), dtype=predictor.dtype)
    for j, i in enumerate(range(n_conditions - 1)):
        X[:, start[j]:stop[j]] = predictor[:, i, None] - predictor[:, i + 1:]
    np.square(X, out=X)
    first_pair_members, second_pair_members = calculate_pair_indices(n_conditions)
    return X, first_pair_members, second_pair_members

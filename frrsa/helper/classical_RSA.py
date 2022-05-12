#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains function to perform classical Representational Similarity Analysis.

@author: Philipp Kaniuth (kaniuth@cbs.mpg.de)
"""

from functools import partial
import numpy as np
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist


def make_RDM(activity_pattern_matrix, distance='pearson'):
    #SOFT-DEPRECATED
    """Compute dissimilarity matrix.

    Parameters
    ----------
    activity_pattern_matrix : ndarray
        Matrix which holds activity patterns for several conditions. Each
        column is one condition, each row one measurement channel.
    distance : {'pearson', 'sqeuclidean'}, optional
        The desired distance measure (defaults to `pearson`).

    Returns
    -------
    rdm : ndarray
        Dissimilarity matrix indicating dissimilarity for all conditions in
        `activity_pattern_matrix`.
    """
    if distance == 'pearson':
        rdm = 1 - np.corrcoef(activity_pattern_matrix, rowvar=False)
    elif distance == 'sqeuclidean':
        rdm = squareform(pdist(activity_pattern_matrix.T, 'sqeuclidean'))
    return rdm


def flatten_matrix(representational_matrix: np.ndarray) -> np.ndarray:
    """Flatten the upper half of a representational matrix to a vector.

     Multiple representational matrices can be fed at once.

    Parameters
    ----------
    representational_matrix : ndarray
        The representational matrix (or matrices) which shall be flattened.
        Expected shape is (n_conditions, n_conditions, n_targets), where
        `n_targets` denotes the number of matrices. If `n_targets == 1`,
        `representational_matrix` can be of shape (n_conditions, n_conditions).

    Returns
    -------
    representational_vector : ndarray
        The unique upper half of `rdm`.
    """
    if representational_matrix.ndim == 3:
        mapfunc = partial(squareform, checks=False)
        representational_vector = np.array(list(map(mapfunc, np.moveaxis(representational_matrix, -1, 0)))).T
    elif representational_matrix.ndim == 2:
        representational_vector = representational_matrix[np.triu_indices(representational_matrix.shape[0], k=1)].reshape(-1, 1)
    return representational_vector

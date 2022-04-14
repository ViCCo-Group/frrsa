#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spyder 4.2.5 | Python 3.8.8 64-bit | Qt 5.9.7 | PyQt5 5.9.2 | Darwin 18.7.0
"""
Contains function to perform classical Representational Similarity Analysis.

@author: Philipp Kaniuth (kaniuth@cbs.mpg.de)
"""

from functools import partial
import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist


def make_RDM(activity_pattern_matrix, distance='pearson'):
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


def flatten_RDM(rdm: np.ndarray) -> np.ndarray:
    """Flatten the upper half of a dissimilarity matrix to a vector.

     Multiple dissimilarity matrices can be fed at once.

    Parameters
    ----------
    rdm : ndarray
        The RDM)s) which shall be flattened. Expected shape is
        (n_conditions, n_conditions, n_targets), where `n_targets` denotes the
        number of RDMs. If `n_targets == 1`, `rdm` can be of shape
        (n_conditions, n_conditions).

    Returns
    -------
    rdv : ndarray
        The unique upper half of `rdm`.
    """
    if rdm.ndim == 3:
        mapfunc = partial(squareform, checks=False)
        rdv = np.array(list(map(mapfunc, np.moveaxis(rdm, -1, 0)))).T
    elif rdm.ndim == 2:
        rdv = rdm[np.triu_indices(rdm.shape[0], k=1)].reshape(-1, 1)
    return rdv


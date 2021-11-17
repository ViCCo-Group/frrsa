#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spyder 4.2.5 | Python 3.8.8 64-bit | Qt 5.9.7 | PyQt5 5.9.2 | Darwin 18.7.0 
"""
Contains function to perform classical Representational Similarity Analysis.

@author: Philipp Kaniuth (kaniuth@cbs.mpg.de)
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr
from functools import partial
from scipy.spatial.distance import squareform

def make_RDM(activity_pattern_matrix, distance='pearson'):
    #TODO: implement for multiple matrices.
    #TODO: implement other distance norms 
    '''Compute dissimilarity matrix.
    
    Parameters
    ----------
    activity_pattern_matrix : ndarray
        Matrix which holds activity patterns for several conditions. Each
        column is one condition, each row one measurement channel.
    distance : str
        The desired distance measure.
        
    Returns
    -------
    rdm : ndarray
        Dissimilarity matrix indicating dissimilarity for all conditions in
        `activity_pattern_matrix`.
    '''
    if distance=='pearson':
        rdm = 1 - np.corrcoef(activity_pattern_matrix, rowvar=False)
    # elif distance=='Euclidian':
    #     rdm = 1 - np.corrcoef(activity_pattern_matrix, rowvar=False)
    return rdm

def flatten_RDM(rdm: np.ndarray) -> np.ndarray:
    '''Flatten the upper half of a dissimilarity matrix to a vector.
    
     Multiple dissimilarity matrices can be fed at once, requiring the shape 
     (n,n,m), where m denotes different matrices of shape (n,n).
    
    Parameters
    ----------
    rdm : ndarray
        The dissimilarity matrix which shall be flattened.
    
    Returns
    -------
    rdv : ndarray
        The unique upper half of `rdm`.
    '''
    if rdm.ndim==3:
        mapfunc = partial(squareform, checks=False)
        rdv = np.array(list(map(mapfunc, np.moveaxis(rdm, -1, 0)))).T
    elif rdm.ndim==2:
        rdv = rdm[np.triu_indices(rdm.shape[0], k=1)].reshape(-1,1)
    return rdv

def correlate_RDMs(rdv1, rdv2, score_type='pearson'):
    '''Relate two flattened dissimilarity matrices to each other
    
    Parameters
    ----------
    rdv1 : array_like
        First flattened dissimilarity matrix. 
    rdv2 : array_like
        Second flattened dissimilarity matrix.
    score_type : {'pearson', 'spearman'}, optional
        Type of association measure to compute (defaults to `pearson`).
        
    Return
    ------
    corr : float
        Correlation coefficient.
    p_value : float
        Two-tailed p-value.
    '''
    if score_type == 'pearson':
        corr, p_value = pearsonr(rdv1, rdv2)
    elif score_type == 'spearman':
        corr, p_value = spearmanr(rdv1, rdv2)
    return corr, p_value
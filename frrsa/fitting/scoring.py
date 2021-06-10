#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 16:44:31 2020

@author: kaniuth
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr


def scoring_unfitted(y_unfitted: np.ndarray, x_unfitted: np.ndarray, score_type: str='pearson') -> float:
    """Returns one of three possible scores to be maximised"""
    
    n_targets = y_unfitted.shape[1]
    
    scores = np.empty(n_targets)
     
    for target in range(n_targets):
        if score_type == 'pearson':
            scores[target] = pearsonr(y_unfitted[:, target], x_unfitted[:, 0])[0]
        elif score_type == 'spearman':
            scores[target] = spearmanr(y_unfitted[:, target], x_unfitted[:, 0])[0]
        elif score_type == 'RSS':
            scores[target] = -((y_unfitted[:, target] - x_unfitted[:, 0]) ** 2).sum()
            
    return scores


def scoring(y_true: np.ndarray, y_pred: np.ndarray, score_type: str='pearson') -> np.ndarray:
    """Returns one of three possible scores to be maximised"""
    
    n_targets = y_true.shape[1]
        
    if (y_pred.ndim == 2 and y_pred.shape[1] != 1 and n_targets == 1) or y_pred.ndim == 3:
        n_alphas = y_pred.shape[1]
    else:
        n_alphas = 0
        
    # Multioutput & evaluating alphas.
    if n_targets != 1 and n_alphas != 0:
        scores = np.empty((n_alphas, n_targets))
        for target in range(n_targets):
            for alpha in range(n_alphas):
                if score_type == 'pearson':
                    scores[alpha, target] = pearsonr(y_true[:, target], y_pred[:, alpha, target])[0]
                elif score_type == 'spearman':
                    scores[alpha, target] = spearmanr(y_true[:,target], y_pred[:, alpha, target])[0]
                elif score_type == 'RSS':
                    # Note: the RSS is converted by multiplying with -1 so that it is a
                    # score to be _maximised_ just as the other two options.
                    scores[alpha, target] = -((y_true[:, target] - y_pred[:, alpha, target]) ** 2).sum()
        
    # (Multioutput or Unioutput) & not evaluating alphas.
    elif n_alphas == 0:
        scores = np.empty(n_targets)
        for target in range(n_targets):
            if score_type == 'pearson':
                scores[target] = pearsonr(y_true[:, target], y_pred[:, target])[0]
            elif score_type == 'spearman':
                scores[target] = spearmanr(y_true[:, target], y_pred[:, target])[0]
            elif score_type == 'RSS':
                scores[target] = -((y_true[:, target] - y_pred[:, target]) ** 2).sum()
     
    # Unioutput & evaluating alphas.    
    elif n_targets == 1 and n_alphas != 0:
        scores = np.empty((n_alphas, n_targets))
        for alpha in range(n_alphas):
            if score_type == 'pearson':
                scores[alpha] = pearsonr(y_true[:, 0], y_pred[:, alpha])[0]
            elif score_type == 'spearman':
                scores[alpha] = spearmanr(y_true[:, 0], y_pred[:, alpha])[0]
            elif score_type == 'RSS':
                scores[alpha] = -((y_true[:, 0] - y_pred[:, alpha]) ** 2).sum()
            
    return scores
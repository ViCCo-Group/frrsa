#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 16:44:31 2020

@author: kaniuth
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr


def scoring_unfitted(y_unfitted, x_unfitted, score_type='pearson'):
    """Returns one of three possible scores to be maximised"""
    
    n_outputs = y_unfitted.shape[1]
    
    scores = np.empty(n_outputs)
     
    for output in range(n_outputs):
        if score_type == 'pearson':
            scores[output] = pearsonr(y_unfitted[:, output], x_unfitted[:, 0])[0]
        elif score_type == 'spearman':
            scores[output] = spearmanr(y_unfitted[:, output], x_unfitted[:, 0])[0]
        elif score_type == 'RSS':
            scores[output] = -((y_unfitted[:, output] - x_unfitted[:, 0]) ** 2).sum()
            
    return scores


def scoring(y_true, y_pred, score_type='pearson'):
    """Returns one of three possible scores to be maximised"""
    
    n_outputs = y_true.shape[1]
        
    if (y_pred.ndim == 2 and y_pred.shape[1] != 1 and n_outputs == 1) or y_pred.ndim == 3:
        n_alphas = y_pred.shape[1]
    else:
        n_alphas = 0
        
    # Multioutput & evaluating alphas.
    if n_outputs != 1 and n_alphas != 0:
        scores = np.empty((n_alphas, n_outputs))
        for output in range(n_outputs):
            for alpha in range(n_alphas):
                if score_type == 'pearson':
                    scores[alpha, output] = pearsonr(y_true[:, output], y_pred[:, alpha, output])[0]
                elif score_type == 'spearman':
                    scores[alpha, output] = spearmanr(y_true[:,output], y_pred[:, alpha, output])[0]
                elif score_type == 'RSS':
                    # Note: the RSS is converted by multiplying with -1 so that it is a
                    # score to be _maximised_ just as the other two options.
                    scores[alpha, output] = -((y_true[:, output] - y_pred[:, alpha, output]) ** 2).sum()
        
    # (Multioutput or Unioutput) & not evaluating alphas.
    elif n_alphas == 0:
        scores = np.empty(n_outputs)
        for output in range(n_outputs):
            if score_type == 'pearson':
                scores[output] = pearsonr(y_true[:, output], y_pred[:, output])[0]
            elif score_type == 'spearman':
                scores[output] = spearmanr(y_true[:, output], y_pred[:, output])[0]
            elif score_type == 'RSS':
                scores[output] = -((y_true[:, output] - y_pred[:, output]) ** 2).sum()
     
    # Unioutput & evaluating alphas.    
    elif n_outputs == 1 and n_alphas != 0:
        scores = np.empty((n_alphas, n_outputs))
        for alpha in range(n_alphas):
            if score_type == 'pearson':
                scores[alpha] = pearsonr(y_true[:, 0], y_pred[:, alpha])[0]
            elif score_type == 'spearman':
                scores[alpha] = spearmanr(y_true[:, 0], y_pred[:, alpha])[0]
            elif score_type == 'RSS':
                scores[alpha] = -((y_true[:, 0] - y_pred[:, alpha]) ** 2).sum()
            
    return scores
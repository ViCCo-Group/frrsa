#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spyder 4.2.5 | Python 3.8.8 64-bit | Qt 5.9.7 | PyQt5 5.9.2 | Darwin 18.7.0
"""
Contains functions that compute correlation between matrices.

These function judge the correspondence between two representational dis-
similarity matrices in different contexts of the `crossvalidation` module.

@author: Philipp Kaniuth (kaniuth@cbs.mpg.de)
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr


def scoring_classical(y_unfitted: np.ndarray, x_unfitted: np.ndarray,
                      score_type: str = 'pearson') -> np.ndarray:
    """Compute the correspondence between two vectors.

    For each column pair of `y_unfitted` one of several possible measures is
    computed to assess the correspondence with `x_unfitted`.

    Parameters
    ----------
    y_unfitted : ndarray
        Vectorized target RDM(s) across all conditions.
    x_unfitted : ndarray
        Vectorized predicting RDM across all conditions.
    score_type : {'pearson', 'spearman', 'RSS'}, optional
        Type of correspondence measure to compute (defaults to `pearson`).

    Returns
    -------
    scores : ndarray
        Association scores.
    """
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


def scoring(y_true: np.ndarray, y_pred: np.ndarray,
            score_type: str = 'pearson') -> np.ndarray:
    """Compute the correspondence between two vectors.

    For each pair of associated columns of `y_unfitted` and `x_unfitted` one
    of several possible correspondence measures is computed.

    Parameters
    ----------
    y_true : ndarray
        Vectorized target RDM(s) for the test set.
    y_pred : ndarray
        Vectorized reweighted predicting RDM(s) for the test set.
    score_type : {'pearson', 'spearman', 'RSS'}, optional
        Type of correspondence measure to compute (defaults to `pearson`).

    Returns
    -------
    scores : ndarray
        Association scores.
    """
    n_targets = y_true.shape[1]
    if (y_pred.ndim == 2 and y_pred.shape[1] != 1 and n_targets == 1) or y_pred.ndim == 3:
        n_hyperparams = y_pred.shape[1]
    else:
        n_hyperparams = 0

    # Multiple targets & evaluating hyperparams.
    if n_targets != 1 and n_hyperparams != 0:
        scores = np.empty((n_hyperparams, n_targets))
        for target in range(n_targets):
            for hyperparam in range(n_hyperparams):
                if score_type == 'pearson':
                    scores[hyperparam, target] = pearsonr(y_true[:, target],
                                                          y_pred[:, hyperparam, target])[0]
                elif score_type == 'spearman':
                    scores[hyperparam, target] = spearmanr(y_true[:, target],
                                                           y_pred[:, hyperparam, target])[0]
                elif score_type == 'RSS':
                    # Note: the RSS is converted by multiplying with -1 so that it is a
                    # score to be _maximised_ just as the other two options.
                    scores[hyperparam, target] = -((y_true[:, target] -
                                                    y_pred[:, hyperparam, target]) ** 2).sum()

    # (Multiple targets or one target) & not evaluating hyperparams.
    elif n_hyperparams == 0:
        scores = np.empty(n_targets)
        for target in range(n_targets):
            if score_type == 'pearson':
                scores[target] = pearsonr(y_true[:, target], y_pred[:, target])[0]
            elif score_type == 'spearman':
                scores[target] = spearmanr(y_true[:, target], y_pred[:, target])[0]
            elif score_type == 'RSS':
                scores[target] = -((y_true[:, target] - y_pred[:, target]) ** 2).sum()

    # One target & evaluating hyperparams.
    elif n_targets == 1 and n_hyperparams != 0:
        scores = np.empty((n_hyperparams, n_targets))
        for hyperparam in range(n_hyperparams):
            if score_type == 'pearson':
                scores[hyperparam] = pearsonr(y_true[:, 0], y_pred[:, hyperparam])[0]
            elif score_type == 'spearman':
                scores[hyperparam] = spearmanr(y_true[:, 0], y_pred[:, hyperparam])[0]
            elif score_type == 'RSS':
                scores[hyperparam] = -((y_true[:, 0] - y_pred[:, hyperparam]) ** 2).sum()
    return scores

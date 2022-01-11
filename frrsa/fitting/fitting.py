#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spyder 4.2.5 | Python 3.8.8 64-bit | Qt 5.9.7 | PyQt5 5.9.2 | Darwin 18.7.0 
"""
Contains wrapper functions for fitting regularized regression models (currently
only L2-regularization). Used in the `crossvalidation` module.

@author: Philipp Kaniuth (kaniuth@cbs.mpg.de)
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from pathlib import Path
import os
if ('dev' not in str(Path(os.getcwd()).parent)) and ('draco' not in str(Path(os.getcwd()).parent)) and ('cobra' not in str(Path(os.getcwd()).parent)):
    from fitting.fracridge import fracridge
else:
    from frrsa.frrsa.fitting.fracridge import fracridge

def count_targets(y):
    '''Compute number of target RDMs.
    
    Parameters
    ----------
    y : ndarray
        Data of all target RDMs.
    
    Returns
    -------
    number : int
        The number of target RDMs.
    '''
    if y.ndim==2:
        return y.shape[1]
    else: 
        return 1

def prepare_variables(X_train, y_train, X_test=None):
    '''Compute column-wise transformed versions of variables.
    
    Parameters
    ----------
    X_train : ndarray
        Training predictor matrix, the columns of which shall be z-transformed.
    y_train : ndarray
        Training target(s), the columns of which shall be centered.
    X_test : ndarray, optional
        Test predictor matrix, the columns of which shall be z-transformed.
    
    Returns
    -------
    X_train_z : ndarray
        Column-wise z-transformed version of `X_train`.
    z_score.mean_ : fitted scaler
        Column-wise mean of `X_train`. Only returned if `X_test` is none.
    z_score.scale_ : fitted scaler
        Column-wise std of `X_train`. Only returned if `X_test` is none.
    X_test_z : ndarray
        Column-wise z-transformed version of `X_test`. Only returned if `X_test` 
        is not none.
    y_train_c: ndarray
        Column-wise centered version of `y_train`.
    y_train_mean: ndarray
        Column-wise mean of `y_train`.
    '''
    z_score = StandardScaler(copy=False, with_mean=True, with_std=True)
    X_train_z = z_score.fit_transform(X_train)
    y_train_mean = np.mean(y_train, axis=0)
    y_train_c = y_train - y_train_mean
    if X_test is not None:
        # Scale X_test with _X_train_stds_ to get _nearly_ unstandardised
        # predictions. Note that X_train is standardised. Therefore, scaling
        # X_test with X_train_stds undoes the scaling of X_train.
        X_test_z = z_score.transform(X_test)
        return X_train_z, X_test_z, y_train_c, y_train_mean
    else:
        return X_train_z, z_score.mean_, z_score.scale_, y_train_c, y_train_mean

def find_hyperparameters(X_train, X_test, y_train, hyperparams, nonnegative, rng_state):
    '''Perform crossvalidated Ridge regression with each candidate hyperparameter.
    
    For each target in `y_train`, Ridge regression is fitted with each candidate
    hyperparameter. Then, predictions for each target for each candidate
    hyperparameter are computed and returned.
    
    Parameters
    ----------
    X_train : ndarray
        Training predictor matrix.
    X_test : ndarray
        Test predictor matrix.
    y_train: ndarray
        Training target(s).
    hyperparams : ndarray
        Candidate hyperparameters that shall be evaluated.
    nonnegative : bool
        Indication of whether the betas shall be constrained to be non-negative.
    rng_state : int
        State of the randomness. Should only be set for testing purposes.
    
    Returns
    -------
    y_predicted : ndarray
        Predictions for all targets and all candidate hyperparameters of shape
        (n_samples, n_hyperparams, n_targets).
    '''
    X_train_z, X_test_z, y_train_c, y_train_mean = prepare_variables(X_train,
                                                                     y_train,
                                                                     X_test)
    if not nonnegative:
        y_predicted, *_ = fracridge(X_train_z, y_train_c, X_test_z, hyperparams)
    else:
        n_targets = count_targets(y_train_c)
        n_hyperparams = len(hyperparams)
        n_samples = X_test_z.shape[0]
        y_predicted = np.zeros((n_samples,n_hyperparams,n_targets))
        for idx,hyperparam in enumerate(hyperparams):
            model = Ridge(alpha=hyperparam, fit_intercept=False, tol=0.001,
                          solver='lbfgs', positive=True, random_state=rng_state)
            model.fit(X_train_z, y_train_c)            
            y_predicted[:,idx,:] = model.predict(X_test_z)
    y_predicted += y_train_mean
    return y_predicted

def regularized_model(X_train, X_test, y_train, y_test, hyperparams, nonnegative, rng_state):
    '''Perform crossvalidated Ridge regression for each target
    
    For each target in `y_train`, Ridge Regression is fitted with each target's
    optimal hyperparameter. Then, predictions for each target are computed.
    
    Parameters
    ----------
    X_train : ndarray
        Training predictor matrix.
    X_test : ndarray
        Test predictor matrix.
    y_train: ndarray
        Training target(s).
    y_test: ndarray
        Test target(s).
    hyperparams : ndarray
        Best hyperparameter for each target in `y_train`.
    nonnegative : bool
        Indication of whether the betas shall be constrained to be non-negative.
    rng_state : int
        State of the randomness. Should only be set for testing purposes.
        
    Returns
    -------
    y_predicted : ndarray
        Predictions for all targets of shape (n_samples, n_targets).
    '''
    X_train_z, X_test_z, y_train_c, y_train_mean = prepare_variables(X_train,
                                                                     y_train,
                                                                     X_test)
    if not nonnegative:
        n_targets = count_targets(y_train_c)
        y_predicted = np.zeros((y_test.shape[0],n_targets))
        unique_hyperparams = np.unique(hyperparams)
        for hyperparam in unique_hyperparams:
            idx = np.where(hyperparams==hyperparam)[0]
            n_current_targets = len(idx)
            y_train_current = y_train_c[:, idx]
            y_pred_current, *_ = fracridge(X_train_z, y_train_current, X_test_z, hyperparam)
            y_predicted[:, idx] = y_pred_current.reshape(-1,n_current_targets)
    else:
        model = Ridge(alpha=hyperparams, fit_intercept=False, tol=0.001,
                      solver='lbfgs', positive=True, random_state=rng_state)
        model.fit(X_train_z, y_train_c)            
        y_predicted = model.predict(X_test_z)
    y_predicted += y_train_mean
    return y_predicted

def final_model(X, y, hyperparams, nonnegative, rng_state):
    '''Perform Ridge Regression on the whole dataset for each target.
    
    For each target in `y`, Ridge regression is fitted to the whole dataset using
    each target's best hyperparameter. Then, betas for each target's
    measurement channel are computed and returned.
    
    Parameters
    ----------
    X : ndarray
        Predictor matrix of the whole data set.
    y: ndarray
        Target(s) of the whole data set.
    hyperparams : ndarray
        Best hyperparameter for each target in `y`.
    nonnegative : bool
        Indication of whether the betas shall be constrained to be non-negative.
    rng_state : int
        State of the randomness. Should only be set for testing purposes.
    
    Returns
    -------
    beta_unstandardized : ndarray
        Beta weight for each predictor's measurement channel plus intercept of
        shape (n_channels+1, n_targets).
    '''
    X_z, X_means, X_stds, y_c, y_mean = prepare_variables(X, y)
    n_targets = count_targets(y)
    if not nonnegative:
        beta_standardized = np.zeros((X_z.shape[1],n_targets))
        hyperparams = hyperparams.to_numpy()
        unique_hyperparams = np.unique(hyperparams)
        for hyperparam in unique_hyperparams:
            idx = np.where(hyperparams==hyperparam)[0]
            n_current_targets = len(idx)
            y_current = y_c[:, idx]
            _, beta_standardized_current, _ = fracridge(X_z, y_current, 
                                                        fracs=hyperparam,
                                                        betas_wanted=True,
                                                        pred_wanted=False)
            beta_standardized[:, idx] = beta_standardized_current.reshape(-1,n_current_targets)
    else:
        model = Ridge(alpha=hyperparams, fit_intercept=False, tol=0.001,
                      solver='lbfgs', positive=True, random_state=rng_state)
        model.fit(X_z, y_c)
        beta_standardized = model.coef_.T
    beta_unstandardized = beta_standardized.T / X_stds 
    intercept = y_mean.reshape(n_targets,1) - (beta_unstandardized @ X_means.reshape(-1,1))
    beta_unstandardized = beta_unstandardized.T
    beta_unstandardized = np.concatenate((intercept.T, beta_unstandardized), axis=0)
    return beta_unstandardized

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spyder 4.2.5 | Python 3.8.8 64-bit | Qt 5.9.7 | PyQt5 5.9.2 | Darwin 18.7.0 
"""
Contains wrapper functions for fitting regularized regression models (currently
only L2-regularization in the form of Fraction Ridge Regression) in different
contexts of the `crossvalidation` module.

@author: Philipp Kaniuth (kaniuth@cbs.mpg.de)
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

#TODO: remove the following imports and conditionals before publicising repo.
from pathlib import Path
import os
if ('dev' not in str(Path(os.getcwd()).parent)) and ('draco' not in str(Path(os.getcwd()).parent)) and ('cobra' not in str(Path(os.getcwd()).parent)):
    from fitting.fracridge import fracridge
else:
    from frrsa.frrsa.fitting.fracridge import fracridge

z_score = StandardScaler(copy=False, with_mean=True, with_std=True)

def count_outputs(y):
    '''Compute amount of separate target variables.
    
    Parameters
    ----------
    y : ndarray
        Data for all target variables.
    
    Returns
    -------
    amount : int
        The amount of separate target variables.
    '''
    if y.ndim==2:
        return y.shape[1]
    else: 
        return 1

def prepare_variables(X_train, X_test, y_train):
    '''Compute column-wise transformed versions of variables.
    
    Parameters
    ----------
    X_train : ndarray
        Data, the columns of which shall be z-transformed.
    X_test : ndarray
        Data, the columns of which shall be z-transformed.
    y_train : ndarray
        Data, the columns of which shall be centered.
    
    Returns
    -------
    X_train_z : ndarray
        Column-wise z-transformed version of `X_train`.
    X_test_z : ndarray
        Column-wise z-transformed version of `X_test`.
    y_train_z: ndarray
        Column-wise centered version of `y_train`.
    y_train_mean: ndarray
        Mean of every column of `y_train`.
    '''
    X_train_z = z_score.fit_transform(X_train)
    # Scale X_test with _X_train_stds_ to get _nearly_ unstandardised
    # predictions. Note that X_train is standardised. Therefore, scaling
    # X_test with X_train_stds undoes the scaling of X_train.
    X_test_z = z_score.transform(X_test)
    y_train_mean = np.mean(y_train, axis=0)
    y_train_z = y_train - y_train_mean
    return X_train_z, X_test_z, y_train_z, y_train_mean

def find_hyperparameters(X_train, X_test, y_train, hyperparams, nonnegative, rng_state):
    '''Perform crossvalidated Ridge regression using each candidate hyperparameter.
    
    For each target in `y_train`, Ridge Regression is fitted for each candidate
    hyperparameter. Then, predictions for each target for each candidate
    hyperparameter are computed.
    
    Parameters
    ----------
    X_train : ndarray
        Predictor matrix of the training data.
    X_test : ndarray
        Predictor matrix of the test data.
    y_train: ndarray
        Target(s) of the training data.
    hyperparams : ndarray
        Candidate hyperparameters.
    nonnegative : bool
        Indication of whether the betas shall be constrained to be non-negative.
    rng_state : int
        State of the randomness in the system. Should only
        be set for testing purposes, will be deprecated in release-version.  
    
    Returns
    -------
    y_predicted : ndarray
        Predictions for all targets and all candidate hyperparameters.
    '''
    X_train, X_test_z, y_train, y_train_mean = prepare_variables(X_train, 
                                                                 X_test, 
                                                                 y_train)
    if not nonnegative:
        y_predicted, *_ = fracridge(X_train, y_train, X_test_z, hyperparams)
    else:
        n_targets = count_outputs(y_train)
        n_hyperparams = len(hyperparams)
        n_samples = X_test_z.shape[0]
        y_predicted = np.zeros((n_samples,n_hyperparams,n_targets))
        for idx,hyperparam in enumerate(hyperparams):
            model = Ridge(alpha=hyperparam, fit_intercept=False, 
                          tol=0.001, solver='lbfgs', 
                          positive=True, random_state=rng_state)
            model.fit(X_train, y_train)            
            y_predicted[:,idx,:] = model.predict(X_test_z)
    y_predicted += y_train_mean
    return y_predicted

def regularized_model(X_train, X_test, y_train, y_test, hyperparams, nonnegative, rng_state):
    '''Perform crossvalidated Ridge regression for each target
    
    For each target in `y_train`, Ridge Regression is fitted using each target's
    best hyperparameter. Then, predictions for each target are computed.
    
    Parameters
    ----------
    X_train : ndarray
        Predictor matrix of the training data.
    X_test : ndarray
        Predictor matrix of the test data.
    y_train: ndarray
        Target(s) of the training data.
    y_test: ndarray
        Target(s) of the test data.
    hyperparams : ndarray
        Hyperparameter for each target in `y_train`.
    nonnegative : bool
        Indication of whether the betas shall be constrained to be non-negative.
    rng_state : int
        State of the randomness in the system. Should only
        be set for testing purposes, will be deprecated in release-version.  
        
    Returns
    -------
    y_predicted : ndarray
        Predictions for all targets.
    '''
    X_train, X_test_z, y_train, y_train_mean = prepare_variables(X_train, 
                                                                 X_test, 
                                                                 y_train)
    if not nonnegative:
        n_outputs = count_outputs(y_train)
        y_predicted = np.zeros((y_test.shape[0],n_outputs))
        unique_hyperparams = np.unique(hyperparams)
        for hyperparam in unique_hyperparams:
            idx = np.where(hyperparams==hyperparam)[0]
            n_current_outputs = len(idx)
            y_train_current = y_train[:, idx]
            y_pred_current, *_ = fracridge(X_train, y_train_current, X_test_z, hyperparam)
            y_predicted[:, idx] = y_pred_current.reshape(-1,n_current_outputs)
    else:
        model = Ridge(alpha=hyperparams, fit_intercept=False, 
                      tol=0.001, solver='lbfgs', 
                      positive=True, random_state=rng_state)
        model.fit(X_train, y_train)            
        y_predicted = model.predict(X_test)
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
        Hyperparameter for each target in `y`.
    nonnegative : bool
        Indication of whether the betas shall be constrained to be non-negative.
    rng_state : int
        State of the randomness in the system. Should only
        be set for testing purposes, will be deprecated in release-version.
    
    Returns
    -------
    beta_unstandardized : ndarray
        Weight for each target's measurement channel.
    '''
    X = z_score.fit_transform(X)
    X_means = z_score.mean_.reshape(-1,1)
    X_stds = z_score.scale_
    y_mean = np.mean(y, axis=0)
    y = y - y_mean
    n_outputs = count_outputs(y)
    
    if not nonnegative:
        beta_standardized = np.zeros((X.shape[1],n_outputs))
        hyperparams = hyperparams.to_numpy()
        unique_hyperparams = np.unique(hyperparams)
        for hyperparam in unique_hyperparams:
            idx = np.where(hyperparams==hyperparam)[0]
            n_current_outputs = len(idx)
            y_current = y[:, idx]
            _, beta_standardized_current, _ = fracridge(X, y_current, fracs=hyperparam,
                                                        betas_wanted=True,
                                                        pred_wanted=False)
            beta_standardized[:, idx] = beta_standardized_current.reshape(-1,n_current_outputs)
    else:
        model = Ridge(alpha=hyperparams, fit_intercept=False, 
                      tol=0.001, solver='lbfgs', 
                      positive=True, random_state=rng_state)
        model.fit(X, y)            
        beta_standardized = model.coef_.T
        
    beta_unstandardized = beta_standardized.T / X_stds 
    intercept = y_mean.reshape(n_outputs,1) - (beta_unstandardized @ X_means)
    beta_unstandardized = beta_unstandardized.T
    beta_unstandardized = np.concatenate((intercept.T, beta_unstandardized), axis=0)
    return beta_unstandardized
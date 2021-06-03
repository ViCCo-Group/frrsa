#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Philipp Kaniuth (kaniuth@cbs.mpg.de)
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

#TODO: remove the following imports and conditionals before publicising repo.
from pathlib import Path
import os
if ('dev' not in str(Path(os.getcwd()).parent)) and ('draco' not in str(Path(os.getcwd()).parent)) and ('cobra' not in str(Path(os.getcwd()).parent)):
    from fitting.fracridge import fracridge
else:
    from frrsa.frrsa.fitting.fracridge import fracridge

z_score = StandardScaler(copy=True, with_mean=True, with_std=True)

def count_outputs(y):
    """Returns number of separate output variables"""
    if y.ndim==2:
        return y.shape[1]
    else: 
        return 1

def prepare_variables(X_train, X_test, y_train):
    """"Z-transforms X_train and X_test using X_train and centers y_train"""
    X_train_z = z_score.fit_transform(X_train)
    # Scale X_test with _X_train_stds_ to get _nearly_ unstandardised predictions.
    X_test_z = z_score.transform(X_test)
    y_train_mean = np.mean(y_train, axis=0)
    y_train = y_train - y_train_mean
    return X_train_z, X_test_z, y_train, y_train_mean

def baseline_model(x_train, x_test, y_train):
    #TODO: deprecate.
    """Cross-validates simple linear regression for two dissimilarity vectors"""
    model = LinearRegression(fit_intercept=True,
                             normalize=False,
                             copy_X=False,
                             n_jobs=None)
    model.fit(x_train, y_train)
    y_predicted = model.predict(x_test)
    return y_predicted

def find_hyperparameters(X_train, X_test, y_train, y_test, fracs):
    """"Performs crossvalidated fracridge for possibly several outputs using all candidate hyperparameters for each output"""
    X_train, X_test_z, y_train, y_train_mean = prepare_variables(X_train, X_test, y_train)
    y_predicted, *_ = fracridge(X_train, y_train, X_test_z, fracs)
    y_predicted += y_train_mean
    return y_predicted

def regularized_model(X_train, X_test, y_train, y_test, fracs):
    """Performs crossvalidated fracridge for possibly several outputs using each output's best hyperparameter"""
    X_train, X_test_z, y_train, y_train_mean = prepare_variables(X_train, X_test, y_train)
    n_outputs = count_outputs(y_train)
    y_predicted = np.zeros((y_test.shape[0],n_outputs))
    unique_fracs = np.unique(fracs)
    for frac in unique_fracs:
        idx = np.where(fracs==frac)[0]
        n_current_outputs = len(idx)
        y_train_current = y_train[:, idx]
        y_pred_current, *_ = fracridge(X_train, y_train_current, X_test_z, frac)
        y_predicted[:, idx] = y_pred_current.reshape(-1,n_current_outputs)
    # To have _fully_ undstandardised predictions, one needs to add y_train to y_predicted.
    y_predicted += y_train_mean
    return y_predicted

def final_model(X, y, fracs):
    """Performs fracridge on the whole dataset for possibly several outputs using each output's best hyperparameter"""
    X = z_score.fit_transform(X)
    X_means = z_score.mean_.reshape(-1,1)
    X_stds = z_score.scale_
    y_mean = np.mean(y, axis=0)
    y = y - y_mean
    n_outputs = count_outputs(y)
    beta_standardized = np.zeros((X.shape[1],n_outputs))
    fracs = fracs.to_numpy()
    unique_fracs = np.unique(fracs)
    for frac in unique_fracs:
        idx = np.where(fracs==frac)[0]
        n_current_outputs = len(idx)
        y_current = y[:, idx]
        _, beta_standardized_current, _ = fracridge(X, y_current, fracs=frac, betas_wanted=True, pred_wanted=False)
        beta_standardized[:, idx] = beta_standardized_current.reshape(-1,n_current_outputs)
    beta_unstandardized = beta_standardized.T / X_stds 
    intercept = y_mean.reshape(n_outputs,1) - (beta_unstandardized @ X_means)
    beta_unstandardized = beta_unstandardized.T
    beta_unstandardized = np.concatenate((intercept.T, beta_unstandardized), axis=0)
    return beta_unstandardized
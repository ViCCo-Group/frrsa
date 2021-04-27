#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 16:47:07 2020

@author: kaniuth
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

def baseline_model(X_train, X_test, y_train):
    model = LinearRegression(fit_intercept=True,
                             normalize=False,
                             copy_X=False,
                             n_jobs=None)
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    return y_predicted

def find_hyperparameters(X_train, X_test, y_train, y_test, fracs):
    """"Fits an RDM to another RDM and returns scores, predictions, and parameters"""
    X_train, X_test_z, y_train, y_train_mean = prepare_variables(X_train, X_test, y_train)
    y_predicted, *_ = fracridge(X_train, y_train, X_test_z, fracs)
    y_predicted += y_train_mean
    return y_predicted

def regularized_model(X_train, X_test, y_train, y_test, fracs):    
    X_train, X_test_z, y_train, y_train_mean = prepare_variables(X_train, X_test, y_train)
    n_outputs = count_outputs(y_train)
    y_predicted = np.zeros((y_test.shape[0],n_outputs))
    best_frac_uni = np.unique(fracs)      
    for frac in best_frac_uni:
        frac_indx = np.where(fracs==frac)[0]
        n_current_outputs = len(frac_indx)
        y_train_current = y_train[:, frac_indx]
        y_pred_current, *_ = fracridge(X_train, y_train_current, X_test_z, frac)
        y_predicted[:, frac_indx] = y_pred_current.reshape(-1,n_current_outputs)
    # To have _fully_ undstandardised predictions, one needs to add y_train to y_predicted.
    y_predicted += y_train_mean
    return y_predicted

def final_model(X, y, fracs, betas_wanted=False, pred_wanted=True):    
    X = z_score.fit_transform(X)
    X_means = z_score.mean_.reshape(-1,1)
    X_stds = z_score.scale_
    y_mean = np.mean(y, axis=0)
    y = y - y_mean
    
    _, beta_standardized, _ = fracridge(X, y, X_test=None, fracs=fracs, tol=1e-10, jit=True, betas_wanted=True, pred_wanted=False)
    
    beta_unstandardized = beta_standardized.T / X_stds 
    n_outputs = count_outputs(y)
    intercept = y_mean.reshape(n_outputs,1) - (beta_unstandardized @ X_means)
    beta_unstandardized = beta_unstandardized.T
    beta_unstandardized = np.concatenate((intercept.T, beta_unstandardized), axis=0)
    return beta_unstandardized






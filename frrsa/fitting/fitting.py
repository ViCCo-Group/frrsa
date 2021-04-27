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

z_scale = StandardScaler(copy=True, with_mean=True, with_std=True)


def prepare_variables(X_train, X_test, y_train, y_test):
    """"Fits an RDM to another RDM and returns predictions and parameters"""
    try:
        n_test = y_test.shape[0]
        n_outputs = y_train.shape[1]
    except IndexError:
        n_outputs = 1

    y_predicted = np.zeros((n_test,n_outputs))

    X_train = z_scale.fit_transform(X_train)
    X_train_means = z_scale.mean_.reshape(-1,1)
    X_train_stds = z_scale.scale_
    y_train_mean = np.mean(y_train, axis=0)
    y_train = y_train - y_train_mean

    # Scale X_test with _X_train_stds_ to get _nearly_ unstandardised predictions.
    X_test_z = (X_test - X_train_means.T) / X_train_stds

    return X_train, X_train_means, X_train_stds, X_test_z, y_train, y_train_mean, n_outputs, y_predicted


def baseline_model(X_train, X_test, y_train):
        
    model = LinearRegression(fit_intercept=True,
                            normalize=False,
                            copy_X=False,
                            n_jobs=None)
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
        
    return y_predicted


def regularized_model(X_train, X_test, y_train, y_test, fracs):    

    X_train, X_train_means, X_train_stds, X_test_z, y_train, y_train_mean, n_outputs, y_predicted = prepare_variables(X_train, X_test, y_train, y_test)
    
    best_frac_uni = np.unique(fracs)      
    
    for frac in best_frac_uni:
        frac_indx = np.where(fracs==frac)[0]
        n_current_outputs = len(frac_indx)
        y_train_current = y_train[:, frac_indx]
        y_pred_current, beta_stand_current, evaluated_alphas = fracridge(X_train, y_train_current, X_test_z, frac)
        y_predicted[:, frac_indx] = y_pred_current.reshape(-1,n_current_outputs)
        
    # To have _fully_ undstandardised predictions, one needs to add y_train to y_predicted.
    y_predicted += y_train_mean
    
    return y_predicted


def find_hyperparameters(X_train, X_test, y_train, y_test, fracs):
    """"Fits an RDM to another RDM and returns scores, predictions, and parameters"""

    X_train, X_train_means, X_train_stds, X_test_z, y_train, y_train_mean, n_outputs, y_predicted = prepare_variables(X_train, X_test, y_train, y_test)

    y_predicted, beta_unstandardized, evaluated_alphas = fracridge(X_train, y_train, X_test_z, fracs)
    
    y_predicted += y_train_mean
    
    return y_predicted









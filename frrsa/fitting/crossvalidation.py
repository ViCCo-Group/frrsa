#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spyder 4.1.4, Python 3.7.6 64-bit | Qt 5.9.6 | PyQt5 5.9.2 | Darwin 18.7.0
"""
@author: Philipp Kaniuth (kaniuth@cbs.mpg.de)
"""


#%% Import packages.
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#TODO: remove the following imports and conditionals before publicising repo.
from pathlib import Path
import os
print(str(Path(os.getcwd())))
if 'dev' not in str(Path(os.getcwd()).parent):
    from helper.classical_RSA import flatten_RDM, make_RDM
    from helper.data_splitter import data_splitter
    from helper.hadamard import hadamard_products
    from fitting.scoring import scoring, scoring_unfitted
    from fitting.fitting import baseline_model, regularized_model, find_hyperparameters
else:
    from frrsa.frrsa.helper.classical_RSA import flatten_RDM, make_RDM
    from frrsa.frrsa.helper.data_splitter import data_splitter
    from frrsa.frrsa.helper.hadamard import hadamard_products
    from frrsa.frrsa.fitting.scoring import scoring, scoring_unfitted
    from frrsa.frrsa.fitting.fitting import baseline_model, regularized_model, find_hyperparameters
# import matplotlib as mpl
# Suppress printing figures to a display.
# mpl.use('Agg')

# Set global variables.
z_scale = StandardScaler(copy=True, with_mean=True, with_std=True)


#%%
#"inputs": channel*condition array.
#"output": condition*condition array.

def frrsa(output, \
          inputs, \
          outer_k=5, \
          outer_reps=10, \
          splitter='random', \
          hyperparams=None, \
          score_type='pearsonr', \
          sanity=False, \
          rng_state=None):
    """ Implements a nested cross-validation, where in each CV, RDMs are fitted."""
#%%    
    if hyperparams is None:
        hyperparams = np.linspace(.05, 1, 20) # following recommendation by Rokem & Kay (2020).

    if not hasattr(hyperparams, "__len__"):
        hyperparams = [hyperparams]
    hyperparams = np.array(hyperparams)
            
    try: 
        n_conditions = output.shape[1]
        n_outputs = output.shape[2]
    except IndexError:
        n_outputs = 1

    predicted_RDM, predicted_RDM_counter = preallocate_predictions(n_conditions, n_outputs)
    y_unfitted = flatten_RDM(output)
    x_unfitted = flatten_RDM(make_RDM(inputs))
    inputs_z = z_scale.fit_transform(inputs)
 
    # Unfitted scores are non-crossvalidated scores between the complete 
    # y_unfitted and x_unfitted, i.e., a classical RSA.
    unfitted_scores = {}
    key_list = ['pearsonr', 'spearmanr', 'RSS']
    for key in key_list:
        unfitted_scores[key] = scoring_unfitted(y_unfitted, x_unfitted, key)
        
        
#%%    
    predicted_RDM_re, predictions, crossval, betas \
                    = start_cross_validation(splitter,
                                             outer_k,
                                             outer_reps,
                                             rng_state,
                                             n_conditions, 
                                             output,
                                             inputs_z,
                                             score_type,
                                             predicted_RDM,
                                             predicted_RDM_counter,
                                             hyperparams,
                                             n_outputs)
            
    # Compute correlation between fitted RDMs and output RDM.
    flattend_pred_RDM = flatten_RDM(predicted_RDM_re)
    
    del_ind = np.where(flattend_pred_RDM[:,0]==9999)[0]
    
    flattend_pred_RDM_del = np.delete(flattend_pred_RDM, del_ind, axis=0)
    y_unfitted_del = np.delete(y_unfitted, del_ind, axis=0)

    fitted_scores = scoring(flattend_pred_RDM_del, y_unfitted_del)

   
    return predicted_RDM_re, predictions, unfitted_scores, crossval, betas, fitted_scores


#%%
def start_cross_validation(splitter,
                           outer_k,
                           outer_reps,
                           rng_state,
                           n_conditions,
                           output,
                           inputs_z,
                           score_type,
                           predicted_RDM,
                           predicted_RDM_counter,
                           hyperparams,
                           n_outputs):

    n_outer_cvs = outer_k * outer_reps
    n_predictors = inputs_z.shape[0] + 1 # +1 because intercept.
    hyperparams_enlarged = hyperparams.reshape(-1,1)
    hyperparams_enlarged = np.repeat(hyperparams_enlarged, repeats=n_outputs, axis=1)
    n_hyperparams = len(hyperparams)
    betas = np.empty((n_predictors*n_outer_cvs, n_outputs+1))  # +1 because for counter of outer CV.
    
    # Pre-allocate empty arrayes in which, for each outer fold, the best hyperparamter,
    # model scores, and a fold-counter will be saved, for each output.
    n_models = 2 # baseline and regularized model.
    score = np.empty(n_outer_cvs * n_outputs * n_models)
    model_type = np.empty(n_outer_cvs * n_outputs * n_models)
    fold = np.empty(n_outer_cvs * n_outputs * n_models)
    hyperparameter = np.empty(n_outer_cvs * n_outputs * n_models)
    
    # Pre-allocate an empty array in which all predictions of the fitted and
    # baseline model, the resepective y_test, factors denoting fold and output
    # and the pairs to which predictions belong will be saved.
    predictions = np.zeros((1,7))

    # Set up outer cross-validation.
    outer_cv = data_splitter(splitter, outer_k, outer_reps, random_state=rng_state)
    outer_loop_count = -1
    list_of_indices = list(range(n_conditions))

    for outer_train_indices, outer_test_indices in outer_cv.split(list_of_indices):

        outer_loop_count += 1

        # Set up inner cross-validation.
        inner_k, inner_reps = 5, 5
        inner_cv = data_splitter(splitter, inner_k, inner_reps, random_state=rng_state)
        inner_loop_count = -1

        # Preallocate an empty array in which the model-scores for each hyperparamter
        # for each inner cross-validation will be put.
        inner_cvs = inner_k * inner_reps
        inner_hyperparams_scores = np.empty((n_hyperparams, n_outputs, inner_cvs))

        # NOTE: In the following loop, rkf.split is applied to the outer_train_indices!
        for inner_train_indices, inner_test_indices in inner_cv.split(outer_train_indices):

            inner_loop_count += 1
            train_idx, test_idx = outer_train_indices[inner_train_indices], outer_train_indices[inner_test_indices]
            
            score_in = fit_and_score_in(inputs_z, output, train_idx, test_idx, score_type, hyperparams)
                        
            inner_hyperparams_scores[:, :, inner_loop_count] = score_in

        # Evalute which hyperparamter(s) is(are) the best for the current
        # outer fold.
        inner_hyperparams_scores_avgs = np.mean(inner_hyperparams_scores, axis=2)
        best_hyperparam_index = np.where(inner_hyperparams_scores_avgs == \
                                         np.amax(inner_hyperparams_scores_avgs, \
                                                 axis=0))
        output_order = best_hyperparam_index[1]
        best_hyperparam = hyperparams_enlarged[best_hyperparam_index][output_order]
        # Note: "best_hyperparam" gives the best fraction for each output, 
        # sorted in the same order as the outputs were supplied in "outputs".

        # Compute y_train and y_test for baseline and regularised model.
        y_train, y_test = vectorise_rdm_to_train_and_test(outer_train_indices, outer_test_indices, output)

        # Fit and score baseline model.
        start_idx = outer_loop_count * n_outputs * n_models
        score[start_idx:start_idx+n_outputs], y_baseline = fit_and_score_baseline(
                                                                    inputs_z, \
                                                                    y_train, \
                                                                    y_test, \
                                                                    outer_train_indices, \
                                                                    outer_test_indices, \
                                                                    score_type)
        model_type[start_idx:start_idx+n_outputs] = 1 # will be a label later.
        fold[start_idx:start_idx+n_outputs] = outer_loop_count+1
        hyperparameter[start_idx:start_idx+n_outputs] = None
            
            
        # Fit and score regularised model.
        start_idx += n_outputs
        score[start_idx:start_idx+n_outputs], \
        first_pair_idx, \
        second_pair_idx, \
        y_regularized, \
        beta_unstandardized = fit_and_score_out(inputs_z, y_train, y_test, \
                                                outer_train_indices, outer_test_indices, \
                                                score_type, best_hyperparam)
        model_type[start_idx:start_idx+n_outputs] = 2
        fold[start_idx:start_idx+n_outputs] = outer_loop_count+1
        hyperparameter[start_idx:start_idx+n_outputs] = best_hyperparam
        
        # Put each predicted dissimilarity in a dissimilarity matrix in which
        # all predictions of all outer CVs are collected.
        first_pair_obj, second_pair_obj = outer_test_indices[first_pair_idx], \
                                          outer_test_indices[second_pair_idx]
        predicted_RDM[first_pair_obj, second_pair_obj, :] += y_regularized
        predicted_RDM_counter[first_pair_obj, second_pair_obj, :] += 1
        
        
        # Save all betas of the regularized model next to a counter for the outer loop.
        row_start = outer_loop_count * n_predictors
        row_end = (outer_loop_count + 1) * n_predictors
        betas[row_start:row_end, :-1] = beta_unstandardized
        betas[row_start:row_end, -1] = outer_loop_count+1
        
        # Save all predictions of the current outer CV with extra info.
        # Note: 'n' is a numerical variable; if n_outputs > 1 it either denotes
            # participants (if output kind is MRI or behavior) or time points (MEG),
            # or DNN layer.
        n = np.empty((y_test.shape))
        n[:,:] = list(range(n_outputs))
        n = n.reshape(len(n)*n_outputs, order='F')+1

        y_test = y_test.reshape(len(y_test)*n_outputs, order='F') #make all ys 1D.
        y_regularized = y_regularized.reshape(len(y_regularized)*n_outputs, order='F')
        y_baseline = y_baseline.reshape(len(y_baseline)*n_outputs, order='F')

        first_pair_obj = np.tile(first_pair_obj, n_outputs)
        second_pair_obj = np.tile(second_pair_obj, n_outputs)

        fold_pred = np.array([outer_loop_count+1] * len(y_test))
        current_predictions = np.array([y_test, y_regularized, y_baseline, n, fold_pred, first_pair_obj, second_pair_obj]).T
        predictions = np.concatenate((predictions, current_predictions), axis=0)

        print('Finished outer loop number: ' + str(outer_loop_count + 1))
        
        
    # 'n' is exactly defined as above; needs to be reassigned because it needs
    # a different structure for 'scores' than for 'current_predictions'.
    n = np.array(list(range(n_outputs)) * n_outer_cvs * n_models)+1
    crossval = pd.DataFrame(data=np.array([score, model_type, fold, hyperparameter, n]).T, \
                              columns=['score', 'model_type', 'fold', 'hyperparameter', 'n'])
    crossval.replace(to_replace={'model_type': {1: 'base', 2: 'fitted'}}, inplace=True)

    predictions = pd.DataFrame(data=np.delete(predictions, 0, 0), \
                               columns=['y_test', 'y_regularized', 'y_baseline', 'n', 'fold', 'first_obj', 'second_obj'])
    
    betas = pd.DataFrame(data=betas, \
                         columns=['betas_n_{0}'.format(i+1) for i in range(n_outputs)] + ['fold'])
    
    # Collapse the RDMs along the diagonal, sum & divide, put pack in array.
    predicted_RDM_re = collapse_RDM(n_conditions, n_outputs, predicted_RDM, predicted_RDM_counter)

    return predicted_RDM_re, predictions, crossval, betas



#%%
def collapse_RDM(n_conditions, n_outputs, predicted_RDM, predicted_RDM_counter):
    '''Collapse the RDMs along the diagonal, sum & divide, put pack in array.'''
    idx_low = np.tril_indices(n_conditions, k=-1)
    idx_up = tuple([idx_low[1], idx_low[0]])
    sum_of_preds_halves = predicted_RDM[idx_up] + predicted_RDM[idx_low]
    sum_of_count_halves = predicted_RDM_counter[idx_up] + predicted_RDM_counter[idx_low]
    with np.errstate(divide='ignore', invalid='ignore'):
        average_preds = sum_of_preds_halves / sum_of_count_halves
    np.nan_to_num(average_preds, copy=False, nan=9999)
    predicted_RDM_re = np.zeros((predicted_RDM.shape))
    for var in range(n_outputs):
        predicted_RDM_re[idx_low[0], idx_low[1], var] = average_preds[:,var]
        predicted_RDM_re[:,:,var] = predicted_RDM_re[:,:,var] + predicted_RDM_re[:,:,var].T
        #TODO: get rid of the loop and make this implicitly parallel, maybe using np.transpose()
    return predicted_RDM_re

#%%
def compute_hadamard_and_transpose(inputs_z, train_indices, test_indices):
    '''Compute Hadamard products for train and test set.'''
    
    X_fitted_train, discard, discard = hadamard_products(inputs_z[:, train_indices])
    X_fitted_test, first_pair_idx, second_pair_idx = hadamard_products(inputs_z[:, test_indices])

    X_fitted_train = 1 - X_fitted_train.transpose()
    X_fitted_test = 1 - X_fitted_test.transpose()

    return X_fitted_train, X_fitted_test, first_pair_idx, second_pair_idx

#%%
def vectorise_rdm_to_train_and_test(train_indices, test_indices, output):
    """ Create vectorized RDMs for train and test sets. """
    ixgrid = np.ix_(train_indices, train_indices)
    y_train = flatten_RDM(output[ixgrid])
    ixgrid = np.ix_(test_indices, test_indices)
    y_test = flatten_RDM(output[ixgrid])
    
    return y_train, y_test

#%%
def fit_and_score_in(inputs_z, output, train_idx, test_idx, score_type, hyperparams):
    """ Fit model, find hyperparamter value and score in the inner cross validation."""
    
    X_train, X_test, discard, discard = compute_hadamard_and_transpose(inputs_z, train_idx, test_idx)

    y_train, y_test = vectorise_rdm_to_train_and_test(train_idx, test_idx, output)
                
    # Fit model for each candidate hyperparamter and get its score.
    y_pred = find_hyperparameters(X_train, X_test, y_train, y_test, hyperparams)
    
    score = scoring(y_test, y_pred, score_type=score_type)

    return score

#%%
def fit_and_score_out(inputs_z, y_train, y_test, train_idx, test_idx, score_type, hyperparams):
    """ Fit model and get its score in outer cross validation."""

    X_train, X_test, first_pair_idx, second_pair_idx = compute_hadamard_and_transpose(inputs_z, train_idx, test_idx)
    
    # Fit model and get predictions and parameters.
    y_pred, beta_unstandardized = regularized_model(X_train, \
                                                    X_test, \
                                                    y_train, \
                                                    y_test, \
                                                    hyperparams)
    # Based on predictions and test data, evaluate fit.
    score = scoring(y_test, y_pred, score_type=score_type)
    
    return score, first_pair_idx, second_pair_idx, y_pred, beta_unstandardized

#%%
def fit_and_score_baseline(inputs_z, y_train, y_test, train_idx, test_idx, score_type):
    """ Fit model and get its score in outer cross validation."""
        # Fit and score baseline model is a simple linear regression model.  Here, X is a
        # vector with the same dimensions as y.  One beta is estimated using the
        # training data.  This beta is then applied to the test data.
        # Create flattened RDMs for train and test sets.
    X_train = flatten_RDM(make_RDM(inputs_z[:, train_idx]))
    X_test = flatten_RDM(make_RDM(inputs_z[:, test_idx]))

    # Fit model and get predictions and parameters.
    y_pred = baseline_model(X_train, X_test, y_train)
    
    # Based on predictions and test data, evaluate fit.
    score = scoring(y_test, y_pred, score_type=score_type)
    
    return score, y_pred

#%%
def preallocate_predictions(n_conditions, n_outputs):
    # Pre-allocate an empty array which will be used later to store all
    # similarities as predicted by the best fitted outer model of each outer fold.
    predicted_RDM = np.zeros((n_conditions, n_conditions, n_outputs))

    # Pre-allocate an empty array which will be used later to count how many
    # similarities as predicted by the best fitted outer model of each outer fold
    # are collected for each unique condition-pair.
    predicted_RDM_counter = np.zeros((n_conditions, n_conditions, n_outputs))
    
    return predicted_RDM, predicted_RDM_counter


#%% End of script.







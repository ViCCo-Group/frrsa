#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spyder 4.1.4, Python 3.7.6 64-bit | Qt 5.9.6 | PyQt5 5.9.2 | Darwin 18.7.0
"""
@author: Philipp Kaniuth (kaniuth@cbs.mpg.de)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import psutil
from joblib import Parallel, delayed

#TODO: remove the following imports and conditionals before publicising repo.
from pathlib import Path
import os
print(str(Path(os.getcwd())))
if ('dev' not in str(Path(os.getcwd()).parent)) and ('draco' not in str(Path(os.getcwd()).parent)) and ('cobra' not in str(Path(os.getcwd()).parent)):
    print('within submodule')
    from helper.classical_RSA import flatten_RDM, make_RDM
    from helper.data_splitter import data_splitter
    from helper.predictor_distance import hadamard
    from fitting.scoring import scoring, scoring_unfitted
    from fitting.fitting import regularized_model, find_hyperparameters, final_model
else:
    print('outside submodule')
    from frrsa.frrsa.helper.classical_RSA import flatten_RDM, make_RDM
    from frrsa.frrsa.helper.data_splitter import data_splitter
    from frrsa.frrsa.helper.predictor_distance import hadamard
    from frrsa.frrsa.fitting.scoring import scoring, scoring_unfitted
    from frrsa.frrsa.fitting.fitting import regularized_model, find_hyperparameters, final_model

# Set global variables.
z_scale = StandardScaler(copy=False, with_mean=True, with_std=True)

#%%
#TODO: Add parameters to choose one of the predictor_distance funcs.
def frrsa(target,
          predictor,
          distance,
          outer_k=5,
          outer_reps=10,
          splitter='random',
          hyperparams=None,
          score_type='pearson',
          betas_wanted=False,
          predictions_wanted=False,
          parallel=False,
          rng_state=None):
    """ Implements repated, nested cross-validated FRRRSA."""
    
    if hyperparams is None:
        hyperparams = np.linspace(.05, 1, 20) # following recommendation by Rokem & Kay (2020).

    if not hasattr(hyperparams, "__len__"):
        hyperparams = [hyperparams]
    hyperparams = np.array(hyperparams)
            
    try: 
        n_conditions = target.shape[1]
        n_targets = target.shape[2]
    except IndexError:
        n_targets = 1

    y_classical = flatten_RDM(target)
    x_classical = flatten_RDM(make_RDM(predictor, distance='pearson'))
 
    classical_scores = pd.DataFrame(columns=['target', 'score', 'RSA_kind'])
    classical_scores['score'] = scoring_unfitted(y_classical, x_classical, score_type)
    classical_scores['target'] = list(range(n_targets))
    classical_scores['RSA_kind'] = 'classical'
    
    n_outer_cvs = outer_k * outer_reps
    
    #TODO: possibly condition scaling on which "predictor_distance" is used; if
    #hadamard z-transform. If euclidian, normalizing might be more suitable.
    #Possibly adapt naming of function input in predictor_distance.euclidian
    #accordingly.
    predictor = z_scale.fit_transform(predictor)
    
    predictions, score, fold, hyperparameter, predicted_RDM, \
        predicted_RDM_counter = start_outer_cross_validation(n_conditions, 
                                                             splitter, 
                                                             rng_state, 
                                                             outer_k, 
                                                             outer_reps,
                                                             n_targets, 
                                                             predictor, 
                                                             target, 
                                                             score_type, 
                                                             hyperparams, 
                                                             n_outer_cvs, 
                                                             parallel,
                                                             predictions_wanted,
                                                             distance)

    # 'targets' is a numerical variable and denotes the distinct target RDMs.
    targets = np.array(list(range(n_targets)) * n_outer_cvs)
    reweighted_scores = pd.DataFrame(data=np.array([score, fold, hyperparameter, targets]).T, \
                                     columns=['score', 'fold', 'hyperparameter', 'target'])
    
    if predictions_wanted:
        predictions = pd.DataFrame(data=np.delete(predictions, 0, 0), \
                                   columns=['y_test', 'y_regularized', 'target', 'fold', 'first_obj', 'second_obj'])
    else:
        predictions = None
        
    if betas_wanted:
        idx = list(range(n_conditions)) # all conditions.
        X, *_ = compute_predictor_distance(predictor, idx, distance)
        fracs = reweighted_scores.groupby(['target'])['hyperparameter'].mean()
        betas = final_model(X, y_classical, fracs)
        betas = pd.DataFrame(data=betas, \
                             columns=['betas_target_{0}'.format(i+1) for i in range(n_targets)])
    else:
        betas = None
    
    reweighted_scores['score'] = reweighted_scores['score'].apply(np.arctanh)
    reweighted_scores = reweighted_scores.groupby(['target'])['score'].mean().reset_index()
    reweighted_scores['RSA_kind'] = 'reweighted'
    
    scores = pd.concat([classical_scores, reweighted_scores], axis=0)
    
    # Collapse the RDMs along the diagonal, sum & divide, and put pack in array.
    predicted_RDM_re = collapse_RDM(n_conditions, n_targets, predicted_RDM, predicted_RDM_counter)
    return predicted_RDM_re, predictions, scores, betas

#%%
def start_outer_cross_validation(n_conditions, 
                                 splitter, 
                                 rng_state, 
                                 outer_k, 
                                 outer_reps, 
                                 n_targets, 
                                 predictor, 
                                 target, 
                                 score_type, 
                                 hyperparams,
                                 n_outer_cvs,
                                 parallel,
                                 predictions_wanted,
                                 distance):
    
    predicted_RDM = np.zeros((n_conditions, n_conditions, n_targets))
    predicted_RDM_counter = np.zeros((n_conditions, n_conditions, n_targets))
    
    n_hyperparams = len(hyperparams)
    
    # Pre-allocate empty arrayes in which, for each outer fold, the best hyperparamter,
    # model scores, and a fold-counter will be saved, for each target.
    score = np.empty(n_outer_cvs * n_targets)
    fold = np.empty(n_outer_cvs * n_targets)
    hyperparameter = np.empty(n_outer_cvs * n_targets)
    
    # Pre-allocate an empty array in which all predictions of the fitted
    # model, the resepective y_test, factors denoting fold and target
    # and the pairs to which predictions belong will be saved.
    if predictions_wanted:
        predictions = np.zeros((1,6))

    outer_cv = data_splitter(splitter, outer_k, outer_reps, random_state=rng_state)
    list_of_indices = list(range(n_conditions))
    
    results = []
    outer_loop_count = -1
    
    if parallel:
        outer_runs = []

        for outer_train_indices, outer_test_indices in outer_cv.split(list_of_indices):
            outer_loop_count += 1
            outer_runs.append([outer_train_indices, outer_test_indices, outer_loop_count])

        number_cores = psutil.cpu_count(logical=False) # use physical cores.
        jobs = Parallel(n_jobs=number_cores, prefer='processes')(delayed(run_parallel)(outer_run, 
                                                                                       splitter, 
                                                                                       rng_state, 
                                                                                       n_hyperparams, 
                                                                                       n_targets, 
                                                                                       score_type, 
                                                                                       hyperparams, 
                                                                                       predictor, 
                                                                                       target, 
                                                                                       predictions_wanted, 
                                                                                       distance) for outer_run in np.array_split(outer_runs, number_cores)) 
        for job in jobs:
            results += job
    else:
        for outer_train_indices, outer_test_indices in outer_cv.split(list_of_indices):
            outer_loop_count += 1
            current_predictions, y_regularized, first_pair_obj, second_pair_obj, \
            regularized_score, best_hyperparam, outer_loop_count = run_outer_cross_validation_batch(splitter, 
                                                                                                    rng_state, 
                                                                                                    n_hyperparams, 
                                                                                                    n_targets, 
                                                                                                    outer_train_indices, 
                                                                                                    score_type, 
                                                                                                    hyperparams, 
                                                                                                    outer_test_indices, 
                                                                                                    outer_loop_count, 
                                                                                                    predictor,
                                                                                                    target,
                                                                                                    predictions_wanted, 
                                                                                                    distance)
            results.append([current_predictions, y_regularized, first_pair_obj, second_pair_obj, 
                            regularized_score, best_hyperparam, outer_loop_count])


    for result in results:
        current_predictions, y_regularized, first_pair_obj, second_pair_obj, \
        regularized_score, best_hyperparam, outer_loop_count = result
        
        if predictions_wanted:
            predictions = np.concatenate((predictions, current_predictions), axis=0)
        else:
            predictions = None
        
        start_idx = outer_loop_count * n_targets
        score[start_idx:start_idx+n_targets] = regularized_score
        fold[start_idx:start_idx+n_targets] = outer_loop_count
        hyperparameter[start_idx:start_idx+n_targets] = best_hyperparam

        predicted_RDM[first_pair_obj, second_pair_obj, :] += y_regularized
        predicted_RDM_counter[first_pair_obj, second_pair_obj, :] += 1
    return predictions, score, fold, hyperparameter, predicted_RDM, predicted_RDM_counter

#%%
def run_outer_cross_validation_batch(splitter, 
                                     rng_state, 
                                     n_hyperparams, 
                                     n_targets, 
                                     outer_train_indices, 
                                     score_type, 
                                     hyperparams, 
                                     outer_test_indices, 
                                     outer_loop_count,
                                     predictor,
                                     target,
                                     predictions_wanted,
                                     distance):
    
    inner_hyperparams_scores = start_inner_cross_validation(splitter, 
                                                            rng_state, 
                                                            n_hyperparams, 
                                                            n_targets, 
                                                            outer_train_indices, 
                                                            predictor, 
                                                            target, 
                                                            score_type, 
                                                            hyperparams,
                                                            distance)
    
    best_hyperparam = evaluate_hyperparams(inner_hyperparams_scores, 
                                           hyperparams)
        
    regularized_score, first_pair_idx, second_pair_idx, \
        y_regularized, y_test = fit_and_score(predictor, 
                                             target,
                                             outer_train_indices, 
                                             outer_test_indices, 
                                             score_type, 
                                             best_hyperparam,
                                             distance,
                                             place='out')
 
    first_pair_obj, second_pair_obj = outer_test_indices[first_pair_idx], \
                                      outer_test_indices[second_pair_idx]
    
    # Save predictions of the current outer CV with extra info.
    # Note: 'targets' is a numerical variable and denotes the distinct target RDMs.
    targets = np.empty((y_test.shape))
    targets[:,:] = list(range(n_targets))
    targets = targets.reshape(len(targets)*n_targets, order='F')

    if predictions_wanted:
        y_test_reshaped = y_test.reshape(len(y_test)*n_targets, order='F') #make all ys 1D.
        y_regularized_reshaped = y_regularized.reshape(len(y_regularized)*n_targets, order='F')
        first_pair_obj_tiled = np.tile(first_pair_obj, n_targets)
        second_pair_obj_tiled = np.tile(second_pair_obj, n_targets)
        fold = np.array([outer_loop_count] * len(y_test_reshaped))
        current_predictions = np.array([y_test_reshaped, y_regularized_reshaped, targets, fold, first_pair_obj_tiled, second_pair_obj_tiled]).T
    else:
        current_predictions = None
        
    return current_predictions, y_regularized, first_pair_obj, second_pair_obj, regularized_score, best_hyperparam, outer_loop_count

#%%
def run_parallel(outer_run,
                 splitter,
                 rng_state,
                 n_hyperparams,
                 n_targets,
                 score_type,
                 hyperparams,
                 predictor,
                 target,
                 predictions_wanted,
                 distance):
    results = []
    for batch in outer_run:
        outer_train_indices = batch[0]
        outer_test_indices = batch[1]
        outer_loop_count = batch[2]
        current_predictions, y_regularized, first_pair_obj, second_pair_obj, \
        regularized_score, best_hyperparam, outer_loop_count = run_outer_cross_validation_batch(splitter, 
                                                                                                rng_state, 
                                                                                                n_hyperparams, 
                                                                                                n_targets, 
                                                                                                outer_train_indices, 
                                                                                                score_type, 
                                                                                                hyperparams, 
                                                                                                outer_test_indices, 
                                                                                                outer_loop_count, 
                                                                                                predictor, 
                                                                                                target,
                                                                                                predictions_wanted,
                                                                                                distance)
        results.append([current_predictions, y_regularized, first_pair_obj, second_pair_obj, regularized_score, 
                        best_hyperparam, outer_loop_count])
    return results

#%%
def start_inner_cross_validation(splitter,
                                 rng_state,
                                 n_hyperparams,
                                 n_targets,
                                 outer_train_indices,
                                 predictor,
                                 target,
                                 score_type,
                                 hyperparams,
                                 distance):
    inner_k, inner_reps = 5, 5
    inner_cv = data_splitter(splitter, inner_k, inner_reps, random_state=rng_state)
    inner_hyperparams_scores = np.empty((n_hyperparams, n_targets, (inner_k * inner_reps)))
    # Note: In the following loop, rkf.split is applied to the outer_train_indices!
    inner_loop_count = -1
    for inner_train_indices, inner_test_indices in inner_cv.split(outer_train_indices):
        inner_loop_count += 1
        train_idx, test_idx = outer_train_indices[inner_train_indices], outer_train_indices[inner_test_indices]
        score_in, *_ = fit_and_score(predictor, target, train_idx, test_idx, score_type, hyperparams, distance, place='in')
        inner_hyperparams_scores[:, :, inner_loop_count] = score_in
    return inner_hyperparams_scores

#%%
def evaluate_hyperparams(inner_hyperparams_scores,
                         hyperparams):
    '''Evalute which hyperparamter is the best for each target for the current outer fold.'''
    inner_hyperparams_scores_avgs = np.mean(inner_hyperparams_scores, axis=2)
    best_hyperparam_index = inner_hyperparams_scores_avgs.argmax(axis=0)
    best_hyperparam = hyperparams[best_hyperparam_index]
    return best_hyperparam

#%%
def collapse_RDM(n_conditions,
                 n_targets,
                 predicted_RDM,
                 predicted_RDM_counter):
    '''Collapse the RDMs along the diagonal, sum & divide, put pack in array.'''
    idx_low = np.tril_indices(n_conditions, k=-1)
    idx_up = tuple([idx_low[1], idx_low[0]])
    sum_of_preds_halves = predicted_RDM[idx_up] + predicted_RDM[idx_low]
    sum_of_count_halves = predicted_RDM_counter[idx_up] + predicted_RDM_counter[idx_low]
    with np.errstate(divide='ignore', invalid='ignore'):
        average_preds = sum_of_preds_halves / sum_of_count_halves
    predicted_RDM_re = np.zeros((predicted_RDM.shape))
    predicted_RDM_re[idx_low[0], idx_low[1], :] = average_preds
    predicted_RDM_re = predicted_RDM_re + predicted_RDM_re.transpose((1,0,2))
    predicted_RDM_re[(np.isnan(predicted_RDM_re))] = 9999
    return predicted_RDM_re

#%%
def compute_predictor_distance(predictor,
                               idx,
                               distance):
    '''Compute feature-specific distances for the predictor.'''
    if distance=='pearson':
        X, first_pair_idx, second_pair_idx = hadamard(predictor[:, idx])
        X = X.transpose()
    return X, first_pair_idx, second_pair_idx

#%%
def fit_and_score(predictor,
                  target,
                  train_idx,
                  test_idx,
                  score_type,
                  hyperparams,
                  distance,
                  place):
    """ Fit ridge regression and get its predictions and scores for a given cross validation."""
    X_train, *_ = compute_predictor_distance(predictor, train_idx, distance)
    X_test, first_pair_idx, second_pair_idx = compute_predictor_distance(predictor, test_idx, distance)
    y_train = flatten_RDM(target[np.ix_(train_idx, train_idx)])
    y_test = flatten_RDM(target[np.ix_(test_idx, test_idx)])
    if place=='in':
        y_pred = find_hyperparameters(X_train, X_test, y_train, y_test, hyperparams)
    elif place=='out':
        y_pred = regularized_model(X_train, X_test, y_train, y_test, hyperparams)
    score = scoring(y_test, y_pred, score_type=score_type)
    return score, first_pair_idx, second_pair_idx, y_pred, y_test

#%% End of script.

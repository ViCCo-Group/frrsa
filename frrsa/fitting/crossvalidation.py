#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conducts feature-reweighted Representational Similarity Analysis (frrsa).

@author: Philipp Kaniuth (kaniuth@cbs.mpg.de)
"""

import sys
import numpy as np
import pandas as pd
import psutil

from sklearn.preprocessing import StandardScaler, normalize
from joblib import Parallel, delayed

from frrsa.helper.classical_RSA import flatten_matrix  # , make_RDM
from frrsa.helper.data_splitter import data_splitter
from frrsa.helper.predictor_distance import hadamard, sqeuclidean
from frrsa.fitting.scoring import scoring  # , scoring_classical
from frrsa.fitting.fitting import regularized_model, find_hyperparameters, final_model


def frrsa(target,
          predictor,
          preprocess,
          nonnegative,
          measures,
          generalize=None,
          cv=[5, 10],
          hyperparams=None,
          score_type='pearson',
          wanted=[],
          parallel='1',
          random_state=None):
    """Conduct repeated, nested, cross-validated FR-RSA.

    This high-level wrapper function conducts some preparatory data
    processing, calls the function actually doing the work, and finally
    reorganizes output data in easily processable data objects.

    Parameters
    ----------
    target : ndarray
        The representational matrix (either an RDM or RSM) which shall
        be predicted. Expected shape is (n_conditions, n_conditions,n_targets),
        where `n_targets` denotes the number of target matrices. If
        `n_targets == 1`, `targets` can be of shape (n_conditions, n_conditions).
    predictor : ndarray
        The data that shall be used as a predictor. Expected shape is
        (n_channels, n_conditions). For each channel, a separate representational
        matrix will be computed and reweighted.
    preprocess : bool
        Indication of whether to initially preprocess the condition patterns
        of `predictor`. If `distance` is set to `pearson`, this amounts to
        z-transforming each condition pattern. If `distance` is set to
        `sqeuclidean`, this amounts to normalizing each condition pattern.
    nonnegative : bool
        Indication of whether the betas shall be constrained to be non-negative.
    measures : list
        A list of two strings that indicate (dis-)similarity measures. The
        first string indicates which (dis-)similarity measure shall be computed
        within each feature of the predictor. It has two possible options: (1)
        'dot' denotes the dot-product, a similarity measure; (2) 'sqeuclidean'
        denotes the squared euclidean distance, a dissimilarity measure. The
        second string must be set to indicate which measure had been used to
        create the target matrix. Its possible dissimilarity measure options
        are: 'minkowski', 'cityblock', 'euclidean', 'mahalanobis', 'cosine_dis',
        'pearson_dis', 'spearman_dis', and 'decoding_dis', and its possible
        similarity measure options are 'cosine_sim', 'pearson_sim', 'spearman_sim',
        'decoding_sim', and 'spose_sim'.
    generalize : dict, optional
        Can hold additional predictors and targets which shall be used to
        generalize the fitted statistical model to other predictor-target pairs.
        The keys must be 'predictor' and 'target'. Each value must be a numpy
        array. The expected shape for the array for the predictor is
        (n_channels, n_conditions, n_predictors), where `n_predictors` denotes
        the number of predictors. If `n_predictors == 1`, the array can be of
        shape (n_channels, n_conditions). The expected shape for the array for
        the target is (n_conditions, n_conditions, n_targets). If
        `n_targets == 1`, `targets` can be of shape (n_conditions, n_conditions).
        Note that it is currently assumed that each predictor-target pair belongs together,
        i.e. the order in which you supply several predictors and targets
        matters! Also, the number of additionally supplied predictors
        and targets must therefore be equal. Further, each additional predictor
        must have the same `n_channels`, which must also be equal to the
        `n_channels` of the original `predictor`. Each predictor in `generalize`
        will be reweighted using the betas as received when reweighting the
        original `predictor` with the original `target`. Currently, `generalize`
        only works if only one original `target` is supplied. It is silently
        assumed that the additional predictor-target pairs in `generalize`
        use the same measures as specified in `measures`.
    cv : list, optional
        A list of integers, where the first integer indicates the fold size of
        the outer cross-validation and the second integer indicates how often the
        outer cross-validation shall be repeated (defaults to [5, 10]).
    hyperparams : array-like, optional
        The hyperparameter candidates to evaluate in the regularization scheme
        (defaults to `None`). Should be in strictly ascending order.
        If `None`, a sensible default is chosen internally.
    score_type : {'pearson', 'spearman'}, optional
        Type of association measure to compute between predictor and target
        (defaults to `pearson`).
    wanted : list, optional
        A list of strings that indicate which output the user wants the
        function to return. Possible elements are 'predicted_matrix', 'betas'
        and 'predictions'. If the first string is present, then the reweighted
        predicted representational matrix will be returned. If the second string
        is present, betas for each measurement channel will be returned. If the
        third string is present, predicticted (dis-)similarities for all outer
        cross-validations will be returned. There is no mandatory order of the
        strings. Defaults to an empty list, i.e. only `scores` will be returned.
    parallel : str, optional
        Number of parallel jobs that shall be set up to parallelize the outer
        cross-validation, `max` would lead to using all of the machine's CPUs
        cores (defaults to `1`).
    random_state : int, optional
        State of the randomness (defaults to `None`). Should only be set for
        testing purposes. If set, leads to reproducible output across multiple
        function calls.

    Returns
    -------
    scores : pd.DataFrame
        Holds the the representational correspondency scores between each target
        and the predictor. Columns are as follows:

    ======   ================================================================
    target   Target to which scores belong (as `int`)
    scores   Correspondence between predicting and target matrix (as `float`)
    ======   ================================================================

    predicted_matrix : ndarray, optional
        The reweighted predicted representational matrix averaged across outer
        folds with shape (n_conditions, n_conditions, n_targets).
        The value `9999` denotes condition pairs for which no (dis-)similarity
        was predicted.

    betas : pd.DataFrame, optional
        Holds the weights for each target's measurement channel with the shape
        (n_conditions, n_targets). Note that the first weight for each target
        is not a channel-weight but an offset.

    predictions : pd.DataFrame, optional
        Holds (dis-)similarities for the target and for the predictor,
        and to which condition pairs they belong, for all folds and targets
        separately. This is a potentially very large object. Only request if
        you really need it. Columns are as follows:

        ================   ==============================================================================
        dissim_target      (Dis-)similarities for the targets' condition pairs (as `float`)
        dissim_predicted   Reweighted (dis-)similarities for the predictor's condition pairs (as `float`)
        target             Target to which (dis-)similarities belong (as `int`)
        fold               Fold to which (dis-)similarities belong (as `int`)
        first_obj          First condition of pair to which (dis-)similarities belong (as `int`)
        second_obj         Second condition of pair to which (dis-)similarities belong (as `int`)
        ================   ==============================================================================
    """
    splitter = 'random'

    # Check 'target'.
    if target.shape[0] != target.shape[1]:
        sys.exit('Your "target" is not a symmetrical matrix. Its shape must be \
                 (n_conditions, n_conditions, n_targets) or (n_conditions, n_conditions).')
    try:
        n_conditions = target.shape[1]
        n_targets = target.shape[2]
    except IndexError:
        n_targets = 1

    if n_conditions < 9:
        raise Exception(f'There must at least be 9 conditions to execute frrsa, \
                        your data only has {n_conditions}.')

    # Check 'predictor'.
    if predictor.shape[0] == predictor.shape[1]:
        print('Your "predictor" is symmetrical. "predictor" must not be a RDM or RSM. \
              If it is though, you should abort. Continuing...')
    if predictor.ndim != 2:
        sys.exit('Your "predictor" is of shape {predictor.shape}. It must be 2d.')

    # Check 'preprocess'.
    if type(preprocess) != bool:
        sys.exit('The parameter "preprocess" must be of type bool.')
    if preprocess:
        if measures[0] == 'dot':
            z_scale = StandardScaler(copy=False, with_mean=True, with_std=True)
            predictor = z_scale.fit_transform(predictor)
        elif measures[0] == 'sqeuclidean':
            predictor = normalize(predictor, norm='l2', axis=0)

    # Check 'nonnegative'.
    if type(nonnegative) != bool:
        sys.exit('The parameter "nonnegative" must be of type bool.')

    # Check 'measures'.
    if len(measures) != 2:
        sys.exit(f'You provided {len(measures)} elements to the "measures" \
                 parameter. You must provide exactly 2.')

    allowed_predictor_measures = ['dot', 'sqeuclidean']
    if measures[0] not in allowed_predictor_measures:
        sys.exit(f'The first element of "measures" that you provided is "{measures[0]}", \
                 but it must be one of {allowed_predictor_measures}.')

    allowed_target_measures = ['minkowski', 'cityblock', 'euclidean', 'mahalanobis',
                               'cosine_dis', 'pearson_dis', 'spearman_dis', 'cosine_sim',
                               'pearson_sim', 'spearman_sim', 'decoding_dis', 'decoding_sim', 'spose_sim']
    if measures[1] not in allowed_target_measures:
        sys.exit(f'The second element of "measures" that you provided is "{measures[1]}", \
                 but it must be one of {allowed_target_measures}.')

    if (measures[0] == 'dot' and 'sim' not in measures[1]) or \
       (measures[0] == 'sqeuclidean' and 'sim' in measures[1]):
        print(f'The first argument of "measures" that you provided is "{measures[0]}" (a similarity) \
              while the second is "{measures[1]}" (a dissimilarity). This might yield confusing results. \
              You might want to abort and choose a (dis-)similarity for both. \n\
              Continuing...')

    # Check 'generalize'.
    if (list(generalize.keys()) == ['predictor', 'target']) | \
       (list(generalize.keys()) == ['target', 'predictor']):
        sys.exit(f'The keys of "generalize" must be "predictor" and "target". \
                  Yours are {generalize.keys()}.')
    if (not isinstance(generalize['predictor'], np.ndarray)) | \
       (not isinstance(generalize['target'], np.ndarray)):
        sys.exit('The values of "generalize" must both be numpy arrays.')
    if generalize['target'].shape[0] != generalize['target'].shape[1]:
        sys.exit('The "target" in "generalize" is not a symmetrical matrix. \
                  Its shape must be (n_conditions, n_conditions, n_targets) \
                  or (n_conditions, n_conditions).')
    if generalize['predictor'].shape[0] == generalize['predictor'].shape[1]:
        print('Your "predictor" in "generalize" is symmetrical. It must not \
               be a RDM or RSM. If it is, you should abort. Continuing...')
    if (generalize['target'].ndim != generalize['predictor'].ndim) | \
       (generalize['target'].ndim == 3 &
          (generalize['target'].shape[2] != generalize['predictor'].shape[2])):
        sys.exit('The number of additionally supplied predictors and targets \
                  must be equal.')
    if generalize['predictor'].shape[0] != predictor.shape[0]:
        sys.exit('Each additional predictor must have as many channels as, \
                  the original `predictor`.')
    if (generalize is not None) & (target.ndim == 3):
        print(f'Currently, `generalize` only works if only one original \
               `target` is supplied. You supplied {target.shape[2]} targets. \
               Normal frrsa will be conducted, but `generalize` will be \
               ignored. Continuing...')
    try:
        n_generalize = generalize['target'].shape[2]
    except IndexError:
        n_generalize = 1

    # Check 'cv'.
    if type(cv) != list:
        sys.exit('The parameter "cv" must be a list.')
    if len(cv) != 2:
        sys.exit('The parameter "cv" must have a length of 2.')
    if not all(isinstance(item, int) for item in cv):
        sys.exit('All elements of "cv" must be integers.')
    outer_k, outer_reps = cv

    # Check combination of 'cv' and 'n_conditions'.
    if not (n_conditions / outer_k > 2):
        print('The combination of your data\'s number of conditions and your choice for "outer_k" would break this algorithm.')
        while not (n_conditions / outer_k > 2):
            outer_k -= 1
        print(f'Therefore, "outer_k" is adjusted... to {outer_k}! Hence, an outer {outer_reps} times repeated {outer_k}-fold cross-validation will be carried out now.')
        print(f'If you have more than 14 conditions, this could take much longer than a 5-fold cross-validation. You might want to abort and provide an "outer_k" that is a bit smaller than {outer_k}. Continuing...')

    # Check 'hyperparams'.
    if hyperparams is None:
        if not nonnegative:
            hyperparams = np.linspace(.05, 1, 20)
        else:
            hyperparams = [1e-1, 1e0, 1e1, 5e1, 1e2, 5e2, 1e3, 4e3, 7e3, 1e4, 3e4, 5e4, 7e4, 1e5]
        print(f'You did not provide hyperparams. We chose {hyperparams} for you.\n\
              Continuing...')

    if not hasattr(hyperparams, "__len__"):
        hyperparams = [hyperparams]

    hyperparams = np.array(hyperparams)

    if len(hyperparams) == 1:
        print(f'You only provided one value within "hyperparam", namely {hyperparams}.\n\
              That doesn\'t seem right...\n\
              You might want to abort and provide more values. \n\
              Continuing...')

    if hyperparams.ndim > 1:
        sys.exit(f'Your "hyperparams" should be one-dimensional, but they have \
                 {hyperparams.ndim} dimensions.\nTry to provide your hyperparams \
                 in a non-nested list instead.')

    if np.any(np.diff(hyperparams) < 0):
        print('Your provided hyperparams were not in a strictly increasing order.\n\
              They were sorted internally.\nContinuing...')
        hyperparams.sort()

    # Check 'score_type'.
    allowed_score_types = ['pearson', 'spearman']
    if score_type not in allowed_score_types:
        sys.exit(f'Your "score_type" is "{score_type}". But it must be one of {allowed_score_types}.')

    # Check 'wanted'.
    if type(wanted) != list:
        sys.exit('The parameter "wanted" must be a list.')
    if not all(isinstance(item, str) for item in wanted):
        sys.exit('All elements of "wanted" must be strings.')
    if not wanted:
        print('You did not request additional outputs. You will only receive "scores". Continuing...')

    # Check 'parallel'.
    if type(parallel) != str:
        sys.exit('The parameter "parallel" must be a string.')
    if parallel == 'max':
        parallel = psutil.cpu_count(logical=False)
    else:
        parallel = int(parallel)

    # Check 'random_state'.
    if random_state:
        print('You set "random_state". This will fix the randomness across runs. Continuing...')

    n_outer_cvs = outer_k * outer_reps

    predictions, score, fold, hyperparameter, predicted_matrix, \
        predicted_matrix_counter = start_outer_cross_validation(n_conditions,
                                                                splitter,
                                                                random_state,
                                                                outer_k,
                                                                outer_reps,
                                                                n_targets,
                                                                predictor,
                                                                target,
                                                                score_type,
                                                                hyperparams,
                                                                n_outer_cvs,
                                                                parallel,
                                                                wanted,
                                                                measures,
                                                                nonnegative)

    targets = np.array(list(range(n_targets)) * n_outer_cvs)
    scores = pd.DataFrame(data=np.array([score, fold, hyperparameter, targets]).T,
                          columns=['score', 'fold', 'hyperparameter', 'target'])

    if 'predictions' in wanted:
        predictions = pd.DataFrame(data=np.delete(predictions, 0, 0),
                                   columns=['dissim_target', 'dissim_predicted', 'target', 'fold', 'first_obj', 'second_obj'])

    if 'betas' in wanted:
        idx = list(range(n_conditions))
        X, *_ = compute_predictor_distance(predictor, idx, measures[0])
        hyperparams = scores.groupby(['target'])['hyperparameter'].mean()
        y_classical = flatten_matrix(target)
        betas = final_model(X, y_classical, hyperparams, nonnegative, random_state)
        betas = pd.DataFrame(data=betas,
                             columns=[f'betas_target_{i+1}' for i in range(n_targets)])
    else:
        betas = None

    if 'predicted_matrix' in wanted:
        predicted_matrix = collapse_RDM(n_conditions, predicted_matrix, predicted_matrix_counter)

    if (generalize is not None) & (target.ndim == 2):
        idx = list(range(n_conditions))

        hyperparam_scores = start_inner_cross_validation(splitter,
                                                         random_state,
                                                         n_targets,
                                                         idx,
                                                         predictor,
                                                         target,
                                                         score_type,
                                                         hyperparams,
                                                         measures,
                                                         nonnegative)
        best_hyperparam = evaluate_hyperparams(hyperparam_scores,
                                               hyperparams)
        X, *_ = compute_predictor_distance(predictor, idx, measures[0])
        y = flatten_matrix(target)
        betas = final_model(X, y, best_hyperparam, nonnegative, random_state)
        g_scores = np.zeros((n_generalize))
        for pair in range(n_generalize):
            g_target = generalize['target'][:, :, pair]
            g_y = flatten_matrix(g_target)
            g_predictor = generalize['predictor'][:, :, pair]
            g_X, *_ = compute_predictor_distance(g_predictor, idx, measures[0])
            g_predictions = betas[0] + (g_predictor @ betas[1:])
            g_scores[pair] = scoring(g_y, g_predictions, score_type=score_type)

    scores['score'] = scores['score'].apply(np.arctanh)
    scores = scores.groupby(['target'])['score'].mean().reset_index()
    scores['score'] = scores['score'].apply(np.tanh)
    # reweighted_scores['RSA_kind'] = 'reweighted'
    # y_classical = flatten_matrix(target)
    # x_classical = flatten_matrix(make_RDM(predictor, distance))
    # classical_scores = pd.DataFrame(columns=['target', 'score', 'RSA_kind'])
    # classical_scores['score'] = scoring_classical(y_classical, x_classical, score_type)
    # classical_scores['target'] = list(range(n_targets))
    # classical_scores['RSA_kind'] = 'classical'
    # scores = pd.concat([classical_scores, reweighted_scores], axis=0)
    return scores, predicted_matrix, betas, predictions


def start_outer_cross_validation(n_conditions,
                                 splitter,
                                 random_state,
                                 outer_k,
                                 outer_reps,
                                 n_targets,
                                 predictor,
                                 target,
                                 score_type,
                                 hyperparams,
                                 n_outer_cvs,
                                 parallel,
                                 wanted,
                                 measures,
                                 nonnegative):
    """Conduct repeated, nested, cross-validated FR-RSA.

    Set up and conduct repeated, nested, cross-validated FR-RSA, either in
    parallel or sequentially.

    Parameters
    ----------
    n_conditions : int
        The number of conditions.
    splitter : str
        How the data shall be split. If `random`, data
        is split randomly. If `kfold`, a classical k-fold is set up.
        Soft-deprecated.
    random_state : int
        State of the randomness (defaults to `None`). Should only be set
        for testing purposes. If set, leads to reproducible output
        across multiple function calls.
    outer_k : int
        The fold size of the outer cross-validation.
    outer_reps : int
        How often the outer k-fold is repeated.
    n_targets : int
        Denotes the number of targets.
    predictor : ndarray
        The data that shall be used as a predictor. Expected shape is
        (n_channels, n_conditions). For each channel, a separate
        representational matrix will be computed and reweighted.
    target : ndarray
        The representational matrix (either an RDM or RSM) which shall
        be predicted. Expected shape is (n_conditions, n_conditions, n_targets),
        where `n_targets` denotes the number of target matrices. If
        `n_targets == 1`, `targets` can be of shape (n_conditions, n_conditions).
    score_type : str
        Type of association measure to compute between predictor and target.
    hyperparams : array-like
        The hyperparameter candidates to evaluate in the regularization scheme.
    n_outer_cvs : int
        Denotes how many outer cross-validations are conducted in total.
    parallel : int
        Number of parallel jobs that shall be set up to parallelize the
        outer cross-validation
    wanted : list
        A list of strings that indicate which output the user wants the
        function to return.
    measures : list
        A list of two strings that indicate the (dis-)similarity measures
        used for the predictor and target.
    nonnegative : bool
        Indication of whether the betas shall be constrained to be non-negative.

    Returns
    -------
    predictions : ndarray
        Holds (dis-)similarities for the target and for the predictor,
        and to which condition pairs they belong, for all folds and targets
        separately.
    score : ndarray
        Holds the the representational correspondency scores between each target
        and the predictor, for feature-reweighted RSA.
    fold : ndarray
        Index indicating outer folds.
    hyperparameter : ndarray
        Best hyperparameter for each target within each outer fold.
    predicted_matrix : ndarray
        The reweighted predicted representational matrix averaged across outer
        folds with shape (n_conditions, n_conditions, n_targets).
    predicted_matrix_counter : ndarray
        A counter of how often a (dis-)similarity for specific condition-pair
        has been predicted across outer folds.
        Shape is (n_conditions, n_conditions, n_targets).
    """
    if 'predicted_matrix' in wanted:
        predicted_matrix = np.zeros((n_conditions, n_conditions, n_targets))
        predicted_matrix_counter = np.zeros((n_conditions, n_conditions, n_targets))
    else:
        predicted_matrix, predicted_matrix_counter = None, None

    score = np.empty(n_outer_cvs * n_targets)
    fold = np.empty(n_outer_cvs * n_targets)
    hyperparameter = np.empty(n_outer_cvs * n_targets)

    if 'predictions' in wanted:
        predictions = np.zeros((1, 6))
    else:
        predictions = None

    outer_cv = data_splitter(splitter, outer_k, outer_reps, random_state)
    list_of_indices = list(range(n_conditions))

    results = []
    outer_loop_count = -1

    if parallel > 1:
        outer_runs = []

        for outer_train_indices, outer_test_indices in outer_cv.split(list_of_indices):
            outer_loop_count += 1
            outer_runs.append([outer_train_indices, outer_test_indices, outer_loop_count])

        jobs = Parallel(n_jobs=parallel, prefer='processes')(delayed(run_parallel)(outer_run,
                                                                                   splitter,
                                                                                   random_state,
                                                                                   n_targets,
                                                                                   score_type,
                                                                                   hyperparams,
                                                                                   predictor,
                                                                                   target,
                                                                                   wanted,
                                                                                   measures,
                                                                                   nonnegative) for outer_run in np.array_split(outer_runs, parallel))
        for job in jobs:
            results += job
    else:
        for outer_train_indices, outer_test_indices in outer_cv.split(list_of_indices):
            outer_loop_count += 1
            current_predictions, y_regularized, first_pair_obj, second_pair_obj, \
                regularized_score, best_hyperparam = run_outer_cross_validation_batch(splitter,
                                                                                      random_state,
                                                                                      n_targets,
                                                                                      outer_train_indices,
                                                                                      score_type,
                                                                                      hyperparams,
                                                                                      outer_test_indices,
                                                                                      outer_loop_count,
                                                                                      predictor,
                                                                                      target,
                                                                                      wanted,
                                                                                      measures,
                                                                                      nonnegative)
            results.append([current_predictions, y_regularized,
                            first_pair_obj, second_pair_obj, regularized_score,
                            best_hyperparam, outer_loop_count])

    for result in results:
        current_predictions, y_regularized, first_pair_obj, second_pair_obj, \
            regularized_score, best_hyperparam, outer_loop_count = result

        if 'predictions' in wanted:
            predictions = np.concatenate((predictions, current_predictions), axis=0)

        if 'predicted_matrix' in wanted:
            predicted_matrix[first_pair_obj, second_pair_obj, :] += y_regularized
            predicted_matrix_counter[first_pair_obj, second_pair_obj, :] += 1

        start_idx = outer_loop_count * n_targets
        score[start_idx:start_idx + n_targets] = regularized_score
        fold[start_idx:start_idx + n_targets] = outer_loop_count
        hyperparameter[start_idx:start_idx + n_targets] = best_hyperparam
    return predictions, score, fold, hyperparameter, predicted_matrix, predicted_matrix_counter


def run_outer_cross_validation_batch(splitter,
                                     random_state,
                                     n_targets,
                                     outer_train_indices,
                                     score_type,
                                     hyperparams,
                                     outer_test_indices,
                                     outer_loop_count,
                                     predictor,
                                     target,
                                     wanted,
                                     measures,
                                     nonnegative):
    """Conduct one outer cross-validated FR-RSA run.

    For one outer cross-validation, all hyperparameters are evaluated in
    an inner cross-validation, the best for each target is selected, and
    FRRSA is performed on the outer train/test set.

    Parameters
    ----------
    splitter : str
        How the data shall be split. If `random`, data is split randomly.
        If `kfold`, a classical k-fold is set up.
        Soft-deprecated.
    random_state : int
        State of the randomness (defaults to `None`). Should only be set
        for testing purposes. If set, leads to reproducible output across
        multiple function calls.
    n_targets : int
        Denotes the number of target RDMs.
    outer_train_indices : array_like
        The indices denoting conditions belonging to the outer training set.
    score_type : str
        Type of association measure to compute between predictor and target.
    hyperparams : array-like
        The hyperparameter candidates to evaluate in the regularization scheme.
    outer_test_indices : array_like
        The indices denoting conditions belonging to the outer test set.
    outer_loop_count : int
        Denotes the current outer cross-validation.
    predictor : ndarray
        The data that shall be used as a predictor. Expected shape is
        (n_channels, n_conditions).
    target : ndarray
        The representational matrix (either an RDM or RSM) which shall be predicted.
        Expected shape is (n_conditions, n_conditions, n_targets).
    wanted : list
        A list of strings that indicate which output the user wants the
        function to return.
    measures : list
        A list of two strings that indicate the (dis-)similarity measures
        used for the predictor and target.
    nonnegative : bool
        Indication of whether the betas shall be constrained to be non-negative.

    Returns
    -------
    current_predictions : ndarray
        Predicted and test (dis-)similarities, respective targets, fold,
        and conditions, for the current outer fold.
    y_regularized : ndarray
        Predicted (dis-)similarities for each target for the current outer fold.
    first_pair_obj : ndarray
        The first condition of the condition pair to which the
        (dis-)similarities belong to.
    second_pair_obj : ndarray
        The second condition of the condition pair to which the
        (dis-)similarities belong to.
    regularized_score : ndarray
        Holds the the representational correspondency scores between
        each target and the predictor.
    best_hyperparam : ndarray
        Holds the best hyperparameter for each target, for the current outer fold.
    """
    inner_hyperparams_scores = start_inner_cross_validation(splitter,
                                                            random_state,
                                                            n_targets,
                                                            outer_train_indices,
                                                            predictor,
                                                            target,
                                                            score_type,
                                                            hyperparams,
                                                            measures,
                                                            nonnegative)

    best_hyperparam = evaluate_hyperparams(inner_hyperparams_scores,
                                           hyperparams)
    place = 'out'
    regularized_score, first_pair_idx, second_pair_idx, \
        y_regularized, y_test = fit_and_score(predictor,
                                              target,
                                              outer_train_indices,
                                              outer_test_indices,
                                              score_type,
                                              best_hyperparam,
                                              measures,
                                              place,
                                              nonnegative,
                                              random_state)

    first_pair_obj, second_pair_obj = outer_test_indices[first_pair_idx], \
                                      outer_test_indices[second_pair_idx]

    targets = np.empty((y_test.shape))
    targets[:, :] = list(range(n_targets))
    targets = targets.reshape(len(targets) * n_targets, order='F')
    if 'predictions' in wanted:
        y_test_reshaped = y_test.reshape(len(y_test) * n_targets, order='F')  # make all ys 1D.
        y_regularized_reshaped = y_regularized.reshape(len(y_regularized) * n_targets, order='F')
        first_pair_obj_tiled = np.tile(first_pair_obj, n_targets)
        second_pair_obj_tiled = np.tile(second_pair_obj, n_targets)
        fold = np.array([outer_loop_count] * len(y_test_reshaped))
        current_predictions = np.array([y_test_reshaped, y_regularized_reshaped, targets, fold, first_pair_obj_tiled, second_pair_obj_tiled]).T
    else:
        current_predictions = None
    return current_predictions, y_regularized, first_pair_obj, second_pair_obj, regularized_score, best_hyperparam


def run_parallel(outer_run,
                 splitter,
                 random_state,
                 n_targets,
                 score_type,
                 hyperparams,
                 predictor,
                 target,
                 wanted,
                 measures,
                 nonnegative):
    """Wrap the function `run_outer_cross_validation_batch` to run it in parallel.

    Parameters
    ----------
    outer_run : ndarray
        Holds sets of `outer_train_indices` and `outer_test_indices` with
        `outer_loop_count` and `number_cores`. The amoung of sets depends
        on the proportion of outer cross-validations and number of cores.
    splitter : str
        How the data shall be split. If `random`, data is split randomly.
        If `kfold`, a classical k-fold is set up.
        Soft-deprecated.
    random_state : int
        State of the randomness (defaults to `None`). Should only be set
        for testing purposes. If set, leads to reproducible output across
        multiple function calls.
    n_targets : int
        Denotes the number of targets.
    score_type : str
        Type of association measure to compute between predictor and target.
    hyperparams : array-like
        The hyperparameter candidates to evaluate in the regularization scheme.
    predictor : ndarray
        The data that shall be used as a predictor. Expected shape is
        (n_channels, n_conditions).
    target : ndarray
        The representational matrix (either an RDM or RSM) which shall be predicted.
        Expected shape is (n_conditions, n_conditions, n_targets).
    wanted : list
        A list of strings that indicate which output the user wants the
        function to return.
    measures : list
        A list of two strings that indicate the (dis-)similarity measures
        used for the predictor and target.
    nonnegative : bool
        Indication of whether the betas shall be constrained to be non-negative.

    Returns
    -------
    results : list
        Holds all results of the parallelized calls of
        `run_outer_cross_validation_batch`.
    """
    results = []
    for batch in outer_run:
        outer_train_indices = batch[0]
        outer_test_indices = batch[1]
        outer_loop_count = batch[2]
        current_predictions, y_regularized, first_pair_obj, second_pair_obj, \
            regularized_score, best_hyperparam = run_outer_cross_validation_batch(splitter,
                                                                                  random_state,
                                                                                  n_targets,
                                                                                  outer_train_indices,
                                                                                  score_type,
                                                                                  hyperparams,
                                                                                  outer_test_indices,
                                                                                  outer_loop_count,
                                                                                  predictor,
                                                                                  target,
                                                                                  wanted,
                                                                                  measures,
                                                                                  nonnegative)
        results.append([current_predictions, y_regularized, first_pair_obj, second_pair_obj, regularized_score,
                        best_hyperparam, outer_loop_count])
    return results


def start_inner_cross_validation(splitter,
                                 random_state,
                                 n_targets,
                                 outer_train_indices,
                                 predictor,
                                 target,
                                 score_type,
                                 hyperparams,
                                 measures,
                                 nonnegative):
    """Conduct inner repated cross-validated FR-RSA.

    Conduct inner repated cross-validated FR-RSA to evaluate all possible
    hyperparameter candidates, for each target.

    Parameters
    ----------
    splitter : str
        How the data shall be split. If `random`, data is split randomly.
        If `kfold`, a classical k-fold is set up.
        Soft-deprecated.
    random_state : int
        State of the randomness (defaults to `None`). Should only be set
        for testing purposes. If set, leads to reproducible output across
        multiple function calls.
    n_targets : int
        Denotes the number of targets.
    outer_train_indices : array_like
        The indices denoting conditions belonging to the outer training set.
    predictor : ndarray
        The data that shall be used as a predictor. Expected shape is
        (n_channels, n_conditions).
    target : ndarray
        The representational matrix (either an RDM or RSM) which shall be predicted.
        Expected shape is (n_conditions, n_conditions, n_targets).
    score_type : str
        Type of association measure to compute between predictor and target.
    hyperparams : array-like
        The hyperparameter candidates to evaluate in the regularization scheme.
    measures : list
        A list of two strings that indicate the (dis-)similarity measures
        used for the predictor and target.
    nonnegative : bool
        Indication of whether the betas shall be constrained to be non-negative.

    Returns
    -------
    inner_hyperparams_scores : ndarray
        Holds the score for each hyperparameter candidate, separately
        for each target and inner cross-validation.
    """
    n_hyperparams = len(hyperparams)
    inner_k, inner_reps = 5, 5
    n_conditions = len(outer_train_indices)

    if not (n_conditions / inner_k > 2):
        print('The inner cross-validation had to be adjusted because your data has so few conditions.')
        while not (n_conditions / inner_k > 2):
            inner_k -= 1
        print(f'It is now a {inner_reps} times repeated {inner_k}-fold CV.')

    inner_cv = data_splitter(splitter, inner_k, inner_reps, random_state)
    inner_hyperparams_scores = np.empty((n_hyperparams, n_targets, (inner_k * inner_reps)))
    inner_loop_count = -1
    place = 'in'
    for inner_train_indices, inner_test_indices in inner_cv.split(outer_train_indices):
        inner_loop_count += 1
        train_idx, test_idx = outer_train_indices[inner_train_indices], outer_train_indices[inner_test_indices]
        score_in, *_ = fit_and_score(predictor, target, train_idx, test_idx, score_type, hyperparams, measures, place, nonnegative, random_state)
        inner_hyperparams_scores[:, :, inner_loop_count] = score_in
    return inner_hyperparams_scores


def evaluate_hyperparams(inner_hyperparams_scores,
                         hyperparams):
    """Evalute which hyperparamter is the best for each target for the current outer fold.

    Parameters
    ----------
    inner_hyperparams_scores : ndarray
        Holds the score for each hyperparameter candidate, separately
        for each target and inner cross-validation.
    hyperparams : array-like
        The hyperparameter candidates to evaluate in the regularization
        scheme.

    Returns
    -------
    best_hyperparam : ndarray
        The best hyperparamter for each target.
    """
    inner_hyperparams_scores_avgs = np.mean(inner_hyperparams_scores, axis=2)
    best_hyperparam_index = inner_hyperparams_scores_avgs.argmax(axis=0)
    best_hyperparam = hyperparams[best_hyperparam_index]
    return best_hyperparam


def collapse_RDM(n_conditions,
                 predicted_matrix,
                 predicted_matrix_counter):
    """Average RDM halves.

    Collapse representational matrices along their diagonal, sum the
    respective values, divide them by the counter, and reshape the
    resulting values back to a symmetrical matrix.

    Parameters
    ----------
    n_conditions : int
        The number of conditions.
    predicted_matrix : ndarray
        The reweighted predicted representational matrix summed across outer
        folds with shape (n_conditions, n_conditions, n_targets).
    predicted_matrix_counter : ndarray
        A counter of how often a (dis-)similarity for specific condition-pair
        has been predicted across outer folds. Shape is (n_conditions, n_conditions, n_targets).

    Returns
    -------
    predicted_matrix_re : ndarray
        The reweighted predicted representational matrix averaged across
        outer folds with shape (n_conditions, n_conditions, n_targets).
        The value `9999` denotes condition pairs for which no
        (dis-)similarity was predicted.
    """
    idx_low = np.tril_indices(n_conditions, k=-1)
    idx_up = tuple([idx_low[1], idx_low[0]])
    sum_of_preds_halves = predicted_matrix[idx_up] + predicted_matrix[idx_low]
    sum_of_count_halves = predicted_matrix_counter[idx_up] + predicted_matrix_counter[idx_low]
    with np.errstate(divide='ignore', invalid='ignore'):
        average_preds = sum_of_preds_halves / sum_of_count_halves
    predicted_matrix_re = np.zeros((predicted_matrix.shape))
    predicted_matrix_re[idx_low[0], idx_low[1], :] = average_preds
    predicted_matrix_re = predicted_matrix_re + predicted_matrix_re.transpose((1, 0, 2))
    predicted_matrix_re[(np.isnan(predicted_matrix_re))] = 9999
    return predicted_matrix_re


def fit_and_score(predictor,
                  target,
                  train_idx,
                  test_idx,
                  score_type,
                  hyperparams,
                  measures,
                  place,
                  nonnegative,
                  random_state):
    """Fit regularized regression and get its predictions and scores.

    Parameters
    ----------
    predictor : ndarray
        The data that shall be used as a predictor. Expected shape is
        (n_channels, n_conditions).
    target : ndarray
        The representational matrix (either an RDM or RSM) which shall be predicted.
        Expected shape is (n_conditions, n_conditions, n_targets).
    train_idx : array_like
        The indices denoting conditions belonging to the train set.
    test_idx : array_like
        The indices denoting conditions belonging to the test set.
    score_type : str
        Type of association measure to compute between predictor and target.
    hyperparams : array_like
        Hyperparameters for which regularized model shall be fitted.
    measures : list
        A list of two strings that indicate (dis-)similarity measures
        used for the predictor and target.
    place : str
        Indication of whether this function is applied in inner our outer cross-validation.
    nonnegative : bool
        Indication of whether the betas shall be constrained to be non-negative.
    random_state : int
        State of the randomness (defaults to `None`). Should only be set
        for testing purposes.

    Returns
    -------
    scores : pd.DataFrame
        Holds the the representational correspondency scores between
        each target and the predictor.
    first_pair_idx : ndarray
        The first condition of the condition pair to which the
        (dis-)similarities belong to.
    second_pair_idx : ndarray
        The second condition of the condition pair to which the
        (dis-)similarities belong to.
    y_pred : ndarray
        Predicted (dis-)similarities for each target.
    y_test : ndarray
        Test (dis-)similarities for each target.
    """
    X_train, *_ = compute_predictor_distance(predictor, train_idx, measures[0])
    X_test, first_pair_idx, second_pair_idx = compute_predictor_distance(predictor, test_idx, measures[0])
    y_train = flatten_matrix(target[np.ix_(train_idx, train_idx)])
    y_test = flatten_matrix(target[np.ix_(test_idx, test_idx)])
    if place == 'in':
        y_pred = find_hyperparameters(X_train, X_test, y_train, hyperparams, nonnegative, random_state)
    elif place == 'out':
        y_pred = regularized_model(X_train, X_test, y_train, y_test, hyperparams, nonnegative, random_state)
    # Clip illegal predictions to nearest legal value (i.e. bound predictions).
    if measures[1] in ['minkowski', 'cityblock', 'euclidean', 'mahalanobis']:
        y_pred[y_pred < 0] = 0
    elif measures[1] in ['cosine_dis', 'pearson_dis', 'spearman_dis']:
        y_pred[y_pred < 0] = 0
        y_pred[y_pred > 2] = 2
    elif measures[1] in ['cosine_sim', 'pearson_sim', 'spearman_sim']:
        y_pred[y_pred < -1] = -1
        y_pred[y_pred > 1] = 1
    elif measures[1] in ['decoding_dis', 'decoding_sim']:
        y_pred[y_pred < 0] = 0
        y_pred[y_pred > 100] = 100
    elif measures[1] in ['spose_sim']:
        y_pred[y_pred < 0] = 0
        y_pred[y_pred > 1] = 1
    score = scoring(y_test, y_pred, score_type=score_type)
    return score, first_pair_idx, second_pair_idx, y_pred, y_test


def compute_predictor_distance(predictor,
                               idx,
                               measure):
    """Compute feature-specific (dis-)similarities for the predictor.

    Parameters
    ----------
    predictor : ndarray
        The data that shall be used as a predictor. Expected shape is
        (n_channels, n_conditions).
    idx : array_like
        Holds indices of those conditions for which the feature-specific
        (dis-)similarity shall be computed.
    measure : str
        The (dis-)similarity measure that shall be used for the predictor.

    Returns
    -------
    X : ndarray
        The feature-specific (dis-)similarities for `predictor`.
    first_pair_idx : ndarray
        The first condition of the condition pair to which the
        (dis-)similarities belong to.
    second_pair_idx : ndarray
        The second condition of the condition pair to which the
        (dis-)similarities belong to.
    """
    if measure == 'dot':
        X, first_pair_idx, second_pair_idx = hadamard(predictor[:, idx])
    elif measure == 'sqeuclidean':
        X, first_pair_idx, second_pair_idx = sqeuclidean(predictor[:, idx])
    X = X.transpose()
    return X, first_pair_idx, second_pair_idx

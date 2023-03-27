#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper function to create indices for training and test sets based on
preference of how to split the data.

@author: Philipp Kaniuth (kaniuth@cbs.mpg.de)
"""

from sklearn.model_selection import RepeatedKFold, ShuffleSplit


def data_splitter(choice, k, reps, random_state=None):
    """Create iterator with indices for training and test data.

    Parameters
    ----------
    choice : {'kfold', 'random'}
            The desired type of splitter.
    k : int
            Number of folds.
    reps : int
            How often the k-fold iterator is repeated.
    random_state: int, optional
            State of the randomness for testing purposes (defaults to `None`).

    Returns
    -------
    mysplit : iterable
            Iterable containing `reps` sets of lists of training and test indices.
    """
    # Semantically, kfold's 'n_splits' is random's inverse of 'test_size'.
    # Further, kfold's 'n_repeats' is one factor determining random's 'n_splits'.
    if choice == 'kfold':  # SOFT-DEPRECATED
        mysplit = RepeatedKFold(n_splits=k, n_repeats=reps,
                                random_state=random_state)
    elif choice == 'random':
        mysplit = ShuffleSplit(n_splits=(k * reps), test_size=1/k,
                               train_size=(1 - 1/k), random_state=random_state)
    return mysplit

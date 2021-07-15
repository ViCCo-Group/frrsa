#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spyder 4.1.4, Python 3.7.6 64-bit | Qt 5.9.6 | PyQt5 5.9.2 | Darwin 18.7.0
"""
@author: Philipp Kaniuth (kaniuth@cbs.mpg.de)
"""

from sklearn.model_selection import RepeatedKFold, ShuffleSplit

def data_splitter(choice, k, reps, random_state=None):
    '''Create iterator with indices for training and test data.
    
    Parameters
    ----------
    choice : str
            The desired type of splitter.
    k : int
            Number of folds.
    reps : int
            How often the k-fold iterator is repeated.
    random_state: int, optional
            State of the randomness for testing purposes (defaults to None).
    
    Returns
    -------
    mysplit : iterable
            Iterable containing `reps` sets of lists of training and test indices.
    '''
    # Semantically, kfold's 'n_splits' is random's inverse of 'test_size'. 
    # Further, kfold's 'n_repeats' is one factor determining random's 'n_splits'.
    if choice == 'kfold':
        mysplit = RepeatedKFold(n_splits=k, n_repeats=reps,
                                random_state=random_state)
    elif choice == 'random':
        mysplit = ShuffleSplit(n_splits=(k * reps), test_size=1/k,
                               train_size=(1 - 1/k), random_state=random_state)
    return mysplit
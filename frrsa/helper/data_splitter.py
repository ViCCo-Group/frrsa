#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 13:40:01 2020

@author: kaniuth
"""
from sklearn.model_selection import RepeatedKFold, ShuffleSplit

# array_for_indices = list(range(1, 22))
# rng_state = 1


def data_splitter(choice, k, reps, random_state=None):
    """Returns indices to split data in training and test"""
    # Semantically, kfold's 'n_splits' is random's inverse of 'test_size'. 
    # Further, kfold's 'n_repeats' is one factor determining random's 'n_splits'.
    if choice == 'kfold':
        mysplit = RepeatedKFold(n_splits=k, n_repeats=reps, random_state=random_state)
    elif choice == 'random':
        mysplit = ShuffleSplit(n_splits=(k * reps), test_size=1/k, train_size=(1 - 1/k), random_state=random_state)
    return mysplit


# mysplit = data_splitter('kfold', 3, 1, rng_state)
# for outer_train_indices, outer_test_indices in mysplit.split(array_for_indices):
#     print('Train: ' + str(outer_train_indices))
#     print('Test: ' + str(outer_test_indices))
    

# mysplit = data_splitter('random', 3, 1, rng_state)
# for outer_train_indices, outer_test_indices in mysplit.split(array_for_indices):
#     print('Train: ' + str(outer_train_indices))
#     print('Test: ' + str(outer_test_indices))
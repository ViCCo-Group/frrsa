#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 13:40:01 2020

@author: kaniuth
"""
from sklearn.model_selection import RepeatedKFold, ShuffleSplit


def data_splitter(choice, k, reps, random_state=None):
    """Returns indices to split data in training and test"""
    # Semantically, kfold's 'n_splits' is random's inverse of 'test_size'. 
    # Further, kfold's 'n_repeats' is one factor determining random's 'n_splits'.
    if choice == 'kfold':
        mysplit = RepeatedKFold(n_splits=k, n_repeats=reps, random_state=random_state)
    elif choice == 'random':
        mysplit = ShuffleSplit(n_splits=(k * reps), test_size=1/k, train_size=(1 - 1/k), random_state=random_state)
    return mysplit
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstration script for running 'frrsa'.

@author: Philipp Kaniuth (kaniuth@cbs.mpg.de)
"""
# Uncomment the three lines below and change 'top_folder' to where you have downloaded frrsa.
# import sys
# top_folder = "/User/desired/top/folder/frrsa"
# sys.path.append(f'{top_folder}/frrsa')
from numpy.random import default_rng
from frrsa.fitting.crossvalidation import frrsa

# Simulate target Representational Matrix and predictor data.
rng = default_rng()
n_channels = 100  # How many features?
n_conditions = 50  # How many conditions?
n_targets = 2  # How many different target matrices?

# The next block will essentially simulate a target filled with Pearson similarities
# between -1 and 1.
target_low = -1
target_high = 1
target = (target_high - target_low) * rng.random(size=(n_conditions, n_conditions, n_targets)) + target_low

# The next block will simulate arbitrary predictor data.
predictor_low = 0
predictor_high = 100
predictor = (predictor_high - predictor_low) * rng.random(size=(n_channels, n_conditions)) + predictor_low

# Set the main function's parameters.
preprocess = True
nonnegative = False
measures = ['dot', 'pearson_sim']
cv = [5, 1]
hyperparams = None
score_type = 'pearson'
wanted = ['predicted_matrix', 'betas', 'predictions']
parallel = '2'
random_state = None

# Call the main funtion and enjoy the output.
scores, predicted_matrix, betas, predictions = frrsa(target,
                                                     predictor,
                                                     preprocess,
                                                     nonnegative,
                                                     measures,
                                                     cv,
                                                     hyperparams,
                                                     score_type,
                                                     wanted,
                                                     parallel,
                                                     random_state)

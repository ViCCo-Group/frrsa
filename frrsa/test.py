#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spyder 4.2.5 | Python 3.8.8 64-bit | Qt 5.9.7 | PyQt5 5.9.2 | Darwin 18.7.0
"""
Demonstration script for running 'frrsa'.

@author: Philipp Kaniuth (kaniuth@cbs.mpg.de)
"""
import sys
from numpy.random import default_rng
from fitting.crossvalidation import frrsa

# Uncomment the two lines below and change 'top_folder' to where you have downloaded frrsa.
# top_folder = "/User/desired/top/folder/frrsa"
# sys.path.append(f'{top_folder}/frrsa')

# Simulate target Representational Matrix and predictor data.
rng = default_rng()
n_channels = 100  # How many features?
n_conditions = 50  # How many conditions?
n_targets = 2  # How many different target matrices?

target_low = -1
target_high = 1
target = (target_high - target_low) * rng.random(size=(n_conditions, n_conditions, n_targets)) + target_low

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
wanted = ['betas', 'predictions']
betas_wanted = True
predictions_wanted = True
parallel = '2'
random_state = None

# Call the main funtion and enjoy the output.
predicted_RDM, predictions, scores, betas = frrsa(target,
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

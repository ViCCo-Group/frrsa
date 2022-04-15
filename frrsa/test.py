#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spyder 4.2.5 | Python 3.8.8 64-bit | Qt 5.9.7 | PyQt5 5.9.2 | Darwin 18.7.0
"""
Demonstration script for running 'frrsa'.

@author: Philipp Kaniuth (kaniuth@cbs.mpg.de)
"""

from numpy.random import default_rng
from fitting.crossvalidation import frrsa
rng = default_rng(seed=4)

# Simulate target Representational Matrix and predictor data.
n_channels = 100  # How many features?
n_conditions = 50  # How many conditions?
n_targets = 2  # How many different target matrices?
target = rng.integers(low=-1, high=1, size=(n_conditions, n_conditions, n_targets))
predictor = rng.integers(low=0, high=100, size=(n_channels, n_conditions))

# Set the main function's parameters.
preprocess = True
nonnegative = False
distance = 'pearson'
outer_k = 5
outer_reps = 1
hyperparams = None
score_type = 'pearson'
betas_wanted = True
predictions_wanted = True
parallel = True
rng_state = None

# Call the main funtion and enjoy the output.
predicted_RDM, predictions, scores, betas = frrsa(target,
                                                  predictor,
                                                  preprocess,
                                                  nonnegative,
                                                  distance,
                                                  outer_k,
                                                  outer_reps,
                                                  hyperparams,
                                                  score_type,
                                                  betas_wanted,
                                                  predictions_wanted,
                                                  parallel,
                                                  rng_state)

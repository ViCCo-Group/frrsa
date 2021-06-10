#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Philipp Kaniuth (kaniuth@cbs.mpg.de) 
"""

from numpy.random import default_rng
from fitting.crossvalidation import frrsa

# Specify a seed for reproducible results.
rng = default_rng(seed=4)

n_units = 100 # How many measurement channels?
n_objects = 100 # How many objects aka conditions?
n_outputs = 2 # How many different outputs?

# Simulate output and inputs.
target = rng.integers(low=0, high=100, size=(n_objects,n_objects,n_outputs))
predictor = rng.integers(low=0, high=100, size=(n_units,n_objects))


#%% Call the main funtion.
outer_k = 2
outer_reps = 3
splitter = 'random'
hyperparams = None
score_type = 'pearson'
betas_wanted = True
parallel = True
rng_state = 1

predicted_RDM, predictions, scores, betas = frrsa(target,
                                                 predictor, 
                                                 outer_k, 
                                                 outer_reps, 
                                                 splitter, 
                                                 hyperparams, 
                                                 score_type, 
                                                 betas_wanted,
                                                 parallel,
                                                 rng_state)

#%% End of script
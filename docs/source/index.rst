.. frrsa documentation master file, created by
   sphinx-quickstart on Tue Jan 18 09:20:25 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to frrsa's documentation!
=================================
This projects provides an algorithm which builds on Representational Similarity Analysis (RSA). 
The classical approach of RSA is to correlate two Representational Dissimilarity Matrices (RDM), in which each cell gives a measure of how dissimilar two conditions are represented by a given system (e.g., the human brain or a deep neural network (DNN)). 
However, this might underestimate the true relationship between the systems, since it assumes that all measurement channels (e.g., fMRI voxel or DNN units) contribute equally to the establishment of stimulus-pair dissimilarity, and in turn, to correspondence between RDMs. 
Feature-reweighted RSA (FRRSA) deploys regularized regression techniques (currently: L2-regularization) to maximize the fit between two RDMs; the RDM's cells of one system are explained by a linear reweighted combination of the dissimilarities of the respective stimuli in all measurement channels of the other system.
Importantly, every measurement channel of the predicting system receives its own weight. This all is implemented in a nested cross-validation, which avoids overfitting on the level of (hyper-)parameters.

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Getting started
==================
FRRSA is written in Python 3 using the Anaconda distribution for Python. 
You can find an exhaustive package list in the `Anaconda environment file <https://github.com/ViCCo-Group/frrsa/blob/master/anaconda_env_specs_frrsa.yml>`_.

.. code-block:: python

   from fitting.crossvalidation import frrsa

   # load your "target" RDM
   # load your "predictor" data.
   # set the "preprocess" flag.
   # set the "nonnegative" flag.

   predicted_RDM, predictions, scores, betas = frrsa(target,
                                                   predictor, 
                                                   preprocess,
                                                   nonnegative,
                                                   distance='pearson',
                                                   outer_k=5, 
                                                   outer_reps=10, 
                                                   splitter='random', 
                                                   hyperparams=None, 
                                                   score_type='pearson', 
                                                   betas_wanted=False,
                                                   predictions_wanted=False,
                                                   parallel=False,
                                                   rng_state=None)
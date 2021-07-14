# frrsa

This projects provides an algorithms which builds on Representational Similarity Analysis (RSA). The usual approach of RSA is to correlate two Representational Dissimilarity Matrices (RDM), in which each cell gives a measure of how dissimilar two stimuli are represented by a given system (e.g., the human brain or a deep neural network (DNN)). However, this might underestimate the true relationship between the RDMs, since it assumes that all measurement channels (e.g., fMRI voxel or DNN units) contribute equally to the establishment of stimulus-pair dissimilarity, and in turn, to correspondence between RDMs. Feature-reweighted RSA (frrsa) deploys regularized regression techniques (currently: L2-regularization) to maximize the fit between two RDMs; the RDM's cells of one system are explained by a linear reweighted combination of the dissimilarities of the respective stimuli in all measurement channels of the other system. Importantly, every measurement channel of the explaining system receives its own weight. This is all implemented in a nested cross-validation, which avoids overfitting on the level of (hyper-)parameters. 


## Getting Started

### Prerequisites
FRRSA is written in Python 3. You can find an exhaustive package list in the [Anaconda environment file](https://github.com/PhilippKaniuth/frrsa/blob/master/anaconda_env_specs_frrsa.yml). Use the [Anaconda distribution for Python](https://www.anaconda.com/distribution/#download-section): with the help of the environment file, you can then set up an Anaconda environment which should allow you to run the package.

### Installing


### How to use
See [ffrsa/test.py](https://github.com/PhilippKaniuth/frrsa/blob/master/frrsa/test.py) for a simple demonstration.

Import the main function `from fitting.crossvalidation import frrsa` and run:
```
predicted_RDM, predictions, scores, betas = frrsa(target,
                                                  predictor, 
                                                  distance='Hadamard',
                                                  outer_k=5, 
                                                  outer_reps=10, 
                                                  splitter='random', 
                                                  hyperparams=None, 
                                                  score_type='pearson, 
                                                  betas_wanted=False,
                                                  predictions_wanted=False,
                                                  parallel=False,
                                                  rng_state=None)
Arguments:
- target: the RDM which you want to fit to. Expected format is a (condition*condition*n_targets) numpy array. `n_targets` denotes how many independented target RDMs shall be predicted by the predictor. If n_targets==1, `targets` can be of shape (condition*condition).
- predictor: the RDM you want to use as a predictor. Expected format is a (channel*condition) numpy array. 
- distance: the distance measure used for both, the target and predictor RDM.
- outer_k: the fold size of the outer crossvalidation.
- outer_reps: how often the outer k-fold is repeated.
- splitter: how the data shall be split. If "random", data is split randomly. If "kfold", a classical k-fold crossvalidation is performed.
- hyperparams: which hyperparameters you want to check for the fractional ridge regression (see paper by Rokem & Kay (2020) below). If "None", a sensible default is chosen internally.
- score_type: how your predicted dissimilarity values shall be related to the corresponding target dissimilarity values.
- betas_wanted: a boolean value, indicating whether you want to have betas returned for each measurement channel.
- predictions_wanted: a boolean value, indicating whether you want to receive all predicticted dissimilarities for all outer cross-validations. This is a potentially huge object.
- parallel: a boolean value, indicating whether you want to parallelize the outer cross-validation using all your CPUs cores.
- rng_state: ignore, will be deprecated in release-version. Keep the default.

Returns:
predicted_RDM: a (condition*condition*n_target) numpy array populated with the predicted dissimilarities.
predictions: a pandasDataFrame which, for all folds and outputs separately, holds predicted and target dissimilarities and to which object pairs they belong.
scores: a pandasDataFrame which holds the scores for classical and feature-reweighted RSA for each target.
betas: a pandasDataFrame which holds the betas for each target's measurement channel.

Notes regarding language:
- "Measurement channel": a generic umbrella term denoting things like a voxel, an MEG measurement channel, a unit of a deep neural network layer.
- "Condition": can mean, for example, an image or other stimulus for which you have an activity pattern.
- "n_target" is the amount of separate target-RDMs you want to predict using your predicting RDM. Different targets could for example be MEG RDMs from different time points or RDMs from different participants.
```

## Built With
* [Anaconda for Python](https://www.anaconda.com/distribution/)


## Authors
* **Philipp Kaniuth** - [GitHub](https://github.com/PhilippKaniuth), [MPI CBS](https://www.cbs.mpg.de/employees/kaniuth)
* **Martin Hebart** - [Personal Homepage](http://martin-hebart.de/), [MPI CBS](https://www.cbs.mpg.de/employees/hebart)


## License
This project is licensed under the GNU AFFERO GENERAL PUBLIC LICENSE Version 3 - see the [LICENSE.md](LICENSE.md) file for details.


## Acknowledgments
* For all ridge regression fitting, [Fractional Ridge Regression](https://pubmed.ncbi.nlm.nih.gov/33252656/) by Rokem & Kay (2020) was used.
* Thanks to Katja Seliger ([GitHub](https://github.com/kateiyas), [Personal Homepage](http://seeliger.space/)), Lukas Muttenthaler ([GitHub](https://github.com/LukasMut), [Personal Homepage](https://lukasmut.github.io/index.html)), and Oliver Contier ([GitHub](https://github.com/oliver-contier), [Personal Homepage](https://olivercontier.com)) for valuable discussions and hints.
* Thanks to Hannes Hansen ([GitHub](https://github.com/hahahannes), [Personal Homepage](https://hannesh.de)) for considerable code improvement.

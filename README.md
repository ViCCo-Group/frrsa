# frrsa

This projects provides algorithms which improve Representational Similarity Analysis (RSA). The usual approach of RSA is to correlate two Representational Dissimilarity Matrices (RDM), in which each cell gives a measure of how dissimilar two stimuli are represented by a given system (e.g., the human brain or a deep neural network). However, this might underestimate the true relationship between the RDMs, since it assumes that all measurement channels contribute equally to the establishment of stimulus-pair dissimilarity, and in turn, to dissimilarity between RDMs. Feature reweigted RSA (frrsa) deploys regularized regression techniques to maximize the fit between two RDMs; the RDM's cells of one system are explained by a linear combination of the dissimilarities of the respective stimuli in all measurement channels of the other system. Importantly, every measurement channel of the explaining system receives its own weight. To counterbalance overfitting problems, nested cross-validation is used.


## Getting Started

### Prerequisites
FRRSA is written in Python 3. You can find an exhaustive package list in the [anaconda-env_tuned-rsa.yml](https://github.com/PhilippKaniuth/tuned_rsa/blob/main/anaconda-env-specs_tuned-rsa.yml). Use the [Anaconda distribution for Python](https://www.anaconda.com/distribution/#download-section): with the help of the environment file, you can then set up an Anaconda environment which should allow you to run the algorithms.

### Installing


### How to use
See ffrsa/test.py for a simple demonstration.

Import the main function `from fitting.crossvalidation import frrsa` and run:
```
predicted_RDM, predictions, unfitted_scores, crossval, betas, fitted_scores = frrsa(output, 
                                                                                    inputs, 
                                                                                    outer_k=5, 
                                                                                    outer_reps=10, 
                                                                                    splitter='random', 
                                                                                    hyperparams=None, 
                                                                                    score_type='pearsonr', 
                                                                                    sanity=False, 
                                                                                    rng_state=None)
Arguments:
- output: the RDM which you want to fit to. Expected format is a (condition*condition*n_output) numpy array. I n_output==1, it can be of shape (condition*condition).
- inputs: the RDM you want to use as a predictor. Expected format is a (channel*condition) numpy array. 
- outer_k: the fold size of the outer crossvalidation.
- outer_reps: how often the outer k-fold is repeated.
- splitter: how the data shall be split. If "random", data is split randomly. If "kfold", a classical k-fold crossvalidation is performed.
- hyperparams: which hyperparameters you want to check for the fractional ridge regression (see paper by Rokem & Kay (2020) below). If "None", a sensible default is chosen internally.
- score_type: how your predicted values shall be scored.
- sanity: ignore. Keep the default.
- rng_state: ignore. Keep the default.

Returns:
predicted_RDM: a (condition*condition*n_output) numpy array populated with the predicted dissimilarities.
predictions: a pandasDataFrame which, for all folds and outputs separately, holds predicted and test dissimilarities and to which object pairs they belong.
unfitted_scores: a dictionary which holds classical RSA scores for each output, separately for different scoring methods.
crossval: a pandasDataFrame which, for all folds and outputs separately, holds the scores and hyperparameters of every outer fold.
betas: a pandasDataFrame which, for all folds and outputs separately, holds the betas from each outer crossvalidation.
fitted_scores: a numpy array which holds, for every output, the correlation between the predicted RDM and the output.

Notes regarding language:
- "Channel" is a generic umbrella term and can mean, for example: a voxel, an MEG measurement channel, a unit of a deep neural network layer.
- "Condition" can mean, for example: an image or other stimulus for which you have an activity pattern.
```

## Built With
* [Anaconda for Python 3.7](https://www.anaconda.com/distribution/) - The development framework used.


## Authors
* **Philipp Kaniuth** - [GitHub](https://github.com/PhilippKaniuth), [MPI CBS](https://www.cbs.mpg.de/employees/kaniuth)
* **Martin Hebart** - [Personal Homepage](http://martin-hebart.de/), [MPI CBS](https://www.cbs.mpg.de/employees/hebart)


## License
This project is licensed under the GNU AFFERO GENERAL PUBLIC LICENSE Version 3 - see the [LICENSE.md](LICENSE.md) file for details.


## Acknowledgments
* For all ridge regression fitting, [Fractional Ridge Regression](https://pubmed.ncbi.nlm.nih.gov/33252656/) by Rokem & Kay (2020) was used.
* Thanks to Katja Seliger ([GitHub](https://github.com/kateiyas), [Personal Homepage](http://seeliger.space/)) and Oliver Contier ([GitHub](https://github.com/oliver-contier), [Personal Homepage](https://olivercontier.com)) for valuable discussions and hints.
* Thanks to Hannes Hansen ([GitHub](https://github.com/hahahannes), [Personal Homepage](https://hannesh.de)) for code improvement.


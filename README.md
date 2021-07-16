# frrsa

This projects provides an algorithm which builds on Representational Similarity Analysis (RSA). The classical approach of RSA is to correlate two Representational Dissimilarity Matrices (RDM), in which each cell gives a measure of how dissimilar two conditions are represented by a given system (e.g., the human brain or a deep neural network (DNN)). However, this might underestimate the true relationship between the systems, since it assumes that all measurement channels (e.g., fMRI voxel or DNN units) contribute equally to the establishment of stimulus-pair dissimilarity, and in turn, to correspondence between RDMs. Feature-reweighted RSA (FRRSA) deploys regularized regression techniques (currently: L2-regularization) to maximize the fit between two RDMs; the RDM's cells of one system are explained by a linear reweighted combination of the dissimilarities of the respective stimuli in all measurement channels of the other system. Importantly, every measurement channel of the predicting system receives its own weight. This all is implemented in a nested cross-validation, which avoids overfitting on the level of (hyper-)parameters. 


## Getting Started

### Prerequisites
FRRSA is written in Python 3 using the [Anaconda distribution for Python](https://www.anaconda.com/distribution/#download-section). You can find an exhaustive package list in the [Anaconda environment file](https://github.com/PhilippKaniuth/frrsa/blob/master/anaconda_env_specs_frrsa.yml).

### Installing


### How to use
See [ffrsa/test.py](https://github.com/PhilippKaniuth/frrsa/blob/master/frrsa/test.py) for a simple demonstration.

Just import the main function and call it.

```py
from fitting.crossvalidation import frrsa

predicted_RDM, predictions, scores, betas = frrsa(target,
                                                  predictor, 
                                                  distance='pearson',
                                                  outer_k=5, 
                                                  outer_reps=10, 
                                                  splitter='random', 
                                                  hyperparams=None, 
                                                  score_type='pearson, 
                                                  betas_wanted=False,
                                                  predictions_wanted=False,
                                                  parallel=False,
                                                  rng_state=None)
```                                            


## FAQ
#### _How does my data have to look like to use the FRRSA algorithm?_
At present, the algorithm expects data of two systems (e.g., a specific DNN layer and a brain region measured with fMRI) the representational spaces of which ought to be compared. The predicting system, that is, the one of which the features ought to be reweighted, is expected to be a _p_ x _k_ matrix. The target system contributes its full RDM in the form of a _k_ x _k_ matrix (where `p:=Number of measurement channels` and `k:=Number of conditions` see [Diedrichsen & Kriegeskorte, 2017](https://dx.plos.org/10.1371/journal.pcbi.1005508)). There are no hard-coded limits on the size of each dimension; however, the bigger _k_ and _p_ become, the larger becomes the computational problem to solve.
#### _FRRSA uses regularization. Which kinds of regularization regimes are implemented?_
As of now, only L2-regularization aka Ridge Regression.
#### _You say ridge regression; which hyperparameter space should I check?_
We implemented the L2-regularization using Fractional Ridge Regression (FRR; [Rokem & Kay, 2020](https://pubmed.ncbi.nlm.nih.gov/33252656/)). One advantage of FRR is that the hyperparameter to be optimized is the fraction between ordinary least squares and L2-regularized regression coefficients, which ranges between 0 and 1. Hence, FRR allows assessing the full range of possible regularization parameters. In the context of FRRSA, ten default values between 0.1 and 1 are set. If you want to specify custom regularization values that ought to be assessed, you are able to do so by inputting a list of candidate values into the frrsa algorithm.
#### _What else? What outputs does the output give? Are there other options I can specify when running FR-RSA?_
There are default values for all parameters, which we partly assessed (see our preprint). However, you input custom parameters as you wish.


## Authors
- **Philipp Kaniuth** - [GitHub](https://github.com/PhilippKaniuth), [MPI CBS](https://www.cbs.mpg.de/employees/kaniuth)
- **Martin Hebart** - [Personal Homepage](http://martin-hebart.de/), [MPI CBS](https://www.cbs.mpg.de/employees/hebart)


## License
This project is licensed under the GNU AFFERO GENERAL PUBLIC LICENSE Version 3 - see the [LICENSE.md](LICENSE.md) file for details.


## Acknowledgments
- Thanks to Katja Seliger ([GitHub](https://github.com/kateiyas), [Personal Homepage](http://seeliger.space/)), Lukas Muttenthaler ([GitHub](https://github.com/LukasMut), [Personal Homepage](https://lukasmut.github.io/index.html)), and Oliver Contier ([GitHub](https://github.com/oliver-contier), [Personal Homepage](https://olivercontier.com)) for valuable discussions and hints.
- Thanks to Hannes Hansen ([GitHub](https://github.com/hahahannes), [Personal Homepage](https://hannesh.de)) for considerable code improvement.

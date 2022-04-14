[![Unittests](https://github.com/PhilippKaniuth/frrsa/actions/workflows/tests.yml/badge.svg)](https://github.com/PhilippKaniuth/frrsa/actions/workflows/tests.yml)

# frrsa

This repository provides a Python package to run Feature-Reweighted Representational Similarity Analysis (FR-RSA). The classical approach of Representational Similarity Analysis (RSA) is to correlate two Representational Matrices, in which each cell gives a measure of how (dis-)similar two conditions are represented by a given system (e.g., the human brain or a model like a deep neural network (DNN)). However, this might underestimate the true correspondence between the systems' representational spaces, since it assumes that all features (e.g., fMRI voxel or DNN units) contribute equally to the establishment of condition-pair (dis-)similarity, and in turn, to correspondence between representational matrices. FR-RSA deploys regularized regression techniques (currently: L2-regularization) to maximize the fit between two representational matrices. The core idea behind FR-RSA is to recover a subspace of the predicting matrix that best fits to the target matrix. To do so, the matrices' cells of the target system are explained by a linear reweighted combination of the feature-specific (dis-)similarities of the respective conditions in the predicting system. Importantly, the Representational Matrix of each feature of the predicting system receives its own weight. This all is implemented in a nested cross-validation, which avoids overfitting on the level of (hyper-)parameters.


## Getting Started

### Prerequisites
The package is written in Python 3 using the [Anaconda distribution for Python](https://www.anaconda.com/distribution/#download-section). You can find an exhaustive package list in the [Anaconda environment file](https://github.com/PhilippKaniuth/frrsa/blob/master/anaconda_env_specs_frrsa.yml) which you should use to create an Anaconda environment.

### Installing
For now, since no setup.py exists yet, just download the package to a location of your choosing (`top_folder`). Then, you could set up via Python:

```py
import os
top_folder = "/User/desired/top/folder/frrsa"
os.system(f'git clone https://github.com/ViCCo-Group/frrsa.git {top_folder}')
# create Anaconda environment.
# activate Anaconda environment.
```

### How to use
See [frrsa/test.py](https://github.com/PhilippKaniuth/frrsa/blob/master/frrsa/test.py) for a simple demonstration of how to use the package.

Activate the Anaconda environment, temporarily append to your Python's sys.path, and then import the main function to call it with your loaded matrices.

```py
import sys
top_folder = "/User/desired/top/folder/frrsa"
sys.path.append(f'{top_folder}/frrsa')
from frrsa.fitting.crossvalidation import frrsa

# load your "target" RDM or RSM.
# load your "predictor" data.
# set the necessary flags ("preprocess", "nonnegative", "distance", ...)

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
```                                            


## FAQ
#### _How does my data have to look like to use the FR-RSA package?_
At present, the package expects data of two systems (e.g., a specific DNN layer and a brain region measured with fMRI) the representational spaces of which ought to be compared. The predicting system, that is, the one of which the feature-specific (dis-)similarities shall be reweighted, is expected to be a _p_ x _k_ numpy array. The target system contributes its full representational matrix in the form of a _k_ x _k_ numpy array (where `p:=Number of measurement channels aka features` and `k:=Number of conditions` see [Diedrichsen & Kriegeskorte, 2017](https://dx.plos.org/10.1371/journal.pcbi.1005508)). There are no hard-coded *upper* limits on the size of each dimension; however, the bigger _k_ and _p_ become, the larger becomes the computational problem to solve. See [Known issues](https://github.com/ViCCo-Group/frrsa#known-issues) for a *lower* limit of _k_.
#### _You say that every feature gets its own weight - can those weights take on any value or are they restricted to be non-negative?_
The function's parameter `nonnegative` can be set to either `True` or `False` and forces weights to be nonnegative (or not), accordingly.
#### _What about the covariances / interactive effects between predicting features?_
One may argue that it could be the case that the interaction of (dis-)similarities in two or more features in one system could help in the prediction of overall (dis-)similarity in another system. Currently, though, feature reweighting does not take into account these interaction terms (nor does classical RSA), which probably also is computationally too expensive for predicting systems with a lot of features (e.g. early DNN layers).
#### _FR-RSA uses regularization. Which kinds of regularization regimes are implemented?_
As of now, only L2-regularization aka Ridge Regression.
#### _You say ridge regression; which hyperparameter space should I check?_
If you set the parameter `nonnegative` to `False`, L2-regularization is implemented using Fractional Ridge Regression (FRR; [Rokem & Kay, 2020](https://pubmed.ncbi.nlm.nih.gov/33252656/)). One advantage of FRR is that the hyperparameter to be optimized is the fraction between ordinary least squares and L2-regularized regression coefficients, which ranges between 0 and 1. Hence, FRR allows assessing the full range of possible regularization parameters. In the context of FR-RSA, twenty evenly spaced values between 0.1 and 1 are pre-set. If you want to specify custom regularization values that shall be assessed, you are able to do so by providing a list of candidate values as the `hyperparams` argument of the frrsa function. <br/> If you set the parameter `nonnegative` to `True`,  L2-regularization is currently implemented using Scikit-Learn functions. They have the disadvantage that one has to define the hyperparameter space oneself, which can be tricky. If you do not provide hyerparameter candidates yourself, [14 pre-set values](https://github.com/ViCCo-Group/frrsa/blob/0b6d7ab35d9eb6962bc6a4eeabfb2b8e345a9969/frrsa/fitting/crossvalidation.py#L142) will be used which *might* be sufficient (see our [preprint](https://github.com/ViCCo-Group/frrsa#citation)).
#### _Which `distance` measures can be used?_
The parameter `distance` lets you choose which (dis-)similarity measure will be computed for the Representational Matrices of the predicting system. Note that this choice is also applied to when computing a global Representational Matrix for the predicting system for computing the classical RSA score. Currently, you can either select `sqeuclidean` or `pearson`. <br/> The former will compute the squared Euclidean distance between the condition pairs, globally (for classical RSA) and within each predicting feature (for FR-RSA). <br/> The latter will compute Pearson's correlation coefficient between the condition pairs. Wait, what?, you might ask at this point, how can one compute a *correlation* between two vectors within each feature, that is, if there is only one data point per condition?! Well spotted! Actually, when applying reweighting, the closest you can get to a feature-specific correlation is the dot-product after having z-transformed the condition pairs. In that case, if you were to average across features (equivalent to weighting each feature equally), you would end up again with the Pearson correlation. In any case, selecting `pearson` computes the feature-specific dot-product for the condition pairs. When computing the global Representational Matrix for the predicting system, the Pearson correlation is computed. <br/> You might further think that the parameter's name (`distance`) is a slight misnomer, as a feature-specific dot-product is not a dissimilarity but rather a similarity measure (and I agree, see #25). Which one you should choose depends on your data. Further, if the Representational Matrix of your target system holds similarities, it is likely more intuitive to select `pearson` (to have similarities on both sides of the equation). If, though, your `target` holds dissimilarities, it might conversely be more intuitive to select `sqeuclidean`.
#### _What else? What objects does the function return? Are there other parameters I can specify when running FR-RSA?_
There are default values for all parameters, which we partly assessed (see our [preprint](https://github.com/ViCCo-Group/frrsa#citation)). However, you can input custom parameters as you wish. For now, see the [respective docstring](https://github.com/ViCCo-Group/frrsa/blob/0008ba45c44ac469624b99175672f241696c0b3a/frrsa/fitting/crossvalidation.py#L48) for an explanation of all returned objects. A more elaborate documentation is work in progress (see [#14](/../../issues/14)).


## Known issues
Note that the data (i.e. `target` and `predictor`) are split along the condition dimension to conduct a nested cross-validation. Therefore, there exists a logical lower bound regarding the number of different conditions, _k_, below which `frrsa` cannot be executed successfully. Below this bound, inner test folds occur that contain data from just two conditions, which leads to just one predicted dissimilarity (for one condition pair). However, to determine the goodness-of-fit, currently the predicted dissimilarities of each cross-validation are _correlated_ with the respective target dissimilarities. This does not work with vectors that have a length < 2.

The exact lower bound depends on the values set for `outer_k` and `splitter`. 

If `outer_k=5` and `splitter='random'` (the default values for these parameters), the lower sufficient size of _k_ is 14. This is due to the inner workings of [data_splitter](https://github.com/ViCCo-Group/frrsa/blob/master/frrsa/helper/data_splitter.py). If `splitter='random'`, `data_splitter` uses [klearn.model_selection.ShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html) which allows to specifically set the proportion of the current dataset to be included in the test fold. Currently, in `data_splitter` this is set to _1/outer_k_ in order to be comparable to `splitter='kfold'`. Therefore, when _k_ is 14, this leads to an outer test fold size of 2.8 ≈ 3, and to an outer training fold size of (14-3)=11. This in turn guarantees an inner test fold size of 2.2 ≈ 3 (note that the sklearn function in question rounds up). However, if _k_ is 13, 2.6 ≈ 3 conditions are allocated to an outer test fold, which leads to an outer training fold size of (13-3)=10. This leads to inner test folds sizes of 2.

If `outer_k=5` and `splitter='kfold'`, the lower sufficient size of _k_ is 19. If `splitter='kfold'`, `data_splitter` uses [klearn.model_selection.RepeatedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html) which does not allow to specifically set the size of the test fold. Therefore, when _k_ is 19, at least data of 15 conditions enter the inner cross-validation, which leads to inner test folds with 3 conditions. However, if _k_ is 18, only data of 14 conditions enter the inner cross-validation, which leads to some inner test folds with data from only 2 conditions.

With different values for `outer_k` the lower bound of _k_ changes accordingly. An automatic check of the parameters with a respective custom warning is work in progress (see [#17](/../../issues/17)), as might be fixing this situation altogether (see [#22](/../../issues/22)).


## Authors
- **Philipp Kaniuth** - [GitHub](https://github.com/PhilippKaniuth), [MPI CBS](https://www.cbs.mpg.de/employees/kaniuth)
- **Martin Hebart** - [Personal Homepage](http://martin-hebart.de/), [MPI CBS](https://www.cbs.mpg.de/employees/hebart)


## Citation
If you use `frrsa` (or any of its modules), please cite our [associated preprint](https://www.biorxiv.org/content/10.1101/2021.09.27.462005v2) as follows:

```
@article{Kaniuth_preprint_2021,
        author = {Kaniuth, Philipp and Hebart, Martin N.},
        title = {Feature-reweighted representational similarity analysis: A method for improving the fit between computational models, brains, and behavior},
        journal = {bioRxiv},
        pages = {2021.09.27.462005},
        year = {2021},
        doi = {10.1101/2021.09.27.462005},
        URL = {https://www.biorxiv.org/content/10.1101/2021.09.27.462005v3}
}
```

## License
This GitHub repository is licensed under the GNU AFFERO GENERAL PUBLIC LICENSE Version 3 - see the [LICENSE.md](LICENSE.md) file for details.


## Acknowledgments
- Thanks to Katja Seliger ([GitHub](https://github.com/kateiyas), [Personal Homepage](http://seeliger.space/)), Lukas Muttenthaler ([GitHub](https://github.com/LukasMut), [Personal Homepage](https://lukasmut.github.io/index.html)), and Oliver Contier ([GitHub](https://github.com/oliver-contier), [Personal Homepage](https://olivercontier.com)) for valuable discussions and hints.
- Thanks to Hannes Hansen ([GitHub](https://github.com/hahahannes), [Personal Homepage](https://hannesh.de)) for considerable code improvement.

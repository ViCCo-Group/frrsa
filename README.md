[![Unittests](https://github.com/PhilippKaniuth/frrsa/actions/workflows/tests.yml/badge.svg)](https://github.com/PhilippKaniuth/frrsa/actions/workflows/tests.yml) ![Maintenance](https://img.shields.io/maintenance/yes/2022) ![License](https://img.shields.io/badge/license-AGPL--3.0-blue) ![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue)

# frrsa

This repository provides a Python package to run Feature-Reweighted Representational Similarity Analysis (FR-RSA). The classical approach of Representational Similarity Analysis (RSA) is to correlate two Representational Matrices, in which each cell gives a measure of how (dis-)similar two conditions are represented by a given system (e.g., the human brain or a model like a deep neural network (DNN)). However, this might underestimate the true correspondence between the systems' representational spaces, since it assumes that all features (e.g., fMRI voxel or DNN units) contribute equally to the establishment of condition-pair (dis-)similarity, and in turn, to correspondence between representational matrices. FR-RSA deploys regularized regression techniques (currently: L2-regularization) to maximize the fit between two representational matrices. The core idea behind FR-RSA is to recover a subspace of the predicting matrix that best fits to the target matrix. To do so, the matrices' cells of the target system are explained by a linear reweighted combination of the feature-specific (dis-)similarities of the respective conditions in the predicting system. Importantly, the Representational Matrix of each feature of the predicting system receives its own weight. This all is implemented in a nested cross-validation, which avoids overfitting on the level of (hyper-)parameters.


## Getting Started

### Prerequisites
The package is written in Python 3.8. You can find an exhaustive package list in the [conda environment file](https://github.com/PhilippKaniuth/frrsa/blob/master/conda_env_frrsa.yml) which you should use to [create a conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) before using `frrsa`.

### Installing
For now, since no setup.py exists yet, just download the package to a location of your choosing (`top_folder`). Then, you could set up e.g. via Python:

```py
import os
top_folder = "/User/desired/top/folder/frrsa"
os.system(f'git clone https://github.com/ViCCo-Group/frrsa.git {top_folder}')
# create conda environment.
# activate conda environment.
```

### How to use
See [frrsa/test.py](https://github.com/PhilippKaniuth/frrsa/blob/master/frrsa/test.py) for a simple demonstration of how to use the package.

Activate the conda environment, temporarily append to your Python's sys.path, and then import the main function to call it with your loaded matrices.

```py
import sys
top_folder = "/User/desired/top/folder/frrsa"
sys.path.append(f'{top_folder}/frrsa')
from frrsa.fitting.crossvalidation import frrsa

# load your "target" RDM or RSM.
# load your "predictor" data.
# set the necessary flags ("preprocess", "nonnegative", "measures", ...)

scores, predicted_matrix, betas, predictions = frrsa(target,
                                                     predictor,
                                                     preprocess,
                                                     nonnegative,
                                                     measures,
                                                     cv=[5, 10],
                                                     hyperparams=None,
                                                     score_type='pearson',
                                                     wanted=[],
                                                     parallel='1',
                                                     random_state=None)
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
#### _Which (dis-)similarity `measures` can be used?_
The parameter `measures` is a list that expects two strings. The first string lets you choose which (dis-)similarity measure will be computed within each feature of the predictor. It has two possible options: (1) `'dot'` denotes the dot-product, a similarity measure; (2) `'sqeuclidean'` denotes the squared euclidean distance, a dissimilarity measure. The second string must be set to indicate which measure had been used to create the target matrix. Its possible dissimilarity measure options are: `'minkowski'`, `'cityblock'`, `'euclidean'`, `'mahalanobis'`, `'cosine_dis'`, `'pearson_dis'`, `'spearman_dis'`, and `'decoding_dis'`, and its possible similarity measure options are `'cosine_sim'`, `'pearson_sim'`, `'spearman_sim'`, and `'decoding_sim'`. <br/> Which measure you should choose for the predictor depends on your data. Additionally, if your `target` holds similarities, it is likely more intuitive to select `'dot'` (to have similarities on both sides of the equation). If, though, your `target` holds dissimilarities, it might conversely be more intuitive to select `'sqeuclidean'`.
#### _What else? What objects does the function return? Are there other parameters I can specify when running FR-RSA?_
There are default values for all parameters, which we partly assessed (see our [preprint](https://github.com/ViCCo-Group/frrsa#citation)). However, you can input custom parameters as you wish. For now, see the [respective docstring](https://github.com/ViCCo-Group/frrsa/blob/master/frrsa/fitting/crossvalidation.py#L45) for an explanation of all returned objects. A more elaborate documentation is work in progress (see [#14](/../../issues/14)).


## Known issues
1. If your data has less than 9 conditions, **`frrsa` cannot be executed successfully**. <br/> Why? Because the data (i.e. `target` and `predictor`) are split along the condition dimension to conduct the nested cross-validation. Therefore, there exists an absolute lower bound for the number of conditions below which inner test folds will occur that contain data from just two conditions, which would lead to just one predicted (dis-)similarity (for one condition pair). However, to determine the goodness-of-fit, currently the predicted (dis-)similarities of each cross-validation are _correlated_ with the respective target (dis-)similarities. This does not work with vectors that have a length < 2. This won't be fixed soon, see [#28](/../../issues/28).

2. The default fold size, outer_k, for the outer crossvalidation is 5 (denoted by the first element of `cv`). In that case, the minimum number of conditions needed is 14. <br/> (This is due to the inner workings of [data_splitter](https://github.com/ViCCo-Group/frrsa/blob/master/frrsa/helper/data_splitter.py). It uses [sklearn.model_selection.ShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html) which allows to specifically set the proportion of the current dataset to be included in the test fold. This proportion is set to _1/outer_k_ (due to historical reasons, so that it was comparable to `splitter='kfold'`, see [#26](/../../issues/26)). Therefore, when there are 14 conditions, this leads to an outer test fold size of 2.8 ≈ 3, and to an outer training fold size of (14-3)=11. This in turn guarantees an inner test fold size of 2.2 ≈ 3 (note that the sklearn's SuffleSplit function rounds up). However, if there are only 13 conditions, still 2.6 ≈ 3 conditions are allocated to an outer test fold, but that leads to an outer training fold size of (13-3)=10 which leads to inner test folds sizes of 2.

3. If your data has between 9 and 13 conditions, `frrsa` will run. However, the default `outer_k` and the hard-coded `inner_k` will be adapted automatically (see [#22](/../../issues/22)).

4. There are other combinations of `outer_k` and the number of conditions (also when the number of conditions is bigger than 14) that would yield too few (inner or outer) test conditions if unchanged, but could be executed successfully otherwise. Therefore, in these cases, `outer_k` and `inner_k` will be adapted automatically [#17](/../../issues/17)).


Long story short: If you have 14 or more conditions, you can (but don't have to) use the default value for `outer_k` which has been used for many analyses (in our [preprint](https://github.com/ViCCo-Group/frrsa#citation)). If you have less than 14 but more than 8 conditions, you can use `frrsa`, just not with `outer_k = 5`. No luck for you for the time being if you only have 8 or fewer conditions.



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

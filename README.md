<a name="readme-top"></a>
<div align="center">
    <a href="https://github.com/ViCCo-Group/frrsa/actions/workflows/tests.yml" rel="nofollow">
        <img src="https://github.com/ViCCo-Group/frrsa/actions/workflows/tests.yml/badge.svg" alt="Tests" />
    </a>
        <img src="https://img.shields.io/maintenance/yes/2023" alt="Maintenance Status" />
    </a>
    <a href="https://www.python.org/" rel="nofollow">
        <img src="https://img.shields.io/badge/python-3.8-blue" alt="Python version" />
    </a>
    <a href="https://github.com/ViCCo-Group/frrsa/blob/master/LICENSE" rel="nofollow">
        <img src="https://img.shields.io/badge/license-AGPL--3.0-blue" alt="License" />
    </a>
</div>



<!-- Table of Contents -->

# :notebook_with_decorative_cover: Table of Contents

- [About the project](#star2-about-the-project)
- [Getting started](#running-getting-started)
  * [Installing](#computer-installing)
  * [How to use](#mag-how-to-use)
  * [Parameters & Returned objects](#repeat-parameters-and-returned-objects)
- [FAQ](#question-faq)
- [Known issues](#spiral_notepad-known-issues)
- [Contributing](#wave-how-to-contribute)
- [License](#warning-license)
- [Citation](#page_with_curl-citation)
- [Contributions](#gem-contributions)


<!-- About the Project -->
## :star2: About the project

`frrsa` is a Python package to conduct Feature-Reweighted Representational Similarity Analysis (FR-RSA). The classical approach of Representational Similarity Analysis (RSA) is to correlate two Representational Matrices, in which each cell gives a measure of how (dis-)similar two conditions are represented by a given system (e.g., the human brain or a model like a deep neural network (DNN)). However, this might underestimate the true correspondence between the systems' representational spaces, since it assumes that all features (e.g., fMRI voxel or DNN units) contribute equally to the establishment of condition-pair (dis-)similarity, and in turn, to correspondence between representational matrices. FR-RSA deploys regularized regression techniques (currently: L2-regularization) to maximize the fit between two representational matrices. The core idea behind FR-RSA is to recover a subspace of the predicting matrix that best fits to the target matrix. To do so, the matrices' cells of the target system are explained by a linear reweighted combination of the feature-specific (dis-)similarities of the respective conditions in the predicting system. Importantly, the Representational Matrix of each feature of the predicting system receives its own weight. This all is implemented in a nested cross-validation, which avoids overfitting on the level of (hyper-)parameters.

:rotating_light: Please also see the published [article](https://www.sciencedirect.com/science/article/pii/S105381192200413X) accompanying this repository. To use this package successfully, follow this `README`. :rotating_light:

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Getting Started -->
## :running: Getting started

<!-- Installing -->
### :computer: Installing
The package is written in Python 3.8. Installation expects you to have a working `conda` on your system (e.g. via `miniconda`). If you have `pip` available already, you can skip the `conda env create` part.

Execute the following lines from a terminal to clone this repository and install it as a local package using pip.
```bash
cd [directory on your system where you want to download this repo to]
git clone https://github.com/ViCCo-Group/frrsa
conda env create --file=./frrsa/environment.yml
conda activate frrsa
cd frrsa
pip install -e .
```

<!-- How to use -->
### :mag: How to use
There is only one user-facing function in `frrsa`. To use it, activate the conda environment, import and then call `frrsa` with your data:

```py
from frrsa import frrsa

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
                                                     wanted=['predicted_matrix', 'betas', 'predictions'],
                                                     parallel='1',
                                                     random_state=None)
```                                            
See [frrsa/test.py](https://github.com/ViCCo-Group/frrsa/blob/master/frrsa/test.py) for another simple demonstration.

<!--params-and-returned-->
### :repeat: Parameters and returned objects

#### Parameters.
There are default values for all parameters, which we partly assessed (see our [paper](https://www.sciencedirect.com/science/article/pii/S105381192200413X)). However, you can input custom parameters as you wish. For an explanation of all parameters please see [the docstring](https://github.com/ViCCo-Group/frrsa/blob/master/frrsa/fitting/crossvalidation.py#L41).

#### Returned objects.
1. `scores`: Holds the the representational correspondency scores between each target and the predictor. These scores can be sensibly used in downstream analyses.
2. `predicted_matrix`: The reweighted predicted representational matrix averaged across outer folds with shape (n_conditions, n_conditions, n_targets). The value `9999` denotes condition pairs for which no (dis-)similarity was predicted ([why?](#in-the-returned-predicted_matrix-why-are-there-some-condition-pairs-for-which-there-are-no-predicted-dis-similarities)). This matrix should only be used for visualizational purposes.
3. `betas`: Holds the weights for each target's measurement channel with the shape (n_conditions, n_targets). Note that the first weight for each target is not a channel-weight but an offset. These betas are currently computed suboptimally and should only be used for informational purposes. Do *not* use them to recreate the `reweighted_matrix` or to reweight something else (see [#43](/../../issues/43)).
4. `predictions`: Holds (dis-)similarities for the target and for the predictor, and to which condition pairs they belong, for all cross-validations and targets separately. This is a potentially very large object. Only request if you really need it. For an explanation of the columns see the [docstring](https://github.com/ViCCo-Group/frrsa/blob/master/frrsa/fitting/crossvalidation.py#L121).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- FAQ -->
## :question: FAQ

#### _How does my data have to look like to use the FR-RSA package?_
At present, the package expects data of two systems (e.g., a specific DNN layer and a brain region measured with fMRI) the representational spaces of which ought to be compared. The predicting system, that is, the one of which the feature-specific (dis-)similarities shall be reweighted, is expected to be a _p_ x _k_ numpy array. The target system contributes its full representational matrix in the form of a _k_ x _k_ numpy array (where `p:=Number of measurement channels aka features` and `k:=Number of conditions` see [Diedrichsen & Kriegeskorte, 2017](https://dx.plos.org/10.1371/journal.pcbi.1005508)). There are no hard-coded *upper* limits on the size of each dimension; however, the bigger _k_ and _p_ become, the larger becomes the computational problem to solve. See [Known issues](#spiral_notepad-known-issues) for a *lower* limit of _k_.
#### _You say that every feature gets its own weight - can those weights take on any value or are they restricted to be non-negative?_
The function's parameter `nonnegative` can be set to either `True` or `False` and forces weights to be nonnegative (or not), accordingly.
#### _What about the covariances / interactive effects between predicting features?_
One may argue that it could be the case that the interaction of (dis-)similarities in two or more features in one system could help in the prediction of overall (dis-)similarity in another system. Currently, though, feature reweighting does not take into account these interaction terms (nor does classical RSA), which probably also is computationally too expensive for predicting systems with a lot of features (e.g. early DNN layers).
#### _FR-RSA uses regularization. Which kinds of regularization regimes are implemented?_
As of now, only L2-regularization aka Ridge Regression.
#### _You say ridge regression; which hyperparameter space should I check?_
If you set the parameter `nonnegative` to `False`, L2-regularization is implemented using Fractional Ridge Regression (FRR; [Rokem & Kay, 2020](https://pubmed.ncbi.nlm.nih.gov/33252656/)). One advantage of FRR is that the hyperparameter to be optimized is the fraction between ordinary least squares and L2-regularized regression coefficients, which ranges between 0 and 1. Hence, FRR allows assessing the full range of possible regularization parameters. In the context of FR-RSA, twenty evenly spaced values between 0.1 and 1 are pre-set. If you want to specify custom regularization values that shall be assessed, you are able to do so by providing a list of candidate values as the `hyperparams` argument of the frrsa function. <br/> If you set the parameter `nonnegative` to `True`,  L2-regularization is currently implemented using Scikit-Learn functions. They have the disadvantage that one has to define the hyperparameter space oneself, which can be tricky. If you do not provide hyerparameter candidates yourself, [14 pre-set values](https://github.com/ViCCo-Group/frrsa/blob/0b6d7ab35d9eb6962bc6a4eeabfb2b8e345a9969/frrsa/fitting/crossvalidation.py#L142) will be used which *might* be sufficient (see our [paper](https://www.sciencedirect.com/science/article/pii/S105381192200413X)).
#### _Which (dis-)similarity `measures` can/should be used?_
Use the parameter `measures` to indicate which (dis-)similarity measures to use. See the [docstring](https://github.com/ViCCo-Group/frrsa/blob/master/frrsa/fitting/crossvalidation.py#L59) for possible arguments. <br/> Which measure you should choose for the predictor depends on your data. Additionally, if your `target` holds similarities, it is likely more intuitive to select `'dot'` (to have similarities on both sides of the equation). If, though, your `target` holds dissimilarities, it might conversely be more intuitive to select `'sqeuclidean'`.
#### _In the returned `predicted_matrix`, why are there some condition pairs for which there are no predicted (dis-)similarities?_
To conduct a proper cross-validation that does not lead to leakge, one needs to split the data based on *conditions* not pairs. However, if conditions A, B, C, D are in the training set, and E, F, G are in the test set, then e.g. the pair (A, E) would never be used for either fitting or testing the statistical model. Therefore, even if one repeats such a k-fold cross-validation a few times it could be that a few pairs never receive a predicted (dis-)similarity. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Known Issues -->
## :spiral_notepad: Known issues
1. If your data has less than 9 conditions, **`frrsa` cannot be executed successfully**. This won't be fixed (see [#28](/../../issues/28)). <br/> 
    <details><summary>Expand for details.</summary> 
    
    The data (i.e. `target` and `predictor`) are split along the condition dimension to conduct the nested cross-validation. Therefore, there exists an absolute lower bound for the number of conditions below which inner test folds will occur that contain data from just two conditions, which would lead to just one predicted (dis-)similarity (for one condition pair): this absolute lower bound is 9. However, to determine the goodness-of-fit, currently the predicted (dis-)similarities of each cross-validation are _correlated_ with the respective target (dis-)similarities. This does not work with vectors that have a length < 2.
    
    </details>

2. The default fold size, `outer_k`, for the outer crossvalidation is 5 (denoted by the first element of `cv`). In that case, the minimum number of conditions needed is 14. <br/> 
    <details><summary>Expand for details.</summary> 
    
    This is due to the inner workings of [data_splitter](https://github.com/ViCCo-Group/frrsa/blob/master/frrsa/helper/data_splitter.py). It uses [sklearn.model_selection.ShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html) which allows to specifically set the proportion of the current dataset to be included in the test fold. This proportion is set to _1/outer_k_ due to historical reasons so that it was comparable to `splitter='kfold'` (see [#26](/../../issues/26)). Therefore, when there are 14 conditions, this leads to an outer test fold size of 2.8 ≈ 3, and to an outer training fold size of (14 - 3) = 11. This in turn guarantees an inner test fold size of 2.2 ≈ 3 (note that the sklearn's SuffleSplit function rounds up). <br/>
    
    </details>
    
    However, if there are only 13 or less conditions and `outer_k` is set to 5, 2.6 ≈ 3 conditions are allocated to an outer test fold, but that leads to an outer training fold size of (13 - 3) = 10 which leads to inner test folds sizes of only 2, which wouldn't work (as explained in 1.). Therefore:

3. If your data has between 9 and 13 conditions, `frrsa` will run. However, the default `outer_k` and the hard-coded `inner_k` will be adapted automatically (see [#22](/../../issues/22)).

4. There are other combinations of `outer_k` and the number of conditions (also when the number of conditions is bigger than 14) that would yield too few (inner or outer) test conditions if unchanged, but could be executed successfully otherwise. Therefore, in these cases, `outer_k` and `inner_k` will be adapted automatically (see [#17](/../../issues/17)).

5. The optionally returned `betas` are currently computed suboptimally and should only be used for informational purposes. Do *not* use them to recreate the `reweighted_matrix` or to reweight something else (see [#43](/../../issues/43)).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Contributing -->
## :wave: How to contribute
If you come across problems or have suggestions please submit an issue!

<p align="right">(<a href="#readme-top">back to top</a>)</p


<!-- License -->
## :warning: License
This GitHub repository is licensed under the GNU AFFERO GENERAL PUBLIC LICENSE Version 3 - see the [LICENSE.md](LICENSE.md) file for details.

<p align="right">(<a href="#readme-top">back to top</a>)</p


<!-- Citation -->
## :page_with_curl: Citation
If you use `frrsa` (or any of its modules), please cite our [associated paper](https://www.sciencedirect.com/science/article/pii/S105381192200413X) as follows:

```
@article{KANIUTH2022119294,
         author = {Philipp Kaniuth and Martin N. Hebart},
         title = {Feature-reweighted representational similarity analysis: A method for improving the fit between computational models, brains, and behavior},
         journal = {NeuroImage},
         pages = {119294},
         year = {2022},
         issn = {1053-8119},
         doi = {https://doi.org/10.1016/j.neuroimage.2022.119294},
         url = {https://www.sciencedirect.com/science/article/pii/S105381192200413X}
}
```

<p align="right">(<a href="#readme-top">back to top</a>)</p


<!-- Contributions -->
## :gem: Contributions
The Python package itself was mostly written by Philipp Kaniuth ([GitHub](https://github.com/PhilippKaniuth), [MPI CBS](https://www.cbs.mpg.de/employees/kaniuth)), with key contributions and amazing help as well as guidance provided by Martin Hebart ([Personal Homepage](http://martin-hebart.de/), [MPI CBS](https://www.cbs.mpg.de/employees/hebart)) and Hannes Hansen ([GitHub](https://github.com/hahahannes), [Personal Homepage](https://hannesh.de)).

Further thanks go to Katja Seliger ([GitHub](https://github.com/kateiyas), [Personal Homepage](http://seeliger.space/)), Lukas Muttenthaler ([GitHub](https://github.com/LukasMut), [Personal Homepage](https://lukasmut.github.io/index.html)), and Oliver Contier ([GitHub](https://github.com/oliver-contier), [Personal Homepage](https://olivercontier.com)) for valuable discussions and hints.

Check our [lab home page](https://hebartlab.com/) for more information on the cool work we do! :nerd_face:

<p align="right">(<a href="#readme-top">back to top</a>)</p

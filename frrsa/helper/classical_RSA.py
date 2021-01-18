# Import packages.
import numpy as np
from scipy.stats import spearmanr, pearsonr

#%% The function 'make_RDM' requires as its only argument a numpy array.
# In that array, each column is one condition, each row one measurement channel.
# The function returns both, the  dissimilarity matrix and the unique half of
# the dissimilarity matrix reduced to a vector.  Pearson's r is used to compute
# the dissimilarity matrix.  For an explanation in which way 'dissim_vec' is
# put together, see the function 'reduce_RDM'.

def make_RDM(activity_pattern_matrix):
    """Based on an activation pattern matrix, a dissimilarity matrix is returned"""
    dissim_mat = 1 - np.corrcoef(activity_pattern_matrix, rowvar=False)
    return dissim_mat


#%% The function 'classical_RSA' performs a classical RSA between two RDMs.
# Hence, it requires to vectorised RDMs as inputs. By default, it performs
# Spearman's correlation. If the parameter 'correlation' is set to 'Pearson',
# Pearson's correlation will be performed.

def correlate_RDMs(RDM1, RDM2, correlation='Spearman'):
    if correlation == 'Spearman':
        corr, p_value = spearmanr(RDM1, RDM2)
    elif correlation == 'Pearson':
        corr, p_value = pearsonr(RDM1, RDM2)
    return corr, p_value


#%% The funcion 'flatten_RDM' requires as its only argument a dissimilarity
# matrix and reduces its unique lower half to a vector using np.triu_indices.
# This makes rows have precedence.  This
# means that, first, all elements from the first row will be selected
# and put into the vector, then all elements from the second row and so on.
# Therefore, the first element in the vector will be the element in
# the first row and second column of the matrix.  The second element in the
# vector wil be the element in the first row and third column of the matrix.
# Note that this function can also handle multiple 'dissim_mat' fed at once.

def flatten_RDM(dissim_mat: np.ndarray) -> np.ndarray:
    """Flattens the upper half of a dissimilarity matrix to a vector"""
    
    n_conditions = dissim_mat.shape[0]
    
    if not dissim_mat.ndim==3:
        dissim_mat = dissim_mat.reshape(n_conditions, n_conditions, 1)

    n_outputs = dissim_mat.shape[2]

    n_pairs = ((n_conditions * n_conditions) - n_conditions ) // 2

    dissim_vec = np.empty((n_pairs, n_outputs))

    idx = np.triu_indices(n_conditions, k=1)

    for output in range(n_outputs):
        dissim_vec[:, output] = dissim_mat[:, :, output][idx]

    return dissim_vec


#%%

def complete_RSA(activity_pattern_matrix_1, activity_pattern_matrix_2, correlation='Spearman'):

    discard, dissim_vec_1 = make_RDM(activity_pattern_matrix_1)
    discard, dissim_vec_2 = make_RDM(activity_pattern_matrix_2)

    corr, p_value = correlate_RDMs(dissim_vec_1, dissim_vec_2, correlation)

    return corr, p_value
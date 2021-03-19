# Import packages.
import numpy as np
from scipy.stats import spearmanr, pearsonr, rankdata

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

#%%
def noise_ceiling(reference_rdms, correlation='pearsonr'):
    n_subjects = reference_rdms.shape[2]
    
    reference_rdms = flatten_RDM(reference_rdms)
    
    if correlation=='pearsonr':
        reference_rdms = (reference_rdms - reference_rdms.mean(0)) / reference_rdms.std(0)
    elif correlation=='spearmanr':
        reference_rdms = rankdata(reference_rdms, axis=0)
    #TODO: maybe implement Kendall's tau_a
    
    reference_rdm_average = np.mean(reference_rdms, axis=1)
    
    upper_bound = 0
    lower_bound = 0
    
    for n in range(n_subjects):
        
        index = list(range(n_subjects))
        index.remove(n)
        rdm_n = reference_rdms[:,n]
        reference_rdm_average_loo = np.mean(reference_rdms[:,index], axis=1)
        upper_bound += np.corrcoef(reference_rdm_average, rdm_n)[0][1]
        lower_bound += np.corrcoef(reference_rdm_average_loo, rdm_n)[0][1]
        #TODO: maybe implement Kendall's tau_a
        
    upper_bound /= n_subjects
    lower_bound /= n_subjects
    
    return upper_bound, lower_bound



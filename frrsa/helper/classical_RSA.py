# Import packages.
import numpy as np
from scipy.stats import spearmanr, pearsonr, rankdata
from functools import partial
from scipy.spatial.distance import squareform

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
# matrix and reduces its unique upper half to a vector using np.triu_indices.
# This makes rows have precedence. This function can also handle multiple 
# RDMs fed at once, requiring the shape (n,n,m), where m denotes different 
# RDMs of shape (n,n).

def flatten_RDM(rdms: np.ndarray) -> np.ndarray:
    """Flattens the upper half of a dissimilarity matrix to a vector"""
    if rdms.ndim==3:
        mapfunc = partial(squareform, checks=False)
        V = np.array(list(map(mapfunc, np.moveaxis(rdms, -1, 0)))).T
    elif rdms.ndim==2:
        V = rdms[np.triu_indices(rdms.shape[0], k=1)].reshape(-1,1)
    return V


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



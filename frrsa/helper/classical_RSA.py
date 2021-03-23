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

def make_RDM(activity_pattern_matrix, distance='pearson'):
    """Based on an activation pattern matrix, a dissimilarity matrix is returned"""
    if distance=='pearson':
        rdm = 1 - np.corrcoef(activity_pattern_matrix, rowvar=False)
    # elif distance=='Euclidian':
    #     rdm = 1 - np.corrcoef(activity_pattern_matrix, rowvar=False)
    return rdm
#TODO: implement for multiple matrices.
#TODO: implement other distance norms 

#%% The funcion 'flatten_RDM' requires as its only argument a dissimilarity
# matrix and reduces its unique upper half to a vector using np.triu_indices.
# This makes rows have precedence. This function can also handle multiple 
# RDMs fed at once, requiring the shape (n,n,m), where m denotes different 
# RDMs of shape (n,n).

def flatten_RDM(rdm: np.ndarray) -> np.ndarray:
    """Flattens the upper half of a dissimilarity matrix to a vector"""
    if rdm.ndim==3:
        mapfunc = partial(squareform, checks=False)
        rdv = np.array(list(map(mapfunc, np.moveaxis(rdm, -1, 0)))).T
    elif rdm.ndim==2:
        rdv = rdm[np.triu_indices(rdm.shape[0], k=1)].reshape(-1,1)
    return rdv

#%% The function 'classical_RSA' performs a classical RSA between two RDMs.
# Hence, it requires to vectorised RDMs as inputs. By default, it performs
# Spearman's correlation. If the parameter 'correlation' is set to 'Pearson',
# Pearson's correlation will be performed.

def correlate_RDMs(rdv1, rdv2, correlation='pearson'):
    if correlation == 'pearson':
        corr, p_value = pearsonr(rdv1, rdv2)
    elif correlation == 'spearman':
        corr, p_value = spearmanr(rdv1, rdv2)
    return corr, p_value

#%%

def complete_RSA(activity_pattern_matrix_1, activity_pattern_matrix_2, correlation='pearson'):

    rdm1 = make_RDM(activity_pattern_matrix_1)
    rdm2 = make_RDM(activity_pattern_matrix_2)

    rdv1 = flatten_RDM(rdm1)
    rdv2 = flatten_RDM(rdm2)

    corr, p_value = correlate_RDMs(rdv1, rdv2, correlation)

    return corr, p_value
#TODO: implement multiple matrices.

#%%

def noise_ceiling(reference_rdms, correlation='pearson'):
    n_subjects = reference_rdms.shape[2]
    
    reference_rdms = flatten_RDM(reference_rdms)
    
    if correlation=='pearson':
        reference_rdms = (reference_rdms - reference_rdms.mean(0)) / reference_rdms.std(0)
    elif correlation=='spearman':
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



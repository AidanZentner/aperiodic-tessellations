import numpy as np
from numba import jit
from joblib import Parallel, delayed
from utils import grid_utils, fast_utils

@jit(nopython=True)
def get_distances_forward(list_arr: np.ndarray, nf_arr: np.ndarray, R: float) -> np.ndarray:
    """
    Calculates minimum distances based on a forward-shifted list.
    
    Args:
        list_arr: The 'list' array.
        nf_arr: The 'nf' integer index array. 
                (Assumes nf_arr contains 1-BASED indices)
        R: The radius value.
        
    Returns:
        A 1D array of minimums.
    """
    len_nf = len(nf_arr)
    list_forward = list_arr + (2 * R)

    # --- Vectorized calculation ---
    
    term1 = list_arr[nf_arr] - list_forward[:len_nf]
    term2 = list_arr[1 : len_nf + 1] - list_arr[:len_nf]

    mins = np.minimum(term1, term2)

    return mins

@jit(nopython=True)
def get_distances_backward(list_arr: np.ndarray, nb_arr: np.ndarray, R: float) -> np.ndarray:
    """
    Calculates minimum distances based on a backward-shifted list.
    
    Args:
        list_arr: The 'list' array.
        nb_arr: The 'nb' integer index array.
        R: The radius value.
        
    Returns:
        A 1D array of minimums.
    """
    len_nb = len(nb_arr)
    list_backward = list_arr - (2 * R)
    
    # --- Vectorized calculation ---
    
    len1 = len(list_arr) - len_nb
    idx2_py_arr = np.arange(len_nb) + len1 - 1
    
    term1 = list_arr[nb_arr] - list_backward[idx2_py_arr]
    term2 = list_arr[idx2_py_arr + 1] - list_arr[idx2_py_arr]

    mins = np.minimum(term1, term2)

    return mins

def nv1D(list_arr: np.ndarray, R: float) -> float:
    """
    This function calculates the variance of the number of points in a 1D tessellation.

    Args:
        list_arr: The main 'list' array.
        R: The radius value.
    
    Returns:
        The 'sigma2' (variance) value.
    """
    
    # --- Forward calculation ---

    list_forward = list_arr + 2 * R
    nf = fast_utils.cf_forward_numba(list_arr, list_forward)

    indices = fast_utils.indexFn_numba(np.array([nf], dtype=int), 0, len(list_arr)-1, np.array([1.0]))
    idx_start_f = indices[0]
    idx_end_f = indices[-1]
    nf = 1 + nf[idx_start_f : idx_end_f + 1]

    list_forward = list_forward[idx_start_f : idx_end_f + 1]
    list_indices_f = np.arange(0, len(nf))
    num_f = nf - list_indices_f
    
    nf_dists = get_distances_forward(list_arr, nf, R)

    # --- Backward calculation ---

    list_backward = list_arr - 2 * R
    nb = fast_utils.cf_forward_numba(list_arr, list_backward)

    indices = fast_utils.indexFn_numba(np.array([nb], dtype=int), -1, len(list_arr)-1, np.array([1.0]))
    idx_start_b = indices[0]
    idx_end_b = indices[-1]
    nb = 1 + nb[idx_start_b : idx_end_b]

    list_backward = list_backward[idx_start_b : idx_end_b]
    list_indices_b = np.arange(len(list_arr) - len(nb), len(list_arr))
    num_b = list_indices_b - nb + 1

    nb_dists = get_distances_backward(list_arr, nb, R)

    # --- Final calculations ---
    
    # Pre-calculate sums for efficiency
    sum_nf_dists = np.sum(nf_dists)
    sum_nb_dists = np.sum(nb_dists)
    total_dists = sum_nf_dists + sum_nb_dists
    
    mu_num = np.sum(num_f * nf_dists) + np.sum(num_b * nb_dists)
    mu = mu_num / total_dists
    
    moment2_num = np.sum(num_f**2 * nf_dists) + np.sum(num_b**2 * nb_dists)
    moment2 = moment2_num / total_dists
    
    sigma2 = moment2 - mu**2
    
    return sigma2


def get_nv1D(case_name, case_args, radii):

    Nmax = int(np.floor(25 * radii[-1]))         # not quite the right heuristic
    Xgen = grid_utils.get_Xgen(case_name, case_args, 0., Nmax, np.array([[1., 0.]]))

    X1D = Xgen(0)[0] # seed = 0

    results = Parallel(n_jobs=-1)(delayed(nv1D)(X1D, rr) for rr in radii) # parallel execution
    final_data = np.array([radii, results]).T

    return final_data

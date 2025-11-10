import numpy as np
from numba import jit
from joblib import Parallel, delayed

from utils import grid_utils, fast_utils

#------------------------- 2D Number Variance Functions -------------------------#

#--------------------------------------------------#
#-------------- Histogram Functions ---------------#

@jit(nopython=True)
def matchRadius_numba(a, b):
    """
    Linear scan histogram.
    """
    n_bins = a.shape[0] + 1
    n_vals = b.shape[0]
    idx = np.zeros(n_bins, dtype=np.int64) 
    i = 0 # bin index

    for j in range(n_vals):
        bj = b[j]
        while i < (n_bins - 1) and a[i] < bj:
            i += 1
        while i > 0 and a[i-1] > bj:
            i -= 1
        idx[i] += 1
    return idx

#--------------------------------------------------#
#-------------- Counting Functions ---------------#

@jit(nopython=True)
def countFn(i, j, X, center, radii, e):

    #----------- Define necessary constants -----------#

    lenX = X.shape[1]

    Q = e.shape[0]
    etrans = e.T  # (2, Q) matrix

    orthos = np.zeros(e.shape)
    orthos[:, 0] = -e[:, 1]
    orthos[:, 1] =  e[:, 0]

    n_lines = lenX - 2
    n_radii = radii.shape[0]
    
    indices = np.empty(Q - 2, dtype=np.int64)
    k = 0
    for idx in range(Q):
        if idx != i and idx != j:
            indices[k] = idx
            k += 1
            
    dotij = np.dot(e[i], e[j])
    inv_denom = 1.0 / (1.0 - dotij**2)
    vec1 = (e[i] - dotij * e[j]) * inv_denom
    vec2 = (e[j] - dotij * e[i]) * inv_denom

    #----------- Define coefficients -----------#

    coeffs1 = np.empty(Q - 2)
    coeffs2 = np.empty(Q - 2)
    for k in range(Q - 2):
        coeffs1[k] = np.dot(e[indices[k]], vec1)
        coeffs2[k] = np.dot(e[indices[k]], vec2)

    Xj_slice = X[j, 1 : lenX - 1].copy() # 1 : 2*Nmax
    term2 = coeffs2.reshape(-1, 1) * Xj_slice.reshape(1, -1)

    val_scalar = np.dot(e[j], orthos[i])
    val_vector = np.empty(Q - 2)
    for k in range(Q - 2):
        val_vector[k] = np.dot(e[indices[k]], orthos[i])
    values = val_scalar * val_vector
    
    Qtuples = np.zeros((Q, n_lines))
    Qtuples[i, :] = 0.5 
    Qtuples[j, :] = np.arange(2, lenX) - 0.5 # 2 : 2 * Nmax + 1
    
    ctr = np.zeros(n_radii + 1, dtype=np.int64)
    center_col = center.reshape(2, 1)

    for nn in range(1, n_lines + 1):
        Qtuples[i, :] = nn + 0.5
        
        X_i_val = X[i, nn]
        
        proj = coeffs1.reshape(-1, 1) * X_i_val + term2

        qtuples_sub_array = np.empty((Q - 2, n_lines))
        for k in range(Q - 2):
            k_idx = indices[k]
            X_k = X[k_idx, :]
            if values[k] > 0.:
                Qtuples[k_idx, :] = fast_utils.cf_forward_numba(X_k, proj[k, :])
            else:
                Qtuples[k_idx, :] = fast_utils.cf_backward_numba(X_k, proj[k, :])
            qtuples_sub_array[k, :] = Qtuples[k_idx, :]
        
        #----------- Find the min_val for invalid ordinals -----------#

        lower, upper = fast_utils.indexFn_numba(qtuples_sub_array, -1, lenX - 1, values)

        #----------- Add 1 to the Qtuples -----------#

        Qtuples[indices] = 1 + Qtuples[indices] # Added for now. FIGURE OUT HOW TO DO THIS PROPERLY.

        if lower <= upper:
            valid_tuples = Qtuples[:, lower : upper + 1].copy() 
            pts = etrans @ valid_tuples
            diff = pts - center_col
            norms = np.sqrt(diff[0,:]**2 + diff[1,:]**2)
            ctr += matchRadius_numba(radii, norms)
            
    return np.cumsum(ctr[:-1])

def _run_rendition(s, radii, e, Xgen):
    """
    Helper function to run a single "rendition" (one seed).
    """
    X = Xgen(s)
    rng = np.random.default_rng(s)
    center = rng.uniform(-1.0, 1.0, 2)
    
    total_counts = np.zeros(len(radii), dtype=np.int64)
    
    Q = e.shape[0]
    indexPairs = np.array([(i, j) for i in range(Q-1) for j in range(i+1, Q)])

    for pair in indexPairs:
        i = pair[0]
        j = pair[1]
        
        total_counts += countFn(i, j, X, center, radii, e)
    return total_counts

def hypParameters(case_name, case_args, gamma, nRends, radii, e):

    seed_array = np.arange(1, nRends + 1)

    Nmax = int(np.floor(2 * radii[-1] / 2.35))
    Xgen = grid_utils.get_Xgen(case_name, case_args, gamma, Nmax, e)

    print(f"Running {nRends} renditions in *parallel* (using all cores)...")
    
    # --- PARALLEL EXECUTION ---
    # n_jobs=-1 uses all available cores
    # We pass all necessary arguments to the delayed function
    rawData_list = Parallel(n_jobs=-1)(
        delayed(_run_rendition)(s, radii, e, Xgen) 
        for s in seed_array
    )
    # --- END PARALLEL EXECUTION ---
    
    print("All renditions complete. Post-processing...")
    
    # --- Post-processing (this is identical to your serial code) ---
    rawData = np.array(rawData_list) # (nRends, nn+1)
    rawData_by_radius = rawData.T    # (nn+1, nRends)
    
    # Get counts for the last radius
    counts_at_Rmax = rawData_by_radius[-1, :]
    densities = counts_at_Rmax / (np.pi * radii[-1]**2)
    meanDensities = np.mean(densities)
    print(f"Computed Mean Density (for bookkeeping): {meanDensities}")
    
    # Compute variance for each radius
    variances = np.var(rawData_by_radius, axis=1) # (nn+1,)
    
    # Stack radii and variances as columns
    NVArray = np.stack([radii, variances], axis=1) # (nn+1, 2)

    return NVArray


#------------------------- 1D Number Variance Functions -------------------------#


import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import grid_utils
from tiling_utils import cf_forward_numba, cf_backward_numba, indexFn_numba

# --- Snippet 1: Initial Variable Definitions ---

tau = (1 + np.sqrt(5)) / 2  # Golden Ratio
Q = 5  # The number of star vectors

# Create Q star vectors
theta = np.pi / 2
j = np.arange(Q)  # [0, 1, 2, 3, 4]
angles = theta + j * 2 * np.pi / Q
e = np.stack([np.cos(angles), np.sin(angles)], axis=1) # (Q, 2) array

T = 1.0 / np.linalg.norm(e, axis=1)  # (Q,) array of period lengths
eNorm = T[:, np.newaxis] * e  # (Q, 2) array of normalized vectors
etrans = e.T  # (2, Q) matrix
orthos = np.stack([-e[:, 1], e[:, 0]], axis=1) # (Q, 2) array of orthogonal vectors

CaseName = "Penrose"
ptsType = "Tiles"  # Using "Tiles" to match your final hypParameters snippet
Rmax = 1000
Rmin = 1
nn = 10**4
nRends = 5000  # Note: 5000 renditions will take a long time!
grids = np.arange(Q)  # [1, 2, 3, 4]

# Create 1-based index pairs
index_pairs_list = []
for i_idx in grids[:-1]:
    for j_idx in np.arange(i_idx + 1, Q):
        index_pairs_list.append((i_idx, j_idx))
indexPairs = np.array(index_pairs_list)

ShortCaseName = "Penrose"
args = None

seed = np.arange(1, nRends + 1)  # Creates the list of random seeds

# --- Nmax and Radii Setup ---
Nmax = 0
if ptsType == "Grid":
    Nmax = np.floor(Rmax / 2.35)
    Rmin = 0.95 * Rmin * (avgL / 2.35)
    Rmax = 0.95 * avgL * Nmax
elif ptsType == "Tiles":
    Nmax = np.floor(2 * Rmax / 2.35)

Nmax = int(Nmax) # Ensure Nmax is an integer

radii = np.logspace(np.log10(Rmin), np.log10(Rmax), num=nn + 1)

# !!! CRITICAL: 'gamma' was never defined in your snippets.
# I am setting it to 0.0 as a placeholder.
# You MUST change this to your intended value.
gamma = 0.0

Xgen = grid_utils.get_Xgen(CaseName, args, gamma, Nmax, T, Q)

def get_centres(seed_val):
    """
    Generates a deterministic centre point
    """
    rng = np.random.default_rng(seed_val)
    # Generate centres from the *same* seed for reproducibility
    centres = rng.uniform(-1.0, 1.0, 2)
    return centres

# --- Snippets 2 & 3: Numba-JIT Helper Functions ---

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

@jit(nopython=True)
def countFn(i, j, X, centres, radii, e, etrans, orthos, Q, Nmax):

    n_lines = 2 * Nmax - 1
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

    coeffs1 = np.empty(Q - 2)
    coeffs2 = np.empty(Q - 2)
    for k in range(Q - 2):
        coeffs1[k] = np.dot(e[indices[k]], vec1)
        coeffs2[k] = np.dot(e[indices[k]], vec2)

    Xj_slice = X[j, 1 : 2*Nmax].copy()
    term2 = coeffs2.reshape(-1, 1) * Xj_slice.reshape(1, -1)

    val_scalar = np.dot(e[j], orthos[i])
    val_vector = np.empty(Q - 2)
    for k in range(Q - 2):
        val_vector[k] = np.dot(e[indices[k]], orthos[i])
    values = val_scalar * val_vector
    
    Qtuples = np.zeros((Q, n_lines))
    Qtuples[i, :] = 0.5 
    Qtuples[j, :] = np.arange(2, 2 * Nmax + 1) - 0.5
    
    ctr = np.zeros(n_radii + 1, dtype=np.int64)
    centres_col = centres.reshape(2, 1)

    for nn in range(1, n_lines + 1):
        Qtuples[i, :] = nn + 0.5
        
        X_i_val = X[i, nn]
        
        proj = coeffs1.reshape(-1, 1) * X_i_val + term2

        qtuples_sub_array = np.empty((Q - 2, n_lines))
        for k in range(Q - 2):
            k_idx = indices[k]
            X_k = X[k_idx, :]
            if values[k] > 0.:
                Qtuples[k_idx, :] = cf_forward_numba(X_k, proj[k, :])
            else:
                Qtuples[k_idx, :] = cf_backward_numba(X_k, proj[k, :])
            qtuples_sub_array[k, :] = Qtuples[k_idx, :]
        
        # --- Find the min_val for invalid ordinals ---
        lower, upper = indexFn_numba(qtuples_sub_array, -1, 2*Nmax, values)

        # ---

        Qtuples[indices] = 1 + Qtuples[indices] # Added for now. FIGURE OUT HOW TO DO THIS PROPERLY.

        if lower <= upper:
            valid_tuples = Qtuples[:, lower : upper + 1].copy() 
            pts = etrans @ valid_tuples
            diff = pts - centres_col
            norms = np.sqrt(diff[0,:]**2 + diff[1,:]**2)
            ctr += matchRadius_numba(radii, norms)
            
    return np.cumsum(ctr[:-1])

# --- Snippet 5: 'hypParameters' and its Helper (Serial Version) ---

def _run_rendition(s, indexPairs, radii, e, etrans, orthos, Q, Nmax, gamma):
    """
    Helper function to run a single "rendition" (one seed).
    This is what will be executed in parallel.
    """
    X = Xgen(s) 
    rng = np.random.default_rng(s)
    centres = rng.uniform(-1.0, 1.0, 2)
    
    total_counts = np.zeros(len(radii), dtype=np.int64)
    
    for pair in indexPairs:
        i = pair[0]
        j = pair[1]
        
        total_counts += countFn(i, j, X, centres, radii, e, etrans, orthos, Q, Nmax)
    return total_counts

# ---
# --- FIX 3: Modify hypParameters to use joblib.Parallel ---
# ---
def hypParameters(seed_array, indexPairs, radii, e, etrans, orthos, Q, Nmax, Rmax, gamma):

    print(f"Running {len(seed_array)} renditions in *parallel* (using all cores)...")
    
    # --- PARALLEL EXECUTION ---
    # n_jobs=-1 uses all available cores
    # We pass all necessary arguments to the delayed function
    rawData_list = Parallel(n_jobs=-1)(
        delayed(_run_rendition)(s, indexPairs, radii, e, etrans, orthos, Q, Nmax, gamma) 
        for s in seed_array
    )
    # --- END PARALLEL EXECUTION ---
    
    print("All renditions complete. Post-processing...")
    
    # --- Post-processing (this is identical to your serial code) ---
    rawData = np.array(rawData_list) # (nRends, nn+1)
    rawData_by_radius = rawData.T    # (nn+1, nRends)
    
    # Get counts for the last radius (Rmax)
    counts_at_Rmax = rawData_by_radius[-1, :]
    densities = counts_at_Rmax / (np.pi * Rmax**2)
    meanDensities = np.mean(densities)
    print(f"Computed Mean Density (for bookkeeping): {meanDensities}")
    
    # Compute variance for each radius
    variances = np.var(rawData_by_radius, axis=1) # (nn+1,)
    
    # Stack radii and variances as columns
    NVArray = np.stack([radii, variances], axis=1) # (nn+1, 2)

    return NVArray

# --- Main Execution ---
if __name__ == "__main__":
    
    # --- FOR TESTING: Use a smaller number of renditions ---
    # Using 5000 will be very slow in series.
    # Let's override 'nRends' and 'seed' for a quick test.
    nRends_test = 100
    seed = np.arange(1, nRends_test + 1)
    # ---
    
    print(f"Starting calculation with {len(seed)} renditions...")

    import time
    start_time = time.time()

    # This is the line you wanted to run:
    outputs = hypParameters(
        seed, 
        indexPairs, 
        radii, 
        e, 
        etrans, 
        orthos, 
        Q, 
        Nmax, 
        Rmax,
        gamma
    )

    end_time = time.time()    

    print("\n--- Finished ---")
    print(f"Time taken: {end_time - start_time} seconds")

    print("Output shape:", outputs.shape)

    # Export as csv file
    np.savetxt("/Users/aidenzentner/Desktop/number_variance_full.csv", outputs, delimiter=',')

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    ax.plot(outputs[:, 0], outputs[:, 1], '-', linewidth=1, color='black')
    ax.set_label(r'$R$')
    ax.set_ylabel(r'$\sigma^2(R)$')
    ax.set_title('Number Variance')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.show()
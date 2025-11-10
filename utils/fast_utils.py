import numpy as np
from numba import jit

#--------------------------------------------------#
#             Numba-JIT Helper Functions           #
#--------------------------------------------------#

#--------------------------------------------------#
#-------------- Marching Algorithms ---------------#

@jit(nopython=True)
def cf_forward_numba(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Finds the index of the nearest LESSER element in a for each element in b.
    Assumes a and b are sorted ascending.
    This is an O(n+m) "marching" algorithm.

    Args:
        a: 1D NumPy array, sorted ascending.
        b: 1D NumPy array, sorted ascending.

    Returns:
        A 1D NumPy array of indices, the same length as b.
        idx[j] = i, where a[i] is the nearest element <= b[j].
        Returns -1 if b[j] is less than all elements in a.
    """
    n = a.shape[0]
    m = b.shape[0]
    # Use np.empty for performance, as we will fill every element.
    # We'll use np.int64 for robust index typing.
    idx = np.empty(m, dtype=np.int64)
    
    i = 0 # Pointer for 'a' array (0-indexed)
    
    # Outer loop over 'b'
    for j in range(m):
        bj = b[j]
        
        # Inner "while" loop advances the 'i' pointer
        # We march 'i' forward as long as it's in bounds and a[i] < b[j]
        while (i < n) and (a[i] < bj):
            i += 1
        
        # When the loop stops, a[i] is the first element >= bj.
        # Therefore, the nearest *lesser* element is at index i - 1.
        idx[j] = i - 1
        
    return idx

@jit(nopython=True)
def cf_backward_numba(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Finds the index of the nearest LESSER element in a for each element in b.
    Assumes a is sorted ascending and b is sorted descending.
    This is an O(n+m) "marching" algorithm.

    Args:
        a: 1D NumPy array, sorted ascending.
        b: 1D NumPy array, sorted descending (reverse).

    Returns:
        A 1D NumPy array of indices, the same length as b.
        idx[j] = i, where a[i] is the nearest element <= b[j].
        Returns -1 if b[j] is less than all elements in a.
    """
    n = a.shape[0]
    m = b.shape[0]
    idx = np.empty(m, dtype=np.int64)
    
    # Start 'i' at the *last* valid index of 'a'
    i = n - 1 
    
    # Outer loop over 'b'
    for j in range(m):
        bj = b[j]
        
        # Inner "while" loop moves the 'i' pointer *backward*
        # We move 'i' back as long as it's in bounds and a[i] >= b[j]
        while (i >= 0) and (a[i] >= bj):
            i -= 1
            
        # When the loop stops, a[i] is the first element (from the right)
        # that is <= bj.
        # This 'i' is the correct index.
        idx[j] = i
        
    return idx

#--------------------------------------------------#
#--------------- Indexing Algorithms --------------#

@jit(nopython=True)
def lowerIndex_numba(list_arr, target):
    """
    Finds the 0-based index *after* the last occurrence of 'target'.
    """
    return np.searchsorted(list_arr, target, side='right')

@jit(nopython=True)
def upperIndex_numba(list_arr, target):
    """
    Finds the 0-based index *of the first* occurrence of 'target'.
    """
    return np.searchsorted(list_arr, target, side='left')

@jit(nopython=True)
def indexFn_numba(qtuples_indices, min_val, max_val, values):
    """
    Finds the min and max indices of the valid tuples.
    """
    m = qtuples_indices.shape[0]
    if m == 0:
        return (0, -1) # Return invalid slice

    n = qtuples_indices.shape[1]
    current_lower = 0       # Start with the full slice
    current_upper = n - 1   # (inclusive)

    for ii in range(m):
        ll = qtuples_indices[ii, :]
        
        # --- Find the local bounds for this list (ll) ---
        local_lower = 0
        local_upper = n - 1
        
        if values[ii] > 0.: # Forward-sorted
            # Find index *after* last 'min_val'
            local_lower = lowerIndex_numba(ll, min_val)
            # Find index *of first* 'max_val'
            local_upper = upperIndex_numba(ll, max_val) - 1
        
        else: # Reverse-sorted
            # Find index *after* last 'max_val'
            local_lower = qtuples_indices.shape[1] - upperIndex_numba(np.flip(ll), max_val)
            # Find index *of first* 'min_val'
            local_upper = qtuples_indices.shape[1] - lowerIndex_numba(np.flip(ll), min_val) - 1
        
        # --- Intersect the bounds ---
        if local_lower > current_lower:
            current_lower = local_lower
        if local_upper < current_upper:
            current_upper = local_upper

        # Check for non-intersection
        if current_lower > current_upper:
            return (0, -1) # Return invalid slice
            
    # Return the final, intersected (inclusive) bounds
    return (current_lower, current_upper)
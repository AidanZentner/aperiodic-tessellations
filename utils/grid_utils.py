import numpy as np

def get_Xgen(name, args, gamma, Nmax, e):
    """
    Initializes the Xgen function for a given case name.
    """

    #-------------------------------------------------#
    #------------ Penrose Initialization -------------#
    
    Q = e.shape[0]
    T = 1.0 / np.linalg.norm(e, axis=1)  # (Q,) array of period lengths

    if name == "Penrose":

        avgL = min(T)

        def Xgen(seed_val):

            rng = np.random.default_rng(seed_val)
            gamma_vec = rng.uniform(-avgL, avgL, Q - 1)

            last_gamma = gamma - np.sum(gamma_vec)
            gamma_vec = np.append(gamma_vec, last_gamma)

            N_vals = np.arange(-Nmax, Nmax + 1)

            sum_arr = N_vals[:, np.newaxis] + gamma_vec[np.newaxis, :]
            result = sum_arr * T[np.newaxis, :]

            return result.T

    #--------------------------------------------------#
    #------------- Poisson Initialization -------------#

    elif name == "Poisson":

        avgL = min(T) / 2
        seq_len = 2 * Nmax + 1

        def Xgen(seed_val):

            # Set gamma offsets

            rng = np.random.default_rng(seed_val)
            gamma_vec = rng.uniform(-avgL, avgL, Q - 1)

            last_gamma = gamma - np.sum(gamma_vec)
            gamma_vec = np.append(gamma_vec, last_gamma)

            # Get spacings

            xArray = rng.uniform(-Nmax / 2, Nmax / 2, size=(Q, seq_len))
            xArray = np.sort(xArray, axis=1)
            x0 = gamma_vec - np.median(xArray, axis=1)
            result = (x0[:, None] + xArray) * T[:, None]

            return result

    #-------------------------------------------------#
    #----------- Perturbed Integer Lattice -----------#

    elif name == "Perturbed Integer Lattice":

        ll = args[0]
        avgL = min(T)

        def Xgen(seed_val):

            # Set gamma offsets

            rng = np.random.default_rng(seed_val)
            gamma_vec = rng.uniform(-avgL, avgL, Q - 1)

            last_gamma = gamma - np.sum(gamma_vec)
            gamma_vec = np.append(gamma_vec, last_gamma)

            # Get spacings

            N_vals = np.arange(-Nmax, Nmax + 1)

            sum_arr = N_vals[:, np.newaxis] + gamma_vec[np.newaxis, :]
            result = (sum_arr + np.random.uniform(-ll, ll, size=[2 * Nmax + 1, Q])) * T[np.newaxis, :]

            return result.T

    #-----------------------------------------------------#
    #--------------- Substitution Sequence ---------------#

    elif name == "Substitution Sequence":

        a, b, c, d = args

        # Length-related constants

        xi = (d - a + np.sqrt((a - d)**2 + 4 * b * c)) / (2 * c)
        S = 1.0
        L = xi

        Ssub = np.array([S] * a + [L] * c)
        Lsub = np.array([S] * b + [L] * d)

        avgL = min(T) * (b + c * xi**2) / (b + c * xi)

        max_len = 10**7
        seq_len = 2 * Nmax + 1

        # Compute substitutions
        spac = np.array([S], dtype=float)
        while spac.size < max_len:
            # Boolean mask for S vs L
            mask = spac == S
            # Substitute using list comprehension (efficient because of NumPy indexing)
            spac = np.concatenate([Ssub if m else Lsub for m in mask])
        # Truncate to max_len
        spac = spac[:max_len]

        def Xgen(seed_val):

            # Set gamma offsets

            rng = np.random.default_rng(seed_val)
            gamma_vec = rng.uniform(-avgL, avgL, Q - 1)

            last_gamma = gamma - np.sum(gamma_vec)
            gamma_vec = np.append(gamma_vec, last_gamma)

            # Get spacings

            starts = rng.integers(0, max_len - seq_len + 1, size=Q)
            offsets = np.arange(seq_len)
            spacings = spac[starts[:, None] + offsets[None, :]]
            xArray = np.cumsum(spacings, axis=1)
            x0 = gamma_vec - np.median(xArray, axis=1)
            result = (x0[:, None] + xArray) * T[:, None]

            return result

    #---------------------------------------------------#
    #----------------- 2-Length Random -----------------#

    elif name == "2-Length Random":

        length, weight = args

        lengths = np.array([1, length])
        weights = np.array([1 - weight, weight])

        avgL = min(T) * np.dot(lengths, weights)

        seq_len = 2 * Nmax + 1

        def Xgen(seed_val):

            # Set gamma offsets

            rng = np.random.default_rng(seed_val)
            gamma_vec = rng.uniform(-avgL, avgL, Q - 1)

            last_gamma = gamma - np.sum(gamma_vec)
            gamma_vec = np.append(gamma_vec, last_gamma)

            # Get spacings

            spacings = rng.choice(lengths, size=(Q, seq_len), p=weights)
            xArray = np.cumsum(spacings, axis=1)
            x0 = gamma_vec - np.median(xArray, axis=1)

            result = (x0[:, None] + xArray) * T[:, None]

            return result

    else:
        raise ValueError(f"Invalid case name: {name}")

    return Xgen
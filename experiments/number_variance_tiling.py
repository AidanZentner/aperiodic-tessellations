import numpy as np
import matplotlib.pyplot as plt

from utils import tiling_utils

# --- Initial Variable Definitions ---

tau = (1 + np.sqrt(5)) / 2  # Golden Ratio
Q = 5  # The number of star vectors

# Create Q star vectors
theta = np.pi / 2
angles = theta + (2 * np.pi / Q) * np.arange(Q)
e = np.stack([np.cos(angles), np.sin(angles)], axis=1) # (Q, 2) array

# --- Define Tiling ---

case_name = "Penrose"
gamma = 0.0

if case_name == "Penrose" or case_name == "Poisson":
    case_args = None
elif case_name == "Perturbed Integer Lattice":
    ll = 0.5
    case_args = (ll,)
elif case_name == "Substitution Sequence":
    a, b, c, d = 0, 2, 2, 2
    case_args = (a, b, c, d)
elif case_name == "2-Length Random":
    length, weight = tau, 1./tau
    case_args = (length, weight)
else:
    raise ValueError(f"Invalid case name: {case_name}")

# --- Define evaluation parameters

nRends = 100

Rmin = 1
Rmax = 1000
nn = 10**3
radii = np.logspace(np.log10(Rmin), np.log10(Rmax), num=nn + 1)

# --- Main Execution ---

if __name__ == "__main__":
    
    print(f"Starting calculation with {nRends} renditions...")

    import time
    start_time = time.time()

    # This is the line you wanted to run:
    outputs = tiling_utils.hypParameters(
        case_name,
        case_args,
        gamma,
        nRends, 
        radii, 
        e
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
import numpy as np


# ---------------
# DATA GENERATION
# ---------------
    

def generate_random_mixed_state(n_qubits: int) -> np.ndarray:
    """
    Generate a random mixed state for n_qubits qubits using the Ginibre ensemble.
    Helper function for data generation.

    This guarantees Hermiticity and positive semi-definiteness with trace 1
    Only mixed states are produced here.
    """
    dim = 2 ** n_qubits
    G = (np.random.normal(size=(dim, dim))
         + 1j * np.random.normal(size=(dim, dim)))
    rho = G @ G.conj().T
    rho /= np.trace(rho)
    return rho

def generate_dataset_of_states_and_probabilities(N: int, n_qubits: int):
    """
    A function to generate a dataset of N random mixed states and their corresponding Cholesky decompositions.
    
    :param N: Number of random mixed states and probabilities to generate
    :param n_qubits: Number of qubits in the system

    Outputs:
        rhos: (N,) array of 2^n_qubits x 2^n_qubits density matrices
        taus: (N,) array of 2^n_qubits x 2^n_qubits Cholesky decompositions of the density matrices
    """

    rhos = np.empty(N, dtype=object)
    taus = np.empty(N, dtype=object)
    for k in range(N):
        rho = generate_random_mixed_state(n_qubits)
        rhos[k] = rho
        taus[k] = np.linalg.cholesky(rho)

    return rhos, taus

#--------------------------------------------------
# SETTING UP IDEAL/ NOISY PROJECTOR MATRICES
#--------------------------------------------------

def proj(ket: np.ndarray) -> np.ndarray:
    """|ket><ket| projector."""
    return ket @ ket.conj().T

def build_projector_matrix(n_qubits: int) -> np.ndarray:
    """
    Build the measurement projector matrix P for n_qubits qubits.
    Projector matrix will be 6^n_qubits x 6^n_qubits, with each entry being a 2^n_qubits x 2^n_qubits projector.

    The projectors represent different measurements of the photon polarisations
    """
    H = np.array([[1.0], [0.0]], dtype=complex)
    V = np.array([[0.0], [1.0]], dtype=complex)

    D = (H + V) / np.sqrt(2)
    A = (H - V) / np.sqrt(2)
    R = (H + 1j * V) / np.sqrt(2)
    L = (H - 1j * V) / np.sqrt(2)

    h = proj(H)
    v = proj(V)
    d = proj(D)
    a = proj(A)
    r = proj(R)
    l = proj(L)

    projs = [h, v, d, a, r, l]

    if n_qubits == 1:
        P = np.empty((6), dtype=object)
        for i in range(6):
            P[i] = projs[i]
        return P
    
    if n_qubits == 2:
        P = np.empty((6, 6), dtype=object)
        for i in range(6):
            for j in range(6):
                P[i, j] = np.kron(projs[i], projs[j])
        return P
    
    if n_qubits == 3:
        P = np.empty((6, 6, 6), dtype=object)
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    P[i, j, k] = np.kron(np.kron(projs[i], projs[j]), projs[k])
        return P
    
    if n_qubits == 4:
        P = np.empty((6, 6, 6, 6), dtype=object)
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    for l in range(6):
                        P[i, j, k, l] = np.kron(np.kron(np.kron(projs[i], projs[j]), projs[k]), projs[l])
        return P
    
    else: 
        raise NotImplementedError("Projector matrix construction only implemented for up to 4 qubits.")
        

def simulate_waveplate_misalignment(sigma: float, P: np.ndarray, n_qubits: int) -> np.ndarray:
    """Rotate the entire projector matrix P by independent noisy unitaries on each qubit.
    
    This is a helper function designed to simulate the effect of misalignment in the measurement apparatus, 
    ie. waveplates not perfectly correct.
    """

    def noisy_unitary(sigma: float) -> np.ndarray:
        """Random SU(2) unitary from Gaussian-distributed Euler angles."""
        theta, phi, zeta = np.random.normal(0, sigma, 3)
        U = np.array([
            [np.exp(1j * phi / 2) * np.cos(theta),
             -1j * np.exp(1j * zeta) * np.sin(theta)],
            [-1j * np.exp(-1j * zeta) * np.sin(theta),
             np.exp(-1j * phi / 2) * np.cos(theta)]
        ], dtype=complex)
        return U
    
   
    unitaries = [noisy_unitary(sigma) for _ in range(n_qubits)]
    U = unitaries[0]
    for i in range(1, n_qubits):
        U = np.kron(U, unitaries[i])
    noisy_P = np.empty_like(P, dtype=object)

# Unitaries are applied to each projector in the same way, as it can be assumed that the misalignment is fixed for the entire measurement process.
# Motors are so reproducible that the same misalignment will occur for each measurement setting, with minimal variational error between them.
        
    if n_qubits == 1:
        for i in range(6):
            noisy_P[i] = U @ P[i] @ U.conj().T
        return noisy_P

    if n_qubits == 2:
        for i in range(6):
            for j in range(6):
                noisy_P[i, j] = U @ P[i, j] @ U.conj().T
        return noisy_P

    if n_qubits == 3:
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    noisy_P[i, j, k] = U @ P[i, j, k] @ U.conj().T
        return noisy_P

    if n_qubits == 4:
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    for l in range(6):
                        noisy_P[i, j, k, l] = U @ P[i, j, k, l] @ U.conj().T
        return noisy_P
    

def get_measurement_probs_from_P_and_rho(rho: np.ndarray, P: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Return 6^n_qubits matrix of measurement probabilities M_ij = Tr(rho P_ij).
    Gives the measurement probabilities for a given density matrix rho and projector matrix P.
    Helper function for data generation.

    The measurement probabilities are for each of the 6^n n-qubit projectors
    """
    shape = tuple([6] * n_qubits)
    M = np.empty(shape, dtype=float)

    if n_qubits == 1:
        for i in range(6):
            M[i] = np.real(np.trace(rho @ P[i]))
        return M / 3
    
    if n_qubits == 2:
        for i in range(6):
            for j in range(6):
                M[i, j] = np.real(np.trace(rho @ P[i, j]))
        return M / 9
    
    if n_qubits == 3:
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    M[i, j, k] = np.real(np.trace(rho @ P[i, j, k]))
        return M / 27
    
    if n_qubits == 4:
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    for l in range(6):
                        M[i, j, k, l] = np.real(np.trace(rho @ P[i, j, k, l]))
        return M / 81
    
    else: 
        raise NotImplementedError("Measurement only implemented for up to 4 qubits.")
    

def simulate_interference_visibility(p: np.ndarray, visibility: float) -> np.ndarray:
    """
    Mix probabilities with uniform noise to model limited measurement visibility/contrast.
    visibility=1 gives p unchanged; visibility<1 washes outcomes towards uniform.
    """

    u = np.ones_like(p, dtype=float) / p.size
    p_washed = visibility * p + (1.0 - visibility) * u

    p_washed = np.maximum(p_washed, 0.0)
    p_washed /= p_washed.sum()

    return p_washed


def retrieve_counts_from_n_shots_per_state(p: np.ndarray, n_shots: int) -> np.ndarray:
    """
    Sample integer outcome counts from probabilities p using 'shots' repetitions
    (multinomial counting statistics).
    """

    p_flat = np.asarray(p, dtype=float).ravel()
    p_flat = np.maximum(p_flat, 0.0)
    p_flat /= p_flat.sum()

    counts_flat = np.random.multinomial(n_shots, p_flat)
    return counts_flat.reshape(p.shape)

# ---------------------
# STOKES RECONSTRUCTION
# ---------------------

def counts_to_frequencies(counts: np.ndarray, shots: int) -> np.ndarray:
    """
    Convert outcome counts to empirical frequencies (probability estimates).
    """

    f = np.asarray(counts, dtype=float) / shots
    return f


def build_stokes_matrix(P: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Build the Stokes matrix A such that f = A @ vec(rho),
    where f are flattened measurement frequencies.
    """
    d = 2 ** n_qubits
    m = P.shape[0]  # number of single-qubit projectors (e.g. 6)

    A = np.zeros((m**n_qubits, d*d), dtype=complex)

    for t in range(m**n_qubits):

        # Iterating over t gives full tuple index, (0,0) through (5,5) for a 6x6 for example
        idx = [0] * n_qubits
        x = t
        for q in range(n_qubits - 1, -1, -1):
            idx[q] = x % m
            x //= m # shifts everything right by one digit in base m

        # Row = vec(P_alpha^T) : projector matrix for that outcome, transposed and flattened
        A[t] = P[tuple(idx)].T.reshape(-1)

    return A


def solve_stokes_linear_system(A: np.ndarray, f: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Solve the Stokes linear system A vec(rho) = f in the least-squares sense.
    Returns the reconstructed density matrix (with heriticity and unit trace ensured).
    """
    # Flatten frequencies
    f_flat = np.asarray(f, dtype=complex).reshape(-1)

    # Least-squares solve
    x, *_ = np.linalg.lstsq(A, f_flat, rcond=None)

    # Reshape back into matrix form
    d = 2 ** n_qubits
    rho = x.reshape(d, d)
    rho = 0.5 * (rho + rho.conj().T)  # Ensure Hermiticity
    rho /= np.trace(rho)  # Normalize to trace 1

    return rho


def stokes_reconstruct_dataset(P: np.ndarray,
                              counts,
                              shots: int,
                              n_qubits: int) -> np.ndarray:
    """
    Reconstruct an array of density matrices from a dataset of outcome counts using Stokes (linear inversion).
    Returns an array of length N with (2**n_qubits, 2**n_qubits) density matrices.
    """
    # Build Stokes matrix once
    A = build_stokes_matrix(P, n_qubits)

    # Reconstruct each state
    stokes_rhos = np.empty(len(counts), dtype=object)
    for k, counts_k in enumerate(counts):
        f = counts_to_frequencies(counts_k, shots)
        rho_hat = solve_stokes_linear_system(A, f, n_qubits)
        stokes_rhos[k] = rho_hat

    return stokes_rhos
import numpy as np

# -----------------------------
# SETTING UP PROJECTOR MATRICES
# -----------------------------

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
        

def rotate_projector_matrix_by_noisy_unitary(sigma: float, P: np.ndarray, n_qubits: int) -> np.ndarray:
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
    U = np.kron(*unitaries)

# Unitaries are applied to each projector in the same way, as it can be assumed that the misalignment is fixed for the entire measurement process.
# Motors are so reproducible that the same misalignment will occur for each measurement setting, with minimal variational error between them.
        
    if n_qubits == 1:
        for i in range(6):
            P[i] = U @ P[i] @ U.conj().T
        return P

    if n_qubits == 2:
        for i in range(6):
            for j in range(6):
                P[i, j] = U @ P[i, j] @ U.conj().T
        return P

    if n_qubits == 3:
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    P[i, j, k] = U @ P[i, j, k] @ U.conj().T
        return P

    if n_qubits == 4:
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    for l in range(6):
                        P[i, j, k, l] = U @ P[i, j, k, l] @ U.conj().T
        return P

# -------------------------------------------------------
# DATA GENERATION AND SHOT SIMULATION
# -------------------------------------------------------

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
        return M
    
    if n_qubits == 2:
        for i in range(6):
            for j in range(6):
                M[i, j] = np.real(np.trace(rho @ P[i, j]))
        return M
    
    if n_qubits == 3:
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    M[i, j, k] = np.real(np.trace(rho @ P[i, j, k]))
        return M
    
    if n_qubits == 4:
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    for l in range(6):
                        M[i, j, k, l] = np.real(np.trace(rho @ P[i, j, k, l]))
        return M
    
    else: 
        raise NotImplementedError("Measurement only implemented for up to 4 qubits.")
    

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

def generate_dataset_of_states_and_probabilities(N: int, P: np.ndarray, n_qubits: int):
    """
    A function to generate a dataset of N random mixed states and their corresponding measurement probabilities.
    
    :param N: Number of random mixed states and probabilities to generate
    :param P: The (maybe noisy) projector matrix for the measurement apparatus
    :param n_qubits: Number of qubits in the system

    Outputs:
        rhos: (N,) array of 2^n_qubits x 2^n_qubits density matrices
        Ms:   (N,) array of 6^n_qubits measurement probability matrices
    """

    rhos = np.empty(N, dtype=object)
    Ms = np.empty(N, dtype=object)
    for k in range(N):
        rho = generate_random_mixed_state(n_qubits)
        M = get_measurement_probs_from_P_and_rho(rho, P, n_qubits)
        rhos[k] = rho
        Ms[k] = M

    return rhos, Ms

#--------------------------------------------------
# SIMULATING SHOTS AND EXPERIMENTAL OUTCOMES/ NOISE
#--------------------------------------------------


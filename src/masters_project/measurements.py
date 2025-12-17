import numpy as np


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


def rotate_projector(P_ij: np.ndarray, sigma: float) -> np.ndarray:
    """Rotate a two-qubit projector by independent random unitaries on each qubit."""
    U1 = noisy_unitary(sigma)
    U2 = noisy_unitary(sigma)
    U = np.kron(U1, U2)
    return U @ P_ij @ U.conj().T


def measure_state(rho: np.ndarray, P: np.ndarray, sigma: float) -> np.ndarray:
    """
    Return 6x6 matrix of noisy measurement probabilities M_ij = Tr(rho P_ij^noisy).

    The measurement probabilities are for each of the 36 two-qubit projectors
    """
    M = np.empty((6, 6), dtype=float)
    for i in range(6):
        for j in range(6):
            P_noisy = rotate_projector(P[i, j], sigma)
            M[i, j] = np.real(np.trace(rho @ P_noisy))
    return M

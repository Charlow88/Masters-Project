import numpy as np
from .measurements import measure_state


def random_mixed_state() -> np.ndarray:
    """Generate a random 4x4 mixed state using the Ginibre ensemble."""
    G = (np.random.normal(size=(4, 4))
         + 1j * np.random.normal(size=(4, 4)))
    rho = G @ G.conj().T
    rho /= np.trace(rho)
    return rho


def generate_dataset(N: int, P: np.ndarray, sigma: float):
    """
    Generate N samples:
      X: (N, 36) flattened measurement probabilities
      Y: (N, 32) concat(Re(vec(rho)), Im(vec(rho)))
    """
    X = np.zeros((N, 36), dtype=np.float32)
    Y = np.zeros((N, 32), dtype=np.float32)

    for k in range(N):
        rho = random_mixed_state()
        M = measure_state(rho, P, sigma)

        X[k] = M.reshape(-1)

        rho_flat = rho.reshape(-1)
        Y[k, :16] = np.real(rho_flat)
        Y[k, 16:] = np.imag(rho_flat)

    return X, Y

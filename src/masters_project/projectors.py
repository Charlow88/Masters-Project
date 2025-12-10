import numpy as np


def proj(ket: np.ndarray) -> np.ndarray:
    """|ket><ket| projector."""
    return ket @ ket.conj().T


def build_projector_matrix() -> np.ndarray:
    """
    Build the 6x6 matrix P of two-qubit projectors as in Lohani.
    P[i, j] is a 4x4 projector.
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
    P = np.empty((6, 6), dtype=object)
    for i in range(6):
        for j in range(6):
            P[i, j] = np.kron(projs[i], projs[j])
    return P

import numpy as np
from .tau_param import fidelity_from_vecs

def build_stokes_matrix(P):
    A = np.zeros((36, 16), dtype=complex)
    idx = 0
    for i in range(6):
        for j in range(6):
            A[idx] = P[i, j].T.reshape(-1)
            idx += 1
    return A

def stokes_reconstruct(M_flat, A):
    x, *_ = np.linalg.lstsq(A, M_flat.astype(complex), rcond=None)
    rho = x.reshape(4, 4)
    rho = 0.5 * (rho + rho.conj().T)
    rho /= np.trace(rho)
    return rho

def precompute_stokes_fidelities(X_test, Y_test, P):
    A = build_stokes_matrix(P)

    fids = []
    flats = []

    for k in range(len(X_test)):
        rho_stokes = stokes_reconstruct(X_test[k], A)

        flat = rho_stokes.reshape(-1)
        flat = np.concatenate([np.real(flat), np.imag(flat)]).astype(np.float32)

        flats.append(flat)
        fids.append(fidelity_from_vecs(flat, Y_test[k]))

    return float(np.mean(fids)), np.stack(flats)

import numpy as np
import torch
from scipy.linalg import sqrtm


def fidelity_from_vecs(pred_flat: np.ndarray, true_flat: np.ndarray) -> float:
    """
    Both pred_flat and true_flat: length-32 real arrays (Re||Im of vec(rho)).
    Returns Uhlmann fidelity F in [0,1].
    """
    pred_rho = pred_flat[:16] + 1j * pred_flat[16:]
    true_rho = true_flat[:16] + 1j * true_flat[16:]

    pred_rho = pred_rho.reshape(4, 4)
    true_rho = true_rho.reshape(4, 4)

    pred_rho = (pred_rho + pred_rho.conj().T) / 2
    true_rho = (true_rho + true_rho.conj().T) / 2
    pred_rho /= np.trace(pred_rho)
    true_rho /= np.trace(true_rho)

    sqrt_true = sqrtm(true_rho)
    inner = sqrt_true @ pred_rho @ sqrt_true
    F = np.real(np.trace(sqrtm(inner)) ** 2)
    return float(np.clip(F, 0.0, 1.0))


def tau_from_params_np(params: np.ndarray) -> np.ndarray:
    """
    params: shape (16,) real
    Returns a 4x4 complex lower-triangular tau.
    """
    p = params
    assert p.shape[-1] == 16

    t00 = p[0]
    t11 = p[3]
    t22 = p[8]
    t33 = p[15]

    t10 = p[1] + 1j * p[2]
    t20 = p[4] + 1j * p[5]
    t21 = p[6] + 1j * p[7]
    t30 = p[9] + 1j * p[10]
    t31 = p[11] + 1j * p[12]
    t32 = p[13] + 1j * p[14]

    tau = np.zeros((4, 4), dtype=complex)
    tau[0, 0] = t00

    tau[1, 0] = t10
    tau[1, 1] = t11

    tau[2, 0] = t20
    tau[2, 1] = t21
    tau[2, 2] = t22

    tau[3, 0] = t30
    tau[3, 1] = t31
    tau[3, 2] = t32
    tau[3, 3] = t33

    return tau


def rho_from_tau_np(params: np.ndarray) -> np.ndarray:
    """Return 4x4 density matrix rho = tau^† tau / Tr(tau^† tau)."""
    tau = tau_from_params_np(params)
    rho = tau.conj().T @ tau
    rho = (rho + rho.conj().T) / 2
    rho /= np.trace(rho)
    return rho


def fidelity_from_tau_params(pred_params: np.ndarray,
                             true_flat: np.ndarray) -> float:
    """
    pred_params: length-16 real array (tau params)
    true_flat:   length-32 real array (Re||Im vec(rho_true))
    """
    rho_pred = rho_from_tau_np(pred_params)

    true_rho = true_flat[:16] + 1j * true_flat[16:]
    true_rho = true_rho.reshape(4, 4)
    true_rho = (true_rho + true_rho.conj().T) / 2
    true_rho /= np.trace(true_rho)

    sqrt_true = sqrtm(true_rho)
    inner = sqrt_true @ rho_pred @ sqrt_true
    F = np.real(np.trace(sqrtm(inner)) ** 2)
    return float(np.clip(F, 0.0, 1.0))


# -------- Torch versions + losses --------

def tau_from_params_torch(params: torch.Tensor) -> torch.Tensor:
    """
    params: (batch, 16) real
    Returns tau: (batch, 4, 4) complex lower-triangular.
    """
    assert params.shape[-1] == 16
    p = params

    t00 = p[:, 0]
    t11 = p[:, 3]
    t22 = p[:, 8]
    t33 = p[:, 15]

    t10 = p[:, 1] + 1j * p[:, 2]
    t20 = p[:, 4] + 1j * p[:, 5]
    t21 = p[:, 6] + 1j * p[:, 7]
    t30 = p[:, 9] + 1j * p[:, 10]
    t31 = p[:, 11] + 1j * p[:, 12]
    t32 = p[:, 13] + 1j * p[:, 14]

    batch = p.shape[0]
    tau = torch.zeros((batch, 4, 4), dtype=torch.cfloat, device=p.device)

    tau[:, 0, 0] = t00

    tau[:, 1, 0] = t10
    tau[:, 1, 1] = t11

    tau[:, 2, 0] = t20
    tau[:, 2, 1] = t21
    tau[:, 2, 2] = t22

    tau[:, 3, 0] = t30
    tau[:, 3, 1] = t31
    tau[:, 3, 2] = t32
    tau[:, 3, 3] = t33

    return tau


def rho_from_tau_torch(params: torch.Tensor) -> torch.Tensor:
    """
    params: (batch, 16) real
    returns rho: (batch, 4, 4) complex
    """
    tau = tau_from_params_torch(params)
    rho = torch.matmul(tau.conj().transpose(-2, -1), tau)
    rho = 0.5 * (rho + rho.conj().transpose(-2, -1))
    tr = torch.real(torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1, keepdim=True))
    rho = rho / (tr.unsqueeze(-1) + 1e-8)
    return rho


def rho_true_from_vec_torch(target: torch.Tensor) -> torch.Tensor:
    """
    target: (batch, 32) real = Re||Im vec(rho_true)
    returns rho_true: (batch, 4, 4) complex
    """
    re = target[:, :16]
    im = target[:, 16:]
    rho = torch.complex(re, im).reshape(-1, 4, 4)
    rho = 0.5 * (rho + rho.conj().transpose(-2, -1))
    tr = torch.real(torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1, keepdim=True))
    rho = rho / (tr.unsqueeze(-1) + 1e-8)
    return rho


def fidelity_loss_tau(pred_params: torch.Tensor,
                      target_vec: torch.Tensor) -> torch.Tensor:
    """
    1 - mean fidelity between rho(pred_params) and rho_true.
    pred_params: (batch, 16)
    target_vec:  (batch, 32)
    """
    rho_p = rho_from_tau_torch(pred_params)
    rho_t = rho_true_from_vec_torch(target_vec)

    evals_t, vecs_t = torch.linalg.eigh(rho_t)
    diag_sqrt = torch.diag_embed(torch.sqrt(torch.clamp(evals_t, min=0))).to(torch.cfloat)
    sqrt_t = vecs_t @ diag_sqrt @ vecs_t.conj().transpose(-2, -1)

    inner = sqrt_t @ rho_p @ sqrt_t
    evals_inner, _ = torch.linalg.eigh(inner)
    F = (torch.sqrt(torch.clamp(evals_inner, min=0)).sum(dim=-1)) ** 2

    return 1.0 - F.mean()


def mse_rho_loss_tau(pred_params: torch.Tensor,
                     target_vec: torch.Tensor) -> torch.Tensor:
    """MSE between rho(pred_params) and rho_true at the matrix level."""
    rho_p = rho_from_tau_torch(pred_params)
    rho_t = rho_true_from_vec_torch(target_vec)
    diff = rho_p - rho_t
    return (diff.real**2 + diff.imag**2).mean()


def get_loss_fn(loss_type: str):
    if loss_type == "mse":
        return mse_rho_loss_tau
    elif loss_type == "fidelity":
        return fidelity_loss_tau
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

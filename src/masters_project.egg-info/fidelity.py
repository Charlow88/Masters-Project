"""
fidelity.py

State comparison metrics for quantum states, primarily Uhlmann fidelity.

Responsibilities
----------------
- Provide numerically stable functions to compute fidelity between two
  density matrices, in both NumPy and PyTorch.
- Offer helpers to work with the flattened real representations used by
  the neural networks (Re||Im vec(rho)).

Core API (suggested)
--------------------
- NumPy:
    - `vector_to_rho(vec: np.ndarray, d: int) -> np.ndarray`
    - `fidelity(rho_true: np.ndarray, rho_pred: np.ndarray) -> float`
    - `fidelity_from_flat(pred_flat: np.ndarray, true_flat: np.ndarray, d: int) -> float`
- PyTorch:
    - `rho_true_from_vec_torch(target: torch.Tensor, d: int) -> torch.Tensor`
    - `fidelity_loss(rho_pred: torch.Tensor, rho_true: torch.Tensor) -> torch.Tensor`
    - `fidelity_loss_from_params(pred_params: torch.Tensor, target_vec: torch.Tensor, d: int) -> torch.Tensor`

Design notes
------------
- Do not assume d=4 in the core logic; only shapes and d should control the
  reshaping.
- Implement Uhlmann fidelity via matrix square roots or eigen-decomposition
  with appropriate clamping for numerical robustness.
- Higher-level losses (e.g. `fidelity_loss_tau`) can live here or in the
  parametrisations module, but aim for a clean separation of concerns.
"""
# copilot: implement fidelity utilities in numpy and torch, general in d but
# compatible with the existing 2-qubit usage.

"""
parametrisations.py

Parameterisations of quantum states for use with neural networks, especially
those based on Cholesky-like decompositions.

Responsibilities
----------------
- Define mappings between real parameter vectors and valid density matrices,
  for arbitrary Hilbert space dimension `d` where possible.
- Provide both NumPy and PyTorch implementations to support:
    * offline analysis (NumPy),
    * differentiable training (PyTorch).

Core examples
-------------
- Cholesky / tau parameterisation:
    params -> lower-triangular tau -> rho = tau† tau / Tr(tau† tau).
- Utilities to convert between:
    * tau-parameters,
    * rho as a complex matrix,
    * flattened real vectors used as NN outputs.

Suggested API
-------------
- NumPy:
    - `tau_from_params_np(params: np.ndarray, d: int) -> np.ndarray`
    - `rho_from_tau_np(params: np.ndarray, d: int) -> np.ndarray`
- PyTorch:
    - `tau_from_params_torch(params: torch.Tensor, d: int) -> torch.Tensor`
    - `rho_from_tau_torch(params: torch.Tensor, d: int) -> torch.Tensor`

Design notes
------------
- For now you can implement the d=4 (two-qubit) case explicitly, but keep
  function signatures and docstrings dimension-agnostic so generalisation
  is straightforward later.
- Ensure Hermiticity and normalisation numerically, and document any
  assumptions about parameter layout for tau.
"""
# copilot: move the existing tau / Cholesky parameterisation code into this
# module and make signatures dimension-aware where reasonable.

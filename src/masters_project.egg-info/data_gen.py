"""
data_gen.py

Dataset generation utilities for training and evaluating tomography models.

Responsibilities
----------------
- Sample random physical quantum states (density matrices) of dimension `d`
  using general constructions (e.g. Ginibre ensemble for mixed states,
  pure states, or user-specified ensembles).
- Use measurement operators and measurement utilities to generate input
  feature vectors (measurement outcomes) with optional noise.
- Provide flexible flattening / encoding of density matrices into real-valued
  vectors suitable as neural network targets, without hard-coding d=4.

Core API (suggested)
--------------------
- `random_mixed_state(d: int, rng=None) -> np.ndarray`
    Draw a random d×d density matrix that is PSD and trace-1.
- `flatten_rho_complex(rho: np.ndarray) -> np.ndarray`
    Map a d×d complex matrix to a 2*d*d real vector [Re(vec(rho)), Im(vec(rho))].
- `generate_dataset(N: int, meas_ops, d: int, sigma: float = 0.0, rng=None)`
    -> tuple[np.ndarray, np.ndarray]
    Return (X, Y), where:
      * X has shape (N, n_meas_flat) = measurement outcomes,
      * Y has shape (N, 2*d*d)      = flattened density matrices.

Design notes
------------
- Do not assume a particular layout (6×6 grid etc.) in the core logic;
  only the calling code should decide how to reshape/flatten measurements.
- Keep everything in NumPy and return float32 arrays for downstream ML.
- The current two-qubit Lohani setup should be implementable as a thin
  wrapper around these general routines (d=4, specific meas_ops).
"""
# copilot: implement general dataset generation routines using numpy and the
# measurement utilities, avoiding any hard-coded dimension or layout.

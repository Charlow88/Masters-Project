"""
projectors.py

Construction and management of measurement operators (projectors or POVM
elements) for quantum state tomography.

Goals
-----
- Provide a general interface for obtaining measurement operators on a Hilbert
  space of dimension `d`, independent of the specific physical platform.
- Support multiple measurement "schemes" (e.g. Pauli bases, mutually unbiased
  bases, polarisation projectors, custom user-defined POVMs).
- Expose convenience functions for the current 2-qubit, 6x6 polarisation grid
  used in the Lohani-style experiment, without hard-coding this everywhere.

Core API (suggested)
--------------------
- `build_projectors(scheme: str, d: int, **kwargs) -> np.ndarray`
    Return an array of measurement operators with shape
    (n_meas, d, d) or (n1, n2, d, d) for grid-like layouts.
- `lohanii_two_qubit_projectors() -> np.ndarray`
    Convenience wrapper returning the current 6x6 grid of two-qubit projectors.

Design notes
------------
- Represent all operators as complex NumPy arrays with shape (d, d).
- Use pure functions with no global state so other modules (data_gen,
  measurements, tomography, etc.) can reuse them for different systems.
- Keep the interface general enough that different systems (qubits, qutrits,
  multi-qubit registers) can plug in their own schemes.
"""
# copilot: implement generic projector builders plus a helper that reproduces
# the current two-qubit 6x6 Lohani-style projector grid.

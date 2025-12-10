"""
measurements.py

Generic measurement utilities for quantum state tomography.

Responsibilities
----------------
- Given a quantum state rho and a collection of measurement operators
  {M_k} (projectors or POVM elements), compute ideal outcome probabilities:
      p_k = Tr(M_k rho).
- Apply configurable noise models to these ideal outcomes to emulate
  experimental imperfections (e.g. misalignment, random unitary rotation,
  detector noise, or shot noise).

Core API (suggested)
--------------------
- `measure_state(rho, meas_ops, noise_model=None, rng=None) -> np.ndarray`
    Compute (possibly noisy) measurement outcomes for a single state.
- `apply_unitary_noise(meas_ops, sigma, rng=None) -> np.ndarray`
    Example noise model: locally rotate measurement operators by random
    unitaries drawn from some distribution on SU(d_local).
- Additional noise models can be added with a common interface:
    `noise_model(meas_ops, rho, rng, **params)`.

Design notes
------------
- Treat measurement operators as arbitrary complex matrices; do not assume
  projective measurements unless needed.
- Keep noise models modular so that the same measurement set can be used
  with different physical noise assumptions.
- Current 2-qubit polarisation + local SU(2) noise is just one specific
  instance of this general interface.
"""
# copilot: implement measurement and noise utilities using numpy, including a
# general measure_state function and a unitary-rotation noise model compatible
# with the existing 2-qubit example.

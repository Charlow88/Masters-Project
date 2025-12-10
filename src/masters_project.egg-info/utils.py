"""
utils.py

General-purpose helpers shared across the tomography project.

Examples of functionality
-------------------------
- Reproducibility helpers (seeding NumPy, PyTorch, etc.).
- Conversions between different data representations:
    * grid-shaped measurements ↔ flattened vectors,
    * complex matrices ↔ stacked real/imag arrays.
- Simple timing, logging, and configuration helpers.
- Plotting utilities for standard curves (e.g. fidelity vs N) that do not
  belong inside the core training code.

Design notes
------------
- Keep this module lightweight and focused on things genuinely reused across
  multiple other modules.
- Avoid importing heavy ML libraries here unless absolutely necessary.
"""
# copilot: implement small, reusable helpers for seeding, reshaping, and
# plotting, based on what is currently duplicated in the monolithic script.
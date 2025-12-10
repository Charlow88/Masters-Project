"""
models.py

PyTorch neural network architectures for quantum state tomography.

Responsibilities
----------------
- Define model classes that map measurement feature vectors or grids to
  parameterisations of quantum states (e.g. tau parameters, flattened rho).
- Keep architectures flexible in input size, output size, depth, activation,
  and regularisation, so they can be reused for different systems and
  measurement schemes.

Core models (suggested)
-----------------------
- `DensityMLP(nn.Module)`
    Generic fully-connected network; constructor should accept:
      * input_dim (e.g. n_meas_flat),
      * output_dim (e.g. number of tau parameters or 2*d*d),
      * hidden_dims, activation, dropout, etc.
- `ConvMeasurementNet(nn.Module)`
    CNN operating on measurement grids (e.g. 6×6) with physics-motivated
    kernels; should still be configurable in input shape and channels.

Design notes
------------
- Do not hard-code 36 → 720 → 450 → 32; pass these as arguments or use
  reasonable defaults while allowing overrides.
- For tau-based models, treat the network output as a parameter vector
  which is converted to rho by the parametrisations module.
- Keep models pure PyTorch with minimal external dependencies.
"""
# copilot: generalise the existing FC and CNN models into parameterisable
# classes/functions without hard-coding input/output sizes.

"""
train_nn.py

Training, evaluation, and hyperparameter sweep utilities for tomography
neural networks.

Responsibilities
----------------
- Provide general training loops that:
    * accept arbitrary model constructors,
    * work with different loss functions (MSE on rho, fidelity-based losses,
      parameter-space losses, etc.),
    * support both vector and image-like (grid) inputs.
- Offer helpers to run experiments over:
    * different training set sizes N,
    * different learning rates, epochs, regularisation settings, etc.
- Optionally compute and return classical baselines (e.g. Stokes / linear
  inversion) for comparison.

Core API (suggested)
--------------------
- `train_model(model, X_train, Y_train, loss_fn, n_epochs, lr, batch_size, is_cnn=False, device=None)`
- `evaluate_model_fidelity(model, X_test, Y_test, fidelity_fn, is_cnn=False)`
- `run_model_curve(model_factory, loss_type, hyperparams, N_list, system_config, rng_seeds, ...)`

Design notes
------------
- Do not assume a specific model (FC vs CNN) or a fixed input shape; use
  flags or callables (e.g. `model_factory`) to keep it generic.
- Use PyTorch's DataLoader for batching, and handle CPU/GPU selection cleanly.
- Hyperparameter sweeps should return structured results (e.g. dicts or
  arrays) that plotting code can consume, rather than plotting inside the
  training utilities themselves.
"""
# copilot: refactor the existing training loops and sweep code into reusable
# functions that are not hard-wired to the two-qubit, 36-feature setup.

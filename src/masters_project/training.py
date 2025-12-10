import numpy as np
import torch
import torch.optim as optim

from .data_gen import generate_dataset
from .baselines import precompute_stokes_fidelities
from .models import make_fc_model, make_cnn_model, OUTPUT_DIM
from .tau_param import get_loss_fn, fidelity_from_tau_params

BATCH_SIZE = 4


def train_model(model: torch.nn.Module,
                X_train: torch.Tensor,
                Y_train: torch.Tensor,
                loss_type: str,
                n_epochs: int,
                lr: float,
                is_cnn: bool = False) -> torch.nn.Module:

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = get_loss_fn(loss_type)

    for epoch in range(n_epochs):
        model.train()
        for i in range(0, len(X_train), BATCH_SIZE):
            X_batch = X_train[i:i + BATCH_SIZE]
            Y_batch = Y_train[i:i + BATCH_SIZE]

            if is_cnn:
                X_batch = X_batch.view(-1, 1, 6, 6)  # (batch, 1, 6, 6)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, Y_batch)
            loss.backward()
            optimizer.step()

    return model


def evaluate_model_fidelity(model: torch.nn.Module,
                            X_test_t: torch.Tensor,
                            Y_test: np.ndarray,
                            is_cnn: bool = False) -> float:
    """
    Run the model on the test set and return mean fidelity vs true rho (from Y_test).
    """
    model.eval()
    with torch.no_grad():
        if is_cnn:
            X_in = X_test_t.view(-1, 1, 6, 6)
        else:
            X_in = X_test_t
        pred = model(X_in).cpu().numpy()  # (N_test, OUTPUT_DIM)

    fids = [
        fidelity_from_tau_params(pred[k], Y_test[k])
        for k in range(len(pred))
    ]
    return float(np.mean(fids))


def run_model_curve(model_type: str,
                    loss_type: str,
                    lr: float,
                    n_epochs: int,
                    N_list,
                    P,
                    sigma,
                    seed):
    """
    Run one model family (FC or CNN) with fixed hyperparameters over all N in N_list.
    Returns:
        avg_fids_model: np.array, shape (len(N_list),)
        avg_fids_stokes: np.array, same shape
    """
    avg_fids_model = []
    avg_fids_stokes = []

    for N in N_list:
        print(f"\n[{model_type.upper()} | {loss_type}] N = {N}, lr={lr}, epochs={n_epochs}")

        np.random.seed(seed)
        torch.manual_seed(seed)

        X, Y = generate_dataset(N, P, sigma)
        split = int(0.8 * N)
        X_train, X_test = X[:split], X[split:]
        Y_train, Y_test = Y[:split], Y[split:]

        X_train_t = torch.from_numpy(X_train.astype(np.float32))
        Y_train_t = torch.from_numpy(Y_train.astype(np.float32))
        X_test_t = torch.from_numpy(X_test.astype(np.float32))

        # build Stokes matrix outside and pass it in if you want to avoid recomputing
        # here we recompute via baselines only on test data
        # (you can precompute A_stokes in your notebook and call precompute_stokes_fidelities there instead)
        # For now, assume you pass in avg_fid_stokes externally if needed.
        avg_fid_stokes, _ = precompute_stokes_fidelities(X_test, Y_test, P)
        avg_fids_stokes.append(avg_fid_stokes)

        torch.manual_seed(seed)
        if model_type.lower() == "fc":
            model = make_fc_model()
            is_cnn = False
        elif model_type.lower() == "cnn":
            model = make_cnn_model()
            is_cnn = True
        else:
            raise ValueError("model_type must be 'fc' or 'cnn'")

        model = train_model(model,
                            X_train_t, Y_train_t,
                            loss_type=loss_type,
                            n_epochs=n_epochs,
                            lr=lr,
                            is_cnn=is_cnn)

        avg_fid_model = evaluate_model_fidelity(model, X_test_t, Y_test, is_cnn=is_cnn)
        avg_fids_model.append(avg_fid_model)

        print(f"    Avg model fidelity:  {avg_fid_model:.6e}")
        print(f"    Avg Stokes fidelity: {avg_fid_stokes:.6e}")

    return np.array(avg_fids_model), np.array(avg_fids_stokes)

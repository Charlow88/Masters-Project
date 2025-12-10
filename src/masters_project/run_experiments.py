import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from .projectors import build_projector_matrix
from .data_gen import generate_dataset
from .baselines import precompute_stokes_fidelities
from .models import make_fc_model, make_cnn_model
from .tau_param import get_loss_fn, fidelity_from_tau_params



seed = 42
sigma = 0.564                      # noise strength
N_list = np.linspace(20, 200, 10, dtype=int)

BATCH_SIZE   = 4
LR_DEFAULT   = 1e-3
N_EPOCHS_FC  = 400
N_EPOCHS_CNN = 400
OUTPUT_DIM   = 16



def train_model(model, X_train_t, Y_train_t,
                loss_type, n_epochs, lr,
                is_cnn=False):
    """Train FC or CNN model on (X_train_t, Y_train_t)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = get_loss_fn(loss_type)

    for epoch in range(n_epochs):
        model.train()
        for i in range(0, len(X_train_t), BATCH_SIZE):
            X_batch = X_train_t[i:i + BATCH_SIZE]
            Y_batch = Y_train_t[i:i + BATCH_SIZE]

            if is_cnn:
                X_batch = X_batch.view(-1, 1, 6, 6)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, Y_batch)
            loss.backward()
            optimizer.step()

    return model


def evaluate_model_fidelity(model, X_test_t, Y_test, is_cnn=False):
    """Mean fidelity between predicted rhos (via tau) and true rhos."""
    model.eval()
    with torch.no_grad():
        if is_cnn:
            X_in = X_test_t.view(-1, 1, 6, 6)
        else:
            X_in = X_test_t
        pred = model(X_in).cpu().numpy()   # (N_test, 16)

    fids = [
        fidelity_from_tau_params(pred[k], Y_test[k])
        for k in range(len(pred))
    ]
    return float(np.mean(fids))



def main():
    os.makedirs("plots", exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)

    P = build_projector_matrix()

    avg_fids_fc_mse     = []
    avg_fids_fc_fid     = []
    avg_fids_cnn_mse    = []
    avg_fids_cnn_fid    = []
    avg_fids_stokes_all = []

    for N in N_list:
        print(f"\n=== Experiments for N = {N} ===")

        np.random.seed(seed)
        torch.manual_seed(seed)

        # dataset
        X, Y = generate_dataset(N, P, sigma)
        split = int(0.8 * N)
        X_train, X_test = X[:split], X[split:]
        Y_train, Y_test = Y[:split], Y[split:]

        X_train_t = torch.from_numpy(X_train.astype(np.float32))
        Y_train_t = torch.from_numpy(Y_train.astype(np.float32))
        X_test_t  = torch.from_numpy(X_test.astype(np.float32))

        # Stokes baseline
        avg_fid_stokes, _ = precompute_stokes_fidelities(X_test, Y_test, P)
        avg_fids_stokes_all.append(avg_fid_stokes)
        print(f"    Stokes Avg fidelity:           {avg_fid_stokes:.6e}")

        # FC, MSE loss
        fc_mse_model = make_fc_model()
        fc_mse_model = train_model(fc_mse_model,
                                   X_train_t, Y_train_t,
                                   loss_type="mse",
                                   n_epochs=N_EPOCHS_FC,
                                   lr=LR_DEFAULT,
                                   is_cnn=False)
        fc_mse_fid = evaluate_model_fidelity(fc_mse_model, X_test_t, Y_test, is_cnn=False)
        avg_fids_fc_mse.append(fc_mse_fid)
        print(f"    FC (MSE) Avg fidelity:         {fc_mse_fid:.6e}")

        # FC, fidelity loss
        fc_fid_model = make_fc_model()
        fc_fid_model = train_model(fc_fid_model,
                                   X_train_t, Y_train_t,
                                   loss_type="fidelity",
                                   n_epochs=N_EPOCHS_FC,
                                   lr=LR_DEFAULT,
                                   is_cnn=False)
        fc_fid_fid = evaluate_model_fidelity(fc_fid_model, X_test_t, Y_test, is_cnn=False)
        avg_fids_fc_fid.append(fc_fid_fid)
        print(f"    FC (Fidelity loss) Avg fidelity: {fc_fid_fid:.6e}")

        # CNN, MSE loss
        cnn_mse_model = make_cnn_model()
        cnn_mse_model = train_model(cnn_mse_model,
                                    X_train_t, Y_train_t,
                                    loss_type="mse",
                                    n_epochs=N_EPOCHS_CNN,
                                    lr=LR_DEFAULT,
                                    is_cnn=True)
        cnn_mse_fid = evaluate_model_fidelity(cnn_mse_model, X_test_t, Y_test, is_cnn=True)
        avg_fids_cnn_mse.append(cnn_mse_fid)
        print(f"    CNN (MSE) Avg fidelity:        {cnn_mse_fid:.6e}")

        # CNN, fidelity loss
        cnn_fid_model = make_cnn_model()
        cnn_fid_model = train_model(cnn_fid_model,
                                    X_train_t, Y_train_t,
                                    loss_type="fidelity",
                                    n_epochs=N_EPOCHS_CNN,
                                    lr=LR_DEFAULT,
                                    is_cnn=True)
        cnn_fid_fid = evaluate_model_fidelity(cnn_fid_model, X_test_t, Y_test, is_cnn=True)
        avg_fids_cnn_fid.append(cnn_fid_fid)
        print(f"    CNN (Fidelity loss) Avg fidelity: {cnn_fid_fid:.6e}")


    # MSE loss plot
    plt.figure()
    plt.plot(N_list, avg_fids_fc_mse,     label='FC (MSE)',  linestyle='-')
    plt.plot(N_list, avg_fids_cnn_mse,    label='CNN (MSE)', linestyle='-.')
    plt.plot(N_list, avg_fids_stokes_all, label='Stokes',    linestyle='--')
    plt.xlabel("Number of training states N")
    plt.ylabel("Average test fidelity")
    plt.title(f"Fidelity vs training set size (σ = {sigma}) with MSE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/fid_vs_N_mse.png", dpi=300)
    plt.close()

    # Fidelity-loss plot
    plt.figure()
    plt.plot(N_list, avg_fids_fc_fid,     label='FC (Fidelity loss)',  linestyle='-')
    plt.plot(N_list, avg_fids_cnn_fid,    label='CNN (Fidelity loss)', linestyle='-.')
    plt.plot(N_list, avg_fids_stokes_all, label='Stokes',              linestyle='--')
    plt.xlabel("Number of training states N")
    plt.ylabel("Average test fidelity")
    plt.title(f"Fidelity vs training set size (σ = {sigma}) with Fidelity loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/fid_vs_N_fidloss.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()

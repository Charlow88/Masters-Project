import numpy as np


# ---------------
# DATA GENERATION
# ---------------
    

def generate_random_mixed_state(n_qubits: int) -> np.ndarray:
    """
    Generate a random mixed state for n_qubits qubits using the Ginibre ensemble.
    Helper function for data generation.

    This guarantees Hermiticity and positive semi-definiteness with trace 1
    Only mixed states are produced here.
    """
    dim = 2 ** n_qubits
    G = (np.random.normal(size=(dim, dim))
         + 1j * np.random.normal(size=(dim, dim)))
    rho = G @ G.conj().T
    rho /= np.trace(rho)
    return rho

def generate_random_pure_state(n_qubits: int, eps: float = 1e-6) -> np.ndarray:
    """
    Generate a random pure state density matrix rho = |psi><psi| via a random complex vector.
    Made nearly pure by mixing with a small amount of identity (eps) to ensure numerical stability in Cholesky decomposition.
    """
    d = 2 ** n_qubits
    psi = np.random.normal(size=(d,)) + 1j * np.random.normal(size=(d,))
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    rho = (1.0 - eps) * rho + eps * np.eye(d, dtype=complex) / d
    return rho

def generate_dataset_of_states_and_probabilities(
        N: int,
        n_qubits: int,
        seed: int = 0,
        p_pure: float = 0.3,
        eps_pure: float = 1e-6
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset of N states containing a mix of mixed and (nearly) pure states,
    along with Cholesky factors taus.
    
    p_pure: fraction of samples generated as (nearly) pure states.
    eps_pure: depolarising amount used to make pure states full-rank for Cholesky.
    """
    np.random.seed(seed)

    rhos = np.empty(N, dtype=object)
    taus = np.empty(N, dtype=object)

    for k in range(N):
        if np.random.rand() < p_pure:
            rho = generate_random_pure_state(n_qubits, eps=eps_pure)
        else:
            rho = generate_random_mixed_state(n_qubits)

        rhos[k] = rho
        taus[k] = np.linalg.cholesky(rho)

    return rhos, taus


def add_train_test_split_to_data(data: dict, train_ratio: float = 0.8, seed: int = 0) -> dict:
    """
    Add train/test indices into data["split"] without duplicating arrays.
    """
    rng = np.random.default_rng(seed)
    N = len(data["rhos"])
    idx = rng.permutation(N)
    split = int(train_ratio * N)

    data["split"] = {
        "train_idx": idx[:split],
        "test_idx": idx[split:],
        "train_ratio": train_ratio,
        "seed": seed,
    }
    return data


def subset_data_by_idx(data: dict, idx: np.ndarray) -> dict:
    """
    Return a smaller data dict containing only samples in idx, while copying shared metadata.
    """
    def subset(arr):
        # list/object array/stacked array all OK
        if isinstance(arr, list):
            return [arr[i] for i in idx]
        return arr[idx]

    out = {}

    # sample-wise keys (subset these if present)
    for k in ["rhos", "taus", "counts"]:
        if k in data:
            out[k] = subset(data[k])

    # shared keys (copy through if present)
    for k in ["shots", "P", "P_noisy", "misalignment_sigma", "visibility", "n_qubits"]:
        if k in data:
            out[k] = data[k]

    return out


def get_split(data: dict, which: str) -> dict:
    """
    Convenience: get_split(data,"train") or get_split(data,"test").
    Requires data["split"] to exist.
    """
    if "split" not in data:
        raise KeyError('No split found. Call add_train_test_split_to_data(...) first.')

    if which == "train":
        idx = data["split"]["train_idx"]
    elif which == "test":
        idx = data["split"]["test_idx"]
    else:
        raise ValueError('which must be "train" or "test".')

    return subset_data_by_idx(data, idx)


#--------------------------------------------------
# SETTING UP IDEAL/ NOISY PROJECTOR MATRICES
#--------------------------------------------------

def proj(ket: np.ndarray) -> np.ndarray:
    """|ket><ket| projector."""
    return ket @ ket.conj().T

def build_projector_matrix(n_qubits: int) -> np.ndarray:
    """
    Build the measurement projector matrix P for n_qubits qubits.
    Projector matrix will be 6^n_qubits x 6^n_qubits, with each entry being a 2^n_qubits x 2^n_qubits projector.

    The projectors represent different measurements of the photon polarisations
    """
    H = np.array([[1.0], [0.0]], dtype=complex)
    V = np.array([[0.0], [1.0]], dtype=complex)

    D = (H + V) / np.sqrt(2)
    A = (H - V) / np.sqrt(2)
    R = (H + 1j * V) / np.sqrt(2)
    L = (H - 1j * V) / np.sqrt(2)

    h = proj(H)
    v = proj(V)
    d = proj(D)
    a = proj(A)
    r = proj(R)
    l = proj(L)

    projs = [h, v, d, a, r, l]

    if n_qubits == 1:
        P = np.empty((6), dtype=object)
        for i in range(6):
            P[i] = projs[i]
        return P
    
    if n_qubits == 2:
        P = np.empty((6, 6), dtype=object)
        for i in range(6):
            for j in range(6):
                P[i, j] = np.kron(projs[i], projs[j])
        return P
    
    if n_qubits == 3:
        P = np.empty((6, 6, 6), dtype=object)
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    P[i, j, k] = np.kron(np.kron(projs[i], projs[j]), projs[k])
        return P
    
    if n_qubits == 4:
        P = np.empty((6, 6, 6, 6), dtype=object)
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    for l in range(6):
                        P[i, j, k, l] = np.kron(np.kron(np.kron(projs[i], projs[j]), projs[k]), projs[l])
        return P
    
    else: 
        raise NotImplementedError("Projector matrix construction only implemented for up to 4 qubits.")
        

def simulate_waveplate_misalignment(sigma: float, P: np.ndarray, n_qubits: int) -> np.ndarray:
    """Rotate the entire projector matrix P by independent noisy unitaries on each qubit.
    
    This is a helper function designed to simulate the effect of misalignment in the measurement apparatus, 
    ie. waveplates not perfectly correct.
    """

    def noisy_unitary(sigma: float) -> np.ndarray:
        """Random SU(2) unitary from Gaussian-distributed Euler angles."""
        theta, phi, zeta = np.random.normal(0, sigma, 3)
        U = np.array([
            [np.exp(1j * phi / 2) * np.cos(theta),
             -1j * np.exp(1j * zeta) * np.sin(theta)],
            [-1j * np.exp(-1j * zeta) * np.sin(theta),
             np.exp(-1j * phi / 2) * np.cos(theta)]
        ], dtype=complex)
        return U
    
   
    unitaries = [noisy_unitary(sigma) for _ in range(n_qubits)]
    U = unitaries[0]
    for i in range(1, n_qubits):
        U = np.kron(U, unitaries[i])
    noisy_P = np.empty_like(P, dtype=object)

# Unitaries are applied to each projector in the same way, as it can be assumed that the misalignment is fixed for the entire measurement process.
# Motors are so reproducible that the same misalignment will occur for each measurement setting, with minimal variational error between them.
        
    if n_qubits == 1:
        for i in range(6):
            noisy_P[i] = U @ P[i] @ U.conj().T
        return noisy_P

    if n_qubits == 2:
        for i in range(6):
            for j in range(6):
                noisy_P[i, j] = U @ P[i, j] @ U.conj().T
        return noisy_P

    if n_qubits == 3:
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    noisy_P[i, j, k] = U @ P[i, j, k] @ U.conj().T
        return noisy_P

    if n_qubits == 4:
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    for l in range(6):
                        noisy_P[i, j, k, l] = U @ P[i, j, k, l] @ U.conj().T
        return noisy_P
    

def get_measurement_probs_from_P_and_rho(rho: np.ndarray, P: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Return 6^n_qubits matrix of measurement probabilities M_ij = Tr(rho P_ij).
    Gives the measurement probabilities for a given density matrix rho and projector matrix P.
    Helper function for data generation.

    The measurement probabilities are for each of the 6^n n-qubit projectors
    """
    shape = tuple([6] * n_qubits)
    M = np.empty(shape, dtype=float)

    if n_qubits == 1:
        for i in range(6):
            M[i] = np.real(np.trace(rho @ P[i]))
        return M / 3
    
    if n_qubits == 2:
        for i in range(6):
            for j in range(6):
                M[i, j] = np.real(np.trace(rho @ P[i, j]))
        return M / 9
    
    if n_qubits == 3:
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    M[i, j, k] = np.real(np.trace(rho @ P[i, j, k]))
        return M / 27
    
    if n_qubits == 4:
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    for l in range(6):
                        M[i, j, k, l] = np.real(np.trace(rho @ P[i, j, k, l]))
        return M / 81
    
    else: 
        raise NotImplementedError("Measurement only implemented for up to 4 qubits.")
    

def simulate_interference_visibility(p: np.ndarray, visibility: float) -> np.ndarray:
    """
    Mix probabilities with uniform noise to model limited measurement visibility/contrast.
    visibility=1 gives p unchanged; visibility<1 washes outcomes towards uniform.
    """

    u = np.ones_like(p, dtype=float) / p.size
    p_washed = visibility * p + (1.0 - visibility) * u

    p_washed = np.maximum(p_washed, 0.0)
    p_washed /= p_washed.sum()

    return p_washed


def retrieve_counts_from_n_shots_per_state(p: np.ndarray, n_shots: int) -> np.ndarray:
    """
    Sample integer outcome counts from probabilities p using 'shots' repetitions
    (multinomial counting statistics).
    """

    p_flat = np.asarray(p, dtype=float).ravel()
    p_flat = np.maximum(p_flat, 0.0)
    p_flat /= p_flat.sum()

    counts_flat = np.random.multinomial(n_shots, p_flat)
    return counts_flat.reshape(p.shape)

# ---------------------
# STOKES RECONSTRUCTION
# ---------------------

def counts_to_frequencies(counts: np.ndarray, shots: int) -> np.ndarray:
    """
    Convert outcome counts to empirical frequencies (probability estimates).
    """

    f = np.asarray(counts, dtype=float) / shots
    return f


def build_stokes_matrix(P: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Build the Stokes matrix A such that f = A @ vec(rho),
    where f are flattened measurement frequencies.
    """
    d = 2 ** n_qubits
    m = P.shape[0]  # number of single-qubit projectors (e.g. 6)

    A = np.zeros((m**n_qubits, d*d), dtype=complex)

    for t in range(m**n_qubits):

        # Iterating over t gives full tuple index, (0,0) through (5,5) for a 6x6 for example
        idx = [0] * n_qubits
        x = t
        for q in range(n_qubits - 1, -1, -1):
            idx[q] = x % m
            x //= m # shifts everything right by one digit in base m

        # Row = vec(P_alpha^T) : projector matrix for that outcome, transposed and flattened
        A[t] = P[tuple(idx)].T.reshape(-1)

    return A


def solve_stokes_linear_system(A: np.ndarray, f: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Solve the Stokes linear system A vec(rho) = f in the least-squares sense.
    Returns the reconstructed density matrix (with heriticity and unit trace ensured).
    """
    # Flatten frequencies
    f_flat = np.asarray(f, dtype=complex).reshape(-1)

    # Least-squares solve
    x, *_ = np.linalg.lstsq(A, f_flat, rcond=None)

    # Reshape back into matrix form
    d = 2 ** n_qubits
    rho = x.reshape(d, d)
    rho = 0.5 * (rho + rho.conj().T)  # Ensure Hermiticity
    rho /= np.trace(rho)  # Normalize to trace 1

    return rho


def stokes_reconstruct_dataset(P: np.ndarray,
                              counts,
                              shots: int,
                              n_qubits: int) -> np.ndarray:
    """
    Reconstruct an array of density matrices from a dataset of outcome counts using Stokes (linear inversion).
    Returns an array of length N with (2**n_qubits, 2**n_qubits) density matrices.
    """
    # Build Stokes matrix once
    A = build_stokes_matrix(P, n_qubits)

    # Reconstruct each state
    stokes_rhos = np.empty(len(counts), dtype=object)
    for k, counts_k in enumerate(counts):
        f = counts_to_frequencies(counts_k, shots)
        rho_hat = solve_stokes_linear_system(A, f, n_qubits)
        stokes_rhos[k] = rho_hat

    return stokes_rhos


def fidelity(rho: np.ndarray, sigma: np.ndarray, eps: float = 1e-12) -> float:
    """
    Uhlmann fidelity F(rho,sigma) in [0,1] (NumPy; for evaluation/plots).
    """
    rho = 0.5 * (rho + rho.conj().T)
    sigma = 0.5 * (sigma + sigma.conj().T)

    w, V = np.linalg.eigh(rho)
    w = np.clip(np.real(w), 0.0, None)
    sqrt_rho = V @ np.diag(np.sqrt(w + eps)) @ V.conj().T

    inner = sqrt_rho @ sigma @ sqrt_rho
    inner = 0.5 * (inner + inner.conj().T)

    w2, _ = np.linalg.eigh(inner)
    w2 = np.clip(np.real(w2), 0.0, None)

    tr_sqrt = np.sum(np.sqrt(w2 + eps))
    F = float(np.clip(tr_sqrt**2, 0.0, 1.0))
    return F

    
# ------------------------------------------
#NEURAL NETWORK IMPLEMENTATION
# ------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim


def tau_params_to_rho_torch(params: torch.Tensor, n_qubits: int, eps: float = 1e-12) -> torch.Tensor:
    """
    Real tau-params -> rho = tau^† tau / Tr(...).
    params: (B, d^2) real. Lower-triangular tau with real diagonal, off-diagonal (re,im) pairs.

    Converts the NN output (Cholesky parameters) into a density matrix rho, ensuring it is Hermitian, positive semi-definite, and has trace 1.
    """
    B = params.shape[0]
    d = 2 ** n_qubits
    if params.shape[1] != d * d:
        raise ValueError(f"Expected params dim {d*d}, got {params.shape[1]}.")

    tau = torch.zeros((B, d, d), dtype=torch.complex64, device=params.device)

    k = 0
    for i in range(d):
        for j in range(i + 1):
            if i == j:
                tau[:, i, j] = params[:, k].to(torch.complex64)
                k += 1
            else:
                re = params[:, k]
                im = params[:, k + 1]
                tau[:, i, j] = (re + 1j * im).to(torch.complex64)
                k += 2

    rho = tau.conj().transpose(-1, -2) @ tau
    rho = 0.5 * (rho + rho.conj().transpose(-1, -2))
    tr = torch.real(torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1))
    rho = rho / (tr.view(-1, 1, 1) + eps)
    return rho


def make_mlp(input_dim: int, output_dim: int, hidden_sizes=(256, 256), dropout: float = 0.0) -> nn.Module:
    """
    Builds a standard fully connected multilayer perceptron (MLP) with ReLU activations and optional dropout.
    """
    layers = []
    prev = input_dim
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        if dropout and dropout > 0:
            layers += [nn.Dropout(dropout)]
        prev = h
    layers += [nn.Linear(prev, output_dim)]
    return nn.Sequential(*layers)


class Flatten(nn.Module):
    """Helper module to flatten CNN outputs before the final linear layer."""
    def forward(self, x):
        return x.view(x.size(0), -1)
    
def make_cnn_2d(
    output_dim: int,
    channels=(8, 16),
    kernel_size=3,
    pool=2,
    hidden_sizes=(256,256),
    dropout=0.0,
):
    """
    Builds a simple 2D CNN for 2-qubit data (6x6 input), with configurable channels, kernel size, pooling, and MLP head.
    """
    layers = []
    in_ch = 1

    # CNN layers
    for out_ch in channels:
        layers += [
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.ReLU()
        ]
        if pool and pool > 1:
            layers += [nn.MaxPool2d(pool)]
        in_ch = out_ch

    layers += [nn.Flatten()]

    # MLP head
    for h in hidden_sizes:
        layers += [
            nn.LazyLinear(h),  # avoids manual flatten dim calc
            nn.ReLU(),
        ]
        if dropout > 0:
            layers += [nn.Dropout(dropout)]

    layers += [nn.LazyLinear(output_dim)]

    return nn.Sequential(*layers)



class NN_Builder:
    """
    Minimal NN trainer for QST.
    - model_type: "mlp" or "cnn" (only MLP implemented for now)
    - loss_type:  "mse" or "fidelity"
    - target:     "tau" (recommended) or "rho" (optional)
    """

    def __init__(self,
                 n_qubits: int,
                 model_type: str = "mlp",
                 loss_type: str = "mse",
                 target: str = "tau",
                 hidden_sizes=(256, 256),
                 dropout: float = 0.0,
                 cnn_channels=(8,16),
                 cnn_kernel_size=3,
                 cnn_kernel_type: str = "standard",         # "standard" or "proj_kernel"
                 proj_kernel_metric: str = "overlap",       # "overlap" or "fidelity" (only used if proj_kernel)             
                 lr: float = 1e-3,
                 batch_size: int = 64,
                 epochs: int = 50,
                 device: str | None = None,
                 seed: int = 0):

        self.n_qubits = n_qubits
        self.d = 2 ** n_qubits

        self.model_type = model_type
        self.loss_type = loss_type
        self.target = target

        self.hidden_sizes = tuple(hidden_sizes)
        self.dropout = dropout
        self.cnn_channels = cnn_channels
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_kernel_type = cnn_kernel_type
        self.proj_kernel_metric = proj_kernel_metric
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.input_dim = 6 ** n_qubits  # flattened freqs
        self.output_dim = (self.d * self.d) if target == "tau" else (2 * self.d * self.d)

        self._prepare_proj_kernel_mixer()
        self.model = self._build_model().to(self.device)


    # Torch fidelity used for training (kept with the NN system)
    @staticmethod
    def uhlmann_fidelity_torch(rho: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        Batched Uhlmann fidelity for training (torch; differentiable).
        """
        rho = 0.5 * (rho + rho.conj().transpose(-1, -2))
        sigma = 0.5 * (sigma + sigma.conj().transpose(-1, -2))

        w, V = torch.linalg.eigh(rho)
        w = torch.clamp(w.real, min=0.0)
        sqrt_rho = V @ torch.diag_embed(torch.sqrt(w + eps)).to(V.dtype) @ V.conj().transpose(-1, -2)

        inner = sqrt_rho @ sigma @ sqrt_rho
        inner = 0.5 * (inner + inner.conj().transpose(-1, -2))

        w2, _ = torch.linalg.eigh(inner)
        w2 = torch.clamp(w2.real, min=0.0)

        tr_sqrt = torch.sum(torch.sqrt(w2 + eps), dim=-1)
        F = tr_sqrt**2
        return torch.clamp(F.real, 0.0, 1.0)


    def _build_model(self) -> nn.Module:
        if self.model_type == "mlp":
            return make_mlp(self.input_dim, self.output_dim, self.hidden_sizes, self.dropout)
        
        if self.model_type == "cnn":
            return make_cnn_2d(output_dim=self.output_dim, channels=self.cnn_channels, kernel_size=self.cnn_kernel_size, hidden_sizes=self.hidden_sizes, dropout=self.dropout)
        
        raise ValueError(f"Unknown model_type: {self.model_type}")
    
    
    def _compute_projector_kernel_matrix(self) -> np.ndarray:
        """
        Build K (36x36) from two-qubit projectors P_ij = P_i ⊗ P_j.

        Metrics:
        - overlap:  K_ab = Tr(P_a P_b)
        - fidelity: K_ab = |Tr(P_a P_b)|^2

        K is a 36x36 matrix that captures the closeness of the 36 projectors to each other
        """
        if self.n_qubits != 2:
            raise NotImplementedError("proj_kernel currently implemented for n_qubits==2 only.")

        metric = getattr(self, "proj_kernel_metric", "overlap")
        if metric not in ("overlap", "fidelity"):
            raise ValueError(f"proj_kernel_metric must be 'overlap' or 'fidelity', got {metric}")

        P = build_projector_matrix(self.n_qubits)  # shape (6,6) object, each entry is (4x4)
        proj_list = [P[i, j] for i in range(6) for j in range(6)]  # row-major: a = 6*i + j

        K = np.zeros((36, 36), dtype=np.float64)
        for a in range(36):
            Pa = proj_list[a]
            for b in range(36):
                Pb = proj_list[b]
                t = np.trace(Pa @ Pb)
                # should be real for projectors, but keep it safe numerically
                if metric == "overlap":
                    K[a, b] = float(np.real(t))
                else:  # fidelity-like
                    K[a, b] = float(np.abs(t) ** 2)

        return K


    @staticmethod
    def _row_normalise(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        row_sums = M.sum(axis=1, keepdims=True)
        return M / (row_sums + eps)


    def _prepare_proj_kernel_mixer(self) -> None:
        """
        Precompute a fixed 36x36 mixing matrix for the CNN input
        Stored as self._cnn_mixer or None.
        """
        self._cnn_mixer = None

        if self.model_type != "cnn":
            return
        if self.cnn_kernel_type == "standard":
            return
        if self.cnn_kernel_type != "proj_kernel":
            raise ValueError(f"Unknown cnn_kernel_type: {self.cnn_kernel_type}")

        K = self._compute_projector_kernel_matrix()
        M = self._row_normalise(K)
        self._cnn_mixer = M.astype(np.float32)


    def _build_X(self, data) -> torch.Tensor:
        shots = data["shots"]
        counts = data["counts"]
        freqs = np.asarray(counts, dtype=float) / shots

        if self.model_type == "mlp":
            X = freqs.reshape(len(freqs), -1).astype(np.float32)
            return torch.from_numpy(X).to(self.device)

        if self.model_type == "cnn":
            if self.n_qubits != 2:
                raise NotImplementedError("CNN only implemented for 2 qubits (36 input projectors). Add more channels/filters and reshape logic for more qubits.")
            
            F = freqs.reshape(len(freqs), -1).astype(np.float32)  # (N, 36)

            if getattr(self, "_cnn_mixer", None) is not None:
                F = F @ self._cnn_mixer.T  # (N,36)

            X = F.reshape(len(freqs), 1, 6, 6).astype(np.float32) # CNN expects NCHW format (Number of samples in batch, channels, height, width)
            return torch.from_numpy(X).to(self.device)

    def _build_Y(self, data) -> torch.Tensor:
        """
        Finds the target Y for comparison, useful in the loss_type="mse" case.
        """
        if self.target == "tau":
            taus = data["taus"]
            taus_arr = np.stack(taus, axis=0)

            N = taus_arr.shape[0]
            params = np.zeros((N, self.d * self.d), dtype=np.float32)

            k = 0
            for i in range(self.d):
                for j in range(i + 1):
                    if i == j:
                        params[:, k] = np.real(taus_arr[:, i, j])
                        k += 1
                    else:
                        params[:, k] = np.real(taus_arr[:, i, j])
                        params[:, k + 1] = np.imag(taus_arr[:, i, j])
                        k += 2

            return torch.from_numpy(params).to(self.device)

        if self.target == "rho":
            rhos = data["rhos"]
            rhos_arr = np.stack(rhos, axis=0)
            flat = rhos_arr.reshape(len(rhos_arr), -1)
            Y = np.concatenate([flat.real, flat.imag], axis=1).astype(np.float32)
            return torch.from_numpy(Y).to(self.device)

        raise ValueError(f"Unknown target: {self.target}")


    def _pred_to_rho(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Convert the NN output back into a density matrix rho for loss calculation. Useful if loss_type is "fidelity", as the fidelity loss needs to compare density matrices directly.
        """
        if self.target == "tau":
            return tau_params_to_rho_torch(pred, self.n_qubits)

        B = pred.shape[0]
        d = self.d
        re = pred[:, : d * d]
        im = pred[:, d * d :]
        rho = (re + 1j * im).to(torch.complex64).reshape(B, d, d)
        rho = 0.5 * (rho + rho.conj().transpose(-1, -2))
        tr = torch.real(torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1))
        rho = rho / (tr.view(-1, 1, 1) + 1e-12)
        return rho


    def _true_rho_tensor(self, data) -> torch.Tensor:
        """
        Convert the true density matrices from the dataset into a torch tensor for comparison in the fidelity loss.
        """
        rhos = data["rhos"]
        rhos_arr = np.stack(rhos, axis=0) if (isinstance(rhos, list) or (isinstance(rhos, np.ndarray) and rhos.dtype == object)) else np.asarray(rhos)
        return torch.from_numpy(rhos_arr).to(self.device).to(torch.complex64)


    def _loss(self, pred: torch.Tensor, Y: torch.Tensor, rho_true: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss between the predicted output and the true target, depending on the specified loss type.

        pred: (B, output_dim) tensor from the model
        Y: (B, output_dim) tensor of true targets (used for MSE loss)
        rho_true: (B, d, d) tensor of true density matrices (used for fidelity loss)
        """
        if self.loss_type == "mse":
            return torch.mean((pred - Y) ** 2)

        if self.loss_type == "fidelity":
            rho_pred = self._pred_to_rho(pred)
            F = self.uhlmann_fidelity_torch(rho_pred, rho_true)
            return torch.mean(1.0 - F)

        raise ValueError(f"Unknown loss_type: {self.loss_type}")


    def fit(self, data_train):
        X = self._build_X(data_train)
        Y = self._build_Y(data_train) if self.loss_type == "mse" else None
        rho_true_all = self._true_rho_tensor(data_train)

        N = X.shape[0]
        idx = torch.randperm(N, device=self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        history = {"loss": []}

        prog_marker = max(1, self.epochs // 10)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for start in range(0, N, self.batch_size):
                bidx = idx[start:start + self.batch_size]
                xb = X[bidx]
                rb = rho_true_all[bidx]
                yb = Y[bidx] if Y is not None else None

                optimizer.zero_grad()
                pred = self.model(xb)
                loss = self._loss(pred, yb, rb)   # update _loss to allow Y=None
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())

            history["loss"].append(epoch_loss)
            if (epoch + 1) % prog_marker == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}")

        return history

    def predict(self, data_any):
        X = self._build_X(data_any)

        self.model.eval()
        with torch.no_grad():
            pred_all = self.model(X)
            rho_pred_all = self._pred_to_rho(pred_all).detach().cpu().numpy()

        out = np.empty(len(rho_pred_all), dtype=object)
        for k in range(len(rho_pred_all)):
            out[k] = rho_pred_all[k]
        return out


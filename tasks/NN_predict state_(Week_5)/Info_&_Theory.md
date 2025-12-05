# Learning Two-Qubit Quantum States from Noisy Tomography

## 1. Overview

The goal is to learn a mapping from **noisy tomographic measurements** of a two-qubit system to its underlying **density matrix** \( \rho \). 

We follow the setup of Lohani *et al.*:

- Generate random (mixed) two-qubit states \( \rho \) from the **Ginibre ensemble**.
- Simulate **noisy projective measurements** using a fixed set of 36 two-qubit projectors.
- Train neural networks (fully-connected and CNN) to reconstruct \( \rho \) from the 36-dimensional measurement vector.
- Compare performance with a classical **Stokes (linear inversion)** reconstruction.
- Ultimately, Lohani uses a **Cholesky (τ) parametrisation** to ensure physical density matrices. Here we first work with the full complex vectorisation of \( \rho \), and later we can switch to τ.

---

## 2. State Generation (Ginibre ensemble)

We generate random mixed two-qubit states using the Ginibre ensemble:

\[
G = N(0,1) + i\,N(0,1), \quad 
\rho = \frac{G G^{\dagger}}{\mathrm{Tr}(G G^{\dagger})},
\]

where \( G \) is a complex \(4 \times 4\) matrix with i.i.d. Gaussian entries. This guarantees:

- \( \rho \) is Hermitian
- \( \rho \ge 0 \) (positive semidefinite)
- \( \mathrm{Tr}(\rho) = 1 \).

In this project we use **only mixed states** for simplicity.

---

## 3. Tomography Measurement Setup

### 3.1 Single-qubit polarisation basis

We work in the polarisation basis with states

\[
\begin{aligned}
|H\rangle &= 
\begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad 
|V\rangle = 
\begin{bmatrix} 0 \\ 1 \end{bmatrix}, \\
|D\rangle &= \tfrac{1}{\sqrt{2}} (|H\rangle + |V\rangle), \quad 
|A\rangle = \tfrac{1}{\sqrt{2}} (|H\rangle - |V\rangle), \\
|R\rangle &= \tfrac{1}{\sqrt{2}} (|H\rangle + i|V\rangle), \quad 
|L\rangle = \tfrac{1}{\sqrt{2}} (|H\rangle - i|V\rangle).
\end{aligned}
\]

The corresponding projectors are

\[
h = |H\rangle\langle H|,\quad
v = |V\rangle\langle V|,\quad
d = |D\rangle\langle D|,\quad
a = |A\rangle\langle A|,\quad
r = |R\rangle\langle R|,\quad
l = |L\rangle\langle L|.
\]

### 3.2 Two-qubit tomography matrix \( P \)

The two-qubit projectors are tensor products of these single-qubit projectors. We arrange 36 of them in a \(6 \times 6\) matrix \( P \):

\[
P =
\begin{bmatrix}
h\!\otimes\!h & h\!\otimes\!v & v\!\otimes\!v & v\!\otimes\!h & v\!\otimes\!r & v\!\otimes\!l \\
h\!\otimes\!l & h\!\otimes\!r & h\!\otimes\!d & h\!\otimes\!a & v\!\otimes\!a & v\!\otimes\!d \\
a\!\otimes\!d & a\!\otimes\!a & d\!\otimes\!a & d\!\otimes\!d & d\!\otimes\!r & d\!\otimes\!l \\
a\!\otimes\!l & a\!\otimes\!r & a\!\otimes\!h & a\!\otimes\!v & d\!\otimes\!v & d\!\otimes\!h \\
r\!\otimes\!h & r\!\otimes\!v & l\!\otimes\!v & l\!\otimes\!h & l\!\otimes\!r & l\!\otimes\!l \\
r\!\otimes\!l & r\!\otimes\!r & r\!\otimes\!d & r\!\otimes\!a & l\!\otimes\!a & l\!\otimes\!d
\end{bmatrix}.
\]

Each element \( P_{ij} \) is a projector on the two-qubit Hilbert space.

For a given two-qubit state \( \rho \), the ideal (noiseless) measurement probability is

\[
M_{ij} = \mathrm{Tr}(\rho\,P_{ij}),
\]

leading to a \(6 \times 6\) matrix \( M \) of probabilities, or 36-dimensional vector \( \mathrm{vec}(M) \).

---

## 4. Noise Model: Misaligned Projectors

Lohani introduces noise by modelling **small misalignments** in the measurement devices. Each single-qubit projector is rotated by a random unitary drawn from Gaussian-distributed Euler angles.

- Sample Euler angles \( (\theta, \phi, \zeta) \sim \mathcal{N}(0, \sigma^2) \).
- Construct a random SU(2) unitary \( U(\theta,\phi,\zeta) \).
- For two qubits, use \( U = U_1 \otimes U_2 \) with independent draws for each qubit.

The noisy two-qubit projector is then

\[
P_{ij}^{\text{(noisy)}} = U\, P_{ij}\, U^\dagger.
\]

The measured probabilities are

\[
M_{ij} = \mathrm{Tr}\big(\rho\,P_{ij}^{\text{(noisy)}}\big).
\]

Here \( \sigma \) controls the strength of misalignment (e.g. \( \sigma \approx \pi/6 \) in the experiments).

---

## 5. Dataset Generation

For a given dataset size \( N \):

1. For each \( k = 1,\dots,N \):
   - Sample a random mixed state \( \rho_k \) from the Ginibre ensemble.
   - Compute noisy measurement matrix \( M_k \) using the projector set \( P \) and noise level \( \sigma \).
2. Flatten \( M_k \) into a 36-dimensional feature vector \( x_k = \mathrm{vec}(M_k) \).
3. Vectorise the density matrix:
   - \( \rho_k \) is \(4 \times 4\), so \( \mathrm{vec}(\rho_k) \in \mathbb{C}^{16} \).
   - Split into real and imaginary parts, giving a 32-dimensional target
     \[
     y_k = (\Re(\mathrm{vec}(\rho_k)),\, \Im(\mathrm{vec}(\rho_k))).
     \]

This yields a supervised dataset

\[
\{(x_k, y_k)\}_{k=1}^N,
\]

with \( x_k \in \mathbb{R}^{36} \), \( y_k \in \mathbb{R}^{32} \).

Later, this representation can be replaced by a **τ-parametrisation** (16 real outputs) without changing the data-generation pipeline.

---

## 6. Stokes Reconstruction (Linear Inversion)

As a classical baseline, we use **linear inversion** based on Stokes parameters.

For each projector \( P_k \) (one of the 36 projectors from \( P \)), the measurement probability is

\[
M_k = \mathrm{Tr}(\rho P_k).
\]

Using the identity

\[
\mathrm{Tr}(\rho P_k) = \mathrm{vec}(P_k^T)^\dagger\, \mathrm{vec}(\rho),
\]

we stack all 36 equations into matrix form:

\[
M = A \, \mathrm{vec}(\rho),
\]

where

- \( M \in \mathbb{R}^{36} \) is the measurement vector,
- \( \mathrm{vec}(\rho) \in \mathbb{C}^{16} \) is the vectorised density matrix,
- \( A \in \mathbb{C}^{36 \times 16} \) has rows \( \mathrm{vec}(P_k^T)^\dagger \).

The **Stokes reconstruction** solves the least-squares problem

\[
\mathrm{vec}(\rho_{\text{Stokes}}) = A^{+} M,
\]

with \( A^{+} \) the Moore–Penrose pseudoinverse.

The result is reshaped to \( 4 \times 4 \), then projected back onto physical states by:

\[
\rho_{\text{Stokes}}
= \frac{1}{\mathrm{Tr}(\tilde\rho)}
\frac{\tilde\rho + \tilde\rho^\dagger}{2},
\quad
\tilde\rho = \text{reshape}(A^+ M).
\]

This gives a **fast, classical** reconstruction method to compare against the neural networks.

---

## 7. Neural Network Reconstruction

### 7.1 Vector representation and fidelity

The neural networks output a 32-dimensional vector:

\[
\hat{y} = (\Re(\mathrm{vec}(\hat\rho)),\, \Im(\mathrm{vec}(\hat\rho))).
\]

We reconstruct a predicted density matrix \( \hat{\rho} \) by

1. Combining real and imaginary parts back into a complex vector.
2. Reshaping to \( 4 \times 4 \).
3. Symmetrising and renormalising:

\[
\hat{\rho} \leftarrow \frac{\hat{\rho} + \hat{\rho}^\dagger}{2}, \quad
\hat{\rho} \leftarrow \frac{\hat{\rho}}{\mathrm{Tr}(\hat{\rho})}.
\]

The **state fidelity** between predicted \( \hat{\rho} \) and true \( \rho \) is

\[
F(\hat{\rho}, \rho) =
\left(
\mathrm{Tr}\left[
\sqrt{
\sqrt{\rho}\,\hat{\rho}\,\sqrt{\rho}
}
\right]
\right)^2.
\]

We use this both as an evaluation metric and (optionally) as a **fidelity-based loss** during training.

---

### 7.2 Fully-connected network (NN)

A baseline network maps the 36-dimensional measurement vector to 32 outputs:

- Input: \( \mathbb{R}^{36} \)
- Two hidden layers (e.g. 720 and 450 neurons, ReLU)
- Dropout for regularisation
- Output: 32 neurons (16 real + 16 imaginary components of \( \rho \))

We experiment with two loss choices:

- **MSE loss** on the 32-dimensional output.
- **Fidelity loss**, computed from the reconstructed density matrices.

By repeating the experiment for different training set sizes \( N \), we study how the **average test fidelity** scales with dataset size.

---

### 7.3 Convolutional neural network (CNN)

Lohani uses a CNN that operates on the **6×6 grid structure** of the measurement matrix \( M \). Our CNN follows the same idea:

- Input: measurement matrix \( M \) reshaped as a \(1 \times 6 \times 6\) image.
- Convolutional layers (e.g. 3×3 kernels, stride 1, padding 1).
- ReLU activations.
- Max pooling (2×2) to reduce from \(6 \times 6\) to \(3 \times 3\) feature maps.
- Flattening and fully-connected layers (e.g. 128 neurons).
- Dropout for regularisation.
- Output: 32 neurons (again, real and imaginary parts of \( \rho \)).

Key CNN concepts in this context:

- **Kernel / filter**: a small (e.g. 3×3) matrix that scans over the input, detecting local patterns.
- **Stride**: how far the kernel moves at each step (here stride 1).
- **Feature maps**: each filter produces one feature map; multiple filters detect different structures in the measurement pattern.
- **Max pooling**: down-samples by taking the maximum in local regions, emphasising dominant features and reducing dimensionality.

Again, we compare MSE vs fidelity-based loss and examine how CNN performance scales with training set size \( N \).

---

## 8. Cholesky (τ) Parametrisation (Lohani’s approach)

Lohani does **not** train the network to output the density matrix directly. Instead, the network outputs a **lower-triangular complex matrix** \( \tau \) (the Cholesky factor), with 16 real parameters.

The density matrix is then reconstructed as

\[
\rho_{\text{pred}} = 
\frac{\tau_{\text{pred}}^{\dagger} \tau_{\text{pred}}}
{\mathrm{Tr}(\tau_{\text{pred}}^{\dagger} \tau_{\text{pred}})}.
\]

Advantages:

- By construction, \( \tau^\dagger \tau \) is positive semidefinite.
- Normalising the trace enforces \( \mathrm{Tr}(\rho) = 1 \).
- The network inherently produces **physical density matrices**.

In our current implementation we still use the full \( \mathrm{vec}(\rho) \) representation with 32 outputs. A later step will be to switch the output layer to 16 real numbers parameterising \( \tau \), and modify the fidelity loss accordingly.

---

## 9. Physical Meaning of the Nucrypt Projectors

In the experiment, single-photon polarisations are measured using a combination of:

- **Polarising beam splitter (PBS)**:
  - Transmits horizontal (H) and reflects vertical (V).
  - Measuring in H/V basis corresponds to placing detectors at the two PBS outputs.

- **Half-wave plate (HWP)**:
  - Rotates linear polarisation by twice the angle between the input polarisation and the plate’s fast axis.
  - Setting the HWP to \(22.5^\circ\) rotates H/V into D/A.
  - The PBS still measures in H/V, but after the HWP this effectively corresponds to a D/A measurement.

- **Quarter-wave plate (QWP)**:
  - Converts between linear and circular polarisations.
  - With the QWP at \( \pm 45^\circ \) before the PBS, the setup measures R/L polarisations.

In this language:

- \( |H\rangle \): horizontal polarisation, measured directly by the PBS.
- \( |V\rangle \): vertical polarisation.
- \( |D\rangle = (|H\rangle + |V\rangle)/\sqrt{2} \): diagonal (45°) polarisation, realised by rotating into this basis with a HWP then measuring H/V.
- \( |A\rangle = (|H\rangle - |V\rangle)/\sqrt{2} \): anti-diagonal (−45°).
- \( |R\rangle = (|H\rangle + i|V\rangle)/\sqrt{2} \): right-circular, obtained using a QWP before the PBS.
- \( |L\rangle = (|H\rangle - i|V\rangle)/\sqrt{2} \): left-circular.

A two-qubit projector such as \( h \otimes d \) (with \( h = |H\rangle\langle H| \), \( d = |D\rangle\langle D| \)) asks:

> “What is the probability that **the first photon** is horizontally polarised and **the second** is diagonally polarised?”

By varying these projectors across the 36-element set, we perform **full two-qubit tomography**.

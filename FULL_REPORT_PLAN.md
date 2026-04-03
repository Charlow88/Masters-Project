Introduction - 5-6 pages
Methodology - 10-12 pages
Results/ Discussion - 11-12 pages
limits/ Future Work - 1 page
Conclusion - 1 page

# Introduction

## What is QST

Explain quantum state tomography (QST) as the process of reconstructing the quantum state of a system from repeated measurements on identically prepared states.

Explain that quantum states are represented by density matrices ρ which must satisfy:

- Hermiticity
- Positive semidefiniteness
- Unit trace

Explain that tomography reconstructs ρ from measurement probabilities given by Born’s rule:

p_k = Tr(ρ E_k)

The measurement probabilities are inferred from measurement outcomes of *n* measurements, known as shots.

Although its stochastic, if there are enough shots, can reduce uncertainty

Explain that the number of independent parameters grows exponentially with the number of qubits:

4^n − 1 parameters for an n-qubit system.

Explain the scaling problem and why tomography quickly becomes computationally difficult.

Explain measurement operators and POVMs and how these correspond to experimentally accessible projectors.

## The physcial system

BE CAREFUL, THIS IS A METHODOLOGICAL EXPLORATION, NOT SPECIFIC TO THIS SYSTEM. WHILE ITS BEING USED, SHOWCASE THAT THE METHODS APPLY MORE GENERALLY. KEEP THE SCOPE TIGHT EXPERIMENTALLY, FOCUS MORE ON THE METHOD, GIVE GOOD REASOMN TO QUESTION DODGE SPDC QUESTIONS IN VIVA

Describe the physical system used as inspiration for the simulation.

Applications in quantum communication, metrology etc etc, why it's a good setup

Explain photonic polarisation qubits:

|H⟩ and |V⟩ represent horizontal and vertical polarisation.

Explain generation of entangled photon pairs using SPDC. Important that this is made clear enopugh for viva especially

Give a diagram for this and explain the physics behind generation slightly.

Short pulses generate pure states, and longer pulses generate mixed states. Mention this as important for further analysis. Very good reference

Explain typical experimental tomography setup:

- waveplates
- polarising beam splitters
- detectors

Introduce the six single-qubit measurement states:

H, V, D, A, R, L

Explain that these correspond to three mutually unbiased bases.

These are determined by the setup, not arbitrarily

Explain that two-qubit tomography uses tensor products of these states.

Explain that this produces **36 projectors** forming an informationally complete measurement set.

These projectors and bases are able to be found through a combo of half wave plates, quarter wave plates and polarising beam splitters.

Explain that although the project is motivated by photonic experiments, all results are obtained through **simulation at the density matrix level**.

Explain why this allows controlled comparison between reconstruction methods.

Mention that measurement noise in experiments arises from:

- waveplate misalignment
- polarisation drift
- detector noise
- imperfect state preparation

Explain that the simulation attempts to capture relevant experimental imperfections.

## Classical approaches

Introduce classical reconstruction approaches:

- Stokes reconstruction (linear inversion)
- Maximum Likelihood Estimation (MLE)

Explain briefly how these work and that they will be discussed in detail in the methodology.

Introduce the limitations motivating this project:

- Stokes reconstruction can produce **unphysical density matrices**, doesnt enforce positivity
- MLE guarantees physicality but is **computationally expensive**
- Tomography becomes difficult to perform quickly in practical experiments

Both also struggle with noise and can be affected by this

## Machine learning approaches

Introduce the idea of **machine learning assisted tomography**.

Explain that neural networks can learn a mapping from measurement probabilities to density matrices, good at learning relationships

Good explanation on what NNs are and how they work: backpropagation etc.

Shifts computation to an offline training phase

Non trivial to apply, mapping highly structured

Explain the potential advantages:

- fast inference
- robustness to noise
- avoiding optimisation at reconstruction time

Explain that naïve ML architectures may not incorporate the physics of the problem.

Mention similar work in other projects, RNNs, Lohani etc.

Introduce the key hypothesis of the project:

Neural networks can perform tomography effectively, but **architectures informed by physical structure should perform best.**

State the central aim of the project:

Benchmark classical tomography methods against neural network approaches and investigate the importance of incorporating physical structure into neural network architectures.

Mention limitations faced regarding physicality meaningfulness etc.

## Scope of this project (what to explore)

Generally the idea of implementing a NN to improve upon the limitations of the baseline model

Documents architecture used etc.

### Cholesky parameterisation

Mention how Cholesky parameterisation can be used to enforce physicality, show key equations

15 dimension real space for 2 qubits

Frees NN to focus on optimsiing without penalisation of unphysical states

Cholesky always give physical states, but small deviations from physicality can occer when reconstructing states that are near the the boundary of state space (pure states), where floating point errors can give very small negative eigenvalues

Dont explain implementation yet just the theory behind it

Lower triangular matrix predicted

### More informed loss function

Standard loss used can be MSE between elements of predicted matrix.

Using a fidelity based loss is more physically meaningful, explain fidelity and give equation. 

### Exploiting architecture via CNNs

Fully connected NN just takes the 36 inputs and applies the NN. A CNN can explore the input projector matrix to see if there are patterns that can be used to help predict the outcome.

## Briefly explain reprot sstructure

explain report structure

---

# Methodology

## Overview of reconstruction pipeline

Explain the full pipeline used in the project:

1. Generate random quantum states
2. Simulate tomographic measurements
3. Apply experimental noise models
4. Reconstruct density matrices using different algorithms
5. Compare reconstruction fidelity and runtime

Include diagram of full reconstruction pipeline.

Explain that all methods are evaluated on identical simulated datasets.

---

## Quantum state generation

Explain generation of random density mixed matrices using the **Ginibre ensemble**.

Pure states generated via a random complex vector

Explain that this guarantees valid density matrices.

Explain that the ensemble produces states with varying purity.

Explain that both **pure and mixed states are analysed separately**.

Explain physical motivation:

- SPDC sources can produce highly pure states under ideal conditions
- realistic experimental imperfections lead to mixed states

Explain dataset composition:

- datasets include both pure and mixed states
- separate analysis performed for each category

Explain that this allows reconstruction performance to be studied across different regions of state space.

---

## Tomographic measurement model

Explain construction of measurement projectors.

Explain tensor product structure giving 36 projectors.

Explain calculation of probabilities using Born’s rule.

Explain representation of measurement outcomes.

For neural networks:

- measurement probabilities arranged as a **6×6 grid**
- CNN models use grid structure
- fully connected models use flattened 36-dimensional vector

Explain that the ordering of the grid is a modelling choice rather than a physical spatial structure.

Explain implications for CNN inductive biases.

---

## Noise model

Explain sources of noise in photonic tomography experiments.

Explain simulation of measurement imperfections through **rotations of measurement operators**.

Explain that local SU(2) rotations are applied to projectors along one axis to simulate waveplate miscalibration

Explain that the rotation magnitude controls noise strength.

Explain that noise parameters are chosen based on experimentally realistic values.

Include reference values used:

- visibility ≈ 93% typical in realistic systems
- misalignment errors ~0.08 degrees
give references for these

Explain that probabilities are computed using noisy projectors.

Explain that noise is applied independently to each measurement operator.

Explain that this models calibration errors in experiments.

---

## Shot noise simulation

Explain the role of shot noise in experiments.

Explain that measurement probabilities are estimated from finite counts.

Explain that statistical uncertainty scales as:

1 / √N

Explain that tomography is therefore affected by the number of measurement shots.

Describe **shot sweep experiments** performed in the project. References describe a range of shots that can be used, so a seep is best practice here

Explain that reconstructions are evaluated for multiple shot numbers.

Explain that the shot sweep explores how reconstruction fidelity depends on available measurement statistics.

Explain that shot ranges explored cover typical experimental regimes.

---

## Dataset construction

Explain dataset generation process. 10,000 states, spanning shots 10, 20, 40, 80, 160, 320, 640, 1280

For each quantum state:

- generate noisy measurement probabilities
- store measurement matrix
- store corresponding density matrix parameters

Explain train/test split.

Explain validation set used for hyperparameter optimisation.

Explain dataset structure stored in results files.

Explain that datasets are constructed separately for:

- different numbers of shots in the sweep
- different qubit systems
- pure and mixed states

---

## Classical reconstruction methods

### Stokes reconstruction

Explain linear inversion method.

Explain formulation as linear system.

Explain least squares reconstruction.

Explain advantages:

- extremely fast
- simple to implement

Explain disadvantages:

- positivity not guaranteed
- reconstructed matrices may be unphysical

Explain that Stokes reconstruction is used as the **classical baseline** alongside MLE.

---

### Maximum Likelihood Estimation (MLE)

Explain formulation as optimisation problem.

Explain that reconstruction maximises likelihood of observed measurement outcomes.

Explain constraints imposed:

- positive semidefinite density matrix
- trace normalisation

Explain optimisation procedure.

Explain computational complexity.

Explain that MLE is widely used in experimental tomography.

Explain its major limitation: **slow runtime due to iterative optimisation**. 

Explain why runtime matters in real applications

---

## Neural network models

Explain motivation for learning-based reconstruction.

Explain that networks learn mapping from measurement probabilities to density matrix parameters.

Explain that once trained they reconstruct states using a single forward pass.

---

### Fully connected neural network

Describe architecture used. Optimiser, layers etc. include all

Explain input layer representing flattened measurement probabilities.

Explain hidden layer structure.

Explain output layer producing Cholesky parameters.

Explain that this model makes minimal assumptions about input structure.

---

### CNN architectures

Explain motivation for CNNs.

Explain representation of measurement probabilities as 6×6 grid.

Explain inductive biases of convolutional layers.

Explain that CNN assumes local relationships between neighbouring measurements.

Explain that this assumption may not correspond directly to tomography physics.

Explain that multiple CNN architectures were explored.

These include:

- standard CNN
- overlap mixing CNN
- fidelity mixing CNN
- graph CNN for sparsity

explain the maths of the mixing types etc

Explain that these architectures attempt to exploit structure in measurement data.

---

### Physics informed neural network

Explain motivation for physics-informed design.

Explain that measurement relationships are determined by known projectors.

Explain that architecture incorporates knowledge of tomography structure.

Explain that this network is designed to better reflect relationships between measurements.

Explain hypothesis that physics-informed architecture should outperform naïve CNN designs.

Include a graph with full strcuture CNN + NN + Cholesky etc. etc.

---

## Hyperparameter tuning

Explain hyperparameter optimisation procedure.

For ones not in CV, use literature to justify the choice

Explain use of **separate validation dataset**.

Explain that validation set uses balanced mixture of pure and mixed states. 320 shots, 50% pure, 50% mixed

Explain hyperparameters tuned include:

- learning rate
- network depth
- dropout
- batch size

Explain use of dropout as regularisation.

Explain that hyperparameters are selected using validation performance.

Explain that final model evaluation uses held-out test set.

Explain that cross-validation experiments were used to verify robustness.

Explain that CV done first on a larger range of values for the params, then a more lcoal set once found. Give the values used in appendix maybe, and show some data to justify decision of values here

Can say that performance will vary between sets, so slightly arbitrary when minimal gain in perfomrance

Explain that learning rate has minimal effect, so epochs are chosen to be 100 using a validation curve. Show the figure here - defensible justification, lr chosen to be 5e-4, 1e-4 struggled, 1e-3 more aggressive - balancing accuracy with computational cost. 100 close enough to 106 to be chosen, arbitrary enough

---

## Output parameterisation

Explain why neural networks cannot directly output arbitrary density matrices.

Explain the need to enforce physical constraints.

Describe **Cholesky parameterisation**:

ρ = τ†τ / Tr(τ†τ)

Explain that this guarantees:

- positivity
- Hermiticity
- trace normalisation

Explain that neural networks predict parameters of τ.

Explain that density matrices are constructed deterministically from these parameters.

Explain advantages:

- ensures physical states
- avoids eigenvalue clipping
- allows differentiable mapping during training

---

## Loss functions

Explain loss functions tested during training.

### Mean squared error loss

Explain formulation.

Explain advantages:

- simple optimisation
- stable gradients

Explain disadvantages:

- not physically meaningful metric

---

### Fidelity loss

Explain fidelity between quantum states.

Explain definition of Uhlmann fidelity.

Explain training objective:

1 − fidelity

Explain that fidelity directly measures physical similarity between states.

Explain optimisation challenges due to nonlinear structure.

Explain that fidelity loss better aligns with evaluation metric.

---

## Evaluation metrics

Explain main metric used:

Average Uhlmann fidelity between true and reconstructed states.

Explain that fidelity is computed for each test state.

Explain mean fidelity across test set.

Explain calculation of confidence intervals.

Explain that results are reported with error bars.

Explain that statistical tests are used to compare model performance.

---

## Statistical testing

Explain use of **paired t-tests** to compare reconstruction methods.

Explain that t-tests are used to determine whether performance differences are statistically significant.

Example used in project:

Comparison between **overlap mixing CNN** and **physics-informed neural network**.

Explain that statistical tests ensure model selection is justified rather than based on noise in the data.

Show how the test will be used, save the result for the results

---

# Results

FOR ALL RESULTS MENTION ERROR SLEEVES TO SMALL TO BE VISIBLE, AND IF GRAPHS ARE CUT OFF FOR VISIBILTIY MENTION THIS

MAKE SURE TO EXPLAIN ALL ERRORS AND HOW THEY ARE CALCULATED ALSO, NOT ENOUGH TO REQUIRE AN ERROR APPENDIX THOUGH

## Architecture comparison

Compare different neural network architectures.

Show naive vs informed first
then show CNN vs NN

Include:

- fully connected networks
- CNN variants
- overlap mixing CNN
- physics-informed neural network

Explain results of t-test comparisons.

Explain that overlap mixing CNN does not significantly outperform physics-informed model. Use the t test results as mentioned in the methodology to verify this (do in a table maybe)

Explain that CNNs don't compete as much and this could be due to all infomration globally affecting the outcome. Better at lower shots could imply that when there's less informationally complete data the CNNs find patterns that allow better predicitons, potentially a good low shot solution

Explain that physics-informed model provides best balance of performance and simplicity.

Explain rationale for selecting physics-informed model as final architecture.

---

## Reconstruction fidelity vs number of shots

Present fidelity curves for all reconstruction methods.

Compare:

- Stokes reconstruction
- MLE
- fully connected neural network
- CNN variants
- physics-informed neural network

Explain how fidelity varies with number of measurement shots.

Explain trends observed.

Explain convergence behaviour of neural networks with increasing data.

---

### Performance

Explain the performance of the fidelities in each at different shots and in the different state types

Explain that neural networks achieve reconstruction fidelity comparable to classical methods. In practical terms negiligible difference

---

### Physicality of reconstructed states

Evaluate whether reconstructed matrices satisfy physical constraints.

Show that:

- Stokes reconstruction can produce unphysical matrices
- neural networks always produce physical states for mixed states, nearly for pure
- MLE enforces physicality through optimisation

Explain significance for practical tomography.

For physicality, for pure states, NN can struggle due to the small deviations at the Cholesky edge case mentioned prior, whereas MLE only tests physcial states so is fine by design.
---

### Runtime comparison

Compare runtime of reconstruction methods.

Measure runtime for:

- Stokes reconstruction
- MLE
- neural network inference

Explain difference between:

- training time
- inference time

Explain that neural networks require expensive training but extremely fast inference.

Explain that MLE requires optimisation for every reconstruction.

Discuss implications for real-time tomography applications.

Explain improvements obtained using physics-informed design. MLE suffers on time, Stokes on physicality, NN a good middle ground

---


## Three-qubit scaling experiment

Explain extension to three-qubit tomography.

Explain exponential growth of density matrix size.

Explain increased computational complexity.

Explain that only the physics-informed neural network was implemented for this case.

Explain results obtained.

Explain that experiment demonstrates feasibility of scaling approach.

---

# Limitations/ Further Work

## Advantages of neural network reconstruction

Discuss benefits:

- guaranteed physical states
- extremely fast inference
- potential scalability to larger systems

Explain why these advantages become more important as system size increases.

---

## Limitations of the current study

Discuss limitations:

TBD

---

## Future work

Discuss possible extensions.

Physics-informed architectures with improved inductive biases.

Application to experimental datasets.

Scaling to higher qubit systems.

Exploration of hybrid training strategies.

---

# Conclusion

Summarise aims of the project.

Restate central research question.

Summarise key results:

- neural networks achieve competitive fidelity
- Stokes reconstruction is fast but can produce unphysical states
- MLE guarantees physicality but is computationally expensive
- physics-informed neural networks provide strong balance between speed and physical validity

Conclude that incorporating physical knowledge into machine learning architectures is important for effective quantum state tomography.

Discuss implications for scalable tomography and future quantum experiments.
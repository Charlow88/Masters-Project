# Potential system to replicate

Two photon polarisation qubits from an SPDC source

Basics:

- Photons encode a qubit in their polarisation - Horizontal |H> or vertical |V>. forms 2 qubit system
- SPDC (spintaneous parametric down conversion) - creates entangled photon pairs
    - laser hits nonlinear crystal - occasionally a high-energy photon 'splits' into two photons with entangled polarizations
- each photon passes through waveplates and polarizers to measure in different bases, ie HH, HV etc. Can use these to recreate rho

NN would take the 36 measurements and predict rho

Noisy because optical components misaligned - would need to accurately simulate this

Could use ablation study - use a known effective process and see hwop removing individual parts affects the result. Learn what's most important via this method.

[James paper on qubit measurements and projectors](James_qubit_measurement_paper.pdf)
[Palmieri paper on SPDC specifically (can use to replicate maybe)](Palmieri_SPDC_paper.pdf)
# NuCrypt / Wang paper – raw notes

## experimental setup + how we get the nucrypt projectors

- all–fiber sagnac loop made of polarization-maintaining (PM) fiber  
- pumped by ~1554 nm mode-locked laser  
- nonlinear four-wave mixing in the fiber creates entangled photon pairs

- inside the loop:
  - clockwise pump → produces |V⟩|V⟩ pairs  
  - counter-clockwise pump → produces |H⟩|H⟩ pairs  
- these recombine at a PM fiber PBS → final state ≈ |HH⟩ + e^{iθ}|VV⟩

- after the loop, each photon goes to a polarization analyzer:
  - polarization controller (PC)
  - half-wave plate (HWP)
  - quarter-wave plate (QWP)
  - polarization beam splitter (PBS)
  - single-photon detectors on the two PBS outputs

- by choosing HWP/QWP angles, the analyzer measures in different bases:
  - H, V (PBS directly)
  - D, A (rotate by 22.5° using HWP)
  - R, L (QWP at ±45°)

- these correspond to single-qubit projectors:
  - h = |H⟩⟨H|
  - v = |V⟩⟨V|
  - d = |D⟩⟨D|
  - a = |A⟩⟨A|
  - r = |R⟩⟨R|
  - l = |L⟩⟨L|

- the full set of “nucrypt projectors” is all tensor products:
  - Pᵢⱼ = (one of {h,v,d,a,r,l}) ⊗ (one of {h,v,d,a,r,l})
  - total = 6 × 6 = 36 two-qubit projectors
- these are exactly the 36 projectors stored in the 6×6 matrix P in the simulation


## noise present in the measured photons

- **polarization drift / misalignment**  
  - PM fibers still slowly rotate polarization (temperature, stress, imperfect splices)  
  - waveplates (HWP/QWP) also not perfectly aligned  
  - modelled in the sim by applying random SU(2) rotations drawn from N(0, σ)

- **raman scattering**  
  - pump scatters inside fiber producing broadband noise photons  
  - some fall into signal/idler bands → accidental coincidences  
  - effectively: measurement probabilities get noisy

- **detector dark counts**  
  - SPADs click even without photons  
  - adds baseline noise to each measurement setting

- **pump fluctuations**  
  - pump power drifts → varying pair production → extra randomness in measured probabilities


## very brief: how the photons are produced

- pulsed pump laser enters a Sagnac loop of nonlinear PM fiber  
- four-wave mixing process:
  - two pump photons → signal + idler photons  
- pump travels both directions around the loop:
  - one direction creates |HH⟩  
  - the other creates |VV⟩  
- interference at the PBS produces the entangled superposition |HH⟩ + e^{iθ}|VV⟩


# NuCrypt / Wang paper – raw notes (emphasis on projectors + their relevance)

## experimental setup + why we get THESE EXACT nucrypt projectors

- system has a fiber-based entangled-photon source followed by **polarization analyzers** on each arm  
- the analyzers are what directly produce the **projectors** used in tomography  
- each analyzer consists of:
  - a **polarization controller (PC)** → rough alignment  
  - a **half-wave plate (HWP)** → rotates linear polarization axes  
  - a **quarter-wave plate (QWP)** → converts linear ↔ circular  
  - a **polarizing beam splitter (PBS)** → splits H vs V at the end  
  - detectors on the PBS outputs → measure probabilities for a given projector

### what this physically means
- the PBS only distinguishes H vs V  
- HWP + QWP rotate the incoming Bloch vector so that when the PBS measures “H”, it’s actually measuring:

  - **H** if no rotation  
  - **D** if the HWP is set to +22.5°  
  - **A** if the HWP is set to −22.5°  
  - **R** if QWP is +45°  
  - **L** if QWP is −45°  

- **this is how the *projectors are physically realised*** → by rotating the photon’s polarization just before the PBS

### list of the single-qubit projectors the system can measure
- h = |H⟩⟨H|
- v = |V⟩⟨V|
- d = |D⟩⟨D|
- a = |A⟩⟨A|
- r = |R⟩⟨R|
- l = |L⟩⟨L|

These correspond to **six specific waveplate/PBS settings**, i.e. the six cardinal points on the Bloch sphere the system can reach **reliably and reproducibly**.

### why these 6 matter (the “nucrypt projectors”)
- they are the **actual experimentally accessible polarization bases** in the Wang/Kanter setup  
- Lohani’s tomography paper *directly copies* these 6, because the goal is to mimic a real experimental tomography configuration  
- using all tensor products:


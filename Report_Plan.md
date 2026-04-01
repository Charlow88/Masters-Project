# Easter Report Writing Plan

## High-Level Steps

1. **Finish remaining coding**
   - Implement Maximum Likelihood Estimation (MLE) reconstruction.
   - Ensure Stokes reconstruction, neural networks, and scaling code all run cleanly.
   - Finalise noise models and dataset generation.
   Record mixed and pure states seperately, mention that more pure in short pulse, less otherwise, reference paper
   Reference Wang for numbers on visibility, when cooled can reach 98%, decreases to 90% at room temp, use 93% as default.
   Nasa paper for 0.08 degree misalignemnt
   - sweep over shots, as there's no set value, it explores how this affects results. Lohani goes 5-200, others say 100, up to 1000 is accepted. Just sweep and explain this. Make sure to explain how uncertainty scales with the number of shots 1 / root N

### Notes for what to do tomorrow:
   - Need to change how fit function operates in order to have a more meaningful loss curve, loss over epochs should be evaluated on a hold out not on the training set, so incorporate the test set into this fit function to evaluate on. 
   - Then, once everything is being stored correctly, can save best epoch and final epoch to check not overfitting and run CV for param optimisation

2. **Run final experiments**
   - Generate full datasets.
   - Run all reconstruction methods (Stokes, MLE, FC NN, CNN).
   - Explore hyperparameters where relevant.
   - Run scaling experiment to three qubits.

3. **Perform analysis**
   - Compute fidelities across all experiments.
   - Generate all plots required for the report.
   - Calculate statistical metrics (mean, variance, confidence intervals).
   - Measure runtime performance where relevant.

4. **Consolidate understanding of the physics**
   - Review the photonic polarisation tomography setup.
   - Understand the experimental apparatus (SPDC, waveplates, PBS, detectors).
   - Identify which states are physically common in SPDC photon sources.

5. **Prepare final figures**
   - Clean, consistent plots with labelled axes and clear captions.
   - Include error bars and number of states used in each experiment.
   - Ensure figures directly support the narrative of the report.

6. **Organise results and insights**
   - Identify the key comparisons between methods.
   - Clarify which results are central to the story.
   - Note limitations and interpretation of results.
   - INCLUDE T TESTS FOR HYPERPARAM OPTIMISED VALUES AND USE CONFIDENCE INTERVALS FOR THINGS

7. **Begin writing once results and understanding are finalised**
   - Write with a clear narrative focused on method comparison.
   - Ensure results, figures, and discussion align with the research question.


---

# Key Points to Remember During the Project

## Core Narrative
- The project is **a methodological exploration**, not an experimental photonics study.
- Focus on **comparing neural networks with classical tomography methods**.
- The physical system provides **context and motivation**, but is not the primary focus.

## Methods and Comparisons
- Include **Stokes reconstruction** as the classical baseline.
- Include **MLE reconstruction** as the standard classical optimisation method.
- Compare against **fully connected neural networks and CNNs**.
- Clearly justify **architecture choices and hyperparameters**.
Need to find papers explaining things such as dropout and justifying them as a regularisation method
Should showcase in report the the hyperparameters were tuned using a seperate validation set (50% pure mixed split) and then report shows test sets

## Physics Context
- Understand the **photonic polarisation tomography experiment**.
- Know the role of **waveplates, polarising beam splitters, and detectors**.
- Explain how measurement projectors arise from the physical setup.
- Justify which states are used (relevant to physics context?)

## Data and Noise
- Explain measurement noise sources.
- Mention **counting statistics scaling (1 / √N)**.
- Clearly state how noise and probabilities are simulated.
- State mix of mixed and pure states and why chosen.

## Statistical Rigor
- Always report:
  - Number of states used
  - Mean fidelity
  - Error bars / confidence intervals
- Be explicit about how statistics are calculated.

## Figures and Results
- Figures must be **clear, labelled, and interpretable**.
- Explain exactly where error bars come from.
- Avoid overly complicated plots.

## Scope Control
- Keep the scope tight to avoid unnecessary complexity.
- Do not overextend into unrelated applications.
- Focus on **method benchmarking and reconstruction performance**.

## Scaling Result
- Include **three-qubit scaling experiment**.
- Clearly state this is **a demonstration of scaling behaviour**, not part of the physical experiment.

## Writing and Narrative
- Maintain a clear motivation throughout the report.
- Anticipate possible questions from readers and examiners.
- Ensure the report concludes with a clear answer to the research question.

## References and Presentation
- Aim for ~20 references.
- Include a **diagram of the physical system**.
- Maintain a consistent scientific presentation style.
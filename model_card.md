# Model Card: GP-NN Hybrid Bayesian Optimisation Strategy

Version: Final (Module 25)

## Overview

- Name: GP-NN Hybrid Bayesian Optimisation Strategy
- Type: Sequential model-guided black-box optimisation
- Primary surrogate: Gaussian Process (scikit-learn GaussianProcessRegressor)
- Secondary surrogate: Neural network (PyTorch, gradient guidance)
- Acquisition functions: Expected Improvement (EI), Upper Confidence Bound (UCB)
- Core libraries: scikit-learn, PyTorch, SciPy, NumPy, Matplotlib
- Repository: github.com/natalialaurr/imperial-captstone-project-nw

## Intended Use

Suitable for:
- Sequential optimisation of expensive unknown scalar functions with continuous inputs in [0,1]
- Budget-constrained optimisation where function evaluations are limited
- Problems with no gradient information available (true black-box setting)
- Low-to-moderate dimensionality (2D-5D) where GP surrogates are well-calibrated

Avoid using for:
- Functions with discontinuities or non-stationary behaviour
- High-dimensional spaces (>6D) with very limited query budgets
- Real-time decision contexts requiring instant responses
- Stochastic functions where the same input may return different outputs

## Strategy Evolution Across 13 Rounds

Rounds 1-2 (Exploration): Space-filling heuristics with no surrogate. Alternating high/low patterns across dimensions to maximise domain coverage.

Rounds 3-4 (Surrogate Introduction): GP surrogate with RBF + WhiteKernel introduced. EI acquisition function selected queries formally. PyTorch neural network added in Round 4 to compute input gradients for higher-dimensional functions.

Rounds 5-7 (Exploitation Lean): Increasing exploitation for 2D/3D functions as GP confidence grew. UCB adopted for higher-dimensional functions. Per-function strategy differentiation formalised.

Rounds 8-10 (Confirmatory Exploitation): Precision targeting for 2D/3D. Gradient ascent for 4D/5D. Boundary-probing for 6D/8D where uncertainty remained high.

Rounds 11-12 (Cluster Convergence): Queries targeted centroid of highest-performing cluster per function. PCA-informed dimension prioritisation applied.

Round 13 (Final Confirmation): Final queries placed at or marginally beyond the current best-observed point for each function.

## Technical Specifications

Gaussian Process Surrogate:
- Kernel: RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
- Hyperparameter optimisation: log marginal likelihood, n_restarts_optimizer=5
- Acquisition: Expected Improvement, xi = 0.005 to 0.05 (function-specific)
- EI optimisation: SciPy differential_evolution over [0,1]^n

Neural Network Surrogate:
- Architecture: 2 hidden layers, 32 neurons each, ReLU activations
- Training: Adam optimiser, lr=0.01, full-batch
- Gradient use: torch.autograd computes dy/dx for gradient ascent query selection
- Role: Directional guidance for high-dimensional functions

## Performance

Primary metric: cumulative maximum observed output (best y) per function across all 13 rounds.

| Function | Dimensions | Round 1 Best y | Round 7 Best y | Round 13 Best y |
|----------|-----------|---------------|---------------|----------------|
| F1 | 2D | - | - | - |
| F2 | 2D | - | - | - |
| F3 | 3D | - | - | - |
| F4 | 4D | - | - | - |
| F5 | 4D | - | - | - |
| F6 | 5D | - | - | - |
| F7 | 6D | - | - | - |
| F8 | 8D | - | - | - |

Key findings:
- 2D and 3D functions showed consistent improvement throughout all 13 rounds
- Neural network gradient guidance improved query efficiency for 4D/5D functions from Round 4
- 6D and 8D functions showed the greatest residual uncertainty at Round 13
- The query budget was insufficient to fully characterise high-dimensional spaces

## Assumptions and Limitations

- Stationarity: RBF kernel assumes uniform function variation. Violated by sharp localised peaks.
- Determinism: assumes same input always returns same output. Noise would invalidate exploitation decisions.
- Continuity: GP interpolation assumes smooth functions. Discontinuous functions poorly approximated.
- Curse of dimensionality: GP reliability degrades significantly for 6D/8D with only 13 observations.
- Selection bias: dataset biased toward early-identified high-output regions.
- No ground truth: surrogate accuracy unmeasurable without function definition.

## Ethical Considerations

All modelling decisions are documented in this repository alongside implementing code. A reviewer with access to the query-response log can reproduce every query decision made across 13 rounds. This level of transparency reflects the principle that interpretability is a responsibility, not just a technical property. Before applying this strategy to real-world high-stakes problems, practitioners should validate the stationarity and determinism assumptions for the specific function being optimised.

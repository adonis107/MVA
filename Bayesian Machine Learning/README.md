# NUTS from Scratch — No-U-Turn Sampler Implementation & Review

A from-scratch Python implementation of the **No-U-Turn Sampler (NUTS)** and **Hamiltonian Monte Carlo (HMC)**, reproducing and extending the results in the article "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo" (Hoffman & Gelman, 2014). Built by Adonis Jamal and Jean-Vincent Martini as a course project for the "Bayesian Machine Learning" course of MVA master's degree (ENS Paris-Saclay).

## Overview

This project implements three NUTS variants and an HMC baseline, validates them on the four benchmarks from the original paper (Multivariate Normal, Logistic Regression, Hierarchical Logistic Regression, Stochastic Volatility), and pushes farther with three new stress tests on pathological posteriors (multimodal BNN, dense-correlation GPC, Neal's funnel).

### Key results

- **NUTS matches or exceeds the best hand-tuned HMC** on all benchmarks without requiring trajectory-length tuning.
- New stress tests expose failure modes of HMC that NUTS handles gracefully: multimodality, dense correlations, and extreme geometric variation.

## Features

| Component | Description |
|-----------|-------------|
| **Samplers** | `DualAveragingHMC`, `NaiveNUTS`, `EfficientNUTS`, `DualAveragingNUTS` |
| **Target distributions** | Multivariate Normal (250-D), Logistic Regression (25-D), Hierarchical LR (302-D), Stochastic Volatility (3001-D), BNN (191-D), GP Classification (150-D), Neal's Funnel (10-D) |
| **Metrics** | Autocorrelation, Effective Sample Size (ESS), ESS per gradient evaluation |
| **Deliverables** | Full review paper (NeurIPS format) and Beamer presentation |

## Project Structure

```
src/bml/
├── samplers/          # HMC & NUTS implementations
│   ├── hmc.py         # HMC with dual-averaging step-size adaptation
│   ├── nuts.py        # Naive, Efficient, and Dual-Averaging NUTS
│   └── utils.py       # Leapfrog integrator, step-size heuristic
├── distributions/     # Target posteriors (log-density + gradient)
│   ├── mvn.py         # Multivariate Normal (250-D)
│   ├── lr.py          # Bayesian Logistic Regression
│   ├── hlr.py         # Hierarchical Logistic Regression
│   ├── sv.py          # Stochastic Volatility (S&P 500)
│   ├── bnn.py         # Bayesian Neural Network
│   ├── gpc.py         # Gaussian Process Classification
│   ├── funnel.py      # Neal's Funnel
│   └── counter.py     # Gradient-call counter wrapper
└── metrics.py         # ESS and autocorrelation computation

notebooks/             # Experiments and analysis (see below)
deliverables/          # LaTeX report (NeurIPS format) and slides
results/               # Saved CSV outputs (ESS/grad per model & delta)
```

## Getting Started

### Requirements

- Python ≥ 3.10

### Installation

```bash
# Clone the repository
git clone https://github.com/adonis107/MVA-BML.git
cd MVA-BML

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package and dependencies
pip install -r requirements.txt
pip install -e .
```

Alternatively, you can install the required dependencies with **uv** using the `uv sync` command in your terminal.

### Quick Example

```python
import numpy as np
from bml.distributions.mvn import MultivariateNormal
from bml.samplers.nuts import DualAveragingNUTS

# Set up a 10-D Gaussian target
target = MultivariateNormal(d=10)

# Initialize the sampler
sampler = DualAveragingNUTS(L=target.log_p, grad=target.grad_log_p)

# Draw 2000 samples (1000 warmup)
theta0 = np.zeros(10)
samples, stats = sampler.sample(theta0, delta=0.65, M=2000, M_adapt=1000)
```

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1a | [Multivariate Normal](notebooks/1a.%20Multivariate%20Normal.ipynb) | Benchmark on 250-D Gaussian; ESS/grad comparison |
| 1b | [Bayesian Logistic Regression](notebooks/1b.%20Bayesian%20Logistic%20Regression.ipynb) | German Credit dataset (25-D) |
| 1c | [Hierarchical Logistic Regression](notebooks/1c.%20Hierarchical%20Logistic%20Regression.ipynb) | Interaction terms + hyperpriors (302-D) |
| 1d | [Stochastic Volatility](notebooks/1d.%20Stochastic%20Volatility.ipynb) | S&P 500 returns (3001-D) |
| 2 | [Discrepancies](notebooks/2.%20Discrepancies.ipynb) | Dual-averaging convergence analysis |
| 3 | [Multimodal Stress Test](notebooks/3.%20Multimodal%20Stress%20Test.ipynb) | BNN with 14 400 equivalent posterior modes |
| 4 | [Dense Correlation Test](notebooks/4.%20Dense%20Correlation%20Test.ipynb) | GP classification with fully dense kernel |
| 5 | [Neal's Funnel](notebooks/5.%20Neal's%20Funnel.ipynb) | Pathological funnel geometry |

## Authors

- **Adonis Jamal** — MVA, ENS Paris-Saclay
- **Jean-Vincent Martini** — MVA, ENS Paris-Saclay


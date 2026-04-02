# Neural Network Models of 2D Conformal Field Theory

**Authors**: Brandon Robinson

**Institution**: University of Amsterdam

**Date**: April 2026

## Overview

This repository contains neural network implementations of 2D conformal field theories (CFTs), centered on the **free boson** (c=1) and its supersymmetric extensions. The core innovation is a **variance-reduced simulation framework** that replaces stochastic angular sampling with deterministic Bessel function evaluation, yielding high-precision verification of CFT correlators, operator algebras, and boundary conditions.

## Key Results

### Summary of Observables

| Observable | Theory Target | Error / Accuracy | Method |
|------------|--------------|------------------|--------|
| Vertex operators `⟨V_α V_-α⟩` | `r^(-α²)` | **0.03–3.4%** | Structure function (original) |
| Stress tensor `⟨TT⟩ = B²/128` | `c/(2r⁴)`, c=1 |**99.6%** | Deterministic Bessel J₂ |
| Super-Virasoro fermion `⟨ψψ⟩` | `1/r` | **99.5%**  | Deterministic Bessel J₁ |
| Super-Virasoro boson `⟨∂φ∂φ⟩` | `1/r²` | **~95.1%**  | Deterministic Bessel J₂ |
| Supercurrent `⟨GG⟩` (Wick) | `1/r³` | **~96.5%**  | Wick factorization |
| Boundary boson (Neumann) | `-ln(2y)` slope | **99.8%**  | Method of images |
| Boundary fermion (Reflection) | `1/r` | **99.5**  | Deterministic Bessel J₁ |
| N=1 scalar multiplet (boundary) | `1/r`, `1/r²`, `1/r³` | **>96%*  | Method of images + Wick |


### Variance Reduction Technique

The key methodological advance replaces stochastic angular integration (4-clock sampling) with **analytically exact Bessel function evaluation**:

- **Fermion sector**: `∫₀²π dθ/(2π) e^{-iθ} sin(kr cosθ) = J₁(kr)` — eliminates angular noise entirely
- **Boson sector**: `∫₀²π dθ/(2π) e^{-2iθ} cos(kr cosθ) = -J₂(kr)` — eliminates angular noise entirely
- **Stratified k-sampling**: log-uniform stratification further reduces radial variance

## Repository Structure

```
.
├── README.md                                    # This file
├── INSTALLATION.md                              # Setup instructions
├── requirements.txt                             # Python dependencies
├── experiments/
│   ├── README.md                                # Detailed experiment documentation
│   ├── variance_reduced_correlators.ipynb        # ★ All variance-reduced simulations
│   ├── vertex_operator.py                       # Vertex operators (original, 0.03–3.4%)
│   ├── free_boson.py                            # Current correlators (original, ~30%)
│   ├── vertex_precision.png                     # Vertex operator result figure
│   └── current_verify_OPTIMIZED.png             # Current correlator result figure
├── figures/
│   ├── vertex_precision.pdf                     # Vertex operator (publication)
│   ├── super_virasoro_variance_reduced.pdf      # ★ Super-Virasoro verification
│   ├── stress_tensor_TT_variance_reduced.pdf    # ★ Stress tensor ⟨TT⟩ / central charge
│   ├── super_boundary_multiplet_variance_reduced.pdf  # ★ N=1 boundary multiplet
│   ├── separate_boundary_tests_variance_reduced_1.pdf # ★ Boundary condition tests
│   └── gaussianity_check.pdf                    # Field Gaussianity verification
```

## Quick Start

### Installation
```bash
# Clone repository
git clone [repository-url]
cd NN-2D-CFT

# Install dependencies
pip install numpy matplotlib scipy tqdm

# Optional: for Jupyter notebooks
pip install jupyter
```

### Run the Variance-Reduced Simulations
```bash
cd experiments
jupyter notebook variance_reduced_correlators.ipynb
```

Each cell in the notebook is a self-contained simulation:
1. **N=1 Scalar Multiplet with Boundary** — boundary fermion, boson, and supercurrent
2. **Boundary Conditions via Method of Images** — Neumann boson + fermionic reflection
3. **Super-Virasoro Algebra** — bulk fermion (J₁), boson (J₂), and supercurrent (Wick)
4. **Stress Tensor OPE** — central charge extraction via ⟨TT⟩ = B²/128

### Run the Original Scripts
```bash
python vertex_operator.py     # Vertex operators (0.03–3.4% error)
python free_boson.py           # Current correlators (~30% error)
```

## References

- **Di Francesco, Mathieu, Sénéchal** — *Conformal Field Theory* (vertex operators, stress tensor)
- **Polchinski** — *String Theory Vol. 1* (boson CFT, normal ordering)
- **Ginsparg** — *Applied Conformal Field Theory* (structure functions, Ward identities)

---

**Last Updated**: April 2026
**Python Version**: 3.8+
**Dependencies**: NumPy, Matplotlib, SciPy, tqdm

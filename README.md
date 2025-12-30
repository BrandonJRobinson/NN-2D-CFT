# Neural Network Models of 2D Conformal Field Theory

**Authors**: Brandon Robinson

**Institution**: University of Amsterdam

**Date**: December 2025

## Overview

This repository contains neural network implementations of 2D conformal field theories (CFTs), specifically:
- **Free Boson** (c=1): Direct CFT verification via vertex operators (0.03-3.4% error) ★

## Key Results

### Summary of Models

| Model | Central Charge | Observable | Best Error|
|-------|---------------|------------|------------|
| **Free Boson** ★ | c = 1 | Vertex operators `<V_α V_-α>` | **0.03-3.4%** |
| **Free Boson** | c = 1 | Current correlator `<J*J>` | ~30% |

### Free Boson: Direct c=1 Verification (★ Best Precision)
- **Method**: Vertex operator correlators `<V_α(0) V_-α(r)> ~ r^(-α²)`
- **Accuracy**: 0.03-3.4% error across α = 0.5, 1.0, 1.5, 2.0


## Repository Structure

```
.
├── README.md                    # This file
├── INSTALLATION.md              # Setup instructions
├── free_boson_model/            # ★ Free boson c=1 (best precision)
│   ├── vertex_operator.py               # Vertex operators (0.03-3.4% error)
│   ├── chiral_boson_redux.py            # Current correlators (~30% error)
│   ├── vertex_precision.png             # Main result figure
```

## Quick Start

### Installation
```bash
# Clone repository
git clone [repository-url]
cd virasoro-neural-networks

# Install dependencies
pip install numpy matplotlib scipy tqdm

# Optional: for Jupyter notebooks
pip install jupyter
```

**Last Updated**: December 2025  
**Python Version**: 3.8+  
**Dependencies**: NumPy, Matplotlib, SciPy, tqdm

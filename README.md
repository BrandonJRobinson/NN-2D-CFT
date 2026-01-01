# Neural Network Models of 2D Conformal Field Theory

**Authors**: Brandon Robinson

**Institution**: University of Amsterdam

**Date**: December 2025

## Overview

This repository contains neural network implementations of 2D conformal field theories (CFTs), specifically:
- **Free Boson** (c=1): Direct CFT verification via vertex operators (0.03-3.4% error) ★

## Repository Structure

```
.
├── README.md                    # This file
├── INSTALLATION.md              # Setup instructions
├── free_boson_model/            # ★ Free boson c=1 (best precision)
│   ├── vertex_operator.py               # Vertex operators (0.03-3.4% error)
│   ├── free_boson.py            # Current correlators (~30% error)
│   ├── vertex_precision.png             # Main result figure
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

**Last Updated**: December 2025  
**Python Version**: 3.8+  
**Dependencies**: NumPy, Matplotlib, SciPy, tqdm

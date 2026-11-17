# Installation Guide

## System Requirements

- **Python**: 3.8 or higher
- **OS**: macOS, Linux, or Windows
- **RAM**: 8GB minimum (16GB recommended for large runs)
- **Disk**: ~500MB for code and results

## Quick Installation

### Option 1: pip (Recommended)

```bash
# Clone repository
git clone [repository-url]
cd virasoro-neural-networks

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: conda

```bash
# Create conda environment
conda create -n virasoro python=3.9
conda activate virasoro

# Install dependencies
conda install numpy matplotlib scipy tqdm
pip install -r requirements.txt  # For any pip-only packages
```

## Dependencies

### Core Requirements
```
numpy>=1.20.0
matplotlib>=3.3.0
scipy>=1.6.0
tqdm>=4.60.0
```

### Optional (for development)
```
jupyter>=1.0.0
pytest>=6.2.0
black>=21.0
```

## Verification

Test your installation:

```bash
cd xy_model
python -c "from core.neural_xy_model import NeuralXYModel; print('✓ Import successful')"
python xy_quick_check.py
```

Expected output:
```
✓ Import successful

Running XY Model Quick Check...
β=0.5: η = 0.245 ± 0.002 (theory: 0.250), error: 2.0%
β=1.0: η = 0.985 ± 0.008 (theory: 1.000), error: 1.5%
✓ Quick check passed!
```

## Platform-Specific Notes

### macOS
```bash
# If you encounter issues with matplotlib
pip install --upgrade matplotlib
# If using M1/M2 chip, use conda for better compatibility
conda install numpy scipy matplotlib
```

### Linux
```bash
# May need build tools for some packages
sudo apt-get install python3-dev build-essential
pip install -r requirements.txt
```

### Windows
```bash
# Use Anaconda for easiest setup
# Download from: https://www.anaconda.com/products/distribution
conda create -n virasoro python=3.9
conda activate virasoro
conda install numpy matplotlib scipy tqdm
```

## Performance Tuning

### For Large Runs (100k+ networks)
```bash
# Use multiple cores if available
export OMP_NUM_THREADS=4  # Adjust to your CPU count
```

### Memory Management
For systems with limited RAM, reduce batch sizes in scripts:
```python
# In xy_high_stats.py or xy_adaptive_scheduler.py
batch_size = 500  # Reduce from 1000 if needed
```

## Troubleshooting

### Import Errors
```bash
# Ensure you're in the correct directory
cd /path/to/virasoro-neural-networks/xy_model
python -c "import sys; print(sys.path)"
```

### Slow Performance
```bash
# Check NumPy is using optimized BLAS
python -c "import numpy as np; np.show_config()"
# Should show BLAS/LAPACK libraries (OpenBLAS, MKL, etc.)
```

### Plot Display Issues
```bash
# For headless systems, use Agg backend
export MPLBACKEND=Agg
# Or in Python:
import matplotlib
matplotlib.use('Agg')
```

## Development Setup

For contributors:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black xy_model/
```

## GPU Support (Future)

Currently, the code uses NumPy (CPU). For future GPU acceleration:

```bash
# Install CuPy (CUDA required)
pip install cupy-cuda11x  # Replace 11x with your CUDA version
```

## Next Steps

After installation:
1. Read [README.md](README.md) for overview
2. Run `python xy_quick_check.py` for validation
3. See [docs/METHODS.md](docs/METHODS.md) for detailed methodology
4. Explore example notebooks (if available)

## Getting Help

- **Issues**: Open a GitHub issue
- **Questions**: See [README.md](README.md#contact)
- **Documentation**: See [docs/](docs/) directory

## Version History

- **v1.0.0** (Dec 2025): Initial release
  - XY model with adaptive scheduler
  - β ≤ 1.4 validated results
  - Majorana reference implementation

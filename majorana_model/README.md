# Majorana Fermion CFT (c=1/2)

## Overview

This directory contains reference implementations of the Majorana fermion CFT model. These files are included for **comparison and documentation purposes**, showing an alternative approach that was ultimately superseded by the XY model.

## Key Finding

**The Majorana model exhibits ~25% systematic variance** in correlation measurements, compared to ~3% for the XY model. This 8× difference in stability makes the XY model the **recommended approach** for neural CFT studies.

## Files

### `majorana_theory_derivation_proper.py`
- **Purpose**: Theoretical framework for Majorana fermion correlations
- **Content**: Proper normalization derivation, c=1/2 CFT theory
- **Status**: Validated theoretical predictions

### `majorana_cft_verification.py`
- **Purpose**: Numerical validation of Majorana correlations
- **Content**: Monte Carlo implementation, exponent extraction
- **Result**: η ≈ 0.5 measured (theory: 0.5), but with 25% variance

## Why Majorana Shows Higher Variance

### Technical Reasons
1. **Fermionic vs Bosonic**: Fermion correlations inherently noisier
2. **Finite Network Effects**: More pronounced for fermions
3. **Normalization Challenges**: Complex field requires more careful treatment

### Observed Behavior
- **Systematic r-dependence** in sim/CFT ratio
- **Not fixable** with more networks (tested up to 1M)
- **Architectural limitation** of the finite momentum mode representation

## Comparison Summary

| Feature | Majorana (c=1/2) | XY Model (c=1) |
|---------|-----------------|----------------|
| Scaling exponent | η = 1/2 (fixed) | η = β² (tunable) |
| Error | ~25% variance | ~3% error |
| Stability | Poor | Excellent |
| Convergence | Systematic issues | Clean power-law |
| Recommendation | Reference only | **Use this!** |

## Usage

These files are provided for:
1. **Historical context**: Shows development process
2. **Comparison studies**: Understand why XY model is superior
3. **Alternative approaches**: May inspire future improvements

**For actual research**: Use the XY model in `../xy_model/`

## Lessons Learned

### What Worked
- ✓ Theoretical framework correct
- ✓ Basic CFT implementation functional
- ✓ Identified proper c=1/2 normalization

### What Didn't Work
- ✗ High variance (25% vs 3% for XY)
- ✗ Systematic r-dependence in ratios
- ✗ Not improvable with more statistics

### Key Insight
**Bosonic fields (XY model) are much better suited** to the neural network momentum-mode representation than fermionic fields.

## Citation

If you use these implementations as reference:

```bibtex
@misc{robinson2025majorana,
  title={Majorana Fermion CFT: Reference Implementation},
  author={Robinson, Brandon},
  year={2025},
  note={Comparison study - see XY model for main results}
}
```

## Future Directions

To improve Majorana implementation:
1. **Different field representation**: Maybe Dirac rather than Majorana
2. **Operator product methods**: Direct OPE computation
3. **Lattice-based**: Connect to Ising model on lattice

But given XY model's success, these are **low priority**.

---

**Status**: Reference implementation (archived)  
**Recommendation**: Use XY model for research  
**Last Updated**: December 2025

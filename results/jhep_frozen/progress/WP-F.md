# WP-F progress report — reference current-kernel window

## Command

```bash
python scripts/run_current_kernel_reference.py
```

## Metric definition (emitted in every row)

`amplitude_fit A` minimises `||K_JJ(r) - A r^-2||_2` over the recorded
evaluation interval on the real axis;
`relative_error = ||K_JJ - A r^-2||_2 / ||A r^-2||_2` is the scale-free shape
error; `amplitude_relative_error = |A - 1|` measures the `kappa = 1`
calibration. The interval is strictly inside eq. (2.5) with the margins
recorded per row. **This value is never compared with the broad robustness
grid.**

## Analytic closed-kernel reference rows

| regulator_class          |   lambda_ir |   lambda_uv |   r_min_eval |   r_max_eval |   amplitude_fit |   relative_error |   window_margin_uv |   window_margin_ir |
|:-------------------------|------------:|------------:|-------------:|-------------:|----------------:|-----------------:|-------------------:|-------------------:|
| sharp_annulus_manuscript |      0.0001 |       10000 |        0.003 |      333.3   |          1.686  |          1.142   |                 30 |                 30 |
| sharp_annulus_manuscript |      0.001  |        1000 |        0.03  |       33.33  |          1.332  |          1.323   |                 30 |                 30 |
| sharp_annulus_manuscript |      0.01   |         100 |        0.3   |        3.333 |          1.26   |          1.434   |                 30 |                 30 |
| tapered_log_window       |      0.0001 |       10000 |        0.003 |      333.3   |          1.071  |          0.02178 |                 30 |                 30 |
| tapered_log_window       |      0.001  |        1000 |        0.03  |       33.33  |          0.9904 |          0.02134 |                 30 |                 30 |
| tapered_log_window       |      0.01   |         100 |        0.3   |        3.333 |          1.02   |          0.02235 |                 30 |                 30 |

## The campaign's central regulator finding

The scalar covariance eq. (A.1) converges cleanly. The current kernel is its
*second* derivative, and the sharp annular support of eq. (2.2) contributes an
endpoint term `b J_1(b)` with `b = Lambda_UV r` whose amplitude grows like
`b^(1/2)`. Therefore:

* `K^Lambda_JJ` has **no conformal limit** under eq. (2.2) — widening the
  annulus makes the deviation from `(z-w)^-2` *larger*. Asserted in
  `tests/test_kernels.py::test_current_kernel_has_no_sharp_cutoff_conformal_limit`.
* the shape error is 114–143% anywhere in the nominal physical window;
* the same endpoint term makes `d_zbar K_JJ` of eq. (3.15) grow like `b^(3/2)`,
  so Proposition 3.4 already predicts the failure quantitatively;
* a 15% raised-cosine taper in `log k` — the minimal smooth modification of the
  same `|k|^-2` prior — reduces the shape error to 2.1–2.3% with amplitude
  `A = 0.99–1.07`, and reduces `|d_zbar K_JJ|` by more than four orders of
  magnitude at fixed cutoffs.

Verified three independent ways: closed-form Bessel derivatives against
numerical second differentiation of the scalar kernel (7 digits); direct Monte
Carlo of the finite-feature field; and the tapered spectral integrator against
the sharp closed form at `taper = 0`.

**The architecture claim survives; the specific regulator written in eq. (2.2)
does not.**

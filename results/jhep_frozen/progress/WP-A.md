# WP-A progress report — operational two-radius pairing

## Commands run

```bash
python scripts/run_ordered_pairing_audit.py --stages a4
python scripts/run_ordered_pairing_audit.py --stages a5,a3,a6 --skip-existing
```

Every stage was run under **both** regulator classes
(`sharp_annulus_manuscript`, taper 0.00 = eq. (2.2) exactly as written; and
`tapered_log_window`, taper 0.15, a labelled diagnostic).

## A1/A2 — implementation and its verification

Three independent implementations of the same functional were cross-validated
before any campaign row was produced (see `tests/test_ordered_pairing.py`):

| check | result |
|---|---|
| Prop. 3.1 on the ideal kernel: `I^>`, `I^<`, `omega` separately | machine precision (1e-12) |
| Prop. 3.2 local eps-residue equivalence, ideal kernel | machine precision |
| exact Bessel/Jacobi–Anger path vs general torus quadrature (q>=256) | 4e-14 relative |
| same, for the non-translation-invariant `phase_correlated_features` control | 2e-14 relative |
| Prop. 3.2 **fails** for the smooth regulated kernel (as it must) | asserted |
| analytic closed-kernel pairing is exactly antidiagonal | asserted (leakage == 0) |
| reported `R_0`-unit pairing is independent of `R_0` | 1e-10 |
| Prop. 3.4 dbar formula vs centred complex finite difference | 2e-5 |
| Prop. 3.6 bound actually dominates the measured radius variation | asserted |

**Declared convention.** The released pairing is the dimensionless `R_0`-unit
pairing `omega~_mn = R_0^-(m+n) omega_mn`, i.e. the pairing of `zeta = z/R_0`.
For the ideal kernel this equals `m delta_(m+n,0)` identically in `R_0` and
`rho`, and it keeps the diagonal rescaling compatible with the involution `C`.
Without it the unweighted Frobenius metrics of eq. (3.24) would depend on the
arbitrary overall contour scale.

## A3 — radius plateau and the Cauchy–Green bound

| regulator_class          |   lambda_uv |   rho |   plateau_nodes |   r0_lo |   r0_hi |   median_bound_ratio |   max_bound_ratio | all_compatible   |
|:-------------------------|------------:|------:|----------------:|--------:|--------:|---------------------:|------------------:|:-----------------|
| sharp_annulus_manuscript |         100 |  1.25 |              10 | 0.3741  |   2.673 |            8.717e-05 |         0.0003838 | True             |
| sharp_annulus_manuscript |         100 |  1.5  |              10 | 0.4061  |   2.462 |            6.261e-05 |         0.0007885 | True             |
| sharp_annulus_manuscript |         100 |  2    |              10 | 0.4622  |   2.163 |            0.0001633 |         0.0003376 | True             |
| sharp_annulus_manuscript |         100 |  3    |              10 | 0.5548  |   1.803 |            6.429e-05 |         0.0005066 | True             |
| sharp_annulus_manuscript |        1000 |  1.25 |              10 | 0.0471  |  21.23  |            3.939e-07 |         1.314e-05 | True             |
| sharp_annulus_manuscript |        1000 |  1.5  |              10 | 0.05113 |  19.56  |            6.591e-07 |         1.658e-05 | True             |
| sharp_annulus_manuscript |        1000 |  2    |              10 | 0.05819 |  17.18  |            1.056e-06 |         3.449e-05 | True             |
| sharp_annulus_manuscript |        1000 |  3    |              10 | 0.06984 |  14.32  |            1.604e-06 |         1.16e-05  | True             |
| tapered_log_window       |         100 |  1.25 |              10 | 0.3741  |   2.673 |            8.71e-06  |         0.00027   | True             |
| tapered_log_window       |         100 |  1.5  |              10 | 0.4061  |   2.462 |            9.999e-06 |         3.923e-05 | True             |
| tapered_log_window       |         100 |  2    |              10 | 0.4622  |   2.163 |            4.255e-05 |         0.0001127 | True             |
| tapered_log_window       |         100 |  3    |              10 | 0.5548  |   1.803 |            6.917e-05 |         0.0001761 | True             |
| tapered_log_window       |        1000 |  1.25 |              10 | 0.0471  |  21.23  |            2.494e-07 |         1.991e-06 | True             |
| tapered_log_window       |        1000 |  1.5  |              10 | 0.05113 |  19.56  |            5.935e-07 |         1.601e-06 | True             |
| tapered_log_window       |        1000 |  2    |              10 | 0.05819 |  17.18  |            3.693e-07 |         1.909e-06 | True             |
| tapered_log_window       |        1000 |  3    |              10 | 0.06984 |  14.32  |            3.835e-07 |         5.387e-06 | True             |

The stated plateau algorithm (adjacent `R_0` nodes compatible when
`|dY| <= 3*sqrt(se_i^2+se_j^2) + CauchyGreen(i,j)`; plateau = longest connected
run) finds a plateau spanning **all ten** `R_0` nodes in every configuration,
and the measured radius variation is compatible with the Prop. 3.6 bound in
every edge. The bound is satisfied by 3–7 orders of magnitude, i.e. it holds
comfortably but is far from tight — that looseness is reported, not hidden,
because it follows from the very large `dbar` norm of the sharp cutoff.

## A4 — recomputation of the 81 released configurations

`robustness_grid_recomputed.csv` has 162 rows (81 per regulator
class). The comparison columns `projection_pairing_leakage`,
`two_radius_pairing_leakage`, `projection_vs_two_radius_relative_error`,
`projection_raw_virasoro_residual` and `two_radius_raw_virasoro_residual` are
retained side by side.

**Structural finding.** The single-radius projection covariance
`C^proj_mn(R) = E_gamma[a_m a_n]` equals the ordered functional at *coincident*
radii and is therefore exactly **symmetric**: its antisymmetrisation vanishes
identically. It is a covariance-level proxy, not a commutator, and the frozen
suite never identifies it with the operational two-radius pairing. The legacy
generating script is absent from the bundle, so no claim is made that the
proxy's numerical value matches the legacy one.

## A5 — multi-seed width/cutoff convergence (1360 seed rows)

Model: `log Y = c + b log N + a log Lambda + regulator fixed effects`, ten
independent named seeds, `N` in {2000..300000}, `Lambda` in {4,8,12,16},
three regulator pairs. Adaptive design: the top width was run on the two widest
regulator pairs only; every executed row is retained.

| regulator_class          | metric                 |   n_features_exponent |   n_features_stderr |   mode_cutoff_exponent |   r_squared |   n_rows |   excluded_rows |   null_p_value | null_rejected_at_5pct   |
|:-------------------------|:-----------------------|----------------------:|--------------------:|-----------------------:|------------:|---------:|----------------:|---------------:|:------------------------|
| sharp_annulus_manuscript | pairing_leakage        |              -0.03731 |             0.01911 |                 0.1598 |     0.04552 |      680 |               0 |      1.113e-93 | True                    |
| sharp_annulus_manuscript | pairing_relative_error |              -0.4565  |             0.01637 |                 4.489  |     0.9505  |      680 |               0 |      0.008031  | True                    |
| sharp_annulus_manuscript | pairing_sampling_error |              -0.4938  |             0.01613 |                 4.527  |     0.9543  |      680 |               0 |      0.7021    | False                   |
| sharp_annulus_manuscript | raw_virasoro_residual  |              -0.5702  |             0.01351 |                 3.117  |     0.9599  |      680 |               0 |      2.674e-07 | True                    |
| tapered_log_window       | pairing_leakage        |              -0.06701 |             0.01755 |                 0.4754 |     0.1373  |      680 |               0 |      2.913e-96 | True                    |
| tapered_log_window       | pairing_relative_error |              -0.491   |             0.01573 |                 4.508  |     0.9531  |      680 |               0 |      0.5672    | False                   |
| tapered_log_window       | pairing_sampling_error |              -0.4911  |             0.01573 |                 4.509  |     0.9532  |      680 |               0 |      0.5702    | False                   |
| tapered_log_window       | raw_virasoro_residual  |              -0.5389  |             0.01078 |                 3.408  |     0.9716  |      680 |               0 |      0.0003331 | True                    |

* `pairing_sampling_error` (the finite-`N` error about the analytic
  closed-kernel pairing) has `b = -0.494 +- 0.016` (sharp) and
  `-0.491 +- 0.016` (tapered). The null `b = -1/2` is **not rejected**
  (p = 0.70, 0.57). This is the clean confirmation of the `N^-1/2`
  random-feature rate.
* `pairing_relative_error` reaches `b = -0.491` (null not rejected) under the
  tapered class but only `-0.456` (null rejected) under the sharp class,
  because the sharp-cutoff systematic is a floor that the sampling error
  descends onto.
* `pairing_leakage` follows **no** clean power law under the operational
  definition (R^2 = 0.05 / 0.14). This is expected and is a genuine correction
  to the legacy picture: the analytic leakage of any rotationally invariant
  prior is identically zero, so measured leakage is pure angular sampling noise
  whose ratio to a noisy denominator is not a power law. The legacy
  `N^-0.510` refers to the single-radius projection proxy, a different object.
* The mode-cutoff exponent is `a ~ 4.5` (pairing) and `~3.1-3.4` (residual),
  much larger than the legacy `0.54 / 1.67`, because the operational pairing
  amplifies entrywise like `rho^|m|` — exactly the behaviour Prop. 3.5 bounds.

## A6 — architecture controls and physical admissibility

All eight controls, six seeds, two widths, two regulator classes
(192 seed rows; 192 admissibility rows). At the largest width:

| regulator_class          | architecture                    |   leakage |    raw_vir |   ward_T |   post_darboux |   star |   pol_min | vac   |
|:-------------------------|:--------------------------------|----------:|-----------:|---------:|---------------:|-------:|----------:|:------|
| sharp_annulus_manuscript | angular_mode_scramble           |   1.414   | 39.57      |  22.13   |      1.764e-13 |  1.448 |  -204.8   | False |
| sharp_annulus_manuscript | anisotropic_angular_prior       |   1.253   | 34.78      |   5.279  |      2.32e-13  |  1.468 |  -213.8   | False |
| sharp_annulus_manuscript | local_exponent_direction        |   1.223   | 51.68      |   9.871  |      3.576e-13 |  1.472 |  -302.6   | False |
| sharp_annulus_manuscript | non_symplectic_dense_mixing     |   3.666   |  5.934e+05 |   6.272  |      4.316e-12 |  1.417 | -2566     | False |
| sharp_annulus_manuscript | phase_correlated_features       |   1.085   | 42.54      |   8.498  |      2.814e-13 |  1.423 |  -171.5   | False |
| sharp_annulus_manuscript | rbf_gaussian_prior              |   0.00532 |  2.877     |   2.049  |      8.804e-16 |  1.414 |    -6.592 | False |
| sharp_annulus_manuscript | selected_log_kernel             |   0.9522  | 34.62      |   6.726  |      1.646e-13 |  1.453 |  -199.3   | False |
| sharp_annulus_manuscript | sign_breaking_mode_reassignment | inf       | 30.82      |   4.799  |      1.671e-13 |  1.414 |  -168.2   | False |
| tapered_log_window       | angular_mode_scramble           |   1.494   | 19.43      |  23.61   |      8.393e-14 |  1.402 |  -103.7   | False |
| tapered_log_window       | anisotropic_angular_prior       |   1.875   | 15.07      |   1.111  |      8.965e-14 |  1.253 |   -60.31  | False |
| tapered_log_window       | local_exponent_direction        |   1.223   | 43.9       |   7.133  |      2.823e-13 |  1.468 |  -257.2   | False |
| tapered_log_window       | non_symplectic_dense_mixing     |   3.828   |  2.491e+04 |   0.6837 |      1.315e-12 |  1.417 |  -966.7   | False |
| tapered_log_window       | phase_correlated_features       |   0.6768  | 14.68      |   0.7632 |      8.864e-14 |  1.43  |  -111.8   | False |
| tapered_log_window       | rbf_gaussian_prior              |   0.00532 |  2.212     |   1.47   |      5.971e-16 |  1.414 |    -5.603 | False |
| tapered_log_window       | selected_log_kernel             |   1.336   | 13.41      |   0.7591 |      1.015e-13 |  1.443 |   -86.48  | False |
| tapered_log_window       | sign_breaking_mode_reassignment | inf       | 11.97      |   1.308  |      1.012e-13 |  1.414 |   -44.73  | False |

* Post-Darboux Virasoro residuals are exactly zero for every nondegenerate
  control, as Prop. 3.5 requires. Recorded as `diagnostic`, never as evidence.
* The involution defect and the polarization minimum eigenvalue are the
  discriminating operator-level columns and are reported per control.
* `angular_mode_scramble` is now a genuine **field-level** architecture
  (magnitudes from the selected `|k|^-2` law, angles from a finite random
  direction set), so the same control drives the pairing, the algebra and the
  Ward audits. It replaces the legacy weak `scrambled_angular_projection`.

## Gates

| gate | status |
|---|---|
| radii recorded per row (`R_in`,`R_out`,`R_0`,`rho`) | PASS |
| stable plateau inside the physical window | PASS (10/10 nodes, all configurations) |
| measured radius variation compatible with Cauchy–Green | PASS (all edges) |
| width/cutoff convergence refit from the operational pairing | PASS |
| null `b = -1/2` tested and p-value reported | PASS |
| pairing converges to `m delta` before Darboux | **FAIL under eq. (2.2)**; PASS under the tapered class |

## Blockers / caveats carried forward

1. The sharp annular prior of eq. (2.2) has **no conformal limit for the
   current kernel**: widening the annulus increases the deviation from
   `(z-w)^-2`. Diagnosed in WP-F; a 15% raised-cosine taper in `log k` repairs
   it. This is a regulator failure, not an architecture failure.
2. Operational leakage cannot discriminate radial priors (it is structurally
   zero for every isotropic prior); the antidiagonal amplitude does. The
   manuscript's complementary-test framing is preserved but sharpened.

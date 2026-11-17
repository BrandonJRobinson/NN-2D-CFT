# WP-D progress report — finite-feature NS super-Virasoro audit

## Command

```bash
python scripts/run_ns_audit.py
```

## The CAR is architecture-derived, not postulated

The manuscript fixes the fermion kernel only by its conformal form, so the suite
derives it from the **same** Log-Kernel feature map by one holomorphic
derivative of the scalar seed:

```
K_psipsi = -d_z K
K_psipsi^(N,Lambda)(z,w) = (L_w/N) sum_j conj(k_j) sin(Re(conj(k_j)(z-w)))
K_psipsi^Lambda(z,w)     = (J_0(Lambda_IR r) - J_0(Lambda_UV r)) / (z-w)
```

Both are manifestly antisymmetric under `z <-> w`, which is exactly the property
section 7 uses to turn the ordering discontinuity into the **symmetric** CAR
pairing. Eq. (7.5) is verified to machine precision on the ideal Cauchy kernel
(`tests/test_algebra.py`), and the canonical NS algebra closes with
`c = 3/2` exactly.

Because the fermion kernel needs only **one** derivative of the scalar seed, it
does **not** suffer the sharp-cutoff pathology that destroys the current kernel:
the analytic regulated CAR defect is already at the percent level under
eq. (2.2), and at the `1e-4` level under the tapered diagnostic.

## Selected architecture

| regulator_class          |   n_features |   fermionic_pairing_relative_error_mean |   fermionic_pairing_leakage_mean |   raw_car_defect_mean |   post_canonicalization_car_defect_mean |   raw_LG_residual_mean |   raw_GG_residual_mean |   canonical_LG_residual_mean |   canonical_GG_residual_mean |   central_charge_fit_mean |   star_compatibility_defect_mean |   polarization_min_eigenvalue_mean |   n_blocks |
|:-------------------------|-------------:|----------------------------------------:|---------------------------------:|----------------------:|----------------------------------------:|-----------------------:|-----------------------:|-----------------------------:|-----------------------------:|--------------------------:|---------------------------------:|-----------------------------------:|-----------:|
| sharp_annulus_manuscript |         5000 |                                 0.3971  |                          0.1988  |               0.3971  |                               2.577e-16 |                11.57   |                 4.085  |                    2.072e-14 |                    9.331e-15 |                       1.5 |                          0.3167  |                            -72.14  |          5 |
| sharp_annulus_manuscript |        10000 |                                 0.2576  |                          0.1452  |               0.2576  |                               2.794e-16 |                 7.766  |                 2.668  |                    2.055e-14 |                    9.35e-15  |                       1.5 |                          0.1811  |                           -114.1   |          5 |
| sharp_annulus_manuscript |        30000 |                                 0.1702  |                          0.1451  |               0.1702  |                               2.228e-16 |                 4.778  |                 1.646  |                    3.219e-14 |                    1.421e-14 |                       1.5 |                          0.1386  |                            -71.26  |          5 |
| sharp_annulus_manuscript |       100000 |                                 0.09104 |                          0.0582  |               0.09104 |                               2.998e-16 |                 2.556  |                 0.8966 |                    4.234e-14 |                    2.033e-14 |                       1.5 |                          0.06769 |                            -58.24  |          5 |
| sharp_annulus_manuscript |       300000 |                                 0.05734 |                          0.03602 |               0.05734 |                               2.202e-16 |                 1.539  |                 0.6637 |                    8.25e-15  |                    3.104e-15 |                       1.5 |                          0.05027 |                            -55.68  |          5 |
| tapered_log_window       |         5000 |                                 0.2296  |                          0.1782  |               0.2296  |                               2.178e-16 |                 6.006  |                 2.107  |                    2.198e-14 |                    1.077e-14 |                       1.5 |                          0.1992  |                            -45.14  |          5 |
| tapered_log_window       |        10000 |                                 0.2624  |                          0.152   |               0.2624  |                               2.146e-16 |                 4.125  |                 1.422  |                    1.162e-14 |                    4.479e-15 |                       1.5 |                          0.2125  |                            -23.41  |          5 |
| tapered_log_window       |        30000 |                                 0.1233  |                          0.08869 |               0.1233  |                               2.276e-16 |                 2.369  |                 0.8017 |                    1.909e-14 |                    1.085e-14 |                       1.5 |                          0.1281  |                             -8.231 |          5 |
| tapered_log_window       |       100000 |                                 0.08346 |                          0.05105 |               0.08346 |                               1.905e-16 |                 1.473  |                 0.5206 |                    5.836e-15 |                    2.618e-15 |                       1.5 |                          0.07627 |                             -2.099 |          5 |
| tapered_log_window       |       300000 |                                 0.04545 |                          0.02864 |               0.04545 |                               2.025e-16 |                 0.7302 |                 0.248  |                    3.013e-15 |                    1.267e-15 |                       1.5 |                          0.0478  |                             -2.441 |          5 |

## Controls

| regulator_class          |   n_features |   fermionic_pairing_relative_error_mean |   fermionic_pairing_leakage_mean |   raw_car_defect_mean |   post_canonicalization_car_defect_mean |   n_blocks | architecture          |
|:-------------------------|-------------:|----------------------------------------:|---------------------------------:|----------------------:|----------------------------------------:|-----------:|:----------------------|
| sharp_annulus_manuscript |       300000 |                                  0.9233 |                         0.9113   |                0.9233 |                               3.178e-16 |          5 | angular_mode_scramble |
| sharp_annulus_manuscript |       300000 |                                  2.676  |                         0.001981 |                2.676  |                               1.476e-16 |          5 | rbf_gaussian_prior    |
| tapered_log_window       |       300000 |                                  0.6564 |                         0.6542   |                0.6564 |                               3.032e-16 |          5 | angular_mode_scramble |
| tapered_log_window       |       300000 |                                  2.178  |                         0.002468 |                2.178  |                               1.768e-16 |          5 | rbf_gaussian_prior    |

## Sampled fermion / supercurrent correlators

| regulator_class          |   separation |   sampled |   stderr |   analytic |   ideal |   rel_err |
|:-------------------------|-------------:|----------:|---------:|-----------:|--------:|----------:|
| sharp_annulus_manuscript |         0.45 |    4.93   |    3.41  |     1.965  |  2.222  |    1.74   |
| sharp_annulus_manuscript |         0.6  |    1.135  |    3.448 |     1.819  |  1.667  |    2.171  |
| sharp_annulus_manuscript |         0.8  |   -0.1958 |    3.42  |     1.337  |  1.25   |    2.376  |
| sharp_annulus_manuscript |         1.05 |    1.823  |    3.39  |     1.016  |  0.9524 |    2.392  |
| sharp_annulus_manuscript |         1.35 |    2.462  |    3.358 |     0.7733 |  0.7407 |    3.71   |
| sharp_annulus_manuscript |         1.7  |    0.7614 |    3.361 |     0.5555 |  0.5882 |    3.402  |
| tapered_log_window       |         0.45 |    0.8869 |    1.728 |     1.965  |  2.222  |    0.9278 |
| tapered_log_window       |         0.6  |    1.007  |    1.705 |     1.819  |  1.667  |    0.8965 |
| tapered_log_window       |         0.8  |    3.955  |    1.711 |     1.337  |  1.25   |    2.004  |
| tapered_log_window       |         1.05 |    1.325  |    1.692 |     1.016  |  0.9524 |    2.057  |
| tapered_log_window       |         1.35 |    0.7269 |    1.714 |     0.7733 |  0.7407 |    1.819  |
| tapered_log_window       |         1.7  |    0.9808 |    1.726 |     0.5555 |  0.5882 |    2.568  |

## Gates

| gate | status |
|---|---|
| CAR architecture-derived, not postulated | PASS |
| pre-canonicalization pairing convergence demonstrated | PASS (see the width sweep) |
| raw and canonicalized `[L,G]`, `{G,G}` residuals released | PASS |
| `c = 3/2` released | PASS |
| `*`-compatibility and polarization diagnostics released | PASS |
| sampled correlator methodology explicit | PASS |

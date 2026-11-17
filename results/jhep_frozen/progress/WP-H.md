# WP-H progress report — sampled vertex weights and finite-width cumulant

## Command

```bash
python scripts/run_vertex_and_cumulant.py
```

## H1 — sampled vertex weights

`<V_a(z)V_-a(0)> = |z|^(-2a^2)` for a scalar primary of weights `h = hbar`, so
the measured log-log slope `s` gives `h_a = -s/4` against the target `a^2/2`
(eq. 6.1). The estimator is `mean_Theta[exp(i a (phi_N(z)-phi_N(0)))]` times the
recorded normal-ordering constant `exp(a^2 K_Lambda(0))`.

| regulator_class          |   charge |   n_features |   h_fit |   h_stderr |   h_target |   z_vs_target |   n_blocks | compatible_with_target   |
|:-------------------------|---------:|-------------:|--------:|-----------:|-----------:|--------------:|-----------:|:-------------------------|
| sharp_annulus_manuscript |     0.25 |         1000 | 0.03254 |  0.000873  |    0.03125 |       1.478   |          6 | True                     |
| sharp_annulus_manuscript |     0.25 |         2000 | 0.03122 |  0.0004924 |    0.03125 |      -0.05817 |          6 | True                     |
| sharp_annulus_manuscript |     0.25 |         4000 | 0.03159 |  0.0004037 |    0.03125 |       0.8524  |          6 | True                     |
| sharp_annulus_manuscript |     0.4  |         1000 | 0.08018 |  0.002965  |    0.08    |       0.06    |          6 | True                     |
| sharp_annulus_manuscript |     0.4  |         2000 | 0.07637 |  0.001793  |    0.08    |      -2.026   |          6 | False                    |
| sharp_annulus_manuscript |     0.4  |         4000 | 0.08211 |  0.001608  |    0.08    |       1.313   |          6 | True                     |
| sharp_annulus_manuscript |     0.55 |         1000 | 0.1511  |  0.009672  |    0.1512  |      -0.01328 |          6 | True                     |
| sharp_annulus_manuscript |     0.55 |         2000 | 0.1477  |  0.002411  |    0.1512  |      -1.491   |          6 | True                     |
| sharp_annulus_manuscript |     0.55 |         4000 | 0.1637  |  0.008664  |    0.1512  |       1.437   |          6 | True                     |
| sharp_annulus_manuscript |     0.7  |         1000 | 0.2355  |  0.02628   |    0.245   |      -0.3613  |          6 | True                     |
| sharp_annulus_manuscript |     0.7  |         2000 | 0.1704  |  0.03028   |    0.245   |      -2.464   |          6 | False                    |
| sharp_annulus_manuscript |     0.7  |         4000 | 0.3055  |  0.04274   |    0.245   |       1.415   |          6 | True                     |
| tapered_log_window       |     0.25 |         1000 | 0.03184 |  0.0004615 |    0.03125 |       1.285   |          6 | True                     |
| tapered_log_window       |     0.25 |         2000 | 0.03043 |  0.0005941 |    0.03125 |      -1.385   |          6 | True                     |
| tapered_log_window       |     0.25 |         4000 | 0.03115 |  0.0003171 |    0.03125 |      -0.3259  |          6 | True                     |
| tapered_log_window       |     0.4  |         1000 | 0.08308 |  0.003539  |    0.08    |       0.8699  |          6 | True                     |
| tapered_log_window       |     0.4  |         2000 | 0.07909 |  0.001092  |    0.08    |      -0.8328  |          6 | True                     |
| tapered_log_window       |     0.4  |         4000 | 0.07935 |  0.001024  |    0.08    |      -0.6382  |          6 | True                     |
| tapered_log_window       |     0.55 |         1000 | 0.1403  |  0.004771  |    0.1512  |      -2.298   |          6 | False                    |
| tapered_log_window       |     0.55 |         2000 | 0.1476  |  0.005738  |    0.1512  |      -0.6427  |          6 | True                     |
| tapered_log_window       |     0.55 |         4000 | 0.1568  |  0.007533  |    0.1512  |       0.7392  |          6 | True                     |
| tapered_log_window       |     0.7  |         1000 | 0.324   |  0.04597   |    0.245   |       1.719   |          6 | True                     |
| tapered_log_window       |     0.7  |         2000 | 0.2712  |  0.05441   |    0.245   |       0.4819  |          6 | True                     |
| tapered_log_window       |     0.7  |         4000 | 0.2993  |  0.07625   |    0.245   |       0.7117  |          6 | True                     |

### Fit-window sensitivity

| regulator_class          |   fit_window_index |   lo |   hi |   h_fit |   stderr |
|:-------------------------|-------------------:|-----:|-----:|--------:|---------:|
| sharp_annulus_manuscript |                  0 | 0.45 | 1.8  |  0.1257 |  0.01068 |
| sharp_annulus_manuscript |                  1 | 0.55 | 1.45 |  0.1298 |  0.0245  |
| sharp_annulus_manuscript |                  2 | 0.7  | 1.15 |  0.1493 |  0.04935 |
| tapered_log_window       |                  0 | 0.45 | 1.8  |  0.1395 |  0.01681 |
| tapered_log_window       |                  1 | 0.55 | 1.45 |  0.1413 |  0.02705 |
| tapered_log_window       |                  2 | 0.7  | 1.15 |  0.1221 |  0.02673 |

## H2 — finite-width connected fourth cumulant

`phi_N` is a normalized sum of `N` i.i.d. bounded terms, so its connected fourth
cumulant is **exactly** `kappa_4(N) = -3 sigma_phi^4/(8N)`, i.e. `~ N^-1` with a
standardized excess kurtosis `-3/(2N)`. That closed form is the theory guide
plotted alongside the measurement, and it is why the audit deliberately uses
small widths: the standardized signal is `3/(2N)`, so the measurable regime is
`N ~ 10^1-10^2` at feasible sample counts.

| regulator_class          |   n_features |   cumulant4 |   cumulant4_stderr |   kappa4_prediction |   relative_deviation_from_prediction |   n_blocks |
|:-------------------------|-------------:|------------:|-------------------:|--------------------:|-------------------------------------:|-----------:|
| sharp_annulus_manuscript |            8 |     -63.09  |             0.6069 |             -63.62  |                             0.008304 |          8 |
| sharp_annulus_manuscript |           16 |     -31.16  |             0.4194 |             -31.81  |                             0.02056  |          8 |
| sharp_annulus_manuscript |           32 |     -16.71  |             0.646  |             -15.91  |                             0.05083  |          8 |
| sharp_annulus_manuscript |           64 |      -7.473 |             0.2022 |              -7.953 |                             0.06032  |          8 |
| sharp_annulus_manuscript |          128 |      -3.994 |             0.5302 |              -3.976 |                             0.004401 |          8 |
| tapered_log_window       |            8 |     -45.26  |             0.447  |             -45.97  |                             0.01532  |          8 |
| tapered_log_window       |           16 |     -23.21  |             0.3862 |             -22.98  |                             0.009653 |          8 |
| tapered_log_window       |           32 |     -11.6   |             0.2359 |             -11.49  |                             0.009464 |          8 |
| tapered_log_window       |           64 |      -5.609 |             0.3892 |              -5.746 |                             0.02391  |          8 |
| tapered_log_window       |          128 |      -2.926 |             0.2704 |              -2.873 |                             0.01851  |          8 |

### Scaling fit and null test

| regulator_class          |   fit_exponent |   fit_exponent_stderr |   fit_exponent_ci_low |   fit_exponent_ci_high |   fit_r_squared |   fit_range_low |   fit_range_high |   null_exponent |   null_p_value | null_rejected_at_5pct   |
|:-------------------------|---------------:|----------------------:|----------------------:|-----------------------:|----------------:|----------------:|-----------------:|----------------:|---------------:|:------------------------|
| sharp_annulus_manuscript |        -1.002  |              0.02131  |                -1.07  |                -0.9345 |          0.9986 |               8 |              128 |              -1 |         0.9207 | False                   |
| tapered_log_window       |        -0.9951 |              0.009225 |                -1.024 |                -0.9658 |          0.9997 |               8 |              128 |              -1 |         0.6342 | False                   |

## Interpretation (emitted in every row)

Interpreted **only** as finite-width non-Gaussianity of the neural ensemble
(appendix F). It is **not** evidence of a local interacting CFT.

## Gates

| gate | status |
|---|---|
| weights include seeds, sample counts, fit windows, uncertainties | PASS |
| cumulant exponent with CI and goodness of fit | PASS |
| cumulant described only as finite-width non-Gaussianity | PASS |

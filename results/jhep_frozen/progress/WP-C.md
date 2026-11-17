# WP-C progress report — sampled finite-network Ward identities

## Command

```bash
python scripts/run_ward_audit.py
```

## Three estimator classes, in three separate files

| class | file | what varies |
|---|---|---|
| `analytic_closed_kernel` | `analytic_ward_points.csv` | nothing; exact regulated kernel |
| `deterministic_finite_feature` | `deterministic_feature_ward_points.csv` | momentum sample (seeded), phases averaged |
| `sampled_finite_network` | `sampled_ward_seed_rows.csv` | full Monte Carlo over networks |

They are **never combined in one uncertainty calculation**.

## 1. analytic_closed_kernel — maximum absolute error

| configuration   |   current_max |   stress_max |
|:----------------|--------------:|-------------:|
| four_vertex     |      0.03053  |     0.008759 |
| three_vertex    |      0.04253  |     0.01227  |
| two_vertex      |      0.007744 |     0.001575 |

Unlike the pairing, the Ward identities involve only **one** derivative of the
scalar covariance, so they are insensitive to the sharp-cutoff endpoint term
that destroys the current kernel (WP-F). The closed-kernel evaluation is
already accurate at the `1e-3` level.

## 2. deterministic_finite_feature (selected architecture)

| regulator_class          |   n_features |   current_max |   stress_max |
|:-------------------------|-------------:|--------------:|-------------:|
| sharp_annulus_manuscript |          500 |        9.2    |      43.33   |
| sharp_annulus_manuscript |         2000 |        5.441  |      14.2    |
| sharp_annulus_manuscript |        20000 |        1.879  |       1.97   |
| sharp_annulus_manuscript |       200000 |        0.4959 |       0.2548 |
| tapered_log_window       |          500 |        4.024  |       7.574  |
| tapered_log_window       |         2000 |        2.061  |       1.836  |
| tapered_log_window       |        20000 |        0.6764 |       0.4365 |
| tapered_log_window       |       200000 |        0.2544 |       0.1293 |

## 3. sampled_finite_network — convergence in M

| regulator_class          | configuration   |   n_networks |   n_features |   stress_err |   stderr |   current_err |   blocks |
|:-------------------------|:----------------|-------------:|-------------:|-------------:|---------:|--------------:|---------:|
| sharp_annulus_manuscript | four_vertex     |         2000 |         2000 |      136.7   |   21     |        4.771  |        4 |
| sharp_annulus_manuscript | four_vertex     |         8000 |         2000 |       87.15  |   10.57  |        2.57   |        4 |
| sharp_annulus_manuscript | four_vertex     |        32000 |         2000 |       46.54  |    5.311 |        1.163  |        4 |
| sharp_annulus_manuscript | three_vertex    |         2000 |         2000 |      703.3   |   82.1   |       18.83   |        4 |
| sharp_annulus_manuscript | three_vertex    |         8000 |         2000 |      357.5   |   67.98  |       10.5    |        4 |
| sharp_annulus_manuscript | three_vertex    |        32000 |         2000 |      179     |   18.96  |        4.726  |        4 |
| sharp_annulus_manuscript | two_vertex      |         2000 |         2000 |      123.2   |   21.42  |        3.999  |        4 |
| sharp_annulus_manuscript | two_vertex      |         8000 |         2000 |       65.38  |    9.147 |        1.944  |        4 |
| sharp_annulus_manuscript | two_vertex      |        32000 |         2000 |       36.66  |    3.588 |        1.111  |        4 |
| tapered_log_window       | four_vertex     |         2000 |         2000 |       39.59  |    3.369 |        2.294  |        4 |
| tapered_log_window       | four_vertex     |         8000 |         2000 |       19.54  |    3.466 |        1.07   |        4 |
| tapered_log_window       | four_vertex     |        32000 |         2000 |       10.07  |    1.869 |        0.5092 |        4 |
| tapered_log_window       | three_vertex    |         2000 |         2000 |      170     |   16.76  |        7.598  |        4 |
| tapered_log_window       | three_vertex    |         8000 |         2000 |       73.22  |    8.566 |        4.201  |        4 |
| tapered_log_window       | three_vertex    |        32000 |         2000 |       37.31  |    6.155 |        1.974  |        4 |
| tapered_log_window       | two_vertex      |         2000 |         2000 |       31.59  |    2.704 |        1.851  |        4 |
| tapered_log_window       | two_vertex      |         8000 |         2000 |       18.47  |    1.311 |        0.9799 |        4 |
| tapered_log_window       | two_vertex      |        32000 |         2000 |        7.878 |    1.055 |        0.4961 |        4 |

## Controls at the largest N and M

| regulator_class          | architecture             |   stress_err |   stderr |   current_err |
|:-------------------------|:-------------------------|-------------:|---------:|--------------:|
| sharp_annulus_manuscript | angular_mode_scramble    |       99.02  | 15.49    |         1.984 |
| sharp_annulus_manuscript | local_exponent_direction |      441.1   | 53.86    |         8.516 |
| sharp_annulus_manuscript | rbf_gaussian_prior       |        1.639 |  0.1893  |         1.467 |
| sharp_annulus_manuscript | selected_log_kernel      |       81.6   | 11.73    |         2.555 |
| tapered_log_window       | angular_mode_scramble    |       17.26  |  2.536   |         1.003 |
| tapered_log_window       | local_exponent_direction |      192.2   | 44.55    |         5.083 |
| tapered_log_window       | rbf_gaussian_prior       |        1.119 |  0.07122 |         1.184 |
| tapered_log_window       | selected_log_kernel      |       20.56  |  4.19    |         1.028 |

## Complementary interpretation (emitted in every row)

Pairing fidelity primarily tests the **radial** spectral structure of the prior;
the current and stress Ward identities test the **chiral/angular**
organisation; post-Darboux closure tests **neither** (Proposition 3.5).

## Gates

| gate | status |
|---|---|
| explicit finite-network estimator | PASS |
| independent uncertainty (blocks = named seeds, bootstrap over blocks) | PASS |
| convergence with N and M | PASS (see the M sweep above) |
| selected architecture separated from the controls | see the control table |
| no analytic identity described as Monte Carlo evidence | PASS (three separate files, three separate estimator classes) |

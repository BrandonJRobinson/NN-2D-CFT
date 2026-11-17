# WP-B progress report — sampled <TT> central charge

## Command

```bash
python scripts/run_sampled_tt.py
```

## Estimator provenance (mission B1)

All items are recorded in `sampled_tt_summary.md` and in
`nnft_virasoro.stress_tensor.ESTIMATOR_PROVENANCE`: neural field eq. (2.1);
prior eq. (2.2); `sigma_phi^2 = 4 L_w`; `T = -(1/2):d phi d phi:`; per-realization
normal-ordering subtraction `c_N = (L_w/2N) sum_j conj(k_j)^2`; insertion geometry
= 48 points on a circle of radius 0.6 so each chord separation is averaged over
48 rotated pairs; fit target `<TT> = (c/2)(z1-z2)^-4` de-rotated as
`U_ab = 2 T_a T_b (z_a-z_b)^4`; **one block per independent named seed**, with
the bootstrap resampling blocks.

## Two sampled estimators, reported side by side and never merged

`direct_TT`
: the raw four-field Monte Carlo average. **No analytic input.** Its
  per-realization signal-to-noise is `~ (1/2) corr^2` with
  `corr = |K_JJ(r)| / E|d phi|^2 ~ 1e-2` anywhere inside the conformal window,
  so its variance is quartic in the field and it is the harder estimator by far.

`wick_factorized`
: `c = (mean_(Theta,pairs) S_ab)^2` with `S_ab = -d phi_a d phi_b (z_a-z_b)^2`,
  using the infinite-width Wick relation `<TT> = (1/2)<d phi d phi>^2`. The
  Gaussian factorization is an **explicitly disclosed analytic input**, exact at
  `N = infinity` with an `O(1/N)` correction bounded by the measured fourth
  cumulant of WP-H2. Its noise is linear in the field, so it is far better
  conditioned.

## Selected architecture, reference fit window

| regulator_class          | estimator_variant   |   n_features |   n_networks |   c_exp |   c_exp_stderr |   c_exp_boot_ci_low |   c_exp_boot_ci_high |   n_blocks | compatible_with_c1   |   z_vs_c1 |
|:-------------------------|:--------------------|-------------:|-------------:|--------:|---------------:|--------------------:|---------------------:|-----------:|:---------------------|----------:|
| sharp_annulus_manuscript | direct_TT           |          250 |        60000 |  5.263  |       11.1     |            -16.53   |               25.14  |         12 | True                 |    0.384  |
| sharp_annulus_manuscript | direct_TT           |          750 |        60000 |  7.013  |        8.905   |             -9.68   |               23.43  |         12 | True                 |    0.6753 |
| sharp_annulus_manuscript | wick_factorized     |          250 |        60000 |  1.22   |        0.1236  |              0.988  |                1.452 |         12 | True                 |    1.78   |
| sharp_annulus_manuscript | wick_factorized     |          750 |        60000 |  1.381  |        0.116   |              1.161  |                1.595 |         12 | False                |    3.284  |
| tapered_log_window       | direct_TT           |          250 |        60000 |  2.057  |        0.5767  |              1.07   |                3.197 |         12 | True                 |    1.833  |
| tapered_log_window       | direct_TT           |          750 |        60000 |  0.7509 |        0.6166  |             -0.4463 |                1.844 |         12 | True                 |   -0.404  |
| tapered_log_window       | wick_factorized     |          250 |        60000 |  1.016  |        0.01125 |              0.9939 |                1.035 |         12 | True                 |    1.397  |
| tapered_log_window       | wick_factorized     |          750 |        60000 |  1.023  |        0.0168  |              0.9926 |                1.055 |         12 | True                 |    1.379  |

## Fit-window sensitivity (wick_factorized, selected)

| regulator_class          |   fit_window_index |   lo |   hi |   c_exp |   stderr |
|:-------------------------|-------------------:|-----:|-----:|--------:|---------:|
| sharp_annulus_manuscript |                  0 | 0.3  | 0.55 |  1.301  | 0.1198   |
| sharp_annulus_manuscript |                  1 | 0.23 | 0.8  |  1.136  | 0.2331   |
| sharp_annulus_manuscript |                  2 | 0.31 | 0.53 |  1.27   | 0.1206   |
| sharp_annulus_manuscript |                  3 | 0.23 | 0.46 |  2.454  | 0.1688   |
| tapered_log_window       |                  0 | 0.3  | 0.55 |  1.019  | 0.01402  |
| tapered_log_window       |                  1 | 0.23 | 0.8  |  0.8906 | 0.01037  |
| tapered_log_window       |                  2 | 0.31 | 0.53 |  1.022  | 0.01491  |
| tapered_log_window       |                  3 | 0.23 | 0.46 |  0.889  | 0.008621 |

## analytic_closed_kernel reference at the same geometry

| regulator_class          |   c_exp |   chi2_per_dof |
|:-------------------------|--------:|---------------:|
| sharp_annulus_manuscript |   3.47  |      13.13     |
| tapered_log_window       |   1.013 |       0.002118 |

This row quantifies the **regulator systematic** at the identical insertion
geometry, separately from the sampling uncertainty.

## Controls

| regulator_class          | architecture          | estimator_variant   |      c_exp |   c_exp_stderr |   n_blocks |
|:-------------------------|:----------------------|:--------------------|-----------:|---------------:|-----------:|
| sharp_annulus_manuscript | angular_mode_scramble | direct_TT           | 438.4      |      8.283     |          4 |
| sharp_annulus_manuscript | angular_mode_scramble | wick_factorized     |   1.375    |      0.07906   |          4 |
| sharp_annulus_manuscript | rbf_gaussian_prior    | direct_TT           |   0.005309 |      0.0008842 |          4 |
| sharp_annulus_manuscript | rbf_gaussian_prior    | wick_factorized     |   0.002753 |      0.0001252 |          4 |
| tapered_log_window       | angular_mode_scramble | direct_TT           |  47.95     |      1.153     |          4 |
| tapered_log_window       | angular_mode_scramble | wick_factorized     |   1.043    |      0.01818   |          4 |
| tapered_log_window       | rbf_gaussian_prior    | direct_TT           |   0.004269 |      0.0004984 |          4 |
| tapered_log_window       | rbf_gaussian_prior    | wick_factorized     |   0.002092 |      2.747e-05 |          4 |

## Sampled-window caveat (recorded in every row)

The connected `<TT>` signal-to-local-variance ratio falls like
`(Lambda_eff r)^-2`, so the strict `margin_uv = 30` used by the pairing audits
would require `M ~ 1e8` realizations. This audit therefore uses a relaxed,
explicitly recorded evaluation margin, and the accompanying
`analytic_closed_kernel` row at the same geometry states exactly what that
costs in systematic terms. Nothing is hidden in a footnote: the margins are
columns.

## Gates

| gate | status |
|---|---|
| estimator provenance complete | PASS |
| uncertainty from independent blocks/seeds | PASS |
| no hidden analytic substitution | PASS (both variants released; the analytic input in `wick_factorized` is named) |
| result statistically compatible with c=1 | see the `compatible_with_c1` column above |

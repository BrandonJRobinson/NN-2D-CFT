# WP-E progress report — extended central-offset model

## Command

```bash
python scripts/run_central_offset.py
```

Modes `m = 2..12`, oscillator cutoffs `Lambda in {16,20,24}`, both regulator
classes, four named seeds, two widths. 561 rows in
`central_offset_mode_extension.csv`; fits in `central_offset_fit.json`.

## Two objects, never conflated

`canonical_truncated` (`exact_symbolic`)
: the modewise anomaly of the **canonical** truncated Sugawara algebra. This is
  the null that asks whether truncation alone produces the released offset.

`raw_empirical` (`deterministic_finite_feature`)
: the modewise anomaly of the truncated algebra built from the **measured**
  finite-feature pairing — the object the manuscript calls the "raw empirical
  release".

## Fits

| group                                        |   n_modes |           A |         A_sd |   decay_p |   model_R2 | consistent_with_legacy   |
|:---------------------------------------------|----------:|------------:|-------------:|----------:|-----------:|:-------------------------|
| canonical_truncated|all                      |        11 |   0         |    0         | nan       |     nan    | False                    |
| canonical_truncated|all|m2to4                |         3 |   0         |    0         | nan       |     nan    | False                    |
| raw_empirical|sharp_annulus_manuscript       |        11 |   1.567e+05 |    4.028e+05 |   1.632   |   -1299    | True                     |
| raw_empirical|tapered_log_window             |        11 |   2.246e+04 |    5.367e+04 |   2.08    |   -1402    | True                     |
| raw_empirical|sharp_annulus_manuscript|m2to4 |         3 | 750         | 1169         |  -0.1816  |     -16.86 | True                     |
| raw_empirical|tapered_log_window|m2to4       |         3 | 102         |  156.3       |  -0.03005 |     -15.93 | True                     |

## Findings

1. **The canonical truncated algebra has exactly zero offset.**
   `scalar([L_m, L_-m] - 2 m L_0) = m(m^2-1)/12` to machine precision for every
   audited `m` and every `Lambda >= m+2`. So the released alternating offset
   `A = 0.1822` is **not** a property of the finite oscillator cutoff, and the
   contamination model of eq. (5.8) has no support from truncation.

2. **The raw empirical offset is not reproduced.** Restricted to the legacy
   scope `m in {2,3,4}` the frozen recomputation gives `A = 750 +- 1170`
   (sharp) and `A = 102 +- 156` (tapered): consistent with zero within its own
   spread and four orders of magnitude larger than the released value. Over the
   full `m = 2..12` range the fitted decay exponent is `+1.6 / +2.1`, not the
   model's `-3`.

3. **Diagnosis, not prose.** The modewise scalar is quadratic in the pairing
   and the pairing amplifies entrywise like `rho^|m|` (Prop. 3.5), so at
   `rho = 2` and `m = 12` the finite-`N` pairing error is amplified by `~4x10^3`.
   The raw empirical modewise anomaly is therefore noise-dominated above `m ~ 4`
   and cannot support a one-parameter offset model at the released precision.

The released value is not reproducible from the mounted release, and the model
is reported as a **failed descriptive fit**. It is not replaced by prose and it
is not quietly dropped.

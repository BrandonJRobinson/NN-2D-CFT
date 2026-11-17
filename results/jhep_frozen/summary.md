# JHEP Virasoro frozen evidence campaign - summary

Generated 2026-08-01T09:13:15Z | commit `b3fd73dd887b` | config hash `278e438b94465564`

## 1. How to read this release

Every quantitative statement is one row of `results/jhep_frozen/claim_ledger.csv`, carrying its estimator class, seed set, uncertainty, generating script, config hash, input and output artifacts, and git commit. Nothing below is hand-entered.

### Estimator classes (never pooled in one uncertainty calculation)

| class | meaning |
|---|---|
| `exact_symbolic` | closed-form identity on the ordered-word algebra or the filtered Fock space; machine-precision zero |
| `analytic_closed_kernel` | exact regulated Bessel kernel; deterministic, no seed |
| `deterministic_finite_feature` | fixed momentum sample with the phases averaged analytically; seeded but exact given the sample |
| `sampled_finite_network` | Monte Carlo over independent network realizations; carries sampling uncertainty |

### Regulator classes

Every kernel-level metric is reported under both:

- **`sharp_annulus_manuscript`** (taper 0.00) - eq. (2.2) exactly as written. **This is the manuscript claim.**
- **`tapered_log_window`** (taper 0.15) - the same `|k|^-2` prior with a raised-cosine taper in `log k`. **A labelled regulator diagnostic only**, used to localize a sharp-cutoff systematic. It is never substituted for the manuscript claim.

## 2. Ledger status counts

| status | rows |
|---|---|
| accepted | 176 |
| diagnostic | 32 |
| failed_gate | 19 |
| accepted_with_caveat | 8 |

## 3. Results by evidence category

### 3.1 Exact mathematical results

Closed-form identities. Theorem 4.6 holds exactly on the filtered Fock subspace; the canonical truncated Sugawara algebra reproduces `m(m^2-1)/12` exactly; the NS super-Virasoro algebra closes with `c = 3/2` exactly.

| claim | metric | value | uncertainty | 95% CI | status |
|---|---|---|---|---|---|
| `wpA_admissibility_angular_mode_scramble_star_compatibility_defect_sharp_annulus_manuscript` | involution defect ||conj(S)C - CS||_F/||S||_F (angular_mode_scramble) | 1.21 | 0.3066 | [0.414471, 1.51509] | accepted |
| `wpA_admissibility_angular_mode_scramble_polarization_min_eigenvalue_sharp_annulus_manuscript` | minimum eigenvalue of the induced Fock polarization Gram (angular_mode_scramble) | -1.8 | 2.331 | [-7.47668, 1.07631] | failed_gate |
| `wpA_admissibility_angular_mode_scramble_symplectic_defect_sharp_annulus_manuscript` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (angular_mode_scramble) | 0.0000000000000019 | 1.638e-15 | [3.37684e-16, 7.47767e-15] | diagnostic |
| `wpA_admissibility_angular_mode_scramble_condition_number_sharp_annulus_manuscript` | cond(S) (angular_mode_scramble) | 13 | 10.62 | [2.25562, 37.3049] | accepted |
| `wpA_admissibility_anisotropic_angular_prior_star_compatibility_defect_sharp_annulus_manuscript` | involution defect ||conj(S)C - CS||_F/||S||_F (anisotropic_angular_prior) | 1.27 | 0.3134 | [0.441239, 1.74602] | accepted |
| `wpA_admissibility_anisotropic_angular_prior_polarization_min_eigenvalue_sharp_annulus_manuscript` | minimum eigenvalue of the induced Fock polarization Gram (anisotropic_angular_prior) | -1.5 | 1.521 | [-4.94933, 0.785492] | failed_gate |
| `wpA_admissibility_anisotropic_angular_prior_symplectic_defect_sharp_annulus_manuscript` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (anisotropic_angular_prior) | 0.0000000000000027 | 2.842e-15 | [3.82808e-16, 1.08573e-14] | diagnostic |
| `wpA_admissibility_anisotropic_angular_prior_condition_number_sharp_annulus_manuscript` | cond(S) (anisotropic_angular_prior) | 15 | 12.51 | [2.2561, 43.915] | accepted |
| `wpA_admissibility_local_exponent_direction_star_compatibility_defect_sharp_annulus_manuscript` | involution defect ||conj(S)C - CS||_F/||S||_F (local_exponent_direction) | 1.35 | 0.2161 | [0.996042, 1.8959] | accepted |
| `wpA_admissibility_local_exponent_direction_polarization_min_eigenvalue_sharp_annulus_manuscript` | minimum eigenvalue of the induced Fock polarization Gram (local_exponent_direction) | -3.9 | 3.186 | [-12.2213, -0.481234] | failed_gate |
| `wpA_admissibility_local_exponent_direction_symplectic_defect_sharp_annulus_manuscript` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (local_exponent_direction) | 0.0000000000000042 | 5.577e-15 | [4.97555e-16, 2.17175e-14] | diagnostic |
| `wpA_admissibility_local_exponent_direction_condition_number_sharp_annulus_manuscript` | cond(S) (local_exponent_direction) | 26 | 42.44 | [4.9624, 181.153] | accepted |
| `wpA_admissibility_non_symplectic_dense_mixing_star_compatibility_defect_sharp_annulus_manuscript` | involution defect ||conj(S)C - CS||_F/||S||_F (non_symplectic_dense_mixing) | 1.417 | 0.07481 | [1.22894, 1.5564] | accepted |
| `wpA_admissibility_non_symplectic_dense_mixing_polarization_min_eigenvalue_sharp_annulus_manuscript` | minimum eigenvalue of the induced Fock polarization Gram (non_symplectic_dense_mixing) | -35 | 20.91 | [-76.9072, -10.0332] | failed_gate |
| `wpA_admissibility_non_symplectic_dense_mixing_symplectic_defect_sharp_annulus_manuscript` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (non_symplectic_dense_mixing) | 0.000000000000031 | 5.716e-14 | [1.96867e-15, 2.15365e-13] | diagnostic |
| `wpA_admissibility_non_symplectic_dense_mixing_condition_number_sharp_annulus_manuscript` | cond(S) (non_symplectic_dense_mixing) | 136 | 136 | [23.7021, 413.272] | accepted |
| `wpA_admissibility_phase_correlated_features_star_compatibility_defect_sharp_annulus_manuscript` | involution defect ||conj(S)C - CS||_F/||S||_F (phase_correlated_features) | 1.17 | 0.264 | [0.492764, 1.49907] | accepted |
| `wpA_admissibility_phase_correlated_features_polarization_min_eigenvalue_sharp_annulus_manuscript` | minimum eigenvalue of the induced Fock polarization Gram (phase_correlated_features) | -0.7 | 1.695 | [-5.16564, 0.96438] | failed_gate |
| `wpA_admissibility_phase_correlated_features_symplectic_defect_sharp_annulus_manuscript` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (phase_correlated_features) | 0.000000000000008 | 2.692e-14 | [3.11151e-16, 1.2535e-13] | diagnostic |
| `wpA_admissibility_phase_correlated_features_condition_number_sharp_annulus_manuscript` | cond(S) (phase_correlated_features) | 24 | 63.42 | [2.14402, 298.847] | accepted |
| `wpA_admissibility_rbf_gaussian_prior_star_compatibility_defect_sharp_annulus_manuscript` | involution defect ||conj(S)C - CS||_F/||S||_F (rbf_gaussian_prior) | 1.036 | 0.01004 | [1.01277, 1.05441] | accepted |
| `wpA_admissibility_rbf_gaussian_prior_polarization_min_eigenvalue_sharp_annulus_manuscript` | minimum eigenvalue of the induced Fock polarization Gram (rbf_gaussian_prior) | 0.826 | 0.02198 | [0.786992, 0.879553] | accepted |
| `wpA_admissibility_rbf_gaussian_prior_symplectic_defect_sharp_annulus_manuscript` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (rbf_gaussian_prior) | 0.000000000000000152 | 5.188e-17 | [6.41406e-17, 2.50014e-16] | diagnostic |
| `wpA_admissibility_rbf_gaussian_prior_condition_number_sharp_annulus_manuscript` | cond(S) (rbf_gaussian_prior) | 19.51 | 0.5253 | [18.2507, 20.3887] | accepted |
| `wpA_admissibility_selected_log_kernel_star_compatibility_defect_sharp_annulus_manuscript` | involution defect ||conj(S)C - CS||_F/||S||_F (selected_log_kernel) | 1.28 | 0.1822 | [0.915098, 1.51546] | accepted |
| `wpA_admissibility_selected_log_kernel_polarization_min_eigenvalue_sharp_annulus_manuscript` | minimum eigenvalue of the induced Fock polarization Gram (selected_log_kernel) | -1.6 | 2.321 | [-8.63437, 0.72419] | failed_gate |
| `wpA_admissibility_selected_log_kernel_symplectic_defect_sharp_annulus_manuscript` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (selected_log_kernel) | 0.000000000000005 | 1.002e-14 | [3.37929e-16, 4.52401e-14] | diagnostic |
| `wpA_admissibility_selected_log_kernel_condition_number_sharp_annulus_manuscript` | cond(S) (selected_log_kernel) | 24 | 32.69 | [2.41811, 136.632] | accepted |
| `wpA_admissibility_sign_breaking_mode_reassignment_star_compatibility_defect_sharp_annulus_manuscript` | involution defect ||conj(S)C - CS||_F/||S||_F (sign_breaking_mode_reassignment) | 1.41421356237000029 | 2.22e-16 | [1.41421, 1.41421] | accepted |
| `wpA_admissibility_sign_breaking_mode_reassignment_polarization_min_eigenvalue_sharp_annulus_manuscript` | minimum eigenvalue of the induced Fock polarization Gram (sign_breaking_mode_reassignment) | -1.07 | 0.9143 | [-3.01099, -0.0432872] | failed_gate |
| `wpA_admissibility_sign_breaking_mode_reassignment_symplectic_defect_sharp_annulus_manuscript` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (sign_breaking_mode_reassignment) | 0.0000000000000028 | 8.029e-15 | [2.6406e-16, 3.77207e-14] | diagnostic |
| `wpA_admissibility_sign_breaking_mode_reassignment_condition_number_sharp_annulus_manuscript` | cond(S) (sign_breaking_mode_reassignment) | 6.3 | 6.953 | [1.89882, 34.9889] | accepted |
| `wpA_admissibility_angular_mode_scramble_star_compatibility_defect_tapered_log_window` | involution defect ||conj(S)C - CS||_F/||S||_F (angular_mode_scramble) | 0.52 | 0.4313 | [0.10882, 1.42304] | accepted |
| `wpA_admissibility_angular_mode_scramble_polarization_min_eigenvalue_tapered_log_window` | minimum eigenvalue of the induced Fock polarization Gram (angular_mode_scramble) | 0.66 | 0.5407 | [-0.827925, 1.05136] | failed_gate |
| `wpA_admissibility_angular_mode_scramble_symplectic_defect_tapered_log_window` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (angular_mode_scramble) | 0.00000000000000063 | 8.749e-16 | [1.06636e-16, 3.87714e-15] | diagnostic |
| `wpA_admissibility_angular_mode_scramble_condition_number_tapered_log_window` | cond(S) (angular_mode_scramble) | 4.9 | 6.06 | [1.25465, 20.7742] | accepted |
| `wpA_admissibility_anisotropic_angular_prior_star_compatibility_defect_tapered_log_window` | involution defect ||conj(S)C - CS||_F/||S||_F (anisotropic_angular_prior) | 0.40 | 0.283 | [0.215202, 1.2096] | accepted |
| `wpA_admissibility_anisotropic_angular_prior_polarization_min_eigenvalue_tapered_log_window` | minimum eigenvalue of the induced Fock polarization Gram (anisotropic_angular_prior) | 0.68 | 0.7981 | [-2.20857, 1.13296] | failed_gate |
| `wpA_admissibility_anisotropic_angular_prior_symplectic_defect_tapered_log_window` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (anisotropic_angular_prior) | 0.00000000000000037 | 2.048e-16 | [1.35001e-16, 9.62176e-16] | diagnostic |
| `wpA_admissibility_anisotropic_angular_prior_condition_number_tapered_log_window` | cond(S) (anisotropic_angular_prior) | 2.3 | 1.352 | [1.39343, 7.09378] | accepted |
| `wpA_admissibility_local_exponent_direction_star_compatibility_defect_tapered_log_window` | involution defect ||conj(S)C - CS||_F/||S||_F (local_exponent_direction) | 1.35 | 0.1974 | [1.00462, 1.86961] | accepted |
| `wpA_admissibility_local_exponent_direction_polarization_min_eigenvalue_tapered_log_window` | minimum eigenvalue of the induced Fock polarization Gram (local_exponent_direction) | -3.3 | 2.708 | [-10.3881, -0.409049] | failed_gate |
| `wpA_admissibility_local_exponent_direction_symplectic_defect_tapered_log_window` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (local_exponent_direction) | 0.0000000000000046 | 5.994e-15 | [4.77428e-16, 2.29124e-14] | diagnostic |
| `wpA_admissibility_local_exponent_direction_condition_number_tapered_log_window` | cond(S) (local_exponent_direction) | 31 | 50.01 | [4.22386, 212.646] | accepted |
| `wpA_admissibility_non_symplectic_dense_mixing_star_compatibility_defect_tapered_log_window` | involution defect ||conj(S)C - CS||_F/||S||_F (non_symplectic_dense_mixing) | 1.420 | 0.03698 | [1.346, 1.49006] | accepted |
| `wpA_admissibility_non_symplectic_dense_mixing_polarization_min_eigenvalue_tapered_log_window` | minimum eigenvalue of the induced Fock polarization Gram (non_symplectic_dense_mixing) | -83 | 78.14 | [-383.073, -11.183] | failed_gate |
| `wpA_admissibility_non_symplectic_dense_mixing_symplectic_defect_tapered_log_window` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (non_symplectic_dense_mixing) | 0.000000000000013 | 1.039e-14 | [4.37995e-15, 3.8925e-14] | diagnostic |
| `wpA_admissibility_non_symplectic_dense_mixing_condition_number_tapered_log_window` | cond(S) (non_symplectic_dense_mixing) | 241 | 194.5 | [46.8143, 690.256] | accepted |
| `wpA_admissibility_phase_correlated_features_star_compatibility_defect_tapered_log_window` | involution defect ||conj(S)C - CS||_F/||S||_F (phase_correlated_features) | 0.30 | 0.3285 | [0.0322485, 1.44152] | accepted |
| `wpA_admissibility_phase_correlated_features_polarization_min_eigenvalue_tapered_log_window` | minimum eigenvalue of the induced Fock polarization Gram (phase_correlated_features) | 0.85 | 0.3868 | [-0.777287, 1.05876] | failed_gate |
| `wpA_admissibility_phase_correlated_features_symplectic_defect_tapered_log_window` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (phase_correlated_features) | 0.0000000000000013 | 2.813e-15 | [1.49634e-16, 1.33189e-14] | diagnostic |
| `wpA_admissibility_phase_correlated_features_condition_number_tapered_log_window` | cond(S) (phase_correlated_features) | 5 | 13.86 | [1.04474, 65.1936] | accepted |
| `wpA_admissibility_rbf_gaussian_prior_star_compatibility_defect_tapered_log_window` | involution defect ||conj(S)C - CS||_F/||S||_F (rbf_gaussian_prior) | 1.0871 | 0.009121 | [1.06556, 1.10345] | accepted |
| `wpA_admissibility_rbf_gaussian_prior_polarization_min_eigenvalue_tapered_log_window` | minimum eigenvalue of the induced Fock polarization Gram (rbf_gaussian_prior) | 0.702 | 0.01869 | [0.668943, 0.74762] | accepted |
| `wpA_admissibility_rbf_gaussian_prior_symplectic_defect_tapered_log_window` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (rbf_gaussian_prior) | 0.000000000000000127 | 5.341e-17 | [2.17172e-17, 2.54451e-16] | diagnostic |
| `wpA_admissibility_rbf_gaussian_prior_condition_number_tapered_log_window` | cond(S) (rbf_gaussian_prior) | 19.51 | 0.5253 | [18.2508, 20.3887] | accepted |
| `wpA_admissibility_selected_log_kernel_star_compatibility_defect_tapered_log_window` | involution defect ||conj(S)C - CS||_F/||S||_F (selected_log_kernel) | 0.30 | 0.4045 | [0.0164104, 1.45851] | accepted |
| `wpA_admissibility_selected_log_kernel_polarization_min_eigenvalue_tapered_log_window` | minimum eigenvalue of the induced Fock polarization Gram (selected_log_kernel) | 0.87 | 0.3845 | [-0.545052, 1.22745] | failed_gate |
| `wpA_admissibility_selected_log_kernel_symplectic_defect_tapered_log_window` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (selected_log_kernel) | 0.0000000000000008 | 1.761e-15 | [5.98002e-17, 8.32727e-15] | diagnostic |
| `wpA_admissibility_selected_log_kernel_condition_number_tapered_log_window` | cond(S) (selected_log_kernel) | 14 | 51.72 | [1.03939, 239.231] | accepted |
| `wpA_admissibility_sign_breaking_mode_reassignment_star_compatibility_defect_tapered_log_window` | involution defect ||conj(S)C - CS||_F/||S||_F (sign_breaking_mode_reassignment) | 1.41421356237000029 | 2.22e-16 | [1.41421, 1.41421] | accepted |
| `wpA_admissibility_sign_breaking_mode_reassignment_polarization_min_eigenvalue_tapered_log_window` | minimum eigenvalue of the induced Fock polarization Gram (sign_breaking_mode_reassignment) | -1.78 | 0.9318 | [-3.29041, -0.0165573] | failed_gate |
| `wpA_admissibility_sign_breaking_mode_reassignment_symplectic_defect_tapered_log_window` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (sign_breaking_mode_reassignment) | 0.00000000000000040 | 4.544e-16 | [1.4839e-16, 2.27267e-15] | diagnostic |
| `wpA_admissibility_sign_breaking_mode_reassignment_condition_number_tapered_log_window` | cond(S) (sign_breaking_mode_reassignment) | 2.2 | 3.043 | [1.07584, 15.1656] | accepted |
| `wpA_admissibility_vs_width_angular_mode_scramble_sharp_annulus_manuscript` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (angular_mode_scramble) | -0.000 | 0.01648 | [-0.0525952, 0.0522869] | accepted |
| `wpA_admissibility_vs_width_anisotropic_angular_prior_sharp_annulus_manuscript` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (anisotropic_angular_prior) | 0.034 | 0.01148 | [-0.00282949, 0.0702255] | accepted |
| `wpA_admissibility_vs_width_local_exponent_direction_sharp_annulus_manuscript` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (local_exponent_direction) | -0.032 | 0.01165 | [-0.0694173, 0.00476334] | accepted |
| `wpA_admissibility_vs_width_non_symplectic_dense_mixing_sharp_annulus_manuscript` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (non_symplectic_dense_mixing) | 0.0004 | 0.006682 | [-0.0208673, 0.0216607] | accepted |
| `wpA_admissibility_vs_width_phase_correlated_features_sharp_annulus_manuscript` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (phase_correlated_features) | -0.003 | 0.03269 | [-0.10704, 0.10105] | accepted |
| `wpA_admissibility_vs_width_rbf_gaussian_prior_sharp_annulus_manuscript` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (rbf_gaussian_prior) | 0.0020 | 0.0023 | [-0.00536201, 0.00927648] | accepted |
| `wpA_admissibility_vs_width_selected_log_kernel_sharp_annulus_manuscript` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (selected_log_kernel) | -0.035 | 0.01393 | [-0.0795322, 0.00912366] | accepted |
| `wpA_admissibility_vs_width_sign_breaking_mode_reassignment_sharp_annulus_manuscript` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (sign_breaking_mode_reassignment) | 0.000000000000000030 | 6.286e-17 | [-1.698e-16, 2.30288e-16] | accepted |
| `wpA_admissibility_vs_width_angular_mode_scramble_tapered_log_window` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (angular_mode_scramble) | -0.327 | 0.08769 | [-0.606064, -0.0479028] | accepted |
| `wpA_admissibility_vs_width_anisotropic_angular_prior_tapered_log_window` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (anisotropic_angular_prior) | -0.265 | 0.03656 | [-0.3814, -0.148698] | accepted |
| `wpA_admissibility_vs_width_local_exponent_direction_tapered_log_window` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (local_exponent_direction) | -0.0261 | 0.009836 | [-0.0574242, 0.00517959] | accepted |
| `wpA_admissibility_vs_width_non_symplectic_dense_mixing_tapered_log_window` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (non_symplectic_dense_mixing) | -0.0058 | 0.005296 | [-0.0226046, 0.0111045] | accepted |
| `wpA_admissibility_vs_width_phase_correlated_features_tapered_log_window` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (phase_correlated_features) | -0.606 | 0.04192 | [-0.739288, -0.472471] | accepted |
| `wpA_admissibility_vs_width_rbf_gaussian_prior_tapered_log_window` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (rbf_gaussian_prior) | 0.0017 | 0.001992 | [-0.00463633, 0.00804393] | accepted |
| `wpA_admissibility_vs_width_selected_log_kernel_tapered_log_window` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (selected_log_kernel) | -0.720 | 0.06705 | [-0.933366, -0.506629] | accepted |
| `wpA_admissibility_vs_width_sign_breaking_mode_reassignment_tapered_log_window` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (sign_breaking_mode_reassignment) | 0.000000000000000000 | 6.098e-17 | [-1.94071e-16, 1.94071e-16] | accepted |
| `wpE_offset_canonical_truncated_all` | alternating central offset A (canonical_truncated|all) | 0 | 0 | [0, 0] | accepted |
| `wpE_offset_canonical_truncated_all_m2to4` | alternating central offset A (canonical_truncated|all|m2to4) | 0 | 0 | [0, 0] | accepted |
| `wpG_edge_remainder_norms` | edge-remainder coefficient norms (max / L1 / L2) | max=16; L1=58; L2=29.0861 |  |  | accepted |
| `wpG_filtered_fock_exactness` | filtered-Fock exact-closure checks passed | 366/366 |  |  | accepted |
| `wp0_release_integrity` | frozen claim-ledger rows | 235 |  |  | accepted |

### 3.2 Analytic closed-kernel computations

Exact regulated Bessel kernel, no sampling. These rows isolate the **regulator systematic** from finite-`N` noise.

| claim | metric | value | uncertainty | 95% CI | status |
|---|---|---|---|---|---|
| `wpA_nonholomorphy_sup_sharp_annulus_manuscript` | sup |d_zbar K_JJ| on the contour tori | 744828 |  | [100.087, 744828] | accepted |
| `wpA_nonholomorphy_sup_tapered_log_window` | sup |d_zbar K_JJ| on the contour tori | 949.155 |  | [3.4362e-06, 949.155] | accepted |
| `wpC_analytic_closed_kernel_four_vertex` | analytic_closed_kernel stress Ward max absolute error (four_vertex) | 0.00875916 |  | [0.000309728, 0.00875916] | accepted |
| `wpC_analytic_closed_kernel_three_vertex` | analytic_closed_kernel stress Ward max absolute error (three_vertex) | 0.0122716 |  | [0.00413523, 0.0122716] | accepted |
| `wpC_analytic_closed_kernel_two_vertex` | analytic_closed_kernel stress Ward max absolute error (two_vertex) | 0.00157508 |  | [0.000713488, 0.00157508] | accepted |
| `wpF_current_reference_sharp_annulus_manuscript_1e-04_1e+04` | reference-window current-kernel relative error | 1.14196 |  |  | failed_gate |
| `wpF_current_reference_sharp_annulus_manuscript_1e-03_1e+03` | reference-window current-kernel relative error | 1.32261 |  |  | failed_gate |
| `wpF_current_reference_sharp_annulus_manuscript_1e-02_1e+02` | reference-window current-kernel relative error | 1.43391 |  |  | failed_gate |
| `wpF_current_reference_tapered_log_window_1e-04_1e+04` | reference-window current-kernel relative error | 0.0217769 |  |  | accepted |
| `wpF_current_reference_tapered_log_window_1e-03_1e+03` | reference-window current-kernel relative error | 0.0213353 |  |  | accepted |
| `wpF_current_reference_tapered_log_window_1e-02_1e+02` | reference-window current-kernel relative error | 0.0223458 |  |  | accepted |

### 3.3 Deterministic finite-feature computations

Fixed momentum sample, phases averaged analytically. Uncertainties are spreads over independent named seeds.

| claim | metric | value | uncertainty | 95% CI | status |
|---|---|---|---|---|---|
| `wpA_pairing_antidiag_error_sharp_annulus_manuscript` | anti_diagonal_amplitude_error at rho=2 over the R_0 grid | 33 | 84.55 | [0.598792, 514.896] | failed_gate |
| `wpA_pairing_antidiag_error_tapered_log_window` | anti_diagonal_amplitude_error at rho=2 over the R_0 grid | 12 | 29.34 | [0.109711, 163.73] | accepted |
| `wpA_radius_plateau_sharp_annulus_manuscript` | largest connected R_0 plateau (stated algorithm) | [0.0470998, 21.2315] | 10 nodes |  | accepted |
| `wpA_cauchy_green_bound_ratio_sharp_annulus_manuscript` | measured radius variation / Cauchy-Green bound | 1.62257e-05 | 0.0001461 | [1.39237e-07, 0.000788513] | accepted |
| `wpA_radius_plateau_tapered_log_window` | largest connected R_0 plateau (stated algorithm) | [0.0470998, 21.2315] | 10 nodes |  | accepted |
| `wpA_cauchy_green_bound_ratio_tapered_log_window` | measured radius variation / Cauchy-Green bound | 0.0173467 | 1.254 | [5.49375e-05, 7.43473] | accepted_with_caveat |
| `wpA_fit_pairing_leakage_sharp_annulus_manuscript` | width exponent b in log pairing_leakage = c + b log N + a log Lambda + regulator FE | -0.037 | 0.01911 | [-0.0748354, 0.000220657] | accepted |
| `wpA_fit_pairing_relative_error_sharp_annulus_manuscript` | width exponent b in log pairing_relative_error = c + b log N + a log Lambda + regulator FE | -0.456 | 0.01637 | [-0.488622, -0.424351] | accepted |
| `wpA_fit_pairing_sampling_error_sharp_annulus_manuscript` | width exponent b in log pairing_sampling_error = c + b log N + a log Lambda + regulator FE | -0.494 | 0.01613 | [-0.525497, -0.462157] | accepted |
| `wpA_fit_raw_virasoro_residual_sharp_annulus_manuscript` | width exponent b in log raw_virasoro_residual = c + b log N + a log Lambda + regulator FE | -0.570 | 0.01351 | [-0.596747, -0.543696] | accepted |
| `wpA_fit_pairing_leakage_tapered_log_window` | width exponent b in log pairing_leakage = c + b log N + a log Lambda + regulator FE | -0.067 | 0.01755 | [-0.101481, -0.0325481] | accepted |
| `wpA_fit_pairing_relative_error_tapered_log_window` | width exponent b in log pairing_relative_error = c + b log N + a log Lambda + regulator FE | -0.491 | 0.01573 | [-0.521882, -0.460109] | accepted |
| `wpA_fit_pairing_sampling_error_tapered_log_window` | width exponent b in log pairing_sampling_error = c + b log N + a log Lambda + regulator FE | -0.491 | 0.01573 | [-0.521946, -0.460188] | accepted |
| `wpA_fit_raw_virasoro_residual_tapered_log_window` | width exponent b in log raw_virasoro_residual = c + b log N + a log Lambda + regulator FE | -0.539 | 0.01078 | [-0.560069, -0.51772] | accepted |
| `wpA_projection_vs_two_radius_sharp_annulus_manuscript` | projection proxy leakage vs operational two-radius leakage | 0.898608 | 0.0963 | [0.637272, 0.991037] | accepted |
| `wpA_projection_vs_two_radius_tapered_log_window` | projection proxy leakage vs operational two-radius leakage | 0.903473 | 0.123 | [0.207917, 0.98579] | accepted |
| `wpC_deterministic_finite_feature_four_vertex` | deterministic_finite_feature stress Ward max absolute error (four_vertex) | 50.0139 |  | [0.00146951, 50.0139] | accepted |
| `wpC_deterministic_finite_feature_three_vertex` | deterministic_finite_feature stress Ward max absolute error (three_vertex) | 101.611 |  | [0.00250754, 101.611] | accepted |
| `wpC_deterministic_finite_feature_two_vertex` | deterministic_finite_feature stress Ward max absolute error (two_vertex) | 25.9123 |  | [0.000697343, 25.9123] | accepted |
| `wpD_raw_car_defect_sharp_annulus_manuscript` | raw CAR defect (N=300000) | 0.0573 | 0.007261 | [0.0452946, 0.0708863] | accepted |
| `wpD_post_canonicalization_car_defect_sharp_annulus_manuscript` | post-canonicalization CAR defect (N=300000) | 0.000000000000000220 | 1.242e-17 | [1.98814e-16, 2.41677e-16] | accepted |
| `wpD_raw_LG_residual_sharp_annulus_manuscript` | raw [L,G] residual (N=300000) | 1.54 | 0.1294 | [1.3162, 1.76148] | accepted |
| `wpD_raw_GG_residual_sharp_annulus_manuscript` | raw {G,G} residual (N=300000) | 0.664 | 0.05974 | [0.557046, 0.770376] | accepted |
| `wpD_canonical_LG_residual_sharp_annulus_manuscript` | canonical [L,G] residual (N=300000) | 0.0000000000000082 | 4.042e-15 | [3.19245e-15, 1.63098e-14] | accepted |
| `wpD_canonical_GG_residual_sharp_annulus_manuscript` | canonical {G,G} residual (N=300000) | 0.0000000000000031 | 1.896e-15 | [9.35082e-16, 6.89605e-15] | accepted |
| `wpD_central_charge_fit_sharp_annulus_manuscript` | NS central charge (N=300000) | 1.50000000000000000 | 1.404e-16 | [1.5, 1.5] | accepted |
| `wpD_raw_car_defect_tapered_log_window` | raw CAR defect (N=300000) | 0.0455 | 0.008699 | [0.0318914, 0.0633109] | accepted |
| `wpD_post_canonicalization_car_defect_tapered_log_window` | post-canonicalization CAR defect (N=300000) | 0.000000000000000202 | 2.948e-17 | [1.5211e-16, 2.56001e-16] | accepted |
| `wpD_raw_LG_residual_tapered_log_window` | raw [L,G] residual (N=300000) | 0.730 | 0.06812 | [0.605147, 0.84009] | accepted |
| `wpD_raw_GG_residual_tapered_log_window` | raw {G,G} residual (N=300000) | 0.248 | 0.02031 | [0.209788, 0.280635] | accepted |
| `wpD_canonical_LG_residual_tapered_log_window` | canonical [L,G] residual (N=300000) | 0.0000000000000030 | 1.324e-15 | [9.18349e-16, 5.58634e-15] | accepted |
| `wpD_canonical_GG_residual_tapered_log_window` | canonical {G,G} residual (N=300000) | 0.00000000000000127 | 5.745e-16 | [3.50717e-16, 2.38084e-15] | accepted |
| `wpD_central_charge_fit_tapered_log_window` | NS central charge (N=300000) | 1.500000000000000000 | 9.93e-17 | [1.5, 1.5] | accepted |
| `wpE_offset_raw_empirical_sharp_annulus_manuscript` | alternating central offset A (raw_empirical|sharp_annulus_manuscript) | 156666 | 4.028e+05 | [2526.18, 413192] | accepted_with_caveat |
| `wpE_offset_raw_empirical_tapered_log_window` | alternating central offset A (raw_empirical|tapered_log_window) | 22455 | 5.367e+04 | [1852.38, 56962.1] | accepted_with_caveat |
| `wpE_offset_raw_empirical_sharp_annulus_manuscript_m2to4` | alternating central offset A (raw_empirical|sharp_annulus_manuscript|m2to4) | 750 | 1169 | [8.15506, 2097.92] | accepted_with_caveat |
| `wpE_offset_raw_empirical_tapered_log_window_m2to4` | alternating central offset A (raw_empirical|tapered_log_window|m2to4) | 102 | 156.3 | [3.74675, 282.153] | accepted_with_caveat |

### 3.4 Sampled neural-ensemble measurements

Monte Carlo over independent network realizations. One block per named seed; the bootstrap resamples blocks, never individual realizations; regulator spread is never used as sampling uncertainty.

| claim | metric | value | uncertainty | 95% CI | status |
|---|---|---|---|---|---|
| `wpB_c_exp_sharp_annulus_manuscript_direct_TT_N250` | sampled <TT> central charge (direct_TT, N=250) | 5 | 11.1 | [-16.5265, 25.1352] | accepted |
| `wpB_c_exp_sharp_annulus_manuscript_direct_TT_N750` | sampled <TT> central charge (direct_TT, N=750) | 7.0 | 8.905 | [-9.68043, 23.4271] | accepted |
| `wpB_c_exp_sharp_annulus_manuscript_wick_factorized_N250` | sampled <TT> central charge (wick_factorized, N=250) | 1.22 | 0.1236 | [0.988032, 1.45194] | accepted |
| `wpB_c_exp_sharp_annulus_manuscript_wick_factorized_N750` | sampled <TT> central charge (wick_factorized, N=750) | 1.38 | 0.116 | [1.16076, 1.59549] | failed_gate |
| `wpB_c_exp_tapered_log_window_direct_TT_N250` | sampled <TT> central charge (direct_TT, N=250) | 2.06 | 0.5767 | [1.07046, 3.19656] | accepted |
| `wpB_c_exp_tapered_log_window_direct_TT_N750` | sampled <TT> central charge (direct_TT, N=750) | 0.75 | 0.6166 | [-0.446336, 1.8443] | accepted |
| `wpB_c_exp_tapered_log_window_wick_factorized_N250` | sampled <TT> central charge (wick_factorized, N=250) | 1.016 | 0.01125 | [0.993859, 1.03548] | accepted |
| `wpB_c_exp_tapered_log_window_wick_factorized_N750` | sampled <TT> central charge (wick_factorized, N=750) | 1.023 | 0.0168 | [0.992643, 1.05501] | accepted |
| `wpC_sampled_ward_four_vertex_sharp_annulus_manuscript` | sampled stress Ward max absolute error (four_vertex) | 41.3 | 5.608 | [32.6673, 52.4007] | accepted |
| `wpC_sampled_ward_three_vertex_sharp_annulus_manuscript` | sampled stress Ward max absolute error (three_vertex) | 175 | 16.32 | [143.233, 199.562] | accepted |
| `wpC_sampled_ward_two_vertex_sharp_annulus_manuscript` | sampled stress Ward max absolute error (two_vertex) | 40.4 | 4.698 | [31.9243, 48.4237] | accepted |
| `wpC_sampled_ward_four_vertex_tapered_log_window` | sampled stress Ward max absolute error (four_vertex) | 10.26 | 0.9527 | [8.50785, 11.9436] | accepted |
| `wpC_sampled_ward_three_vertex_tapered_log_window` | sampled stress Ward max absolute error (three_vertex) | 43.1 | 6.907 | [31.7876, 55.7212] | accepted |
| `wpC_sampled_ward_two_vertex_tapered_log_window` | sampled stress Ward max absolute error (two_vertex) | 8.51 | 0.648 | [7.34394, 9.60514] | accepted |
| `wpH_vertex_h_sharp_annulus_manuscript_a0.25_N1000` | sampled vertex weight h_a (a=0.25, N=1000) | 0.03254 | 0.000873 | [0.0309777, 0.0340378] | accepted |
| `wpH_vertex_h_sharp_annulus_manuscript_a0.25_N2000` | sampled vertex weight h_a (a=0.25, N=2000) | 0.03122 | 0.0004924 | [0.0304277, 0.0321732] | accepted |
| `wpH_vertex_h_sharp_annulus_manuscript_a0.25_N4000` | sampled vertex weight h_a (a=0.25, N=4000) | 0.03159 | 0.0004037 | [0.0310379, 0.0323933] | accepted |
| `wpH_vertex_h_sharp_annulus_manuscript_a0.4_N1000` | sampled vertex weight h_a (a=0.4, N=1000) | 0.0802 | 0.002965 | [0.075448, 0.0856342] | accepted |
| `wpH_vertex_h_sharp_annulus_manuscript_a0.4_N2000` | sampled vertex weight h_a (a=0.4, N=2000) | 0.0764 | 0.001793 | [0.0733222, 0.079272] | accepted_with_caveat |
| `wpH_vertex_h_sharp_annulus_manuscript_a0.4_N4000` | sampled vertex weight h_a (a=0.4, N=4000) | 0.0821 | 0.001608 | [0.0792522, 0.0848466] | accepted |
| `wpH_vertex_h_sharp_annulus_manuscript_a0.55_N1000` | sampled vertex weight h_a (a=0.55, N=1000) | 0.1511 | 0.009672 | [0.133003, 0.167223] | accepted |
| `wpH_vertex_h_sharp_annulus_manuscript_a0.55_N2000` | sampled vertex weight h_a (a=0.55, N=2000) | 0.1477 | 0.002411 | [0.14316, 0.151684] | accepted |
| `wpH_vertex_h_sharp_annulus_manuscript_a0.55_N4000` | sampled vertex weight h_a (a=0.55, N=4000) | 0.1637 | 0.008664 | [0.149317, 0.179018] | accepted |
| `wpH_vertex_h_sharp_annulus_manuscript_a0.7_N1000` | sampled vertex weight h_a (a=0.7, N=1000) | 0.236 | 0.02628 | [0.192715, 0.282617] | accepted |
| `wpH_vertex_h_sharp_annulus_manuscript_a0.7_N2000` | sampled vertex weight h_a (a=0.7, N=2000) | 0.170 | 0.03028 | [0.117087, 0.223837] | accepted_with_caveat |
| `wpH_vertex_h_sharp_annulus_manuscript_a0.7_N4000` | sampled vertex weight h_a (a=0.7, N=4000) | 0.305 | 0.04274 | [0.231466, 0.382932] | accepted |
| `wpH_vertex_h_tapered_log_window_a0.25_N1000` | sampled vertex weight h_a (a=0.25, N=1000) | 0.03184 | 0.0004615 | [0.0310398, 0.0326662] | accepted |
| `wpH_vertex_h_tapered_log_window_a0.25_N2000` | sampled vertex weight h_a (a=0.25, N=2000) | 0.03043 | 0.0005941 | [0.0293479, 0.031479] | accepted |
| `wpH_vertex_h_tapered_log_window_a0.25_N4000` | sampled vertex weight h_a (a=0.25, N=4000) | 0.03115 | 0.0003171 | [0.0305441, 0.0316748] | accepted |
| `wpH_vertex_h_tapered_log_window_a0.4_N1000` | sampled vertex weight h_a (a=0.4, N=1000) | 0.0831 | 0.003539 | [0.077183, 0.0897812] | accepted |
| `wpH_vertex_h_tapered_log_window_a0.4_N2000` | sampled vertex weight h_a (a=0.4, N=2000) | 0.0791 | 0.001092 | [0.0771412, 0.0810215] | accepted |
| `wpH_vertex_h_tapered_log_window_a0.4_N4000` | sampled vertex weight h_a (a=0.4, N=4000) | 0.0793 | 0.001024 | [0.0774534, 0.0810239] | accepted |
| `wpH_vertex_h_tapered_log_window_a0.55_N1000` | sampled vertex weight h_a (a=0.55, N=1000) | 0.1403 | 0.004771 | [0.132933, 0.149623] | accepted_with_caveat |
| `wpH_vertex_h_tapered_log_window_a0.55_N2000` | sampled vertex weight h_a (a=0.55, N=2000) | 0.1476 | 0.005738 | [0.138014, 0.158012] | accepted |
| `wpH_vertex_h_tapered_log_window_a0.55_N4000` | sampled vertex weight h_a (a=0.55, N=4000) | 0.1568 | 0.007533 | [0.14599, 0.171673] | accepted |
| `wpH_vertex_h_tapered_log_window_a0.7_N1000` | sampled vertex weight h_a (a=0.7, N=1000) | 0.324 | 0.04597 | [0.24163, 0.404858] | accepted |
| `wpH_vertex_h_tapered_log_window_a0.7_N2000` | sampled vertex weight h_a (a=0.7, N=2000) | 0.271 | 0.05441 | [0.189962, 0.379535] | accepted |
| `wpH_vertex_h_tapered_log_window_a0.7_N4000` | sampled vertex weight h_a (a=0.7, N=4000) | 0.299 | 0.07625 | [0.192178, 0.455915] | accepted |
| `wpH_cumulant_exponent_sharp_annulus_manuscript` | finite-width connected fourth-cumulant exponent p in |kappa_4| ~ N^p | -1.002 | 0.02131 | [-1.07012, -0.93449] | accepted |
| `wpH_cumulant_exponent_tapered_log_window` | finite-width connected fourth-cumulant exponent p in |kappa_4| ~ N^p | -0.9951 | 0.009225 | [-1.02449, -0.965774] | accepted |

### 3.5 Architecture-discriminating evidence

Radial discrimination is carried by the antidiagonal amplitude and the raw Sugawara residual; chiral/angular discrimination by the Ward identities. Under the *operational* two-radius definition the leakage of a rotationally invariant prior vanishes identically, so leakage discriminates angular, not radial, structure.

| claim | metric | value | uncertainty | 95% CI | status |
|---|---|---|---|---|---|
| `wpA_control_angular_mode_scramble_pre_darboux_leakage_sharp_annulus_manuscript` | pre_darboux_leakage (angular_mode_scramble, N=1000000) | 2.11 | 0.2835 | [1.78849, 2.56189] | accepted |
| `wpA_control_angular_mode_scramble_raw_virasoro_residual_sharp_annulus_manuscript` | raw_virasoro_residual (angular_mode_scramble, N=1000000) | 1.190 | 0.06107 | [1.0926, 1.26105] | accepted |
| `wpA_control_angular_mode_scramble_ward_stress_max_absolute_error_sharp_annulus_manuscript` | ward_stress_max_absolute_error (angular_mode_scramble, N=1000000) | 1.5 | 1.656 | [0.323983, 4.3519] | accepted |
| `wpA_control_anisotropic_angular_prior_pre_darboux_leakage_sharp_annulus_manuscript` | pre_darboux_leakage (anisotropic_angular_prior, N=1000000) | 1.15 | 0.185 | [0.841895, 1.30421] | accepted |
| `wpA_control_anisotropic_angular_prior_raw_virasoro_residual_sharp_annulus_manuscript` | raw_virasoro_residual (anisotropic_angular_prior, N=1000000) | 0.948 | 0.08188 | [0.830678, 1.06023] | accepted |
| `wpA_control_anisotropic_angular_prior_ward_stress_max_absolute_error_sharp_annulus_manuscript` | ward_stress_max_absolute_error (anisotropic_angular_prior, N=1000000) | 0.045 | 0.01291 | [0.0292979, 0.0612852] | accepted |
| `wpA_control_local_exponent_direction_pre_darboux_leakage_sharp_annulus_manuscript` | pre_darboux_leakage (local_exponent_direction, N=1000000) | 0.200 | 0.0529 | [0.153104, 0.285477] | accepted |
| `wpA_control_local_exponent_direction_raw_virasoro_residual_sharp_annulus_manuscript` | raw_virasoro_residual (local_exponent_direction, N=1000000) | 1.324 | 0.07434 | [1.19688, 1.37869] | accepted |
| `wpA_control_local_exponent_direction_ward_stress_max_absolute_error_sharp_annulus_manuscript` | ward_stress_max_absolute_error (local_exponent_direction, N=1000000) | 0.042 | 0.02031 | [0.022402, 0.0740747] | accepted |
| `wpA_control_non_symplectic_dense_mixing_pre_darboux_leakage_sharp_annulus_manuscript` | pre_darboux_leakage (non_symplectic_dense_mixing, N=1000000) | 2.49 | 0.6122 | [1.78772, 3.14128] | accepted |
| `wpA_control_non_symplectic_dense_mixing_raw_virasoro_residual_sharp_annulus_manuscript` | raw_virasoro_residual (non_symplectic_dense_mixing, N=1000000) | 13.0 | 2.769 | [9.59921, 17.0968] | accepted |
| `wpA_control_non_symplectic_dense_mixing_ward_stress_max_absolute_error_sharp_annulus_manuscript` | ward_stress_max_absolute_error (non_symplectic_dense_mixing, N=1000000) | 0.0303 | 0.009223 | [0.0187053, 0.0426603] | accepted |
| `wpA_control_phase_correlated_features_pre_darboux_leakage_sharp_annulus_manuscript` | pre_darboux_leakage (phase_correlated_features, N=1000000) | 0.224 | 0.08543 | [0.10316, 0.344531] | accepted |
| `wpA_control_phase_correlated_features_raw_virasoro_residual_sharp_annulus_manuscript` | raw_virasoro_residual (phase_correlated_features, N=1000000) | 0.741 | 0.05218 | [0.682805, 0.798443] | accepted |
| `wpA_control_phase_correlated_features_ward_stress_max_absolute_error_sharp_annulus_manuscript` | ward_stress_max_absolute_error (phase_correlated_features, N=1000000) | 0.0141 | 0.003638 | [0.0095557, 0.0193604] | accepted |
| `wpA_control_rbf_gaussian_prior_pre_darboux_leakage_sharp_annulus_manuscript` | pre_darboux_leakage (rbf_gaussian_prior, N=1000000) | 0.00138 | 0.0005457 | [0.000740754, 0.00201364] | accepted |
| `wpA_control_rbf_gaussian_prior_raw_virasoro_residual_sharp_annulus_manuscript` | raw_virasoro_residual (rbf_gaussian_prior, N=1000000) | 1.2990 | 0.001605 | [1.29789, 1.3018] | accepted |
| `wpA_control_rbf_gaussian_prior_ward_stress_max_absolute_error_sharp_annulus_manuscript` | ward_stress_max_absolute_error (rbf_gaussian_prior, N=1000000) | 0.8916 | 0.002289 | [0.888511, 0.893959] | accepted |
| `wpA_control_selected_log_kernel_pre_darboux_leakage_sharp_annulus_manuscript` | pre_darboux_leakage (selected_log_kernel, N=1000000) | 0.20 | 0.1207 | [0.0576366, 0.379661] | accepted |
| `wpA_control_selected_log_kernel_raw_virasoro_residual_sharp_annulus_manuscript` | raw_virasoro_residual (selected_log_kernel, N=1000000) | 0.76 | 0.1007 | [0.626191, 0.883034] | accepted |
| `wpA_control_selected_log_kernel_ward_stress_max_absolute_error_sharp_annulus_manuscript` | ward_stress_max_absolute_error (selected_log_kernel, N=1000000) | 0.0221 | 0.008566 | [0.0123742, 0.0358395] | accepted |
| `wpA_control_sign_breaking_mode_reassignment_raw_virasoro_residual_sharp_annulus_manuscript` | raw_virasoro_residual (sign_breaking_mode_reassignment, N=1000000) | 1.1124 | 0.009846 | [1.0988, 1.12572] | accepted |
| `wpA_control_sign_breaking_mode_reassignment_ward_stress_max_absolute_error_sharp_annulus_manuscript` | ward_stress_max_absolute_error (sign_breaking_mode_reassignment, N=1000000) | 0.0301 | 0.004822 | [0.022988, 0.0361081] | accepted |
| `wpA_control_angular_mode_scramble_pre_darboux_leakage_tapered_log_window` | pre_darboux_leakage (angular_mode_scramble, N=1000000) | 0.42 | 0.158 | [0.167932, 0.576687] | accepted |
| `wpA_control_angular_mode_scramble_raw_virasoro_residual_tapered_log_window` | raw_virasoro_residual (angular_mode_scramble, N=1000000) | 0.58 | 0.2196 | [0.22999, 0.804029] | accepted |
| `wpA_control_angular_mode_scramble_ward_stress_max_absolute_error_tapered_log_window` | ward_stress_max_absolute_error (angular_mode_scramble, N=1000000) | 0.60 | 0.421 | [0.0719183, 1.23668] | accepted |
| `wpA_control_anisotropic_angular_prior_pre_darboux_leakage_tapered_log_window` | pre_darboux_leakage (anisotropic_angular_prior, N=1000000) | 0.317 | 0.01308 | [0.299873, 0.336503] | accepted |
| `wpA_control_anisotropic_angular_prior_raw_virasoro_residual_tapered_log_window` | raw_virasoro_residual (anisotropic_angular_prior, N=1000000) | 0.4091 | 0.009969 | [0.396613, 0.42218] | accepted |
| `wpA_control_anisotropic_angular_prior_ward_stress_max_absolute_error_tapered_log_window` | ward_stress_max_absolute_error (anisotropic_angular_prior, N=1000000) | 0.0538 | 0.009056 | [0.0426273, 0.0661511] | accepted |
| `wpA_control_local_exponent_direction_pre_darboux_leakage_tapered_log_window` | pre_darboux_leakage (local_exponent_direction, N=1000000) | 0.200 | 0.0529 | [0.153104, 0.285477] | accepted |
| `wpA_control_local_exponent_direction_raw_virasoro_residual_tapered_log_window` | raw_virasoro_residual (local_exponent_direction, N=1000000) | 1.268 | 0.06383 | [1.15954, 1.3173] | accepted |
| `wpA_control_local_exponent_direction_ward_stress_max_absolute_error_tapered_log_window` | ward_stress_max_absolute_error (local_exponent_direction, N=1000000) | 0.026 | 0.01041 | [0.0150839, 0.0430425] | accepted |
| `wpA_control_non_symplectic_dense_mixing_pre_darboux_leakage_tapered_log_window` | pre_darboux_leakage (non_symplectic_dense_mixing, N=1000000) | 3.2 | 1.841 | [1.73615, 6.34626] | accepted |
| `wpA_control_non_symplectic_dense_mixing_raw_virasoro_residual_tapered_log_window` | raw_virasoro_residual (non_symplectic_dense_mixing, N=1000000) | 119 | 59.3 | [69.5794, 218.815] | accepted |
| `wpA_control_non_symplectic_dense_mixing_ward_stress_max_absolute_error_tapered_log_window` | ward_stress_max_absolute_error (non_symplectic_dense_mixing, N=1000000) | 0.0119 | 0.00381 | [0.00883533, 0.0182936] | accepted |
| `wpA_control_phase_correlated_features_pre_darboux_leakage_tapered_log_window` | pre_darboux_leakage (phase_correlated_features, N=1000000) | 0.0279 | 0.006434 | [0.0175591, 0.0345504] | accepted |
| `wpA_control_phase_correlated_features_raw_virasoro_residual_tapered_log_window` | raw_virasoro_residual (phase_correlated_features, N=1000000) | 0.066 | 0.01611 | [0.0462337, 0.0842219] | accepted |
| `wpA_control_phase_correlated_features_ward_stress_max_absolute_error_tapered_log_window` | ward_stress_max_absolute_error (phase_correlated_features, N=1000000) | 0.0099 | 0.003799 | [0.00480719, 0.0152858] | accepted |
| `wpA_control_rbf_gaussian_prior_pre_darboux_leakage_tapered_log_window` | pre_darboux_leakage (rbf_gaussian_prior, N=1000000) | 0.00138 | 0.0005457 | [0.000740754, 0.00201364] | accepted |
| `wpA_control_rbf_gaussian_prior_raw_virasoro_residual_tapered_log_window` | raw_virasoro_residual (rbf_gaussian_prior, N=1000000) | 1.0472 | 0.001017 | [1.04639, 1.04894] | accepted |
| `wpA_control_rbf_gaussian_prior_ward_stress_max_absolute_error_tapered_log_window` | ward_stress_max_absolute_error (rbf_gaussian_prior, N=1000000) | 0.6334 | 0.001657 | [0.631156, 0.635113] | accepted |
| `wpA_control_selected_log_kernel_pre_darboux_leakage_tapered_log_window` | pre_darboux_leakage (selected_log_kernel, N=1000000) | 0.0287 | 0.009917 | [0.0138062, 0.0379109] | accepted |
| `wpA_control_selected_log_kernel_raw_virasoro_residual_tapered_log_window` | raw_virasoro_residual (selected_log_kernel, N=1000000) | 0.059 | 0.02623 | [0.0339549, 0.102859] | accepted |
| `wpA_control_selected_log_kernel_ward_stress_max_absolute_error_tapered_log_window` | ward_stress_max_absolute_error (selected_log_kernel, N=1000000) | 0.0128 | 0.002251 | [0.0090993, 0.0148287] | accepted |
| `wpA_control_sign_breaking_mode_reassignment_raw_virasoro_residual_tapered_log_window` | raw_virasoro_residual (sign_breaking_mode_reassignment, N=1000000) | 1.381 | 0.03072 | [1.33771, 1.42362] | accepted |
| `wpA_control_sign_breaking_mode_reassignment_ward_stress_max_absolute_error_tapered_log_window` | ward_stress_max_absolute_error (sign_breaking_mode_reassignment, N=1000000) | 0.0114 | 0.002789 | [0.00718289, 0.0149167] | accepted |

### 3.6 Post-Darboux implementation checks (non-discriminating)

Proposition 3.5 makes symplectic closure universal for nondegenerate antisymmetric forms, so these rows are **not** architecture evidence. The operator-level discriminators are the involution defect `||conj(S)C - CS||_F/||S||_F` and the positivity of the induced Fock polarization; unrestricted Darboux equivalence does not imply unitary equivalence.

| claim | metric | value | uncertainty | 95% CI | status |
|---|---|---|---|---|---|
| `wpA_control_angular_mode_scramble_post_darboux_virasoro_residual_sharp_annulus_manuscript` | post_darboux_virasoro_residual (angular_mode_scramble, N=1000000) | 0.00000000000000045 | 1.247e-16 | [2.76174e-16, 6.19272e-16] | diagnostic |
| `wpA_control_anisotropic_angular_prior_post_darboux_virasoro_residual_sharp_annulus_manuscript` | post_darboux_virasoro_residual (anisotropic_angular_prior, N=1000000) | 0.00000000000000035 | 1.049e-16 | [2.00115e-16, 4.6084e-16] | diagnostic |
| `wpA_control_local_exponent_direction_post_darboux_virasoro_residual_sharp_annulus_manuscript` | post_darboux_virasoro_residual (local_exponent_direction, N=1000000) | 0.000000000000000289 | 4.594e-17 | [2.30913e-16, 3.43434e-16] | diagnostic |
| `wpA_control_non_symplectic_dense_mixing_post_darboux_virasoro_residual_sharp_annulus_manuscript` | post_darboux_virasoro_residual (non_symplectic_dense_mixing, N=1000000) | 0.000000000000016 | 1.378e-14 | [4.68453e-15, 3.88901e-14] | diagnostic |
| `wpA_control_phase_correlated_features_post_darboux_virasoro_residual_sharp_annulus_manuscript` | post_darboux_virasoro_residual (phase_correlated_features, N=1000000) | 0.00000000000000035 | 3.528e-16 | [9.02762e-17, 9.58054e-16] | diagnostic |
| `wpA_control_rbf_gaussian_prior_post_darboux_virasoro_residual_sharp_annulus_manuscript` | post_darboux_virasoro_residual (rbf_gaussian_prior, N=1000000) | 0.000000000000000069 | 3.958e-17 | [2.11293e-18, 9.80433e-17] | diagnostic |
| `wpA_control_selected_log_kernel_post_darboux_virasoro_residual_sharp_annulus_manuscript` | post_darboux_virasoro_residual (selected_log_kernel, N=1000000) | 0.00000000000000035 | 3.733e-16 | [1.06667e-16, 9.95703e-16] | diagnostic |
| `wpA_control_sign_breaking_mode_reassignment_post_darboux_virasoro_residual_sharp_annulus_manuscript` | post_darboux_virasoro_residual (sign_breaking_mode_reassignment, N=1000000) | 0.000000000000000334 | 7.436e-17 | [2.63507e-16, 4.58176e-16] | diagnostic |
| `wpA_control_angular_mode_scramble_post_darboux_virasoro_residual_tapered_log_window` | post_darboux_virasoro_residual (angular_mode_scramble, N=1000000) | 0.000000000000000143 | 7.214e-17 | [4.88898e-17, 2.46559e-16] | diagnostic |
| `wpA_control_anisotropic_angular_prior_post_darboux_virasoro_residual_tapered_log_window` | post_darboux_virasoro_residual (anisotropic_angular_prior, N=1000000) | 0.000000000000000104 | 4.553e-17 | [6.16765e-17, 1.80537e-16] | diagnostic |
| `wpA_control_local_exponent_direction_post_darboux_virasoro_residual_tapered_log_window` | post_darboux_virasoro_residual (local_exponent_direction, N=1000000) | 0.000000000000000298 | 3.742e-17 | [2.38533e-16, 3.30748e-16] | diagnostic |
| `wpA_control_non_symplectic_dense_mixing_post_darboux_virasoro_residual_tapered_log_window` | post_darboux_virasoro_residual (non_symplectic_dense_mixing, N=1000000) | 0.000000000000032 | 2.199e-14 | [6.03608e-15, 6.2411e-14] | diagnostic |
| `wpA_control_phase_correlated_features_post_darboux_virasoro_residual_tapered_log_window` | post_darboux_virasoro_residual (phase_correlated_features, N=1000000) | 0.000000000000000128 | 4.439e-17 | [6.52811e-17, 1.90377e-16] | diagnostic |
| `wpA_control_rbf_gaussian_prior_post_darboux_virasoro_residual_tapered_log_window` | post_darboux_virasoro_residual (rbf_gaussian_prior, N=1000000) | 0.000000000000000012 | 1.227e-17 | [2.01329e-18, 3.21274e-17] | diagnostic |
| `wpA_control_selected_log_kernel_post_darboux_virasoro_residual_tapered_log_window` | post_darboux_virasoro_residual (selected_log_kernel, N=1000000) | 0.0000000000000000605 | 8.608e-18 | [5.15641e-17, 7.4357e-17] | diagnostic |
| `wpA_control_sign_breaking_mode_reassignment_post_darboux_virasoro_residual_tapered_log_window` | post_darboux_virasoro_residual (sign_breaking_mode_reassignment, N=1000000) | 0.000000000000000120 | 2.306e-17 | [8.25661e-17, 1.45111e-16] | diagnostic |
| `wpA_admissibility_angular_mode_scramble_star_compatibility_defect_sharp_annulus_manuscript` | involution defect ||conj(S)C - CS||_F/||S||_F (angular_mode_scramble) | 1.21 | 0.3066 | [0.414471, 1.51509] | accepted |
| `wpA_admissibility_angular_mode_scramble_polarization_min_eigenvalue_sharp_annulus_manuscript` | minimum eigenvalue of the induced Fock polarization Gram (angular_mode_scramble) | -1.8 | 2.331 | [-7.47668, 1.07631] | failed_gate |
| `wpA_admissibility_angular_mode_scramble_symplectic_defect_sharp_annulus_manuscript` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (angular_mode_scramble) | 0.0000000000000019 | 1.638e-15 | [3.37684e-16, 7.47767e-15] | diagnostic |
| `wpA_admissibility_angular_mode_scramble_condition_number_sharp_annulus_manuscript` | cond(S) (angular_mode_scramble) | 13 | 10.62 | [2.25562, 37.3049] | accepted |
| `wpA_admissibility_anisotropic_angular_prior_star_compatibility_defect_sharp_annulus_manuscript` | involution defect ||conj(S)C - CS||_F/||S||_F (anisotropic_angular_prior) | 1.27 | 0.3134 | [0.441239, 1.74602] | accepted |
| `wpA_admissibility_anisotropic_angular_prior_polarization_min_eigenvalue_sharp_annulus_manuscript` | minimum eigenvalue of the induced Fock polarization Gram (anisotropic_angular_prior) | -1.5 | 1.521 | [-4.94933, 0.785492] | failed_gate |
| `wpA_admissibility_anisotropic_angular_prior_symplectic_defect_sharp_annulus_manuscript` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (anisotropic_angular_prior) | 0.0000000000000027 | 2.842e-15 | [3.82808e-16, 1.08573e-14] | diagnostic |
| `wpA_admissibility_anisotropic_angular_prior_condition_number_sharp_annulus_manuscript` | cond(S) (anisotropic_angular_prior) | 15 | 12.51 | [2.2561, 43.915] | accepted |
| `wpA_admissibility_local_exponent_direction_star_compatibility_defect_sharp_annulus_manuscript` | involution defect ||conj(S)C - CS||_F/||S||_F (local_exponent_direction) | 1.35 | 0.2161 | [0.996042, 1.8959] | accepted |
| `wpA_admissibility_local_exponent_direction_polarization_min_eigenvalue_sharp_annulus_manuscript` | minimum eigenvalue of the induced Fock polarization Gram (local_exponent_direction) | -3.9 | 3.186 | [-12.2213, -0.481234] | failed_gate |
| `wpA_admissibility_local_exponent_direction_symplectic_defect_sharp_annulus_manuscript` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (local_exponent_direction) | 0.0000000000000042 | 5.577e-15 | [4.97555e-16, 2.17175e-14] | diagnostic |
| `wpA_admissibility_local_exponent_direction_condition_number_sharp_annulus_manuscript` | cond(S) (local_exponent_direction) | 26 | 42.44 | [4.9624, 181.153] | accepted |
| `wpA_admissibility_non_symplectic_dense_mixing_star_compatibility_defect_sharp_annulus_manuscript` | involution defect ||conj(S)C - CS||_F/||S||_F (non_symplectic_dense_mixing) | 1.417 | 0.07481 | [1.22894, 1.5564] | accepted |
| `wpA_admissibility_non_symplectic_dense_mixing_polarization_min_eigenvalue_sharp_annulus_manuscript` | minimum eigenvalue of the induced Fock polarization Gram (non_symplectic_dense_mixing) | -35 | 20.91 | [-76.9072, -10.0332] | failed_gate |
| `wpA_admissibility_non_symplectic_dense_mixing_symplectic_defect_sharp_annulus_manuscript` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (non_symplectic_dense_mixing) | 0.000000000000031 | 5.716e-14 | [1.96867e-15, 2.15365e-13] | diagnostic |
| `wpA_admissibility_non_symplectic_dense_mixing_condition_number_sharp_annulus_manuscript` | cond(S) (non_symplectic_dense_mixing) | 136 | 136 | [23.7021, 413.272] | accepted |
| `wpA_admissibility_phase_correlated_features_star_compatibility_defect_sharp_annulus_manuscript` | involution defect ||conj(S)C - CS||_F/||S||_F (phase_correlated_features) | 1.17 | 0.264 | [0.492764, 1.49907] | accepted |
| `wpA_admissibility_phase_correlated_features_polarization_min_eigenvalue_sharp_annulus_manuscript` | minimum eigenvalue of the induced Fock polarization Gram (phase_correlated_features) | -0.7 | 1.695 | [-5.16564, 0.96438] | failed_gate |
| `wpA_admissibility_phase_correlated_features_symplectic_defect_sharp_annulus_manuscript` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (phase_correlated_features) | 0.000000000000008 | 2.692e-14 | [3.11151e-16, 1.2535e-13] | diagnostic |
| `wpA_admissibility_phase_correlated_features_condition_number_sharp_annulus_manuscript` | cond(S) (phase_correlated_features) | 24 | 63.42 | [2.14402, 298.847] | accepted |
| `wpA_admissibility_rbf_gaussian_prior_star_compatibility_defect_sharp_annulus_manuscript` | involution defect ||conj(S)C - CS||_F/||S||_F (rbf_gaussian_prior) | 1.036 | 0.01004 | [1.01277, 1.05441] | accepted |
| `wpA_admissibility_rbf_gaussian_prior_polarization_min_eigenvalue_sharp_annulus_manuscript` | minimum eigenvalue of the induced Fock polarization Gram (rbf_gaussian_prior) | 0.826 | 0.02198 | [0.786992, 0.879553] | accepted |
| `wpA_admissibility_rbf_gaussian_prior_symplectic_defect_sharp_annulus_manuscript` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (rbf_gaussian_prior) | 0.000000000000000152 | 5.188e-17 | [6.41406e-17, 2.50014e-16] | diagnostic |
| `wpA_admissibility_rbf_gaussian_prior_condition_number_sharp_annulus_manuscript` | cond(S) (rbf_gaussian_prior) | 19.51 | 0.5253 | [18.2507, 20.3887] | accepted |
| `wpA_admissibility_selected_log_kernel_star_compatibility_defect_sharp_annulus_manuscript` | involution defect ||conj(S)C - CS||_F/||S||_F (selected_log_kernel) | 1.28 | 0.1822 | [0.915098, 1.51546] | accepted |
| `wpA_admissibility_selected_log_kernel_polarization_min_eigenvalue_sharp_annulus_manuscript` | minimum eigenvalue of the induced Fock polarization Gram (selected_log_kernel) | -1.6 | 2.321 | [-8.63437, 0.72419] | failed_gate |
| `wpA_admissibility_selected_log_kernel_symplectic_defect_sharp_annulus_manuscript` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (selected_log_kernel) | 0.000000000000005 | 1.002e-14 | [3.37929e-16, 4.52401e-14] | diagnostic |
| `wpA_admissibility_selected_log_kernel_condition_number_sharp_annulus_manuscript` | cond(S) (selected_log_kernel) | 24 | 32.69 | [2.41811, 136.632] | accepted |
| `wpA_admissibility_sign_breaking_mode_reassignment_star_compatibility_defect_sharp_annulus_manuscript` | involution defect ||conj(S)C - CS||_F/||S||_F (sign_breaking_mode_reassignment) | 1.41421356237000029 | 2.22e-16 | [1.41421, 1.41421] | accepted |
| `wpA_admissibility_sign_breaking_mode_reassignment_polarization_min_eigenvalue_sharp_annulus_manuscript` | minimum eigenvalue of the induced Fock polarization Gram (sign_breaking_mode_reassignment) | -1.07 | 0.9143 | [-3.01099, -0.0432872] | failed_gate |
| `wpA_admissibility_sign_breaking_mode_reassignment_symplectic_defect_sharp_annulus_manuscript` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (sign_breaking_mode_reassignment) | 0.0000000000000028 | 8.029e-15 | [2.6406e-16, 3.77207e-14] | diagnostic |
| `wpA_admissibility_sign_breaking_mode_reassignment_condition_number_sharp_annulus_manuscript` | cond(S) (sign_breaking_mode_reassignment) | 6.3 | 6.953 | [1.89882, 34.9889] | accepted |
| `wpA_admissibility_angular_mode_scramble_star_compatibility_defect_tapered_log_window` | involution defect ||conj(S)C - CS||_F/||S||_F (angular_mode_scramble) | 0.52 | 0.4313 | [0.10882, 1.42304] | accepted |
| `wpA_admissibility_angular_mode_scramble_polarization_min_eigenvalue_tapered_log_window` | minimum eigenvalue of the induced Fock polarization Gram (angular_mode_scramble) | 0.66 | 0.5407 | [-0.827925, 1.05136] | failed_gate |
| `wpA_admissibility_angular_mode_scramble_symplectic_defect_tapered_log_window` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (angular_mode_scramble) | 0.00000000000000063 | 8.749e-16 | [1.06636e-16, 3.87714e-15] | diagnostic |
| `wpA_admissibility_angular_mode_scramble_condition_number_tapered_log_window` | cond(S) (angular_mode_scramble) | 4.9 | 6.06 | [1.25465, 20.7742] | accepted |
| `wpA_admissibility_anisotropic_angular_prior_star_compatibility_defect_tapered_log_window` | involution defect ||conj(S)C - CS||_F/||S||_F (anisotropic_angular_prior) | 0.40 | 0.283 | [0.215202, 1.2096] | accepted |
| `wpA_admissibility_anisotropic_angular_prior_polarization_min_eigenvalue_tapered_log_window` | minimum eigenvalue of the induced Fock polarization Gram (anisotropic_angular_prior) | 0.68 | 0.7981 | [-2.20857, 1.13296] | failed_gate |
| `wpA_admissibility_anisotropic_angular_prior_symplectic_defect_tapered_log_window` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (anisotropic_angular_prior) | 0.00000000000000037 | 2.048e-16 | [1.35001e-16, 9.62176e-16] | diagnostic |
| `wpA_admissibility_anisotropic_angular_prior_condition_number_tapered_log_window` | cond(S) (anisotropic_angular_prior) | 2.3 | 1.352 | [1.39343, 7.09378] | accepted |
| `wpA_admissibility_local_exponent_direction_star_compatibility_defect_tapered_log_window` | involution defect ||conj(S)C - CS||_F/||S||_F (local_exponent_direction) | 1.35 | 0.1974 | [1.00462, 1.86961] | accepted |
| `wpA_admissibility_local_exponent_direction_polarization_min_eigenvalue_tapered_log_window` | minimum eigenvalue of the induced Fock polarization Gram (local_exponent_direction) | -3.3 | 2.708 | [-10.3881, -0.409049] | failed_gate |
| `wpA_admissibility_local_exponent_direction_symplectic_defect_tapered_log_window` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (local_exponent_direction) | 0.0000000000000046 | 5.994e-15 | [4.77428e-16, 2.29124e-14] | diagnostic |
| `wpA_admissibility_local_exponent_direction_condition_number_tapered_log_window` | cond(S) (local_exponent_direction) | 31 | 50.01 | [4.22386, 212.646] | accepted |
| `wpA_admissibility_non_symplectic_dense_mixing_star_compatibility_defect_tapered_log_window` | involution defect ||conj(S)C - CS||_F/||S||_F (non_symplectic_dense_mixing) | 1.420 | 0.03698 | [1.346, 1.49006] | accepted |
| `wpA_admissibility_non_symplectic_dense_mixing_polarization_min_eigenvalue_tapered_log_window` | minimum eigenvalue of the induced Fock polarization Gram (non_symplectic_dense_mixing) | -83 | 78.14 | [-383.073, -11.183] | failed_gate |
| `wpA_admissibility_non_symplectic_dense_mixing_symplectic_defect_tapered_log_window` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (non_symplectic_dense_mixing) | 0.000000000000013 | 1.039e-14 | [4.37995e-15, 3.8925e-14] | diagnostic |
| `wpA_admissibility_non_symplectic_dense_mixing_condition_number_tapered_log_window` | cond(S) (non_symplectic_dense_mixing) | 241 | 194.5 | [46.8143, 690.256] | accepted |
| `wpA_admissibility_phase_correlated_features_star_compatibility_defect_tapered_log_window` | involution defect ||conj(S)C - CS||_F/||S||_F (phase_correlated_features) | 0.30 | 0.3285 | [0.0322485, 1.44152] | accepted |
| `wpA_admissibility_phase_correlated_features_polarization_min_eigenvalue_tapered_log_window` | minimum eigenvalue of the induced Fock polarization Gram (phase_correlated_features) | 0.85 | 0.3868 | [-0.777287, 1.05876] | failed_gate |
| `wpA_admissibility_phase_correlated_features_symplectic_defect_tapered_log_window` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (phase_correlated_features) | 0.0000000000000013 | 2.813e-15 | [1.49634e-16, 1.33189e-14] | diagnostic |
| `wpA_admissibility_phase_correlated_features_condition_number_tapered_log_window` | cond(S) (phase_correlated_features) | 5 | 13.86 | [1.04474, 65.1936] | accepted |
| `wpA_admissibility_rbf_gaussian_prior_star_compatibility_defect_tapered_log_window` | involution defect ||conj(S)C - CS||_F/||S||_F (rbf_gaussian_prior) | 1.0871 | 0.009121 | [1.06556, 1.10345] | accepted |
| `wpA_admissibility_rbf_gaussian_prior_polarization_min_eigenvalue_tapered_log_window` | minimum eigenvalue of the induced Fock polarization Gram (rbf_gaussian_prior) | 0.702 | 0.01869 | [0.668943, 0.74762] | accepted |
| `wpA_admissibility_rbf_gaussian_prior_symplectic_defect_tapered_log_window` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (rbf_gaussian_prior) | 0.000000000000000127 | 5.341e-17 | [2.17172e-17, 2.54451e-16] | diagnostic |
| `wpA_admissibility_rbf_gaussian_prior_condition_number_tapered_log_window` | cond(S) (rbf_gaussian_prior) | 19.51 | 0.5253 | [18.2508, 20.3887] | accepted |
| `wpA_admissibility_selected_log_kernel_star_compatibility_defect_tapered_log_window` | involution defect ||conj(S)C - CS||_F/||S||_F (selected_log_kernel) | 0.30 | 0.4045 | [0.0164104, 1.45851] | accepted |
| `wpA_admissibility_selected_log_kernel_polarization_min_eigenvalue_tapered_log_window` | minimum eigenvalue of the induced Fock polarization Gram (selected_log_kernel) | 0.87 | 0.3845 | [-0.545052, 1.22745] | failed_gate |
| `wpA_admissibility_selected_log_kernel_symplectic_defect_tapered_log_window` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (selected_log_kernel) | 0.0000000000000008 | 1.761e-15 | [5.98002e-17, 8.32727e-15] | diagnostic |
| `wpA_admissibility_selected_log_kernel_condition_number_tapered_log_window` | cond(S) (selected_log_kernel) | 14 | 51.72 | [1.03939, 239.231] | accepted |
| `wpA_admissibility_sign_breaking_mode_reassignment_star_compatibility_defect_tapered_log_window` | involution defect ||conj(S)C - CS||_F/||S||_F (sign_breaking_mode_reassignment) | 1.41421356237000029 | 2.22e-16 | [1.41421, 1.41421] | accepted |
| `wpA_admissibility_sign_breaking_mode_reassignment_polarization_min_eigenvalue_tapered_log_window` | minimum eigenvalue of the induced Fock polarization Gram (sign_breaking_mode_reassignment) | -1.78 | 0.9318 | [-3.29041, -0.0165573] | failed_gate |
| `wpA_admissibility_sign_breaking_mode_reassignment_symplectic_defect_tapered_log_window` | symplectic defect ||S Omega S^T - Omega_can||_F/||Omega_can||_F (sign_breaking_mode_reassignment) | 0.00000000000000040 | 4.544e-16 | [1.4839e-16, 2.27267e-15] | diagnostic |
| `wpA_admissibility_sign_breaking_mode_reassignment_condition_number_tapered_log_window` | cond(S) (sign_breaking_mode_reassignment) | 2.2 | 3.043 | [1.07584, 15.1656] | accepted |
| `wpA_admissibility_vs_width_angular_mode_scramble_sharp_annulus_manuscript` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (angular_mode_scramble) | -0.000 | 0.01648 | [-0.0525952, 0.0522869] | accepted |
| `wpA_admissibility_vs_width_anisotropic_angular_prior_sharp_annulus_manuscript` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (anisotropic_angular_prior) | 0.034 | 0.01148 | [-0.00282949, 0.0702255] | accepted |
| `wpA_admissibility_vs_width_local_exponent_direction_sharp_annulus_manuscript` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (local_exponent_direction) | -0.032 | 0.01165 | [-0.0694173, 0.00476334] | accepted |
| `wpA_admissibility_vs_width_non_symplectic_dense_mixing_sharp_annulus_manuscript` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (non_symplectic_dense_mixing) | 0.0004 | 0.006682 | [-0.0208673, 0.0216607] | accepted |
| `wpA_admissibility_vs_width_phase_correlated_features_sharp_annulus_manuscript` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (phase_correlated_features) | -0.003 | 0.03269 | [-0.10704, 0.10105] | accepted |
| `wpA_admissibility_vs_width_rbf_gaussian_prior_sharp_annulus_manuscript` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (rbf_gaussian_prior) | 0.0020 | 0.0023 | [-0.00536201, 0.00927648] | accepted |
| `wpA_admissibility_vs_width_selected_log_kernel_sharp_annulus_manuscript` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (selected_log_kernel) | -0.035 | 0.01393 | [-0.0795322, 0.00912366] | accepted |
| `wpA_admissibility_vs_width_sign_breaking_mode_reassignment_sharp_annulus_manuscript` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (sign_breaking_mode_reassignment) | 0.000000000000000030 | 6.286e-17 | [-1.698e-16, 2.30288e-16] | accepted |
| `wpA_admissibility_vs_width_angular_mode_scramble_tapered_log_window` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (angular_mode_scramble) | -0.327 | 0.08769 | [-0.606064, -0.0479028] | accepted |
| `wpA_admissibility_vs_width_anisotropic_angular_prior_tapered_log_window` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (anisotropic_angular_prior) | -0.265 | 0.03656 | [-0.3814, -0.148698] | accepted |
| `wpA_admissibility_vs_width_local_exponent_direction_tapered_log_window` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (local_exponent_direction) | -0.0261 | 0.009836 | [-0.0574242, 0.00517959] | accepted |
| `wpA_admissibility_vs_width_non_symplectic_dense_mixing_tapered_log_window` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (non_symplectic_dense_mixing) | -0.0058 | 0.005296 | [-0.0226046, 0.0111045] | accepted |
| `wpA_admissibility_vs_width_phase_correlated_features_tapered_log_window` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (phase_correlated_features) | -0.606 | 0.04192 | [-0.739288, -0.472471] | accepted |
| `wpA_admissibility_vs_width_rbf_gaussian_prior_tapered_log_window` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (rbf_gaussian_prior) | 0.0017 | 0.001992 | [-0.00463633, 0.00804393] | accepted |
| `wpA_admissibility_vs_width_selected_log_kernel_tapered_log_window` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (selected_log_kernel) | -0.720 | 0.06705 | [-0.933366, -0.506629] | accepted |
| `wpA_admissibility_vs_width_sign_breaking_mode_reassignment_tapered_log_window` | involution-defect width exponent p in ||conj(S)C-CS||_F/||S||_F ~ N^p (sign_breaking_mode_reassignment) | 0.000000000000000000 | 6.098e-17 | [-1.94071e-16, 1.94071e-16] | accepted |

### 3.7 Failed gates and inconclusive controls

Retained verbatim. No failed gate is replaced by prose.

| claim | metric | value | uncertainty | 95% CI | status |
|---|---|---|---|---|---|
| `wpA_pairing_antidiag_error_sharp_annulus_manuscript` | anti_diagonal_amplitude_error at rho=2 over the R_0 grid | 33 | 84.55 | [0.598792, 514.896] | failed_gate |
| `wpA_admissibility_angular_mode_scramble_polarization_min_eigenvalue_sharp_annulus_manuscript` | minimum eigenvalue of the induced Fock polarization Gram (angular_mode_scramble) | -1.8 | 2.331 | [-7.47668, 1.07631] | failed_gate |
| `wpA_admissibility_anisotropic_angular_prior_polarization_min_eigenvalue_sharp_annulus_manuscript` | minimum eigenvalue of the induced Fock polarization Gram (anisotropic_angular_prior) | -1.5 | 1.521 | [-4.94933, 0.785492] | failed_gate |
| `wpA_admissibility_local_exponent_direction_polarization_min_eigenvalue_sharp_annulus_manuscript` | minimum eigenvalue of the induced Fock polarization Gram (local_exponent_direction) | -3.9 | 3.186 | [-12.2213, -0.481234] | failed_gate |
| `wpA_admissibility_non_symplectic_dense_mixing_polarization_min_eigenvalue_sharp_annulus_manuscript` | minimum eigenvalue of the induced Fock polarization Gram (non_symplectic_dense_mixing) | -35 | 20.91 | [-76.9072, -10.0332] | failed_gate |
| `wpA_admissibility_phase_correlated_features_polarization_min_eigenvalue_sharp_annulus_manuscript` | minimum eigenvalue of the induced Fock polarization Gram (phase_correlated_features) | -0.7 | 1.695 | [-5.16564, 0.96438] | failed_gate |
| `wpA_admissibility_selected_log_kernel_polarization_min_eigenvalue_sharp_annulus_manuscript` | minimum eigenvalue of the induced Fock polarization Gram (selected_log_kernel) | -1.6 | 2.321 | [-8.63437, 0.72419] | failed_gate |
| `wpA_admissibility_sign_breaking_mode_reassignment_polarization_min_eigenvalue_sharp_annulus_manuscript` | minimum eigenvalue of the induced Fock polarization Gram (sign_breaking_mode_reassignment) | -1.07 | 0.9143 | [-3.01099, -0.0432872] | failed_gate |
| `wpA_admissibility_angular_mode_scramble_polarization_min_eigenvalue_tapered_log_window` | minimum eigenvalue of the induced Fock polarization Gram (angular_mode_scramble) | 0.66 | 0.5407 | [-0.827925, 1.05136] | failed_gate |
| `wpA_admissibility_anisotropic_angular_prior_polarization_min_eigenvalue_tapered_log_window` | minimum eigenvalue of the induced Fock polarization Gram (anisotropic_angular_prior) | 0.68 | 0.7981 | [-2.20857, 1.13296] | failed_gate |
| `wpA_admissibility_local_exponent_direction_polarization_min_eigenvalue_tapered_log_window` | minimum eigenvalue of the induced Fock polarization Gram (local_exponent_direction) | -3.3 | 2.708 | [-10.3881, -0.409049] | failed_gate |
| `wpA_admissibility_non_symplectic_dense_mixing_polarization_min_eigenvalue_tapered_log_window` | minimum eigenvalue of the induced Fock polarization Gram (non_symplectic_dense_mixing) | -83 | 78.14 | [-383.073, -11.183] | failed_gate |
| `wpA_admissibility_phase_correlated_features_polarization_min_eigenvalue_tapered_log_window` | minimum eigenvalue of the induced Fock polarization Gram (phase_correlated_features) | 0.85 | 0.3868 | [-0.777287, 1.05876] | failed_gate |
| `wpA_admissibility_selected_log_kernel_polarization_min_eigenvalue_tapered_log_window` | minimum eigenvalue of the induced Fock polarization Gram (selected_log_kernel) | 0.87 | 0.3845 | [-0.545052, 1.22745] | failed_gate |
| `wpA_admissibility_sign_breaking_mode_reassignment_polarization_min_eigenvalue_tapered_log_window` | minimum eigenvalue of the induced Fock polarization Gram (sign_breaking_mode_reassignment) | -1.78 | 0.9318 | [-3.29041, -0.0165573] | failed_gate |
| `wpB_c_exp_sharp_annulus_manuscript_wick_factorized_N750` | sampled <TT> central charge (wick_factorized, N=750) | 1.38 | 0.116 | [1.16076, 1.59549] | failed_gate |
| `wpF_current_reference_sharp_annulus_manuscript_1e-04_1e+04` | reference-window current-kernel relative error | 1.14196 |  |  | failed_gate |
| `wpF_current_reference_sharp_annulus_manuscript_1e-03_1e+03` | reference-window current-kernel relative error | 1.32261 |  |  | failed_gate |
| `wpF_current_reference_sharp_annulus_manuscript_1e-02_1e+02` | reference-window current-kernel relative error | 1.43391 |  |  | failed_gate |

## 4. Regulator limitations (the campaign's central finding)

The scalar covariance eq. (A.1) converges cleanly, but the current kernel is its **second** derivative. The sharp annular support of eq. (2.2) contributes an endpoint term `b J_1(b)`, `b = Lambda_UV r`, whose amplitude grows like `b^(1/2)`. Consequences, each verified three independent ways (closed-form Bessel derivatives against numerical second differentiation; direct Monte Carlo of the finite-feature field; a tapered spectral integrator that reproduces the sharp closed form at taper 0):

1. `K_JJ^Lambda` has **no conformal limit** under eq. (2.2): widening the annulus increases the deviation from `(z-w)^-2`.
2. The shape error of the amplitude fit is 114-143% anywhere inside the nominal physical window (WP-F).
3. `d_zbar K_JJ` of eq. (3.15) grows like `b^(3/2)`, so Proposition 3.4 predicts the failure quantitatively, and the Cauchy-Green bound of Proposition 3.6 - while satisfied by every measured radius variation - is correspondingly loose.
4. A 15% raised-cosine taper in `log k`, the minimal smooth modification of the same `|k|^-2` prior, reduces the shape error to ~2%, restores the `kappa = 1` amplitude, reduces `|d_zbar K_JJ|` by more than four orders of magnitude, and brings the operational pairing to `eps_pair ~ 1e-4`.

**The architecture claim survives; the regulator written in eq. (2.2) does not.** Both classes are released side by side so a reader can see exactly which claims depend on which.

## 5. Artifacts

| artifact | rows | sha256 (first 16) |
|---|---|---|
| `figures_jhep_frozen/architecture_controls_postdarboux.pdf` | - | `ac32a7a5f4e10269` |
| `figures_jhep_frozen/architecture_controls_predarboux.pdf` | - | `f1b469a9d1f91135` |
| `figures_jhep_frozen/architecture_controls_ward.pdf` | - | `7923988935c3f207` |
| `figures_jhep_frozen/darboux_admissibility_vs_width.pdf` | - | `7841e81a12704dfc` |
| `figures_jhep_frozen/finite_width_cumulant.pdf` | - | `6b6af74f6c00e6b3` |
| `figures_jhep_frozen/ns_pairing_convergence.pdf` | - | `317b18afbdc7e403` |
| `figures_jhep_frozen/ns_super_virasoro_residuals.pdf` | - | `18a805afa2fb79cd` |
| `figures_jhep_frozen/ordered_pairing_radius_plateau.pdf` | - | `8c8d17d78038bb6b` |
| `figures_jhep_frozen/predarboux_pairing_vs_width.pdf` | - | `bfdc5fa1afd70c21` |
| `figures_jhep_frozen/predarboux_sampling_error_vs_width.pdf` | - | `a8cbad17f8bccd7f` |
| `figures_jhep_frozen/raw_modewise_central_charge_offset_model.pdf` | - | `7cb864502f3d82e0` |
| `figures_jhep_frozen/raw_virasoro_vs_width.pdf` | - | `ffb11dc14a3651ff` |
| `figures_jhep_frozen/sampled_tt_central_charge.pdf` | - | `aa18a16bdf132cbe` |
| `figures_jhep_frozen/sampled_vertex_weights.pdf` | - | `97b268e58d05d5ad` |
| `figures_jhep_frozen/sampled_ward_convergence.pdf` | - | `a4114dae337bcdeb` |
| `figures_jhep_frozen/ward_controls.pdf` | - | `76e7fd88928e9036` |
| `results/jhep_frozen/all_metrics.csv` | 9773 | `820831939ba777d0` |
| `results/jhep_frozen/analytic_ward_points.csv` | 12 | `fc19a0bb38b1a5ba` |
| `results/jhep_frozen/architecture_controls_seed_rows.csv` | 320 | `3bc611161d8caf66` |
| `results/jhep_frozen/central_offset_fit.json` | - | `e014122c003d104a` |
| `results/jhep_frozen/central_offset_mode_extension.csv` | 561 | `afd0f5c777b3aa12` |
| `results/jhep_frozen/claim_ledger.csv` | 235 | `ee4feb07533bf306` |
| `results/jhep_frozen/current_kernel_reference.csv` | 78 | `edbea5b0b0579cb9` |
| `results/jhep_frozen/darboux_admissibility_vs_width.csv` | 16 | `2e377ade658cfc7e` |
| `results/jhep_frozen/darboux_physical_admissibility.csv` | 320 | `fbb788d66f6ec012` |
| `results/jhep_frozen/deterministic_feature_ward_points.csv` | 3072 | `7303258461156b1d` |
| `results/jhep_frozen/edge_remainder_summary.csv` | 54 | `4c60f95a182ad0fd` |
| `results/jhep_frozen/edge_remainders.json` | - | `61e08c5e51e5480c` |
| `results/jhep_frozen/finite_width_cumulant.csv` | 10 | `db756408bbb32e46` |
| `results/jhep_frozen/finite_width_cumulant_seed_rows.csv` | 80 | `e4e0ace3775cb5bd` |
| `results/jhep_frozen/ns_frozen_metrics.csv` | 14 | `8adae708c24293a0` |
| `results/jhep_frozen/ns_pairing_seed_rows.csv` | 72 | `c386fb9493eac838` |
| `results/jhep_frozen/ns_sampled_correlators.csv` | 36 | `74e598679634f21b` |
| `results/jhep_frozen/ns_super_virasoro_seed_rows.csv` | 70 | `eb423ba89bff3e49` |
| `results/jhep_frozen/ordered_pairing_grid.csv` | 640 | `f47dbce9d37ebbee` |
| `results/jhep_frozen/ordered_pairing_radius_plateau.csv` | 144 | `188f365348f5a909` |
| `results/jhep_frozen/predarboux_convergence_fits.csv` | 8 | `6f04ec624c594bb3` |
| `results/jhep_frozen/predarboux_convergence_seed_rows.csv` | 1360 | `200bad449768c78c` |
| `results/jhep_frozen/robustness_grid_recomputed.csv` | 162 | `06585216e7f853b1` |
| `results/jhep_frozen/sampled_tt_central_charge.csv` | 64 | `cb740eadbac2eb88` |
| `results/jhep_frozen/sampled_tt_seed_rows.csv` | 520 | `84fca62130adce8d` |
| `results/jhep_frozen/sampled_tt_summary.md` | - | `388593ad3757d983` |
| `results/jhep_frozen/sampled_vertex_weights.csv` | 72 | `614a38e171ccc461` |
| `results/jhep_frozen/sampled_vertex_weights_seed_rows.csv` | 432 | `2fe61f6f6a89263e` |
| `results/jhep_frozen/sampled_ward_seed_rows.csv` | 1584 | `f69b394d78ff1a00` |
| `results/jhep_frozen/sampled_ward_summary.csv` | 72 | `251da3165b900f4f` |

## 6. Complementary interpretation (reported verbatim)

Pairing fidelity primarily tests the radial spectral structure of the prior; the current and stress Ward identities test the chiral/angular organization; post-Darboux closure tests neither (Proposition 3.5).

## 7. Reproduction

```bash
python scripts/run_jhep_frozen_suite.py \
  --config configs/jhep_frozen.yaml \
  --skip-existing
python scripts/check_jhep_claim_gates.py --submission
```

Exact commands as executed are in `run_commands.sh`; the environment is in `environment.txt`; per-work-package reports are in `progress/`.


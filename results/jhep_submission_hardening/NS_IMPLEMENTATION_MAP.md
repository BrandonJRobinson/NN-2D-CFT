<!-- generated 2026-09-16T20:42:36Z from docs/hardening/NS_IMPLEMENTATION_MAP.md | commit 2900ae4aa10f | config f51ad38719b59543 -->

# NS_IMPLEMENTATION_MAP — what the NS sector actually computes

Written before the Gate-7 run, to record what the native code does and why the
gate is additive rather than replacing it.

## 1. Is the native NS code a real architecture?

**Yes — but not an independent one.**

`scripts/run_ns_audit.py` draws genuine random momenta through
`architectures.sample_architecture`, and
`ns_fermion.fermionic_pairing_empirical_exact` builds

    K_psipsi^(N,Λ)(z,w) = (L_w/N) Σ_j conj(k_j) sin(Re(conj(k_j)(z−w)))

then evaluates the ordered contour pairing exactly. The width is `n_features`;
the randomness across seeds is the momentum draw. It does **not** perturb the
target CAR matrix, and it does not postulate the CAR — antisymmetry of the
kernel under `z ↔ w` is what converts the ordering discontinuity into
`{β_r, β_s} = δ_{r+s,0}`.

The limitation is its provenance. The kernel is *derived* from the bosonic
scalar seed by one holomorphic derivative,

    K_psipsi = −∂_z K_phiphi ,

so its width and its randomness **are the boson's**. Nothing in the NS sector
has its own Grassmann degrees of freedom. An NS claim resting only on this
construction is really a restatement of the bosonic one.

## 2. Why neither branch of the gate instruction applies verbatim

The mission's Gate-7 instruction anticipates two cases: either the NS code is a
real architecture (keep it), or it has no architecture at all and a Cauchy
construction would materially redefine the experiment (a stop condition). The
native code is neither — it is a real architecture that is not independent.

**Decision: additive.** Retain the native derived construction as a cross-check,
and *also* implement the explicit randomized Cauchy factorization the gate
specifies, giving the NS claim an independent Grassmann architecture with its
own width and its own randomness. This adds a construction rather than redefining
the experiment, so stop condition 4 does not apply.

## 3. The added architecture

`src/nnft_virasoro/ns_cauchy.py`. For `|z| > |w|`,

    1/(z − w) = Σ_{n≥0} w^n z^{−n−1} ,

so a *random importance-weighted subset of NS modes* is an unbiased finite-width
realization of the Cauchy kernel. Draw `n_1..n_N` i.i.d. from a proposal `P(n)`
and set

    psi_N(z) = (1/√N) Σ_j a_j z^{−n_j−1/2} / √P(n_j),   {a_j, a_l} = δ_jl

giving, radially ordered,

    K^(N)(z,w) = (1/N) Σ_j w^{n_j} z^{−n_j−1} / P(n_j)      (|z| > |w|)
               = −(1/N) Σ_j z^{n_j} w^{−n_j−1} / P(n_j)     (|w| > |z|)

The sign flip between branches is the antisymmetry `1/(z−w) = −1/(w−z)` — the
same ordering discontinuity section 7 uses, now arising from the architecture
rather than from a derivative of the boson.

| | native | randomized Cauchy |
|---|---|---|
| width | bosonic `n_features` | its own `N` = number of drawn modes |
| randomness | bosonic momentum draw | NS mode draw |
| kernel origin | `−∂_z K_phiphi` | Cauchy mode expansion, sampled |
| shares the boson's seed | **yes** | **no** |

## 4. The proposal is part of the architecture

With a geometric `P(n) = (1−q)q^n` the pointwise estimator is unbiased for any
`q ∈ (0,1)`, but its **second moment** carries `Σ_n ((|w|/|z|)²/q)^n`, so

    Var K^(N)(z,w) < ∞   ⟺   q > (|w|/|z|)² = 1/ρ² .

At `ρ = 2` this is `q > 0.25`; the campaign uses `q = 0.75`. Below the threshold
the estimator remains unbiased but has infinite variance and its sample paths do
not settle. This is declared in the config and **scanned across violating and
satisfying proposals** in `ns_kernel_seed_rows.csv`, so the condition is
demonstrated rather than asserted. The runner refuses to start if the configured
proposal violates it.

`n_max = 400` is numerical truncation only: the discarded tail is `q^401 ≈ 1e−50`,
and the proposal is renormalized over the truncation so the estimator stays
exactly unbiased for the truncated law.

## 5. The pairing has a closed form — used to validate, not to replace

The ordered contour integrals select a single Laurent mode, so

    J^>_{rs} = δ_{r+s,0} · 1_{r>0} · p̂(r−1/2) / P(r−1/2)
    J^<_{rs} = −δ_{r+s,0} · 1_{r<0} · p̂(−r−1/2) / P(−r−1/2)
    η_{rs}   = δ_{r+s,0} · p̂(|r|−1/2) / P(|r|−1/2)

with `p̂` the empirical mode frequency. Two consequences worth stating plainly:

1. the CAR **structure** `δ_{r+s,0}` is exact at every width — this architecture
   cannot produce off-antidiagonal leakage, so `fermionic_pairing_leakage` is
   not a discriminating observable for it, and must not be read as one;
2. the only error is the importance ratio `p̂/P`, which has mean 1 and standard
   deviation `√((1−P(n))/(N·P(n)))` — exactly the `N^(−1/2)` law the
   architecture-to-operator argument requires, with a predicted coefficient.

Because (1) makes part of the answer structural, the closed form is **not** used
as the sole evaluation path. `cauchy_pairing_quadrature` recomputes the same
object by the same torus quadrature used for the ideal kernel, and the two are
compared per row (`exact_vs_quadrature_max_gap`). Validation runs showed
agreement at **2.9e-16**. The quadrature path is capped at `N ≤ 10⁴` because its
cost is `O(q²N)` and it is an evaluation check, not a statistical one.

## 6. What Gate 7 can and cannot conclude

It **can** show that an NS architecture with its own width and randomness
reproduces the CAR and the super-Virasoro relations with `N^(−1/2)` convergence,
and that the native derived construction agrees with it. That removes the
objection that the NS sector is bosonic bookkeeping.

It **cannot** use the CAR structure of the Cauchy construction as architecture
discrimination, for the reason in §5(1) — the structure is exact by construction.
Discrimination in the NS sector rests on the same pre-Darboux bosonic quantities
as everywhere else. All Gate-7 super-Virasoro residuals are reported
**pre-Darboux**; canonicalized versions are an implementation check only
(mission rule 3).

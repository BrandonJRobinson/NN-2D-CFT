# Virasoro computational evidence

This deposit candidate contains the September 2026 computational evidence for *Virasoro Symmetry in Neural Network Field Theories*, together with the separate 7 November 2026 audit supplement. It is prepared for author approval and is not a published archive; no DOI has yet been assigned. The associated manuscript is arXiv:2512.24420. The older public manuscript version does not describe every correction in the November submission draft.

## Contents and provenance

The source export is taken directly from generation commit `2900ae4aa10ff33ecf88a527e9cf633f0185f98c`. Source hashes and Git blob identifiers are in `PROVENANCE.json`. The configuration hash is `f51ad38719b59543`. Original generated artifacts were packaged at `2238dc35103a746d616d0a49f22c988611c1ebeb`.

`results/jhep_submission_hardening/manifest.json` is unchanged and identifies 377 preserved artifacts. `results/jhep_frozen/` is the earlier August baseline. Its provenance and weaker or rejected claims must not be substituted for the September adjudications. The hardening claim ledger and architecture synthesis distinguish supported, reformulated, rejected, and diagnostic-only claims. `docs/hardening/` and the original generated reports remain historical records.

`review_20261107/` contains the separate regulator and finite-block statistical checks. In particular, the old TT residual statistic has the same value but its finite-block Gaussian-model calibration changes the direct contrast-test p-value from 0.0269 to 0.1250. The joint four-geometry p-values are 0.2463 and 0.3194. The largest direct separation remains poorly determined. The historical CSVs and reports are not silently rewritten. See `review_20261107/CORRECTIONS.md` before interpreting the old narrative reports.

`manuscript_evidence/` contains the seven evidence figures and their inputs. No unapproved submission manuscript, unpublished collaborative manuscript, private Git history, or account settings are included. Acknowledgment of the collaborative regulator framework is in the associated revised manuscript; this archive does not purport to release that collaborative work.

## Verify without simulation

From the extracted archive root:

```sh
python3 verify_release.py
```

This standard-library check verifies the payload manifest and the original 377-artifact manifest. It does not require network access, a Git checkout, or scientific dependencies.

## Reproduce the independent November checks

Install `requirements.txt` in a virtual environment. The historical production versions are recorded separately in `requirements-historical.txt` and the preserved environment records; validation on a newer environment does not imply bit-identical Monte Carlo output.

```sh
python3 review_20261107/independent_audit.py
```

This reads the saved evidence, checks the regulator identities and Fock canonicalizer, and recomputes the TT finite-block analysis. It writes new results only beneath `review_20261107/rerun/`.

## Scientific tests and Git context

The unmodified suite contains 215 distinct tests. First run `python3 verify_release.py --restore-archive-times`. This verifies all content before setting equal payload modification times, preventing extraction order from tripping the legacy figure-freshness heuristic. Python's ZIP extractor does not restore the stored file times. Content hashes and scientific provenance, rather than extraction timestamps, identify this release.

An extracted source archive lacks Git history, and one provenance-stamper test requires a Git repository. In a Git checkout, run `python3 -m pytest tests -q`. For an archive without Git, run `python3 -m pytest tests -q -k 'not test_stamper_rejects_an_unknown_estimator_class'`; this runs 214 tests and deliberately omits that one repository-metadata test. The public candidate is validated in a temporary local Git checkout, so the full 215-test suite is also exercised.

## Rerun the September production campaign

The production runner requires a committed Git checkout. Use a fresh copy or a public repository checkout, not the preserved evidence copy. New runs carry the new checkout commit, not the historical generation SHA; never relabel them as the original release. The full direct TT extension used approximately 96 million realizations and can take substantial compute time.

```sh
python3 scripts/run_jhep_submission_hardening.py --config configs/jhep_submission_hardening.yaml
```

This deliberately regenerates the standard result and figure locations in that fresh copy. The `--quick` option is a smoke run, not the published production evidence. Do not run the master campaign over the archived originals or interpret a cached `--skip-existing` run as reproduction.

## Scientific scope

The evidence concerns fixed regulated kernels, selected finite mode windows, pairing approximation, and a separately specified quantum readout. It does not establish a generic architecture classifier, an exact finite-network Euclidean QFT, reflection positivity of the neural ensemble, uniform removal of all cutoffs in operator norm, or local interactions from the finite-width fourth cumulant alone.

## Citation and reuse

Use `CITATION.cff`, the deposited version, and the real DOI once assigned. Proposed licensing is stated in `LICENSES_PROPOSED.md` and requires the author's approval before public publication.

# Installation and verification

Use Python 3.10 or newer. Historical production versions remain in
requirements-historical.txt and the evidence environment records.

    git clone https://github.com/BrandonJRobinson/NN-2D-CFT.git
    cd NN-2D-CFT
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    python verify_release.py --restore-archive-times
    python -m pytest tests -q

The standard-library verifier can run before installing dependencies.
Timestamp normalization follows successful checksum verification and
accommodates only the legacy figure-freshness test.

Scripts and tests import the source package from src/; installing the
package itself is unnecessary. The Git checkout runs all 215 original tests.
For an archive extracted without Git history, omit its single metadata test:

    python -m pytest tests -q -k 'not test_stamper_rejects_an_unknown_estimator_class'

The independent numerical audit uses NumPy, SciPy and pandas. Recovered-sector
checks use SymPy. The boundary and exact-phase supplements provide separate
README files, version records and verification scripts. The April notebook
instructions remain under legacy/april_2026/.

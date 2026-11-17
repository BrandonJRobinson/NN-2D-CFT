# Installation

Use Python 3.10 or newer.

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    python verify_release.py --restore-archive-times
    python -m pytest tests -q

The standard-library verifier can run before installing dependencies.
The optional timestamp normalization follows successful content verification.
Scripts and tests import the numerical package from src/.

For a source archive without Git history, omit the single Git-metadata test:

    python -m pytest tests -q -k 'not test_stamper_rejects_an_unknown_estimator_class'

Historical production package versions are in requirements-historical.txt.
The boundary and Ward calculations also record their numerical environments.

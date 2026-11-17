"""Release-integrity tests: ledger schema, figure freshness, excluded citations.

These mirror the submission checker's gates so a regression is caught by the
quick test suite, not only at release time.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from nnft_virasoro.config import load_config
from nnft_virasoro.ledger import (LEDGER_COLUMNS, STATUSES, ledger_row,
                                  significant_figures, validate_ledger)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "jhep_frozen"
FIGURES = ROOT / "figures_jhep_frozen"
CONFIG = ROOT / "configs" / "jhep_frozen.yaml"


# --------------------------------------------------------------------------
# claim-ledger schema
# --------------------------------------------------------------------------
def _minimal_row(**kw):
    base = dict(claim_id="x", work_package="WP-A", manuscript_location="sec:1",
                metric="m", value=1.0, estimator_class="exact_symbolic",
                script="s.py", config_hash="abc", output_artifact="o.csv",
                git_commit="deadbeef")
    base.update(kw)
    return ledger_row(**base)


def test_ledger_schema_accepts_a_complete_row():
    assert validate_ledger([_minimal_row()]) == []


def test_ledger_schema_rejects_missing_and_duplicate_fields():
    assert validate_ledger([]) == ["claim ledger is empty"]
    bad = _minimal_row()
    del bad["metric"]
    assert any("missing columns" in p for p in validate_ledger([bad]))
    dup = [_minimal_row(), _minimal_row()]
    assert any("duplicate claim_id" in p for p in validate_ledger(dup))
    empty = _minimal_row(script="")
    assert any("empty required field" in p for p in validate_ledger([empty]))


def test_ledger_rejects_unknown_status():
    with pytest.raises(ValueError):
        _minimal_row(status="looks_fine_to_me")
    for st in STATUSES:
        assert _minimal_row(status=st)["status"] == st


def test_displayed_precision_never_exceeds_the_uncertainty():
    """PDG convention: uncertainty to two significant digits, value to match."""
    assert significant_figures(1.23456, 0.01) == "1.235"
    assert significant_figures(1.23456, 0.0001) == "1.23456"
    assert significant_figures(0.4938, 0.016) == "0.494"
    # a value is never quoted to more digits than its uncertainty justifies
    assert significant_figures(1.23456789, 1.0) == "1.2"
    # non-finite / absent uncertainty falls back to six significant digits
    assert significant_figures(1.23456789, 0.0) == "1.23457"
    assert significant_figures(float("nan"), 0.1) == "nan"


@pytest.mark.skipif(not (RESULTS / "claim_ledger.csv").exists(),
                    reason="frozen suite has not been run")
def test_frozen_ledger_is_valid_and_complete():
    df = pd.read_csv(RESULTS / "claim_ledger.csv")
    assert list(df.columns) == list(LEDGER_COLUMNS)
    assert validate_ledger(df.to_dict("records")) == []
    assert df["claim_id"].is_unique
    # every emitted estimator class must be one of the four declared ones
    from nnft_virasoro import ESTIMATOR_CLASSES

    assert set(df["estimator_class"]) <= set(ESTIMATOR_CLASSES)


# --------------------------------------------------------------------------
# excluded-citation guard
# --------------------------------------------------------------------------
def test_excluded_citation_guard_flags_a_later_anomaly_paper(tmp_path):
    cfg = load_config(CONFIG)
    pats = cfg.section("wp_0")["excluded_citation_patterns"]
    bib = tmp_path / "x.bib"
    bib.write_text(
        '@article{Later:2026abc,\n    title = "{Anomalies in Neural Network '
        'Field Theories}",\n    year = "2026"\n}\n', encoding="utf-8")
    blob = bib.read_text()
    hits = []
    for key, body in re.findall(r"@\w+\{([^,]+),(.*?)\n\}", blob, re.S):
        title = " ".join(re.findall(r'title\s*=\s*[{"](.+?)[}"]', body,
                                    re.S | re.I)).lower()
        yr = int(re.findall(r'year\s*=\s*[{"]?(\d{4})', body)[0])
        if yr >= 2025 and any(p.lower() in title for p in pats):
            hits.append(key)
    assert hits == ["Later:2026abc"], "the guard must catch a later anomaly paper"


@pytest.mark.skipif(not (ROOT / "Virasoro_JHEP_Frozen.bib").exists(),
                    reason="frozen bib has not been built")
def test_frozen_bibliography_has_no_excluded_later_papers():
    cfg = load_config(CONFIG)
    pats = cfg.section("wp_0")["excluded_citation_patterns"]
    blob = (ROOT / "Virasoro_JHEP_Frozen.bib").read_text(encoding="utf-8")
    for key, body in re.findall(r"@\w+\{([^,]+),(.*?)\n\}", blob, re.S):
        title = " ".join(re.findall(r'title\s*=\s*[{"](.+?)[}"]', body,
                                    re.S | re.I)).lower()
        years = re.findall(r'year\s*=\s*[{"]?(\d{4})', body)
        if years and int(years[0]) >= 2025:
            assert not any(p.lower() in title for p in pats), key


def test_no_collaborator_exponent_sweep_artifacts_anywhere():
    cfg = load_config(CONFIG)
    forbidden = cfg.section("wp_0")["forbidden_artifacts"]
    offenders = []
    for pth in list(ROOT.rglob("*.csv")) + list(ROOT.rglob("*.py")):
        if ".git/" in str(pth):
            continue
        if any(f in pth.name.lower() for f in forbidden):
            offenders.append(str(pth))
    assert offenders == []


# --------------------------------------------------------------------------
# figure freshness
# --------------------------------------------------------------------------
@pytest.mark.skipif(not (FIGURES / "figure_captions.json").exists(),
                    reason="figures have not been generated")
def test_every_generated_figure_has_a_caption_naming_class_and_source():
    caps = json.loads((FIGURES / "figure_captions.json").read_text())
    assert caps, "no captions recorded"
    for name, text in caps.items():
        assert (FIGURES / name).exists(), name
        assert "Estimator class:" in text, name
        assert "Source: results/jhep_frozen/" in text, name


@pytest.mark.skipif(not (FIGURES / "figure_captions.json").exists(),
                    reason="figures have not been generated")
def test_figures_are_not_older_than_their_source_csv():
    from importlib import util

    spec = util.spec_from_file_location(
        "chk", ROOT / "scripts" / "check_jhep_claim_gates.py")
    chk = util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    stale = []
    for fig, src in chk.FIGURE_SOURCES.items():
        fp, sp = FIGURES / fig, RESULTS / src
        if sp.exists() and fp.exists() and fp.stat().st_mtime < sp.stat().st_mtime:
            stale.append(fig)
    assert stale == []


# --------------------------------------------------------------------------
# config integrity
# --------------------------------------------------------------------------
def test_config_declares_named_seeds_and_both_regulator_classes():
    cfg = load_config(CONFIG)
    seeds = cfg.section("named_seeds")
    assert len(seeds) >= 40
    assert all(isinstance(v, int) for v in seeds.values())
    rc = cfg.section("global")["regulator_classes"]
    assert set(rc) == {"sharp_annulus_manuscript", "tapered_log_window"}
    assert rc["sharp_annulus_manuscript"] == 0.0
    assert rc["tapered_log_window"] > 0.0


def test_config_hash_is_deterministic_and_sensitive():
    a = load_config(CONFIG)
    b = load_config(CONFIG)
    assert a.hash == b.hash
    from nnft_virasoro.config import Config

    mutated = Config(path=a.path, data={**a.data, "version": 999})
    assert mutated.hash != a.hash


def test_unknown_seed_name_fails_loudly():
    from nnft_virasoro.config import seed_value

    with pytest.raises(KeyError):
        seed_value({"a": 1}, "not_declared")


def test_provenance_columns_are_enforced():
    from nnft_virasoro.provenance import ProvenanceError, require_columns

    with pytest.raises(ProvenanceError):
        require_columns([], ["a"])
    with pytest.raises(ProvenanceError):
        require_columns([{"a": 1}], ["a", "b"])
    require_columns([{"a": 1, "b": 2}], ["a", "b"])


def test_stamper_rejects_an_unknown_estimator_class():
    from nnft_virasoro.provenance import ProvenanceError, Stamper

    st = Stamper(script="t.py", config_hash="x")
    with pytest.raises(ProvenanceError):
        st.stamp({}, estimator_class="vibes", seed="s")
    row = st.stamp({}, estimator_class="exact_symbolic", seed="s")
    assert row["git_commit"] and row["timestamp_utc"].endswith("Z")


# --------------------------------------------------------------------------
# generated manuscript
# --------------------------------------------------------------------------
@pytest.mark.skipif(not (ROOT / "Draft-JHEP-Frozen.tex").exists(),
                    reason="frozen manuscript has not been built")
def test_generated_manuscript_has_no_unresolved_gate_anchors():
    text = (ROOT / "Draft-JHEP-Frozen.tex").read_text(encoding="utf-8")
    assert "CLAIM-GATE FM-" not in text
    assert "RELEASE-GATE FM-" not in text
    assert "WORKING INTEGRATION DRAFT" not in text
    # every gate is accounted for in the generated header checklist
    for n in range(1, 13):
        assert f"FM-{n:02d} ->" in text, f"FM-{n:02d} missing from the checklist"
    assert "graphicspath{{figures_jhep_frozen/}}" in text
    assert "bibliography{Virasoro_JHEP_Frozen}" in text


@pytest.mark.skipif(not (ROOT / "Draft-JHEP-Frozen.tex").exists(),
                    reason="frozen manuscript has not been built")
def test_generated_latex_is_balanced_and_escaped():
    """Cheap structural checks that catch the escaping bugs a generator makes."""
    text = (ROOT / "Draft-JHEP-Frozen.tex").read_text(encoding="utf-8")
    body = re.sub(r"%.*", "", text)
    assert body.count("\\begin{align}") == body.count("\\end{align}")
    assert body.count("\\begin{split}") == body.count("\\end{split}")
    assert body.count("\\begin{tabular}") == body.count("\\end{tabular}")
    assert body.count("\\begin{center}") == body.count("\\end{center}")
    assert body.count("$") % 2 == 0
    # inside an align/split, at most one alignment point per line
    for env in re.findall(r"\\begin\{split\}(.*?)\\end\{split\}", body, re.S):
        for line in env.split("\\\\"):
            assert line.count("&") <= 1, line[:80]
    # a generated table row must terminate with "\\", never a lone backslash
    for tab in re.findall(r"\\begin\{tabular\}(.*?)\\end\{tabular\}", body, re.S):
        for line in tab.splitlines():
            st = line.rstrip()
            if st.endswith("\\") and not st.endswith("\\\\"):
                raise AssertionError(f"lone trailing backslash: {st[:80]}")


@pytest.mark.skipif(not (ROOT / "tables_jhep_frozen").exists(),
                    reason="tables have not been generated")
def test_generated_tables_put_numbers_in_math_mode():
    for p in (ROOT / "tables_jhep_frozen").glob("*.tex"):
        body = re.sub(r"%.*", "", p.read_text(encoding="utf-8"))
        assert body.count("$") % 2 == 0, p.name
        assert "\\times10^" not in re.sub(r"\$[^$]*\$", "", body), (
            f"{p.name}: scientific notation outside math mode")


def test_git_dirty_parses_every_porcelain_status_form(monkeypatch):
    """The status token must be split by whitespace, not by a fixed offset.

    `_run_git` strips the output, which removes the leading space of an
    unstaged-only line (" M path" -> "M path"); a fixed `line[3:]` slice then
    silently mangles the path and reports a clean tree as dirty.
    """
    from nnft_virasoro import provenance as PV

    cases = {
        " M figures_jhep_frozen/a.pdf": False,   # unstaged, leading space eaten
        "M  results/b.csv": False,               # staged
        "?? tables_jhep_frozen/c.tex": False,    # untracked
        "MM Draft-JHEP-Frozen.tex": False,       # staged + unstaged
        'R  "results/old.csv" -> "results/new.csv"': False,
        " M src/nnft_virasoro/kernels.py": True,  # real code change
        "?? scripts/scratch.py": True,
    }
    for line, expected in cases.items():
        monkeypatch.setattr(PV, "_run_git", lambda *a, _l=line: _l.strip())
        assert PV.git_dirty() is expected, line


def test_release_artifacts_share_one_commit_and_config_hash():
    """A release must be a single clean pass, not an accretion of partial runs."""
    import pandas as pd

    res = ROOT / "results" / "jhep_frozen"
    if not (res / "claim_ledger.csv").exists():
        pytest.skip("no frozen release present")
    commits, hashes = set(), set()
    for p in res.glob("*.csv"):
        if p.name in ("claim_ledger.csv", "all_metrics.csv"):
            continue
        try:
            df = pd.read_csv(p, usecols=["git_commit", "config_hash"])
        except Exception:
            continue
        commits.update(str(x) for x in df["git_commit"].dropna().unique())
        hashes.update(str(x) for x in df["config_hash"].dropna().unique())
    assert len(hashes) <= 1, f"multiple config hashes: {hashes}"
    assert len(commits) <= 1, f"multiple code commits: {sorted(c[:8] for c in commits)}"

"""README numbers must come from the results files (runs in CI; no raw data needed)."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_readme", ROOT / "scripts" / "build_readme.py")
build_readme = importlib.util.module_from_spec(spec); spec.loader.exec_module(build_readme)


def test_readme_matches_results_files():
    rendered = build_readme.build()
    current = (ROOT / "README.md").read_text(encoding="utf-8")
    assert current == rendered, "README.md drifted from results/*.json; run python scripts/build_readme.py"


def test_unresolvable_placeholder_fails():
    with pytest.raises(KeyError):
        build_readme.render("{{m:models/no_such_model/all_rows/roc_auc|.3f}}", build_readme.load_sources())


def test_changed_metric_changes_readme():
    data = build_readme.load_sources()
    t = "{{m:models/lgbm/all_rows/roc_auc|.3f}}"
    before = build_readme.render(t, data)
    data["m"]["models"]["lgbm"]["all_rows"]["roc_auc"] += 0.01
    assert build_readme.render(t, data) != before


def test_results_files_are_internally_consistent():
    m = json.loads((ROOT / "results/metrics.json").read_text())
    assert m["evaluated_split"] == "test" and m["mode"] == "final"
    assert (m["cleaning"]["end_rows"], m["cleaning"]["end_noshows"]) == (109_899, 22_098)
    assert m["splits"]["test"]["rows"] == m["models"]["lgbm"]["all_rows"]["n"]
    assert len((ROOT / "results/test_runs.log").read_text().strip().splitlines()) == 1  # test touched once
    pbi = json.loads((ROOT / "results/powerbi_export_check.json").read_text())
    assert (pbi["rows"], pbi["no_shows"]) == (m["cleaning"]["end_rows"], m["cleaning"]["end_noshows"])

"""Render README.md from docs/README.template.md and the committed results files.

Every number in the README is a placeholder {{source:path|format}} resolved
from a results JSON; nothing is typed by hand. tests/test_readme_drift.py
re-renders and fails if README.md differs, so README and results cannot drift.

  python scripts/build_readme.py          # write README.md
  python scripts/build_readme.py --check  # exit 1 if README.md is stale
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "m": "results/metrics.json",
    "shap": "results/shap_summary.json",
    "post": "results/posthoc_same_day.json",
    "pair": "results/paired_diff_test.json",
    "pbi": "results/powerbi_export_check.json",
    "lift": "results/lift_by_decile_test.json",
    "audit": "reports/raw_audit.json",
}
PATTERN = re.compile(r"\{\{\s*([a-z]+):([^|}]+?)\s*(?:\|\s*([^}]*?)\s*)?\}\}")


def load_sources():
    data = {k: json.loads((ROOT / v).read_text(encoding="utf-8")) for k, v in SOURCES.items()}
    m = data["m"]
    val = m["validation"]["models"]["lgbm"]["calibration_all_rows"]
    tst = m["models"]["lgbm"]["calibration_all_rows"]
    data["d"] = {  # derived values (arithmetic on results, never typed)
        "val_gap_pts": (val["mean_pred"] - val["observed_rate"]) * 100,
        "test_gap_pts": (tst["mean_pred"] - tst["observed_rate"]) * 100,
        "test_lines": len((ROOT / "results/test_runs.log").read_text().strip().splitlines()),
        "raw_sha_short": m["raw_sha256"][:12],
        "rows_explained": data["shap"]["rows_explained"]["valid_all"] + data["shap"]["rows_explained"]["test_all"],
        "unseen_share": 1 - m["sensitivity"]["seen_vs_unseen"]["seen_in_train"]["share_of_eval_rows"],
    }
    return data


def resolve(data, src, path):
    node = data[src]
    for part in path.split("/"):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def render(template, data):
    def sub(mt):
        src, path, fmt = mt.group(1), mt.group(2).strip(), mt.group(3)
        try:
            val = resolve(data, src, path)
        except (KeyError, IndexError, ValueError) as e:
            raise KeyError(f"unresolved placeholder {mt.group(0)}: {e!r}") from None
        return format(val, fmt) if fmt else str(val)
    out = PATTERN.sub(sub, template)
    if "{{" in out:
        raise ValueError("template still contains '{{' after rendering")
    return out


def build():
    return render((ROOT / "docs/README.template.md").read_text(encoding="utf-8"), load_sources())


if __name__ == "__main__":
    text = build()
    readme = ROOT / "README.md"
    if "--check" in sys.argv:
        ok = readme.exists() and readme.read_text(encoding="utf-8") == text
        print("README.md is up to date" if ok else "README.md is STALE: run python scripts/build_readme.py")
        sys.exit(0 if ok else 1)
    readme.write_text(text, encoding="utf-8")
    print(f"wrote README.md ({len(text.splitlines())} lines)")

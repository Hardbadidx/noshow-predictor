"""Write the cleaned, descriptive CSV for Power BI (local only; not committed).

Usage: python scripts/export_powerbi.py data/raw/KaggleV2-May-2016.csv
Writes data/processed/noshow_powerbi.csv and results/powerbi_export_check.json
(the check values your Power BI cards should reproduce).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from noshow.clean import clean, load_raw  # noqa: E402
from noshow.powerbi import build_export, export_checks  # noqa: E402


def main(raw_csv):
    df, _ = clean(load_raw(raw_csv))
    out = build_export(df)
    dest = ROOT / "data/processed/noshow_powerbi.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False, encoding="utf-8")
    checks = export_checks(out)
    (ROOT / "results/powerbi_export_check.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False))
    print(f"wrote {dest.relative_to(ROOT)}: {checks['rows']:,} rows, {len(out.columns)} columns")
    print(json.dumps({k: v for k, v in checks.items() if not k.startswith("by_")}, indent=1))
    print("by lead bucket:")
    for r in checks["by_lead_bucket"]:
        print(f"  {r['lead_bucket']:11s} n={r['appointments']:6,} no-shows={r['no_shows']:5,} rate={r['no_show_rate']:.4f} sms_sent={r['sms_sent']:,}")
    print("by lead bucket x SMS:")
    for r in checks["by_lead_bucket_and_sms"]:
        print(f"  {r['lead_bucket']:11s} {r['sms_received']:9s} n={r['appointments']:6,} rate={r['no_show_rate']:.4f}")


if __name__ == "__main__":
    main(sys.argv[1])

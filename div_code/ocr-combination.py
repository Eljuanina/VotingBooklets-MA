"""
combine_results.py — Merge all pipeline CSVs into combined files
Produces three combined files:
  ocr_eval_by_language_ALL.csv
  ocr_eval_by_year_ALL.csv
  ocr_eval_results_ALL.csv  (per file)
"""

import csv
from pathlib import Path

INPUT_DIR  = Path("ocr-results-final")
OUTPUT_DIR = Path("ocr-results-final")

PIPELINES = [
    "mono",
    "agent",
    "docling",
    "gemini-ocr",
    "gemini-post",
    "gemini-3-post",
    "pytesseract",
    "finepdf",
]

TYPES = [
    "by_language",
    "by_year",
    "results",      # per file
]

for typ in TYPES:
    all_rows = []

    for tag in PIPELINES:
        path = INPUT_DIR / f"ocr_eval_{typ}_{tag}.csv"
        if not path.exists():
            print(f"[WARNING] Not found, skipping: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_rows.append({"pipeline": tag, **row})

    if not all_rows:
        print(f"[WARNING] No data found for type '{typ}', skipping.")
        continue

    out_path = OUTPUT_DIR / f"ocr_eval_{typ}_ALL.csv"
    fieldnames = ["pipeline"] + [k for k in all_rows[0] if k != "pipeline"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Written {len(all_rows)} rows → {out_path}")
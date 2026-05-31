import json
import pandas as pd
from pathlib import Path

YEARS = ["1977", "1985", "2007"]
INPUT_DIR = Path("data/parallel_data")
OUTPUT_DIR = Path(".")  # change if you want outputs elsewhere

def jsonl_to_df(path):
    rows = []
    with open(path, encoding="utf8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)

for year in YEARS:
    input_path  = INPUT_DIR / f"{year}_parallel.jsonl"
    output_path = OUTPUT_DIR / f"{year}.xlsx"

    if not input_path.exists():
        print(f"[WARNING] File not found: {input_path}")
        continue

    df = jsonl_to_df(input_path)
    df.to_excel(output_path, index=False)
    print(f"Saved {len(df)} rows → {output_path}")
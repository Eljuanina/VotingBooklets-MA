"""
jsonl_to_excel.py
-----------------
Converts parallel corpus JSONL files to Excel.
Expects files named:  parallel_corpus_1977.jsonl
                      parallel_corpus_1985.jsonl
                      parallel_corpus_2007.jsonl

Produces one Excel file per year:  parallel_corpus_1977.xlsx  etc.
OR one combined Excel with one sheet per year:  parallel_corpus_all.xlsx

Usage:
    # Convert all three years, one file each:
    python jsonl_to_excel.py --years 1977 1985 2007

    # Convert and merge into one file with one sheet per year:
    python jsonl_to_excel.py --years 1977 1985 2007 --combine

    # Specify custom input/output directories:
    python jsonl_to_excel.py --years 1977 1985 2007 --input-dir data/ --output-dir out/ --combine
"""

import argparse
import json
from pathlib import Path

import pandas as pd


# Preferred column order — any extras are appended at the end
COLUMN_ORDER = ["voting_date", "vote_id", "paragraph_index", "de", "fr", "it", "rm"]


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)

    # Reorder columns: known ones first, then any unexpected extras
    present = [c for c in COLUMN_ORDER if c in df.columns]
    extras  = [c for c in df.columns if c not in COLUMN_ORDER]
    return df[present + extras].fillna("")


def save_single(df: pd.DataFrame, out_path: Path) -> None:
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="corpus")

        # Auto-fit column widths (capped at 80)
        ws = writer.sheets["corpus"]
        for col_cells in ws.columns:
            max_len = max(
                len(str(cell.value)) if cell.value else 0
                for cell in col_cells
            )
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 80)

    print(f"  Saved → {out_path}  ({len(df)} rows)")


def save_combined(dfs: dict[str, pd.DataFrame], out_path: Path) -> None:
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for year, df in dfs.items():
            df.to_excel(writer, index=False, sheet_name=str(year))

            ws = writer.sheets[str(year)]
            for col_cells in ws.columns:
                max_len = max(
                    len(str(cell.value)) if cell.value else 0
                    for cell in col_cells
                )
                ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 80)

    total = sum(len(df) for df in dfs.values())
    print(f"  Saved → {out_path}  ({len(dfs)} sheets, {total} rows total)")


def main():
    parser = argparse.ArgumentParser(description="Convert parallel corpus JSONL to Excel")
    parser.add_argument("--years",      nargs="+", default=["1977", "1985", "2007"],
                        help="Year labels to process (default: 1977 1985 2007)")
    parser.add_argument("--input-dir",  type=Path, default=Path("."),
                        help="Directory containing the JSONL files (default: current dir)")
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                        help="Directory for output Excel files (default: current dir)")
    parser.add_argument("--combine",    action="store_true",
                        help="Write all years into one Excel file (one sheet per year)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dfs = {}
    for year in args.years:
        path = args.input_dir / "Agent-VB-gold/Data/ParallelCorpora"/f"parallel_corpus_{year}.jsonl"
        if not path.exists():
            print(f"  [WARN] File not found, skipping: {path}")
            continue
        print(f"Loading {path} ...")
        dfs[year] = load_jsonl(path)
        print(f"  {len(dfs[year])} rows, columns: {list(dfs[year].columns)}")

    if not dfs:
        print("No files loaded — nothing to write.")
        return

    if args.combine:
        out_path = args.output_dir / "parallel_corpus_all.xlsx"
        save_combined(dfs, out_path)
    else:
        for year, df in dfs.items():
            out_path = args.output_dir / f"parallel_corpus_{year}.xlsx"
            save_single(df, out_path)

    print("Done.")


if __name__ == "__main__":
    main()
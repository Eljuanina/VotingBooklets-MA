"""
alignment_accuracy.py
----------------------
Calculates alignment accuracy per language pair from manually annotated
parallel corpus Excel files.

Expected Excel columns:
    de, fr, it, rm          ← text columns
    de-fr, de-it, fr-it     ← annotation columns (1 = correct, 0 = incorrect)
    de-rm, fr-rm, it-rm     ← optional, if Romansh is present

Usage:
    python alignment_accuracy.py --files parallel_corpus_1977.xlsx parallel_corpus_1985.xlsx parallel_corpus_2007.xlsx
    python alignment_accuracy.py --dir path/to/excels/
"""

import argparse
from pathlib import Path

import pandas as pd


# All possible annotation columns
PAIR_COLS = ["de-fr", "de-it", "de-rm", "fr-it", "fr-rm", "it-rm"]


def compute_accuracy(df: pd.DataFrame) -> dict[str, dict]:
    """
    For each annotation column present in df, compute:
        - n_correct   : number of 1s
        - n_total     : number of non-null rows
        - accuracy    : n_correct / n_total
    """
    results = {}
    for col in PAIR_COLS:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        n_total   = len(series)
        n_correct = int(series.sum())
        results[col] = {
            "correct": n_correct,
            "total":   n_total,
            "accuracy": n_correct / n_total,
        }
    return results


def print_results(label: str, results: dict[str, dict]) -> None:
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    if not results:
        print("  No annotation columns found.")
        return
    print(f"  {'Pair':<10} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*10}")
    for pair, vals in sorted(results.items()):
        print(f"  {pair:<10} {vals['correct']:>8} {vals['total']:>8} {vals['accuracy']:>9.2%}")


def aggregate(all_results: dict[str, dict[str, dict]]) -> dict[str, dict]:
    """Sum correct/total across all files, recompute accuracy."""
    combined: dict[str, dict] = {}
    for year_results in all_results.values():
        for pair, vals in year_results.items():
            if pair not in combined:
                combined[pair] = {"correct": 0, "total": 0}
            combined[pair]["correct"] += vals["correct"]
            combined[pair]["total"]   += vals["total"]
    for pair in combined:
        c = combined[pair]
        c["accuracy"] = c["correct"] / c["total"] if c["total"] > 0 else 0.0
    return combined


def main():
    parser = argparse.ArgumentParser(description="Compute alignment accuracy from annotated Excel files")
    parser.add_argument("--files", nargs="+", type=Path,
                        help="Excel files to evaluate")
    parser.add_argument("--dir",   type=Path,
                        help="Directory containing Excel files (alternative to --files)")
    args = parser.parse_args()

    # Collect file paths
    if args.files:
        paths = args.files
    elif args.dir:
        paths = sorted(args.dir.glob("*.xlsx"))
    else:
        # Default: look for parallel_corpus_*.xlsx in current directory
        paths = sorted(Path(".").glob("parallel_corpus_*.xlsx"))

    if not paths:
        print("No Excel files found. Use --files or --dir.")
        return

    all_results: dict[str, dict[str, dict]] = {}

    for path in paths:
        if not path.exists():
            print(f"[WARN] File not found: {path}")
            continue

        label = path.stem  # e.g. "parallel_corpus_1977"
        print(f"Loading {path} ...")
        df = pd.read_excel(path, engine="openpyxl")
        print(f"  {len(df)} rows, columns: {df.columns.tolist()}")

        results = compute_accuracy(df)
        all_results[label] = results
        print_results(label, results)

    # Print aggregated summary if more than one file
    if len(all_results) > 1:
        agg = aggregate(all_results)
        print_results("OVERALL (all years combined)", agg)

    # Save to CSV
    rows = []
    for label, results in all_results.items():
        for pair, vals in results.items():
            rows.append({
                "file":     label,
                "pair":     pair,
                "correct":  vals["correct"],
                "total":    vals["total"],
                "accuracy": round(vals["accuracy"], 4),
            })

    if len(all_results) > 1:
        agg = aggregate(all_results)
        for pair, vals in sorted(agg.items()):
            rows.append({
                "file":     "OVERALL",
                "pair":     pair,
                "correct":  vals["correct"],
                "total":    vals["total"],
                "accuracy": round(vals["accuracy"], 4),
            })

    out_path = paths[0].parent / "alignment_accuracy.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
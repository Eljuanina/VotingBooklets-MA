"""
Alignment score calculator for 1977.xlsx, 1985.xlsx, 2007.xlsx
Computes standard MT alignment evaluation metrics for all available pairs per file.
"""
import pandas as pd

INPUT_FILES = ["1977.xlsx", "1985.xlsx", "2007.xlsx"]
ALL_PAIRS = ["de-fr", "de-it", "de-rm", "fr-it", "fr-rm", "it-rm"]
OUTPUT_FILE = "alignment_scores_results.txt"

def run(out):
    for filepath in INPUT_FILES:
        try:
            df = pd.read_excel(filepath)
        except FileNotFoundError:
            msg = f"\n[WARNING] File not found: {filepath}, skipping.\n"
            print(msg, end="")
            out.write(msg)
            continue

        pairs = [p for p in ALL_PAIRS if p in df.columns]

        header = "\n" + "=" * 55 + f"\n  ALIGNMENT EVALUATION — {filepath}\n" + "=" * 55
        print(header)
        out.write(header + "\n")

        results = {}
        for pair in pairs:
            col = df[pair].dropna()
            total = len(col)
            correct = int(col.sum())
            incorrect = total - correct
            accuracy = correct / total if total > 0 else 0
            aer = incorrect / total if total > 0 else 0

            results[pair] = {
                "Total segments": total,
                "Correct (1)": correct,
                "Incorrect (0)": incorrect,
                "Accuracy": accuracy,
                "Error Rate (AER)": aer,
            }

            block = (
                f"\n--- {pair.upper()} ---\n"
                f"  Total segments : {total}\n"
                f"  Correct  (1)   : {correct}\n"
                f"  Incorrect (0)  : {incorrect}\n"
                f"  Accuracy       : {accuracy:.4f}  ({accuracy*100:.2f}%)\n"
                f"  Error Rate     : {aer:.4f}  ({aer*100:.2f}%)"
            )
            print(block)
            out.write(block + "\n")

        summary = (
            "\n" + "-" * 55 + "\n"
            "  SUMMARY TABLE\n" +
            "-" * 55 + "\n" +
            f"  {'Pair':<10} {'Accuracy':>10} {'Error Rate':>12} {'Correct':>9} {'Total':>7}\n" +
            "  " + "-" * 50
        )
        print(summary)
        out.write(summary + "\n")

        for pair, r in results.items():
            row = (f"  {pair:<10} {r['Accuracy']:>10.4f} {r['Error Rate (AER)']:>12.4f} "
                   f"{r['Correct (1)']:>9} {r['Total segments']:>7}")
            print(row)
            out.write(row + "\n")

        if len(pairs) > 1:
            score_df = df[pairs].dropna()
            if len(score_df) > 0:
                agree_header = (f"\n  CROSS-PAIR AGREEMENT  (n={len(score_df)} complete rows)\n" +
                                "  " + "-" * 50)
                print(agree_header)
                out.write(agree_header + "\n")

                for i, p1 in enumerate(pairs):
                    for p2 in pairs[i+1:]:
                        agree = (score_df[p1] == score_df[p2]).sum()
                        total = len(score_df)
                        line = f"    {p1} ↔ {p2}: {agree}/{total} agree ({agree/total*100:.1f}%)"
                        print(line)
                        out.write(line + "\n")

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    run(out)

print(f"\n  Results saved to: {OUTPUT_FILE}\n")
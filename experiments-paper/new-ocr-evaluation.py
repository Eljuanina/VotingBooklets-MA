import pandas as pd
import re
from jiwer import process_words, process_characters

# ------------------------------------------------
# Paths
# ------------------------------------------------
excel_path = "gs-2007-new.xlsx"

# ------------------------------------------------
# Normalize spacing around punctuation
# ------------------------------------------------
def normalize_spacing(text):
    text = re.sub(r'(\w)\.\s+(\w)', r'\1.\2', text)
    text = re.sub(r'(\d)\s+(\d{3})', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ------------------------------------------------
# Calculate metrics for one column
# ------------------------------------------------
def calculate_metrics(gold_series, ocr_series, column_name, sheet_name):
    total_ins = total_del = total_sub = total_hits = 0
    total_words = 0
    cer_total = 0
    cer_chars = 0

    print(f"\n{'='*60}")
    print(f"Sheet: {sheet_name} | Column: {column_name}")
    print(f"{'='*60}")

    for idx, (g_raw, o_raw) in enumerate(zip(gold_series, ocr_series)):
        g_raw = g_raw.strip()
        o_raw = o_raw.strip()

        # Skip if gold is empty
        if g_raw == "":
            continue

        # Normalize for comparison only
        g = normalize_spacing(g_raw)
        o = normalize_spacing(o_raw)

        # If OCR is missing — count entire gold sentence as deleted
        if o_raw == "":
            gold_words = g.split()
            total_del += len(gold_words)
            total_words += len(gold_words)
            cer_total += 1.0 * len(g)
            cer_chars += len(g)
            print(f"\nRow {idx+1} [MISSING OCR — full deletion]:")
            print(f"  REF: {g_raw}")
            print(f"  DEL: {len(gold_words)} words")
            continue

        result = process_words(g, o)
        total_ins += result.insertions
        total_del += result.deletions
        total_sub += result.substitutions
        total_hits += result.hits
        total_words += result.hits + result.substitutions + result.deletions

        cer_result = process_characters(g, o)
        cer_total += cer_result.cer * len(g)
        cer_chars += len(g)

        # Print original (non-normalized) text so output is readable
        if result.substitutions + result.deletions + result.insertions > 0:
            print(f"\nRow {idx+1}:")
            print(f"  REF: {g_raw}")
            print(f"  HYP: {o_raw}")
            print(f"  Hits: {result.hits} | Subs: {result.substitutions} | Del: {result.deletions} | Ins: {result.insertions}")

    # Compute overall metrics
    wer = (total_sub + total_del + total_ins) / total_words if total_words > 0 else 0
    cer_final = cer_total / cer_chars if cer_chars > 0 else 0

    print(f"\n  >> WER: {wer:.4f} | CER: {cer_final:.4f}")
    print(f"  >> Insertions: {total_ins} | Deletions: {total_del} | Substitutions: {total_sub}")
    print(f"  >> Rows compared: {len(gold_series)}")

    return {
        "WER": wer,
        "CER": cer_final,
        "insertions": total_ins,
        "deletions": total_del,
        "substitutions": total_sub
    }

# ------------------------------------------------
# Read Excel and get sheet names
# ------------------------------------------------
xls = pd.ExcelFile(excel_path)
sheet_names = xls.sheet_names
results = []

for sheet_name in sheet_names:
    df_sheet = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)

    if df_sheet.shape[1] < 4:
        print(f"Sheet '{sheet_name}' skipped (needs at least 4 columns, found {df_sheet.shape[1]})")
        continue

    # Col 0: Gold Standard
    # Col 1: Gemini
    # Col 2: Pytesseract
    # Col 3: Pytesseract + Gemini
    gold_series         = df_sheet.iloc[:, 0].fillna("").astype(str)
    gemini_series       = df_sheet.iloc[:, 1].fillna("").astype(str)
    pytesseract_series  = df_sheet.iloc[:, 2].fillna("").astype(str)
    pytess_gem_series   = df_sheet.iloc[:, 3].fillna("").astype(str)
    pytess_gem3_series   = df_sheet.iloc[:, 3].fillna("").astype(str)

    # GS vs Gemini
    m1 = calculate_metrics(gold_series, gemini_series, "Gemini", sheet_name)
    # GS vs Pytesseract
    m2 = calculate_metrics(gold_series, pytesseract_series, "Pytesseract", sheet_name)
    # GS vs Pytesseract + Gemini
    m3 = calculate_metrics(gold_series, pytess_gem_series, "Pytesseract + Gemini", sheet_name)
    # GS vs Pytesseract + Gemini
    m4 = calculate_metrics(gold_series, pytess_gem3_series, "Pytesseract + Gemini3", sheet_name)

    results.extend([
        {"sheet": sheet_name, "method": "Gemini",                **m1},
        {"sheet": sheet_name, "method": "Pytesseract",           **m2},
        {"sheet": sheet_name, "method": "Pytesseract + Gemini",  **m3},
        {"sheet": sheet_name, "method": "Pytesseract + Gemini3",  **m4},
    ])

# ------------------------------------------------
# Summary table
# ------------------------------------------------
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'Sheet':<20} {'Method':<25} {'WER':>7} {'CER':>7} {'Ins':>6} {'Del':>6} {'Sub':>6}")
print(f"{'-'*20} {'-'*25} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*6}")
for r in results:
    print(
        f"{r['sheet']:<20} {r['method']:<25} "
        f"{r['WER']:>7.4f} {r['CER']:>7.4f} "
        f"{r['insertions']:>6} {r['deletions']:>6} {r['substitutions']:>6}"
    )
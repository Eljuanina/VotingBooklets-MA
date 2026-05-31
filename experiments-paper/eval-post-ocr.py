import pandas as pd
import re
from jiwer import process_words, process_characters

# ------------------------------------------------
# Paths
# ------------------------------------------------
excel_path = "alignment_agent_gold.xlsx"

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
# Only evaluate selected sheets
# ------------------------------------------------
TARGET_SHEETS = ["rm2007", "de1985"]

results = []

for sheet_name in TARGET_SHEETS:
    print(f"\nProcessing sheet: {sheet_name}")
    
    df_sheet = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    
    # Expecting 4 columns now
    # 0 = GS
    # 1 = Gemini
    # 2 = Pytesseract
    # 3 = Pytesseract + Gemini (post-OCR)
    
    if df_sheet.shape[1] < 4:
        print(f"Sheet {sheet_name} skipped (needs 4 columns)")
        continue
    
    gold = df_sheet.iloc[:, 0].fillna("").astype(str)
    gemini = df_sheet.iloc[:, 1].fillna("").astype(str)
    pytess = df_sheet.iloc[:, 2].fillna("").astype(str)
    pytess_gemini = df_sheet.iloc[:, 3].fillna("").astype(str)
    
    # ---- Run all 3 comparisons ----
    
    gemini_metrics = calculate_metrics(gold, gemini, "Gemini", sheet_name)
    pytess_metrics = calculate_metrics(gold, pytess, "Pytesseract", sheet_name)
    pytess_gemini_metrics = calculate_metrics(gold, pytess_gemini, "Pytesseract+Gemini", sheet_name)
    
    # Store results
    results.append({"sheet": sheet_name, "method": "Gemini", **gemini_metrics})
    results.append({"sheet": sheet_name, "method": "Pytesseract", **pytess_metrics})
    results.append({"sheet": sheet_name, "method": "Pytesseract+Gemini", **pytess_gemini_metrics})

# ------------------------------------------------
# Save results
# ------------------------------------------------
df_results = pd.DataFrame(results)
print("\n" + "="*60)
print("ALL RESULTS:")
print("="*60)
print(df_results)

# Save to CSV
df_results.to_csv("excel_row_level_evaluation.csv", index=False)
print("\nSaved to excel_row_level_evaluation.csv")

# Also create a pivot table for easier comparison
pivot = df_results.pivot_table(
    index='sheet',
    columns='method',
    values=['WER', 'CER', 'insertions', 'deletions', 'substitutions']
)
print("\n" + "="*60)
print("COMPARISON TABLE:")
print("="*60)
print(pivot)
pivot.to_csv("excel_comparison_table.csv")
print("\nSaved comparison table to excel_comparison_table.csv")
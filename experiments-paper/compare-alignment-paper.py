"""
compare-alignment-paper.py
---------------------------
Evaluates all four alignment methods against gold-standard Excel files
and prints a combined comparison table.

Gold standard Excel format:
    Col A: DE  | Col B: FR  | Col C: IT  | Col D: RM (optional)
    One row per aligned segment.

Auto-aligned TSV naming convention (all in ALIGNED_DIR):
    multi_aligned_YEAR.tsv          ← multilingual MiniLM embeddings
    bert_aligned_YEAR.tsv           ← SwissBERT
    gemini_aligned_YEAR.tsv         ← Gemini
    bert_gemini_aligned_YEAR.tsv    ← SwissBERT + Gemini correction

Metrics per language:
    - Fuzzy F1  (precision / recall / F1)
    - CER vs gold
    - Cosine similarity (DE <> lang in the auto output)

Usage:
    python3 compare-alignment-paper.py --year 1977 1985 2007
    python3 compare-alignment-paper.py --year 1985 --gold-dir path/to/golds/ --dir path/to/tsvs/
"""

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from jiwer import cer as jiwer_cer
from sentence_transformers import SentenceTransformer

# ------------------------------------------------
# Config
# ------------------------------------------------
LANGS            = ["fr", "it", "rm"]
FUZZY_CER_THRESH = 0.15
EMBED_MODEL      = "paraphrase-multilingual-MiniLM-L12-v2"
ALIGNED_DIR      = Path("aligned")

ALL_METHODS = ["MiniLM", "SwissBERT", "Gemini", "BERT+Gemini","postgem25"]

# ------------------------------------------------
# Helpers
# ------------------------------------------------
def normalize(text) -> str:
    if not isinstance(text, str):
        text = "" if (text is None or (isinstance(text, float) and np.isnan(text))) else str(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fuzzy_match(a: str, b: str) -> bool:
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return False
    try:
        return jiwer_cer(a, b) <= FUZZY_CER_THRESH
    except Exception:
        return False


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (denom + 1e-9))


# ------------------------------------------------
# Load files
# ------------------------------------------------
def load_gold(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, header=None, engine="openpyxl")
    col_names = ["de", "fr", "it", "rm"]
    df = df.iloc[:, :4]
    df.columns = col_names[:len(df.columns)]
    for col in col_names:
        if col not in df.columns:
            df[col] = ""
    return df.fillna("").map(normalize)


def load_auto(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ["de", "fr", "it", "rm"]:
        if col not in df.columns:
            df[col] = ""
    return df.map(normalize)


# ------------------------------------------------
# Metrics
# ------------------------------------------------
def fuzzy_f1(auto_df, gold_df, lang):
    auto_pairs = [(r["de"], r[lang]) for _, r in auto_df.iterrows() if r["de"] and r[lang]]
    gold_pairs = [(r["de"], r[lang]) for _, r in gold_df.iterrows() if r["de"] and r[lang]]
    if not auto_pairs or not gold_pairs:
        return 0.0, 0.0, 0.0
    tp_auto = sum(
        1 for a_de, a_lang in auto_pairs
        if any(fuzzy_match(a_de, g_de) and fuzzy_match(a_lang, g_lang)
               for g_de, g_lang in gold_pairs)
    )
    tp_gold = sum(
        1 for g_de, g_lang in gold_pairs
        if any(fuzzy_match(g_de, a_de) and fuzzy_match(g_lang, a_lang)
               for a_de, a_lang in auto_pairs)
    )
    p  = tp_auto / len(auto_pairs)
    r  = tp_gold / len(gold_pairs)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def lang_cer(auto_df, gold_df, lang):
    gold_pairs = [(normalize(r["de"]), normalize(r[lang]))
                  for _, r in gold_df.iterrows() if r["de"] and r[lang]]
    scores = []
    for _, row in auto_df.iterrows():
        a_de, a_lang = normalize(row["de"]), normalize(row[lang])
        if not a_de or not a_lang:
            continue
        best_lang, best_d = None, float("inf")
        for g_de, g_lang in gold_pairs:
            if not g_de or not g_lang:
                continue
            try:
                d = jiwer_cer(a_de, g_de)
            except Exception:
                continue
            if d < best_d and d <= FUZZY_CER_THRESH:
                best_d, best_lang = d, g_lang
        if best_lang is None:
            continue
        try:
            scores.append(jiwer_cer(best_lang, a_lang))
        except Exception:
            continue
    return scores


def cosine_scores(auto_df, lang, model):
    pairs = [(r["de"], r[lang]) for _, r in auto_df.iterrows() if r["de"] and r[lang]]
    if not pairs:
        return []
    de_texts, lang_texts = zip(*pairs)
    de_embs   = model.encode(list(de_texts),   convert_to_numpy=True)
    lang_embs = model.encode(list(lang_texts), convert_to_numpy=True)
    return [cos_sim(a, b) for a, b in zip(de_embs, lang_embs)]


# ------------------------------------------------
# Evaluate one method
# ------------------------------------------------
def evaluate_one(auto_df, gold_df, model, method_label, year):
    present_langs = [l for l in LANGS if gold_df[l].str.strip().any()]
    results = []
    for lang in present_langs:
        if not auto_df[lang].str.strip().any():
            print(f"    [{lang.upper()}] not present in auto output — skipping")
            continue
        p, r, f1   = fuzzy_f1(auto_df, gold_df, lang)
        cer_scores = lang_cer(auto_df, gold_df, lang)
        cos        = cosine_scores(auto_df, lang, model)
        results.append({
            "year":       year,
            "method":     method_label,
            "lang":       lang,
            "precision":  p,
            "recall":     r,
            "f1":         f1,
            "cer_mean":   np.mean(cer_scores)   if cer_scores else None,
            "cer_median": np.median(cer_scores) if cer_scores else None,
            "cos_mean":   np.mean(cos)          if cos        else None,
        })
    return results


# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year",     type=str, nargs="+", required=True,
                        help="Year label(s) e.g. --year 1977 1985 2007")
    parser.add_argument("--gold-dir", type=Path, default=Path("."),
                        help="Folder with gold Excel files named gs-alignmentYY.xlsx (default: current dir)")
    parser.add_argument("--dir",      type=Path, default=ALIGNED_DIR,
                        help="Folder with aligned TSVs (default: aligned/)")
    parser.add_argument("--mini",     type=Path, help="MiniLM TSV (overrides auto-detect, single year only)")
    parser.add_argument("--bert",     type=Path, help="SwissBERT TSV (overrides auto-detect, single year only)")
    parser.add_argument("--gemini",   type=Path, help="Gemini TSV (overrides auto-detect, single year only)")
    args = parser.parse_args()

    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    prefixes = {
        "MiniLM":      "multi_aligned",
        "SwissBERT":   "bert_aligned",
        "Gemini":      "gemini_aligned",
        "BERT+Gemini": "bert_gemini_aligned",
        "postgem25":   "postgem25_aligned",
    }
    explicit = {
        "MiniLM":      args.mini,
        "SwissBERT":   args.bert,
        "Gemini":      args.gemini,
        "BERT+Gemini": None,
    }

    print(f"Years to evaluate: {args.year}")
    all_results = []

    for year in args.year:
        # Auto-match gold: try gs-alignment1985.xlsx then gs-alignment85.xlsx
        short_year = year[-2:]
        candidates = [
            args.gold_dir / f"gs-alignment{year}.xlsx",
            args.gold_dir / f"gs-alignment{short_year}.xlsx",
        ]
        gold_path = next((c for c in candidates if c.exists()), None)
        if gold_path is None:
            print(f"\n[{year}] No gold file found (tried: {[str(c) for c in candidates]}) — skipping")
            continue

        gold_df = load_gold(gold_path)
        print(f"\n{'#'*70}")
        print(f"# Year: {year}  |  Gold: {gold_path}  ({len(gold_df)} rows)")
        print(f"{'#'*70}")

        method_paths = {}
        for label, prefix in prefixes.items():
            if explicit.get(label) and len(args.year) == 1:
                method_paths[label] = explicit[label]
            else:
                candidate = args.dir / f"{prefix}_{year}.tsv"
                if candidate.exists():
                    method_paths[label] = candidate
                else:
                    print(f"  [{label}] not found: {candidate} — skipping")

        for label, tsv_path in method_paths.items():
            print(f"\n{'='*60}")
            print(f"  Method: {label}  |  {tsv_path}")
            print(f"{'='*60}")
            auto_df = load_auto(tsv_path)
            print(f"  Rows: {len(auto_df)}")
            res = evaluate_one(auto_df, gold_df, model, label, year)
            for r in res:
                cer_str = f"{r['cer_mean']:.4f}" if r["cer_mean"] is not None else "   N/A"
                cos_str = f"{r['cos_mean']:.4f}" if r["cos_mean"] is not None else "   N/A"
                print(f"  [{r['lang'].upper()}]  F1={r['f1']:.4f}  "
                      f"Prec={r['precision']:.4f}  Rec={r['recall']:.4f}  "
                      f"CER={cer_str}  CosSim={cos_str}")
            all_results.extend(res)

    # ------------------------------------------------
    # Combined summary table
    # ------------------------------------------------
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Year':<6} {'Method':<14} {'Lang':<5} {'F1':>7} {'Prec':>7} {'Rec':>7} "
          f"{'CER':>8} {'CosSim':>8}")
    print(f"{'-'*6} {'-'*14} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*8}")

    for year in args.year:
        first_year = True
        for label in ALL_METHODS:
            rows = [r for r in all_results if r["year"] == year and r["method"] == label]
            if not rows:
                continue
            first_method = True
            for r in rows:
                cer_str    = f"{r['cer_mean']:.4f}" if r["cer_mean"] is not None else "     N/A"
                cos_str    = f"{r['cos_mean']:.4f}" if r["cos_mean"] is not None else "     N/A"
                year_col   = year  if first_year   else ""
                method_col = label if first_method else ""
                print(f"{year_col:<6} {method_col:<14} {r['lang']:<5} {r['f1']:>7.4f} "
                      f"{r['precision']:>7.4f} {r['recall']:>7.4f} "
                      f"{cer_str:>8} {cos_str:>8}")
                first_year   = False
                first_method = False
        print(f"{'-'*6} {'-'*14} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*8}")


if __name__ == "__main__":
    main()





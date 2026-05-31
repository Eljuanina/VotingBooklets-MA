"""
alignment_scores.py — Segment alignment evaluation for booklets dataset
========================================================================
Compares two pipelines (agent, monolithic) against gold JSONL files.

Outputs:
  alignment_results_overall.csv   — AER/F1/precision/recall per year per pipeline
  alignment_results_by_lang.csv   — same metrics broken down by language

Dependencies:
    pip install rapidfuzz numpy
"""

import csv
import json
import re
from pathlib import Path

from rapidfuzz import fuzz
import numpy as np


##################################################
# CONFIG
##################################################

YEARS = ["1977", "1985", "2007"]

GOLD_DIR  = Path("gold_files")
AGENT_DIR = Path("Agent-VB-gold/Data/ParallelCorpora")

SIM_THRESHOLD = 0.70


##################################################
# TEXT NORMALISATION
##################################################

def normalize(text: str, case_sensitive: bool = False,
              strip_punctuation: bool = False) -> str:
    if not text:
        return ""
    if not case_sensitive:
        text = text.lower()
    text = re.sub(r"^\s*(\d+\.|[a-z]\.|[-•–—])\s+", "", text, flags=re.MULTILINE)
    if strip_punctuation:
        text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


##################################################
# SAFE JSONL LOADER
##################################################

def load_jsonl(path):
    data = []
    with open(path, encoding="utf8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                obj.pop("date", None)
                data.append(obj)
            except json.JSONDecodeError:
                print(f"⚠️  Skipping invalid JSON at {path}:{i}")
    print(f"  Loaded {len(data)} segments from {path}")
    return data


##################################################
# LANGUAGE DETECTION
##################################################

def detect_languages(gold):
    langs = set()
    for seg in gold:
        for k, v in seg.items():
            if k != "date" and v:
                langs.add(k)
    return sorted(langs)


##################################################
# SEGMENT SIMILARITY
##################################################

def segment_similarity(a, b, langs):
    scores = []
    for lang in langs:
        ta = normalize(a.get(lang, ""))
        tb = normalize(b.get(lang, ""))
        if ta and tb:
            scores.append(fuzz.token_set_ratio(ta, tb) / 100)
    return sum(scores) / len(scores) if scores else 0.0


##################################################
# AER METRICS
##################################################

def compute_aer(gold, system, langs):
    matches, similarities = [], []
    for g in gold:
        best_score = max(
            (segment_similarity(g, s, langs) for s in system),
            default=0
        )
        if best_score >= SIM_THRESHOLD:
            matches.append(True)
            similarities.append(best_score)

    S, A, M = len(gold), len(system), len(matches)
    precision = M / A if A else 0
    recall    = M / S if S else 0
    f1        = 2 * precision * recall / (precision + recall + 1e-9) if (precision + recall) else 0
    aer       = 1 - (2 * M) / (S + A) if (S + A) else 1

    return {
        "gold_segments":    S,
        "system_segments":  A,
        "matches":          M,
        "precision":        round(precision, 4),
        "recall":           round(recall, 4),
        "f1":               round(f1, 4),
        "aer":              round(aer, 4),
        "over_segmentation":  round(max(0, A - S) / S, 4) if S else 0,
        "under_segmentation": round(max(0, S - A) / S, 4) if S else 0,
        "avg_similarity":   round(float(np.mean(similarities)), 4) if similarities else 0,
    }


##################################################
# LANGUAGE-WISE METRICS
##################################################

def compute_language_scores(gold, system, lang):
    matches, similarities = 0, []
    for g in gold:
        g_text = normalize(g.get(lang, ""))
        if not g_text:
            continue
        best_score = 0
        for s in system:
            s_text = normalize(s.get(lang, ""))
            if not s_text:
                continue
            score = fuzz.token_set_ratio(g_text, s_text) / 100
            if score > best_score:
                best_score = score
        if best_score >= SIM_THRESHOLD:
            matches += 1
            similarities.append(best_score)

    gold_count   = sum(1 for g in gold   if g.get(lang))
    system_count = sum(1 for s in system if s.get(lang))
    precision = matches / system_count if system_count else 0
    recall    = matches / gold_count   if gold_count   else 0
    f1        = 2 * precision * recall / (precision + recall + 1e-9) if (precision + recall) else 0
    aer       = 1 - (2 * matches) / (gold_count + system_count) if (gold_count + system_count) else 1

    return {
        "gold":           gold_count,
        "system":         system_count,
        "matches":        matches,
        "precision":      round(precision, 4),
        "recall":         round(recall, 4),
        "f1":             round(f1, 4),
        "aer":            round(aer, 4),
        "avg_similarity": round(sum(similarities) / len(similarities), 4) if similarities else 0,
    }


##################################################
# CSV WRITERS
##################################################

OVERALL_FIELDS = [
    "year", "pipeline",
    "gold_segments", "system_segments", "matches",
    "precision", "recall", "f1", "aer",
    "over_segmentation", "under_segmentation", "avg_similarity",
]

LANG_FIELDS = [
    "year", "pipeline", "lang",
    "gold", "system", "matches",
    "precision", "recall", "f1", "aer", "avg_similarity",
]

def write_csv(path: str, fields: list, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


##################################################
# PRINTING
##################################################

def print_metrics(label: str, res: dict) -> None:
    print(f"\n  --- {label} ---")
    for k, v in res.items():
        print(f"  {k:25s}: {v:.4f}" if isinstance(v, float) else f"  {k:25s}: {v}")


##################################################
# MAIN
##################################################

def main() -> None:
    overall_rows = []
    lang_rows    = []

    for year in YEARS:
        print(f"\n{'═' * 60}")
        print(f"  Year: {year}")
        print(f"{'═' * 60}")

        # ── Load files ────────────────────────────────────────────
        missing = []
        gp = GOLD_DIR  / f"gold_aligned_{year}.jsonl"
        ap = AGENT_DIR / f"parallel_corpus_{year}.jsonl"

        for p in (gp, ap):
            if not p.exists():
                missing.append(str(p))

        if missing:
            print(f"  [WARNING] Skipping {year} — missing files:")
            for m in missing:
                print(f"    {m}")
            continue

        gold  = load_jsonl(gp)
        agent = load_jsonl(ap)
        langs = detect_languages(gold)

        print(f"  Detected languages: {langs}")

        # ── Overall AER ───────────────────────────────────────────
        res = compute_aer(gold, agent, langs)
        print_metrics("AGENT — overall", res)
        overall_rows.append({"year": year, "pipeline": "agent", **res})

        # ── Per-language ──────────────────────────────────────────
        print(f"\n  Language breakdown:")
        for lang in langs:
            res = compute_language_scores(gold, agent, lang)

            print(
                f"  {lang.upper()} / agent   "
                f"F1={res['f1']:.4f}  "
                f"AER={res['aer']:.4f}  "
                f"P={res['precision']:.4f}  "
                f"R={res['recall']:.4f}"
            )

            lang_rows.append({
                "year": year,
                "pipeline": "agent",
                "lang": lang,
                **res
            })

    # ── Write CSVs ────────────────────────────────────────────────
    write_csv("alignment_results_overall.csv", OVERALL_FIELDS, overall_rows)
    write_csv("alignment_results_by_lang.csv", LANG_FIELDS,    lang_rows)

    print(f"\n  Files written:")
    print(f"     alignment_results_overall.csv")
    print(f"     alignment_results_by_lang.csv\n")


if __name__ == "__main__":
    main()
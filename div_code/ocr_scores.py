"""
ocr_scores.py — Batch OCR evaluation for booklets dataset
==========================================================
Compares two pipelines against the same gold files:
  - monolithic : monolithic-pipeline/cleaned_ocr_txt/gemini-ocr/{Lang}/{lang}_{year}.txt
  - agent      : new-agent/data/booklets_txt/booklets_{lang}_{year}.txt

Metrics: WER, CER (CSV only), substitutions/deletions/insertions (word & char),
         character-class accuracy (digits/letters/spaces/other).

Outputs (per pipeline):
  ocr_eval_results_{tag}.csv
  ocr_eval_by_language_{tag}.csv
  ocr_eval_by_year_{tag}.csv

Dependencies:
    pip install jiwer


IMPORTANT

pls change the route to the files depending on your file structure

the route indicated in this code was how i structured the files on my local computer
it's not the route that works with the structure of this repository

you first need to change the paths if you want to rerun the evaluation

"""

import csv
import re
import sys
from pathlib import Path

try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    print(
        "[WARNING] jiwer not found — install with: pip install jiwer\n"
        "          WER/CER will fall back to built-in implementation.\n",
        file=sys.stderr,
    )


# ─────────────────────────────────────────────
# Text normalisation
# ─────────────────────────────────────────────

def normalise(text: str, case_sensitive: bool = False,
              strip_punctuation: bool = False) -> str:
    if not case_sensitive:
        text = text.lower()
    # Remove list markers: 1. 2. 3., a. b. c., -, •, –, — etc.
    text = re.sub(r"^\s*(\d+\.|[a-z]\.|[-•–—])\s+", "", text, flags=re.MULTILINE)
    if strip_punctuation:
        text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────
# Fallback edit-distance (used when jiwer not installed)
# ─────────────────────────────────────────────

def _levenshtein(seq_a: list, seq_b: list) -> tuple[int, int, int, int]:
    n, m = len(seq_a), len(seq_b)
    dp = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                candidates = [
                    (dp[i-1][j-1][0]+1, dp[i-1][j-1][1]+1, dp[i-1][j-1][2],   dp[i-1][j-1][3]),
                    (dp[i-1][j  ][0]+1, dp[i-1][j  ][1],   dp[i-1][j  ][2]+1, dp[i-1][j  ][3]),
                    (dp[i  ][j-1][0]+1, dp[i  ][j-1][1],   dp[i  ][j-1][2],   dp[i  ][j-1][3]+1),
                ]
                dp[i][j] = min(candidates, key=lambda x: x[0])
    total, subs, dels, ins = dp[n][m]
    return subs, dels, ins, total


# ─────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────

def compute_metrics(ref: str, hyp: str) -> dict:
    if not JIWER_AVAILABLE:
        ref_w, hyp_w = ref.split(), hyp.split()
        s, d, i, total = _levenshtein(ref_w, hyp_w)
        wer_val = total / max(len(ref_w), 1)
        ref_c, hyp_c = list(ref), list(hyp)
        sc, dc, ic, totalc = _levenshtein(ref_c, hyp_c)
        cer_val = totalc / max(len(ref_c), 1)
        return {
            "wer_pct": round(wer_val * 100, 2),
            "cer_pct": round(cer_val * 100, 2),
            "word_subs": s,  "word_dels": d,  "word_ins": i,
            "char_subs": sc, "char_dels": dc, "char_ins": ic,
            "ref_words": len(ref_w), "hyp_words": len(hyp_w),
            "ref_chars": len(ref),   "hyp_chars": len(hyp),
            "source": "built-in (jiwer not installed)",
        }

    word_out = jiwer.process_words(ref, hyp)
    char_out = jiwer.process_characters(ref, hyp)
    return {
        "wer_pct": round(word_out.wer * 100, 2),
        "cer_pct": round(char_out.cer * 100, 2),
        "word_subs": word_out.substitutions,
        "word_dels": word_out.deletions,
        "word_ins":  word_out.insertions,
        "char_subs": char_out.substitutions,
        "char_dels": char_out.deletions,
        "char_ins":  char_out.insertions,
        "ref_words": len(ref.split()), "hyp_words": len(hyp.split()),
        "ref_chars": len(ref),         "hyp_chars": len(hyp),
        "source": "jiwer",
    }


def character_class_accuracy(ref: str, hyp: str) -> dict:
    def dist(text):
        return {
            "digits":  sum(c.isdigit()  for c in text),
            "letters": sum(c.isalpha()  for c in text),
            "spaces":  sum(c.isspace()  for c in text),
            "other":   sum(not (c.isdigit() or c.isalpha() or c.isspace()) for c in text),
        }
    r, h = dist(ref), dist(hyp)
    result = {}
    for cls in ("digits", "letters", "spaces", "other"):
        rv, hv = r[cls], h[cls]
        acc = (1.0 if rv == 0 and hv == 0
               else 0.0 if rv == 0
               else max(0.0, 1.0 - abs(rv - hv) / rv))
        result[f"{cls}_count_ref"] = rv
        result[f"{cls}_count_hyp"] = hv
        result[f"{cls}_acc"]       = round(acc * 100, 2)
    return result


def evaluate(ref_raw: str, hyp_raw: str,
             case_sensitive: bool = False,
             strip_punctuation: bool = False) -> dict:
    ref = normalise(ref_raw, case_sensitive, strip_punctuation)
    hyp = normalise(hyp_raw, case_sensitive, strip_punctuation)
    m  = compute_metrics(ref, hyp)
    cc = character_class_accuracy(ref, hyp)
    return {**m, **cc}


# ─────────────────────────────────────────────
# CSV field definitions
# ─────────────────────────────────────────────

FILE_FIELDS = [
    "year", "lang",
    "wer_pct", "cer_pct",
    "word_subs", "word_dels", "word_ins",
    "char_subs", "char_dels", "char_ins",
    "ref_words", "hyp_words", "ref_chars", "hyp_chars",
    "digits_count_ref", "digits_count_hyp", "digits_acc",
    "letters_count_ref", "letters_count_hyp", "letters_acc",
    "spaces_count_ref", "spaces_count_hyp", "spaces_acc",
    "other_count_ref", "other_count_hyp", "other_acc",
    "status",
]

SUMMARY_FIELDS_LANG = [
    "lang", "n_files",
    "mean_wer_pct", "min_wer_pct", "max_wer_pct",
    "mean_cer_pct", "min_cer_pct", "max_cer_pct",
    "mean_digits_acc", "mean_letters_acc",
    "mean_spaces_acc", "mean_other_acc",
]

SUMMARY_FIELDS_YEAR = [
    "year", "n_files",
    "mean_wer_pct", "min_wer_pct", "max_wer_pct",
    "mean_cer_pct", "min_cer_pct", "max_cer_pct",
    "mean_digits_acc", "mean_letters_acc",
    "mean_spaces_acc", "mean_other_acc",
]


# ─────────────────────────────────────────────
# Row builders
# ─────────────────────────────────────────────

def make_file_row(year: int, lang: str, res: dict, status: str = "ok") -> dict:
    row = {"year": year, "lang": lang, "status": status}
    for f in FILE_FIELDS:
        if f not in ("year", "lang", "status"):
            row[f] = res.get(f)
    return row


def empty_file_row(year: int, lang: str, status: str) -> dict:
    row = {f: None for f in FILE_FIELDS}
    row.update({"year": year, "lang": lang, "status": status})
    return row


# ─────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────

def summarise(rows: list[dict], group_key: str) -> list[dict]:
    groups: dict = {}
    for r in rows:
        if r["status"] != "ok":
            continue
        groups.setdefault(r[group_key], []).append(r)

    summary_rows = []
    for key in sorted(groups, key=str):
        g = groups[key]

        def avg(field):
            vals = [r[field] for r in g if r[field] is not None]
            return round(sum(vals) / len(vals), 2) if vals else None

        def mn(field):
            vals = [r[field] for r in g if r[field] is not None]
            return round(min(vals), 2) if vals else None

        def mx(field):
            vals = [r[field] for r in g if r[field] is not None]
            return round(max(vals), 2) if vals else None

        summary_rows.append({
            group_key:          key,
            "n_files":          len(g),
            "mean_wer_pct":     avg("wer_pct"),
            "min_wer_pct":      mn("wer_pct"),
            "max_wer_pct":      mx("wer_pct"),
            "mean_cer_pct":     avg("cer_pct"),
            "min_cer_pct":      mn("cer_pct"),
            "max_cer_pct":      mx("cer_pct"),
            "mean_digits_acc":  avg("digits_acc"),
            "mean_letters_acc": avg("letters_acc"),
            "mean_spaces_acc":  avg("spaces_acc"),
            "mean_other_acc":   avg("other_acc"),
        })
    return summary_rows


# ─────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────

def write_csv(path: str, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_file_row(row: dict) -> None:
    tag = f"{row['lang'].upper()} {row['year']}"
    if row["status"] != "ok":
        print(f"  [WARNING] {tag}: {row['status']}")
        return
    print(
        f"  {tag:<8}  "
        f"WER={row['wer_pct']:6.2f}%  "
        f"CER={row['cer_pct']:6.2f}%  "
        f"| subs={row['word_subs']} dels={row['word_dels']} ins={row['word_ins']}  "
        f"| digits={row['digits_acc']}%  letters={row['letters_acc']}%  "
        f"spaces={row['spaces_acc']}%  other={row['other_acc']}%"
    )


def print_summary(label: str, rows: list[dict], key: str) -> None:
    print(f"\n  {label}")
    print("  " + "─" * 70)
    for r in rows:
        print(
            f"  {str(r[key]).upper():<6}  n={r['n_files']}  "
            f"WER={r['mean_wer_pct']:6.2f}% [{r['min_wer_pct']}–{r['max_wer_pct']}]  "
            f"CER={r['mean_cer_pct']:6.2f}% [{r['min_cer_pct']}–{r['max_cer_pct']}]  "
            f"| digits={r['mean_digits_acc']}%  letters={r['mean_letters_acc']}%  "
            f"spaces={r['mean_spaces_acc']}%  other={r['mean_other_acc']}%"
        )


def print_overall(ok: list[dict]) -> None:
    if not ok:
        return
    avg = lambda f: sum(r[f] for r in ok if r[f] is not None) / len(ok)
    print(f"\n  Overall averages ({len(ok)} files):")
    print(f"     Mean WER : {avg('wer_pct'):.2f}%")
    print(f"     Mean CER : {avg('cer_pct'):.2f}%")


# ─────────────────────────────────────────────
# Per-pipeline runner
# ─────────────────────────────────────────────

def run_pipeline(tag: str, combos: list, gold_path_fn, hyp_path_fn) -> None:
    out_dir = Path("ocr-results-final")
    out_dir.mkdir(exist_ok=True)

    print(f"\n{'═' * 72}")
    print(f"  Pipeline: {tag.upper()}")
    print(f"{'═' * 72}")

    file_rows = []

    for year, lang in combos:
        gp = gold_path_fn(year, lang)
        hp = hyp_path_fn(year, lang)

        missing = []
        if not gp.exists(): missing.append(f"gold not found: {gp}")
        if not hp.exists(): missing.append(f"hyp not found: {hp}")

        if missing:
            row = empty_file_row(year, lang, status="; ".join(missing))
            file_rows.append(row)
            print_file_row(row)
            continue

        ref_text = gp.read_text(encoding="utf-8", errors="replace")
        hyp_text = hp.read_text(encoding="utf-8", errors="replace")
        res      = evaluate(ref_text, hyp_text)
        row      = make_file_row(year, lang, res)
        file_rows.append(row)
        print_file_row(row)

    # CSVs
    write_csv(out_dir/f"ocr_eval_results_{tag}.csv",     FILE_FIELDS,        file_rows)
    write_csv(out_dir/f"ocr_eval_by_language_{tag}.csv", SUMMARY_FIELDS_LANG, summarise(file_rows, "lang"))
    write_csv(out_dir/f"ocr_eval_by_year_{tag}.csv",     SUMMARY_FIELDS_YEAR, summarise(file_rows, "year"))

    # Printed summaries
    print_summary("By language", summarise(file_rows, "lang"), "lang")
    print_summary("By year",     summarise(file_rows, "year"), "year")

    print("\n  " + "─" * 70)
    print_overall([r for r in file_rows if r["status"] == "ok"])

    print(f"\n  Files written:")
    print(f"     ocr-results-final/ocr_eval_results_{tag}.csv")
    print(f"     ocr-results-final/ocr_eval_by_language_{tag}.csv")
    print(f"     ocr-results-final/ocr_eval_by_year_{tag}.csv")


# ─────────────────────────────────────────────
# Main — edit paths here
# ─────────────────────────────────────────────

def main() -> None:
    COMBOS = [
        (1977, "it"), (1977, "fr"), (1977, "de"),
        (1985, "it"), (1985, "fr"), (1985, "de"), (1985, "rm"),
        (2007, "it"), (2007, "fr"), (2007, "de"), (2007, "rm"),
    ]

    LANG_FOLDER = {
        "de": "Deutsch",
        "fr": "Französisch",
        "it": "Italienisch",
        "rm": "Rätoromanisch",
    }

    def gold_path(year: int, lang: str) -> Path:
        return Path(f"gold_files/gold_ocr/gold_{year}/gold_{lang}_{year}.txt")

    # ── Monolithic pipeline ───────────────────────────────────────
    def mono_hyp_path(year: int, lang: str) -> Path:
        folder = LANG_FOLDER[lang]
        return Path(f"monolithic-pipeline/cleaned_ocr_txt/gemini-ocr/{folder}/{lang}_{year}.txt")

    # ── Agent pipeline ────────────────────────────────────────────
    def agent_hyp_path(year: int, lang: str) -> Path:
        return Path(f"new-agent/data/booklets_txt/booklets_{lang}_{year}.txt")

    run_pipeline("mono",  COMBOS, gold_path, mono_hyp_path)
    run_pipeline("agent", COMBOS, gold_path, agent_hyp_path)


    def docling_hyp_path(year: int, lang: str) -> Path:
        return Path(f"ocr-get-text/docling_output/{lang}_{year}_extracted.txt")

    def gemini_hyp_path(year: int, lang: str) -> Path:
        return Path(f"ocr-get-text/gemini-ocr/{lang}_{year}_extracted.txt")

    #  from compare-ocr folder now
    def pytesseract_hyp_path(year: int, lang: str) -> Path:
        return Path(f"compare-ocr/pytesseract/{lang}_{year}_extracted.txt")


    def gemini_post_hyp_path(year: int, lang: str) -> Path:
        return Path(f"compare-ocr/post-ocr-tesseract/{lang}_{year}_extracted.txt")

    def gemini_3_hyp_path(year: int, lang: str) -> Path:
        return Path(f"compare-ocr/gem3-post-ocr-tesseract/{lang}_{year}_extracted.txt")

    # finepdf
    def finepdf_hyp_path(year: int, lang: str) -> Path:
        return Path(f"finepdf/finepdf_files/{year}/{lang}_{year}.txt")

    run_pipeline("docling", COMBOS, gold_path, docling_hyp_path)
    run_pipeline("gemini-ocr", COMBOS, gold_path, gemini_hyp_path)

    run_pipeline("pytesseract", COMBOS, gold_path, pytesseract_hyp_path)
    run_pipeline("gemini-post", COMBOS, gold_path, gemini_post_hyp_path)
    run_pipeline("gemini-3-post", COMBOS, gold_path, gemini_3_hyp_path)

    run_pipeline("finepdf", COMBOS, gold_path, finepdf_hyp_path)

"""
# uncomment for antigravity evaluation
    from pathlib import Path

    BASE = Path(__file__).parent   # folder where script lives

    COMBOS = [
        (77, "it"), (77, "fr"), (77, "de"),
        (85, "it"), (85, "fr"), (85, "de"), (85, "rm"),
        (7,  "it"), (7,  "fr"), (7,  "de"), (7,  "rm"),
    ]

    def gold_path(year: int, lang: str) -> Path:
        year_full = 1900 + year if year != 7 else 2007
        return (
            BASE
            / "gold_files"
            / "gold_ocr"
            / f"gold_{year_full}"
            / f"gold_{lang}_{year_full}.txt"
        )

    # OCR FILES
    def desktop_ocr_path(year: int, lang: str) -> Path:
        return BASE / f"ocr_antigravity-{lang}{year:02}.txt"

    run_pipeline(
        "desktop-ocr",
        COMBOS,
        gold_path,
        desktop_ocr_path,
    )

"""

if __name__ == "__main__":
    main()



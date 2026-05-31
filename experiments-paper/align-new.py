"""
align_languages.py
------------------
Aligns language text files against a DE anchor using multilingual sentence
embeddings + dynamic programming.

DE anchor  : one line per GS segment (clean, correctly chunked)
Other langs: dot-split text (many small fragments that need merging)

Allowed DP moves (N = 1..MAX_MERGE):
    1:1   direct match
    1:0   anchor line unmatched → other cell empty
    0:1   other fragment dropped
    1:N   one anchor line matched to N consecutive other fragments
    N:1   N anchor lines merged into one other fragment

Expected file names in INPUT_DIR:
    1985_de.txt   1985_fr.txt   1985_it.txt   1985_rm.txt   ...

Output TSVs go to OUTPUT_DIR:
    aligned_1985.tsv   (columns: de  fr  it  rm)
"""

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ------------------------------------------------
# Config
# ------------------------------------------------
INPUT_DIR  = Path("texts")
OUTPUT_DIR = Path("aligned")
OUTPUT_DIR.mkdir(exist_ok=True)

ALL_LANGS = ["de", "fr", "it", "rm"]
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Maximum N for both 1:N and N:1 merges
MAX_MERGE = 5

# DP costs
MATCH_BONUS   =  1.0
SKIP_PENALTY  = -0.3    # penalty per unmatched line on either side
MERGE_PENALTY = -0.05   # small penalty per extra line merged

# Min cosine sim to allow any merge (n > 1)
MIN_SIM_TO_MERGE = 0.30

# ------------------------------------------------
# Helpers
# ------------------------------------------------
def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_lines(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return [normalize(l) for l in f if normalize(l)]


def embed(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (denom + 1e-9))


def mean_vec(vecs: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(vecs), axis=0)


# ------------------------------------------------
# DP aligner
# ------------------------------------------------
def dp_align(
    anchor_embs:  np.ndarray,
    other_embs:   np.ndarray,
    anchor_texts: list[str],
    other_texts:  list[str],
) -> list[tuple[str, str]]:
    """
    Returns list of (anchor_text, other_text) pairs.
    other_text is "" when the anchor line has no match.
    Merged texts are joined with a space.
    """
    A, B = len(anchor_embs), len(other_embs)
    NEG_INF = -1e9

    dp   = np.full((A + 1, B + 1), NEG_INF)
    back = {}   # (i, j) → (pi, pj, a_span, o_span)
    dp[0, 0] = 0.0

    def update(ni, nj, val, fi, fj, a_span, o_span):
        if val > dp[ni, nj]:
            dp[ni, nj] = val
            back[(ni, nj)] = (fi, fj, a_span, o_span)

    for i in range(A + 1):
        for j in range(B + 1):
            cur = dp[i, j]
            if cur == NEG_INF:
                continue

            # skip anchor line (empty other cell)
            if i < A:
                update(i+1, j, cur + SKIP_PENALTY, i, j, 1, 0)

            # skip other fragment (no row emitted)
            if j < B:
                update(i, j+1, cur + SKIP_PENALTY, i, j, 0, 1)

            # 1:N — one anchor line vs N consecutive other fragments
            if i < A:
                merged_other = []
                for n in range(1, MAX_MERGE + 1):
                    jo = j + n - 1
                    if jo >= B:
                        break
                    merged_other.append(other_embs[jo])
                    mv  = mean_vec(merged_other)
                    sim = cos_sim(anchor_embs[i], mv)
                    # always allow n=1; require min sim for merges
                    if n == 1 or sim >= MIN_SIM_TO_MERGE:
                        penalty = (n - 1) * MERGE_PENALTY
                        update(i+1, j+n,
                               cur + MATCH_BONUS * sim + penalty,
                               i, j, 1, n)

            # N:1 — N consecutive anchor lines vs one other fragment
            if j < B:
                merged_anchor = []
                for n in range(2, MAX_MERGE + 1):  # start at 2, 1:1 covered above
                    ia = i + n - 1
                    if ia >= A:
                        break
                    merged_anchor.append(anchor_embs[i + n - 1])
                    mv  = mean_vec([anchor_embs[i + k] for k in range(n)])
                    sim = cos_sim(mv, other_embs[j])
                    if sim >= MIN_SIM_TO_MERGE:
                        penalty = (n - 1) * MERGE_PENALTY
                        update(i+n, j+1,
                               cur + MATCH_BONUS * sim + penalty,
                               i, j, n, 1)

    # Traceback
    raw_pairs = []
    i, j = A, B
    while i > 0 or j > 0:
        if (i, j) not in back:
            break
        pi, pj, a_span, o_span = back[(i, j)]
        if a_span > 0:
            raw_pairs.append((pi, pj, a_span, o_span))
        i, j = pi, pj

    raw_pairs.reverse()

    # Resolve indices → text
    def resolve(texts, start, span):
        if span == 0 or start is None:
            return ""
        return " ".join(texts[start + k] for k in range(span))

    result = []
    for ai, oj, a_span, o_span in raw_pairs:
        a_text = resolve(anchor_texts, ai, a_span)
        o_text = resolve(other_texts,  oj, o_span)
        if a_text:
            result.append((a_text, o_text))

    return result


# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Discover year → {lang: path} from filenames like de_1985.txt
    year_lang_files: dict[str, dict[str, Path]] = {}
    for f in sorted(INPUT_DIR.glob("*.txt")):
        parts = f.stem.split("_", 1)
        if len(parts) == 2:
            lang, year = parts
            year_lang_files.setdefault(year, {})[lang] = f

    if not year_lang_files:
        print(f"No .txt files found in {INPUT_DIR}/")
        return

    for year, lang_files in sorted(year_lang_files.items()):
        if "de" not in lang_files:
            print(f"[{year}] Skipping — no DE anchor file found")
            continue

        present_langs = [l for l in ALL_LANGS if l in lang_files]
        other_langs   = [l for l in present_langs if l != "de"]

        print(f"\n{'='*60}")
        print(f"Year: {year}  |  Languages: {present_langs}")
        print(f"Moves: 1:1, 1:N, N:1  (N up to {MAX_MERGE})")
        print(f"{'='*60}")

        # Encode DE anchor
        de_lines = read_lines(lang_files["de"])
        print(f"  DE: {len(de_lines)} lines — encoding...")
        de_embs = embed(de_lines, model)

        # Align each other language independently to DE
        aligned_cols: dict[str, list[tuple[str, str]]] = {}
        for lang in other_langs:
            other_lines = read_lines(lang_files[lang])
            print(f"  {lang.upper()}: {len(other_lines)} lines — encoding...")
            other_embs = embed(other_lines, model)
            print(f"  {lang.upper()}: aligning...")
            pairs = dp_align(de_embs, other_embs, de_lines, other_lines)
            aligned_cols[lang] = pairs
            print(f"    → {len(pairs)} aligned rows")

        # Build DataFrame: base on DE↔first-other alignment, join remaining langs by DE text
        if other_langs:
            base_lang = other_langs[0]
            rows = [{"de": a, base_lang: b} for a, b in aligned_cols[base_lang]]
        else:
            rows = [{"de": line} for line in de_lines]

        for lang in other_langs[1:]:
            de_to_other: dict[str, str] = {}
            for a, b in aligned_cols[lang]:
                de_to_other.setdefault(a, b)
            for row in rows:
                row[lang] = de_to_other.get(row["de"], "")

        col_order = [l for l in ALL_LANGS if l in present_langs]
        df = pd.DataFrame(rows, columns=col_order).fillna("")

        out_path = OUTPUT_DIR / f"multi_aligned_{year}.tsv"
        df.to_csv(out_path, sep="\t", index=False)
        print(f"  Saved → {out_path}  ({len(df)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
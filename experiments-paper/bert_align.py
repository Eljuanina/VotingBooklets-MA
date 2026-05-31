"""
align_languages_swissbert.py
-----------------------------
Same DP alignment logic as align_languages.py but uses sentence-swissbert
(jgrosjean-mathesis/sentence-swissbert) for embeddings.

SwissBERT routes through language adapters — set_default_language() must be
called before encoding each language. Supported codes: de_CH, fr_CH, it_CH, rm_CH.

Expected file names in INPUT_DIR:
    de_1985.txt   fr_1985.txt   it_1985.txt   rm_1985.txt   ...

Output TSVs go to OUTPUT_DIR:
    bert_aligned_1985.tsv   (columns: de  fr  it  rm)
"""

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# ------------------------------------------------
# Config
# ------------------------------------------------
INPUT_DIR  = Path("texts")
OUTPUT_DIR = Path("aligned")
OUTPUT_DIR.mkdir(exist_ok=True)

ALL_LANGS = ["de", "fr", "it", "rm"]

LANG_ADAPTER = {
    "de": "de_CH",
    "fr": "fr_CH",
    "it": "it_CH",
    "rm": "rm_CH",
}

MODEL_NAME = "jgrosjean-mathesis/sentence-swissbert"
BATCH_SIZE = 32

# DP costs
MATCH_BONUS      =  1.0
SKIP_PENALTY     = -0.3
MERGE_PENALTY    = -0.05

MAX_MERGE        =  5
MIN_SIM_TO_MERGE =  0.30

# ------------------------------------------------
# Load model
# ------------------------------------------------
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model     = AutoModel.from_pretrained(MODEL_NAME)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model on: {device}")


# ------------------------------------------------
# Embedding
# ------------------------------------------------
def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def embed(texts: list[str], lang: str) -> np.ndarray:
    """Encode texts using the SwissBERT language adapter for `lang`."""
    adapter = LANG_ADAPTER[lang]
    # XmodModel exposes set_default_language(); AutoModel loads it as XmodModel
    if hasattr(model, "set_default_language"):
        model.set_default_language(adapter)
    else:
        raise RuntimeError(
            f"Model does not have set_default_language(). "
            f"Make sure transformers>=4.33 is installed and the model loaded correctly."
        )
    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**encoded)
        embs = mean_pooling(out.last_hidden_state, encoded["attention_mask"])
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        all_embs.append(embs.cpu().numpy())
    return np.vstack(all_embs)


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


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (denom + 1e-9))


def mean_vec(vecs: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(vecs), axis=0)


# ------------------------------------------------
# DP aligner  (identical logic to align_languages.py)
# ------------------------------------------------
def dp_align(
    anchor_embs:  np.ndarray,
    other_embs:   np.ndarray,
    anchor_texts: list[str],
    other_texts:  list[str],
) -> list[tuple[str, str]]:
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

            # skip anchor (empty other cell)
            if i < A:
                update(i+1, j, cur + SKIP_PENALTY, i, j, 1, 0)

            # skip other fragment
            if j < B:
                update(i, j+1, cur + SKIP_PENALTY, i, j, 0, 1)

            # 1:N — one anchor vs N other fragments
            if i < A:
                merged_other = []
                for n in range(1, MAX_MERGE + 1):
                    jo = j + n - 1
                    if jo >= B:
                        break
                    merged_other.append(other_embs[jo])
                    mv  = mean_vec(merged_other)
                    sim = cos_sim(anchor_embs[i], mv)
                    if n == 1 or sim >= MIN_SIM_TO_MERGE:
                        update(i+1, j+n,
                               cur + MATCH_BONUS * sim + (n-1) * MERGE_PENALTY,
                               i, j, 1, n)

            # N:1 — N anchor lines vs one other fragment
            if j < B:
                for n in range(2, MAX_MERGE + 1):
                    if i + n - 1 >= A:
                        break
                    mv  = mean_vec([anchor_embs[i + k] for k in range(n)])
                    sim = cos_sim(mv, other_embs[j])
                    if sim >= MIN_SIM_TO_MERGE:
                        update(i+n, j+1,
                               cur + MATCH_BONUS * sim + (n-1) * MERGE_PENALTY,
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

    def resolve(texts, start, span):
        if span == 0 or start is None:
            return ""
        return " ".join(texts[start + k] for k in range(span))

    return [
        (resolve(anchor_texts, ai, a_span), resolve(other_texts, oj, o_span))
        for ai, oj, a_span, o_span in raw_pairs
        if resolve(anchor_texts, ai, a_span)
    ]


# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
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
        de_embs = embed(de_lines, "de")

        # Align each other language independently to DE
        aligned_cols: dict[str, list[tuple[str, str]]] = {}
        for lang in other_langs:
            other_lines = read_lines(lang_files[lang])
            print(f"  {lang.upper()}: {len(other_lines)} lines — encoding...")
            other_embs = embed(other_lines, lang)
            print(f"  {lang.upper()}: aligning...")
            pairs = dp_align(de_embs, other_embs, de_lines, other_lines)
            aligned_cols[lang] = pairs
            print(f"    → {len(pairs)} aligned rows")

        # Build DataFrame
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

        out_path = OUTPUT_DIR / f"bert_aligned_{year}.tsv"
        df.to_csv(out_path, sep="\t", index=False)
        print(f"  Saved → {out_path}  ({len(df)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
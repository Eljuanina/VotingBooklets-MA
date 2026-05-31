import re
import unicodedata
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# ------------------------------------------------
# Config
# ------------------------------------------------
INPUT_DIR  = Path("gemini-ocr-2.5-flash-lite")
OUTPUT_DIR = Path("corpus/aligned-bert")
CACHE_DIR  = OUTPUT_DIR / ".cache"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ALL_LANGS  = ["de", "fr", "it", "rm"]

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

# Romansh month names → month number
RM_MONTHS = {
    "schaner": 1, "favrer": 2, "mars": 3, "avrigl": 4, "matg": 5,
    "zercladur": 6, "fanadur": 7, "avust": 8, "settember": 9,
    "october": 10, "november": 11, "december": 12,
}


# ------------------------------------------------
# Date extraction
# ------------------------------------------------
def extract_date_key(filename: str) -> str | None:
    """
    Try multiple patterns to extract a date from a filename.
    Always returns a normalised 'dd-mm-yyyy' string, or None if nothing matched.
    """
    name = filename

    # Helper: build key, expanding 2-digit years to 20xx
    def make_key(d, m, y):
        d, m, y = int(d), int(m), int(y)
        if y < 100:
            y += 2000
        if y > 9999:          # typo like 20222 → take first 4 digits
            y = int(str(y)[:4])
        return f"{d:02d}-{m:02d}-{y:04d}"

    # 1. Parenthesised dd.mm.yyyy or d.m.yyyy (possibly with typo extra digits)
    #    e.g. (09.02.2003)  (9.2.2020)  (25.09.20222)
    m = re.search(r'\((\d{1,2})\.(\d{1,2})\.(\d{2,5})\)', name)
    if m:
        return make_key(m.group(1), m.group(2), m.group(3))

    # 2. Parenthesised dd.mm.yy  e.g. (09.02.25)
    m = re.search(r'\((\d{1,2})\.(\d{1,2})\.(\d{2})\)', name)
    if m:
        return make_key(m.group(1), m.group(2), m.group(3))

    # 3. Bare date in name: dd.mm.yyyy without parens
    #    e.g. Explicaziuns_...18.05.2014... or ..._30.11.2014...
    m = re.search(r'(\d{1,2})\.(\d{1,2})\.(\d{4})', name)
    if m:
        return make_key(m.group(1), m.group(2), m.group(3))

    # 4. yyyy_mm_dd or yyyy-mm-dd at any position
    #    e.g. 2016_02_28_...  or  2021-03-07  or  Bund_2014_02_09
    m = re.search(r'(\d{4})[_\-](\d{2})[_\-](\d{2})', name)
    if m:
        return make_key(m.group(3), m.group(2), m.group(1))  # swap to dd-mm-yyyy

    # 5. dd-mm-yy with dashes, e.g. 30-11-08
    m = re.search(r'(\d{2})-(\d{2})-(\d{2})(?!\d)', name)
    if m:
        return make_key(m.group(1), m.group(2), m.group(3))

    # 6. Compact yymmdd, e.g. 090613
    m = re.search(r'_(\d{6})[-_]', name)
    if m:
        s = m.group(1)
        return make_key(s[4:6], s[2:4], s[0:2])  # yymmdd

    # 7. Romansh spelled-out date: dals_DD_da_MONTH_YYYY
    m = re.search(r'dals?[_\s](\d{1,2})[_\s]da[_\s](\w+)[_\s](\d{4})', name, re.IGNORECASE)
    if m:
        day, month_word, year = m.group(1), m.group(2).lower(), m.group(3)
        month_num = RM_MONTHS.get(month_word)
        if month_num:
            return make_key(day, month_num, year)

    return None


# ------------------------------------------------
# Load model (lazy — only when needed)
# ------------------------------------------------
tokenizer = None
model     = None
device    = None

def load_model():
    global tokenizer, model, device
    if model is not None:
        return
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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


def embed(texts: list[str], lang: str, cache_path: Path) -> np.ndarray:
    """
    Encode texts using the SwissBERT language adapter for `lang`.
    Loads from `cache_path` (.npy) if it already exists; saves there after encoding.
    """
    if cache_path.exists():
        print(f"    [cache] Loading embeddings from {cache_path.name}")
        return np.load(cache_path)

    load_model()
    adapter = LANG_ADAPTER[lang]
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

    result = np.vstack(all_embs)
    np.save(cache_path, result)
    print(f"    [cache] Saved embeddings → {cache_path.name}")
    return result


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
# DP aligner
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
    back = {}
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

            if i < A:
                update(i+1, j, cur + SKIP_PENALTY, i, j, 1, 0)
            if j < B:
                update(i, j+1, cur + SKIP_PENALTY, i, j, 0, 1)

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
# Cache helpers
# ------------------------------------------------
def emb_cache_path(date_key: str, lang: str) -> Path:
    return CACHE_DIR / f"{date_key}_{lang}.npy"

def pairs_cache_path(date_key: str, lang: str) -> Path:
    return CACHE_DIR / f"{date_key}_pairs_{lang}.jsonl"

def load_pairs_cache(path: Path) -> list[tuple[str, str]] | None:
    if not path.exists():
        return None
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pairs.append((obj["a"], obj["b"]))
    print(f"    [cache] Loaded pairs from {path.name}")
    return pairs

def save_pairs_cache(path: Path, pairs: list[tuple[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for a, b in pairs:
            f.write(json.dumps({"a": a, "b": b}, ensure_ascii=False) + "\n")
    print(f"    [cache] Saved pairs → {path.name}")


# ------------------------------------------------
# Main (JSONL output)
# ------------------------------------------------
def main_jsonl():
    date_lang_files: dict[str, dict[str, Path]] = {}
    for lang in ALL_LANGS:
        lang_dir = INPUT_DIR / lang
        if not lang_dir.is_dir():
            continue
        for f in sorted(lang_dir.glob("*.txt")):
            date_key = extract_date_key(f.name)
            if date_key:
                date_lang_files.setdefault(date_key, {})[lang] = f
            else:
                print(f"  [WARN] No date found in filename, skipping: {f.name}")

    if not date_lang_files:
        print(f"No matching .txt files found in {INPUT_DIR}/<lang>/ subfolders.")
        return

    for date_key, lang_files in sorted(date_lang_files.items()):
        if "de" not in lang_files:
            print(f"[{date_key}] Skipping — no DE anchor file found")
            continue

        present_langs = [l for l in ALL_LANGS if l in lang_files]
        other_langs   = [l for l in present_langs if l != "de"]

        out_path = OUTPUT_DIR / f"bert_aligned_{date_key}.jsonl"
        if out_path.exists():
            print(f"[{date_key}] Skipping — {out_path.name} already exists")
            continue

        print(f"\n{'='*60}")
        print(f"Date: {date_key}  |  Languages: {present_langs}")
        print(f"Moves: 1:1, 1:N, N:1  (N up to {MAX_MERGE})")
        print(f"{'='*60}")

        de_lines = read_lines(lang_files["de"])
        print(f"  DE: {len(de_lines)} lines — encoding...")
        de_embs = embed(de_lines, "de", emb_cache_path(date_key, "de"))

        aligned_cols: dict[str, list[tuple[str, str]]] = {}
        for lang in other_langs:
            # --- pairs cache (fastest: skip embed + align entirely) ---
            cached_pairs = load_pairs_cache(pairs_cache_path(date_key, lang))
            if cached_pairs is not None:
                aligned_cols[lang] = cached_pairs
                print(f"    → {len(cached_pairs)} aligned rows (from cache)")
                continue

            other_lines = read_lines(lang_files[lang])
            print(f"  {lang.upper()}: {len(other_lines)} lines — encoding...")
            other_embs = embed(other_lines, lang, emb_cache_path(date_key, lang))
            print(f"  {lang.upper()}: aligning...")
            pairs = dp_align(de_embs, other_embs, de_lines, other_lines)
            save_pairs_cache(pairs_cache_path(date_key, lang), pairs)
            aligned_cols[lang] = pairs
            print(f"    → {len(pairs)} aligned rows")

        # Build JSONL rows
        if other_langs:
            base_lang = other_langs[0]
            rows = [{"de": a, base_lang: b} for a, b in aligned_cols[base_lang]]
        else:
            rows = [{"de": line} for line in de_lines]

        for lang in other_langs[1:]:
            de_to_other: dict[str, str] = {a: b for a, b in aligned_cols[lang]}
            for row in rows:
                row[lang] = de_to_other.get(row["de"], "")

        # Write JSONL
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                json_line = {k: row[k] for k in present_langs if k in row}
                f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

        print(f"  Saved → {out_path}  ({len(rows)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main_jsonl()
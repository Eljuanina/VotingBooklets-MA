"""
align_languages_bert_gemini.py
-------------------------------
Two-stage alignment pipeline:
  Stage 1: SwissBERT aligns other languages to DE anchor
  Stage 2: Gemini reviews each aligned row and corrects suspicious alignments

Stage 2 uses Gemini via LiteLLM proxy. It receives the DE text and the
SwissBERT-aligned translation, and decides whether the alignment is correct
or should be fixed using the available fragments.

Expected folder structure under INPUT_DIR:
    de/  fr/  it/  rm/
with filenames containing a date like (09.02.2003).

Output JSONLs go to OUTPUT_DIR:
    bert_gemini_aligned_09-02-2003.jsonl   (one JSON object per line)

.env file:
    GEMINI_API_KEY=your_key_here
    GENAI_BASE_URL=http://172.23.205.120:4000
"""

import json
import os
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from transformers import AutoTokenizer, AutoModel

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ------------------------------------------------
# Config
# ------------------------------------------------
INPUT_DIR  = Path("../gemini-ocr-2.5-flash-lite")
OUTPUT_DIR = Path("../corpus/aligned-booklets")
OUTPUT_DIR.mkdir(exist_ok=True)

ALL_LANGS  = ["de", "fr", "it", "rm"]

LANG_ADAPTER = {
    "de": "de_CH",
    "fr": "fr_CH",
    "it": "it_CH",
    "rm": "rm_CH",
}

SWISSBERT_MODEL  = "jgrosjean-mathesis/sentence-swissbert"
GEMINI_MODEL     = "gemini-2.5-flash-lite" # i changed it here
BATCH_SIZE       = 32
RATE_LIMIT_DELAY = 0.5

# DP costs (SwissBERT stage)
MATCH_BONUS      =  1.0
SKIP_PENALTY     = -0.3
MERGE_PENALTY    = -0.05
MAX_MERGE        =  5
MIN_SIM_TO_MERGE =  0.30

# Cosine similarity below this triggers Gemini correction review
CORRECTION_THRESHOLD = 0.65

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
    name = filename

    def make_key(d, m, y):
        d, m, y = int(d), int(m), int(y)
        if y < 100:
            y += 2000
        if y > 9999:
            y = int(str(y)[:4])
        return f"{d:02d}-{m:02d}-{y:04d}"

    m = re.search(r'\((\d{1,2})\.(\d{1,2})\.(\d{2,5})\)', name)
    if m:
        return make_key(m.group(1), m.group(2), m.group(3))

    m = re.search(r'\((\d{1,2})\.(\d{1,2})\.(\d{2})\)', name)
    if m:
        return make_key(m.group(1), m.group(2), m.group(3))

    m = re.search(r'(\d{1,2})\.(\d{1,2})\.(\d{4})', name)
    if m:
        return make_key(m.group(1), m.group(2), m.group(3))

    m = re.search(r'(\d{4})[_\-](\d{2})[_\-](\d{2})', name)
    if m:
        return make_key(m.group(3), m.group(2), m.group(1))

    m = re.search(r'(\d{2})-(\d{2})-(\d{2})(?!\d)', name)
    if m:
        return make_key(m.group(1), m.group(2), m.group(3))

    m = re.search(r'_(\d{6})[-_]', name)
    if m:
        s = m.group(1)
        return make_key(s[4:6], s[2:4], s[0:2])

    m = re.search(r'dals?[_\s](\d{1,2})[_\s]da[_\s](\w+)[_\s](\d{4})', name, re.IGNORECASE)
    if m:
        day, month_word, year = m.group(1), m.group(2).lower(), m.group(3)
        month_num = RM_MONTHS.get(month_word)
        if month_num:
            return make_key(day, month_num, year)

    return None


# clean json

def clean_llm_json(text: str) -> str:
    # remove markdown fences
    text = re.sub(r"^```[a-z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text)

    # keep only JSON array
    start = text.find("[")
    end   = text.rfind("]")
    if start != -1 and end != -1:
        text = text[start:end+1]

    # fix common LLM issues
    text = text.replace("\r", "")
    text = re.sub(r",\s*]", "]", text)   # trailing commas

    return text


# ------------------------------------------------
# Load SwissBERT (lazy)
# ------------------------------------------------
tokenizer  = None
bert_model = None
device     = None

def load_swissbert():
    global tokenizer, bert_model, device
    if bert_model is not None:
        return
    print(f"Loading SwissBERT: {SWISSBERT_MODEL}")
    tokenizer  = AutoTokenizer.from_pretrained(SWISSBERT_MODEL)
    bert_model = AutoModel.from_pretrained(SWISSBERT_MODEL)
    bert_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    print(f"SwissBERT on: {device}")


# ------------------------------------------------
# Load Gemini via LiteLLM
# ------------------------------------------------
llm = ChatOpenAI(
    model=GEMINI_MODEL,
    temperature=0.0,
    api_key=os.environ.get("GEMINI_API_KEY", "placeholder"),
    base_url=os.environ.get("GENAI_BASE_URL", "http://172.23.205.120:4000"),
    model_kwargs={
        "extra_body": {"drop_params": True}
    },
)


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

# json fixers
def json_safe(text: str) -> str:
    """Make text safe for LLM JSON generation / parsing."""
    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace('"', "'")
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")
    text = text.replace("\\", "/")
    return text.strip()


def safe_json_load(text: str):
    """Robust JSON parser for LLM output."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = re.sub(r",\s*]", "]", text)
        text = re.sub(r",\s*}", "}", text)
        return json.loads(text)


def save_jsonl(path: Path, rows: list[dict]):
    """Write JSONL file safely."""
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

# ------------------------------------------------
# SwissBERT embedding
# ------------------------------------------------
def mean_pooling(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def embed(texts: list[str], lang: str) -> np.ndarray:
    load_swissbert()
    if hasattr(bert_model, "set_default_language"):
        bert_model.set_default_language(LANG_ADAPTER[lang])
    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = bert_model(**encoded)
        embs = mean_pooling(out.last_hidden_state, encoded["attention_mask"])
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        all_embs.append(embs.cpu().numpy())
    return np.vstack(all_embs)


# ------------------------------------------------
# DP aligner (SwissBERT stage)
# ------------------------------------------------
def dp_align(
    anchor_embs, other_embs, anchor_texts, other_texts
) -> list[tuple[str, str, float]]:
    """Returns list of (anchor_text, other_text, cosine_sim)."""
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

    result = []
    for ai, oj, a_span, o_span in raw_pairs:
        a_text = resolve(anchor_texts, ai, a_span)
        o_text = resolve(other_texts,  oj, o_span)
        if not a_text:
            continue
        a_emb = mean_vec([anchor_embs[ai + k] for k in range(a_span)])
        if o_span > 0 and oj is not None:
            o_emb = mean_vec([other_embs[oj + k] for k in range(o_span)])
            sim = cos_sim(a_emb, o_emb)
        else:
            sim = 0.0
        result.append((a_text, o_text, sim))

    return result


# ------------------------------------------------
# Gemini correction stage
# ------------------------------------------------
CORRECTION_PROMPT = """You are a multilingual text alignment expert for Swiss official documents.

You will receive a list of aligned segment pairs (German DE and {lang_name} {lang_key}).
Some alignments may be incorrect — the {lang_name} text may be misaligned, incomplete, or merged incorrectly.

You also receive the full list of original {lang_name} fragments the aligner had available.

For each pair:
- If the alignment looks correct, keep it as-is.
- If the {lang_name} text is clearly wrong or misaligned, find the correct fragment(s) from the available fragments and replace it.
- If no good match exists, return an empty string for that pair.
- Do NOT translate or paraphrase — only use text from the original fragments verbatim.

You must return VALID JSON ONLY.

Rules:
- No explanations
- No markdown
- No comments
- No trailing commas
- Escape all quotes
- Output must parse with json.loads()

Return exactly:

[
  {{
    "de": "...",
    "fr": "...",
    "it": "...",
    "rm": "..."
  }}
]

--- ALIGNED PAIRS TO REVIEW ---
{pairs}

--- AVAILABLE {lang_key_upper} FRAGMENTS ---
{fragments}"""


def call_gemini_correction(
    pairs: list[tuple[str, str]],
    all_fragments: list[str],
    lang: str,
    retries: int = 3
) -> list[tuple[str, str]]:
    lang_name = {"fr": "French", "it": "Italian", "rm": "Romansh"}.get(lang, lang.upper())
    pairs_numbered = "\n".join(
        f"{i+1}. DE: {json_safe(de)}\n   {lang.upper()}: {json_safe(other)}"
        for i, (de, other) in enumerate(pairs)
    )

    frags_numbered = "\n".join(
        f"{i+1}. {json_safe(t)}"
        for i, t in enumerate(all_fragments)
    )

    prompt = CORRECTION_PROMPT.format(
        lang_name=lang_name,
        lang_key=lang,
        lang_key_upper=lang.upper(),
        pairs=pairs_numbered,
        fragments=frags_numbered,
    )

    for attempt in range(retries):
        try:
            response = llm.invoke([HumanMessage(content=prompt)])

            # CLEAN GEMINI OUTPUT
            raw = clean_llm_json(response.content)
            items = safe_json_load(raw)

            return [
                (normalize(item.get("de", "")), normalize(item.get(lang, "")))
                for item in items
            ]
        except Exception as e:
            print(f"    Gemini correction error (attempt {attempt+1}/{retries}): {e}")
            time.sleep(2 ** attempt)

    return pairs  # Fallback: return unchanged


# ------------------------------------------------
# Full two-stage alignment for one language
# ------------------------------------------------
def align_lang_two_stage(
    de_lines: list[str],
    other_lines: list[str],
    lang: str,
) -> list[tuple[str, str]]:

    print(f"    Stage 1: SwissBERT embedding + DP alignment...")
    de_embs    = embed(de_lines,    "de")
    other_embs = embed(other_lines, lang)
    pairs_with_sim = dp_align(de_embs, other_embs, de_lines, other_lines)

    n_total     = len(pairs_with_sim)
    n_uncertain = sum(1 for _, _, sim in pairs_with_sim if sim < CORRECTION_THRESHOLD)
    print(f"    Stage 1 done: {n_total} rows, {n_uncertain} uncertain (sim < {CORRECTION_THRESHOLD})")

    uncertain_indices = [
        i for i, (_, _, sim) in enumerate(pairs_with_sim)
        if sim < CORRECTION_THRESHOLD
    ]

    if uncertain_indices:
        print(f"    Stage 2: Sending {len(uncertain_indices)} uncertain rows to Gemini for correction...")

        uncertain_pairs = [
            (pairs_with_sim[i][0], pairs_with_sim[i][1])
            for i in uncertain_indices
        ]

        # --- batching helper ---
        def chunk(lst, size):
            for i in range(0, len(lst), size):
                yield lst[i:i+size]

        corrected = []
        BATCH_CORRECTION_SIZE = 5

        total_batches = (len(uncertain_pairs) + BATCH_CORRECTION_SIZE - 1) // BATCH_CORRECTION_SIZE

        for b_idx, batch in enumerate(chunk(uncertain_pairs, BATCH_CORRECTION_SIZE), start=1):
            print(f"      Gemini batch {b_idx}/{total_batches} ({len(batch)} rows)...")

            try:
                # limit fragments to avoid massive prompts
                result = call_gemini_correction(batch, other_lines[:150], lang)
                corrected.extend(result)
            except Exception as e:
                print(f"      Batch {b_idx} failed: {e}")
                corrected.extend(batch)

            time.sleep(RATE_LIMIT_DELAY)

        # --- apply corrections ---
        if len(corrected) == len(uncertain_indices):
            for idx, (de_text, fixed_lang) in zip(uncertain_indices, corrected):
                orig_de, orig_lang, orig_sim = pairs_with_sim[idx]

                if fixed_lang and normalize(de_text) == normalize(orig_de):
                    pairs_with_sim[idx] = (orig_de, fixed_lang, orig_sim)
                    print(f"      Row {idx+1}: corrected")
        else:
            print(
                f"    Warning: Gemini returned {len(corrected)} corrections "
                f"for {len(uncertain_indices)} rows"
            )

    else:
        print(f"    Stage 2: all rows confident, skipping Gemini correction")

    return [(de, lang_text) for de, lang_text, _ in pairs_with_sim]


# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
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

    def total_size(lang_files: dict[str, Path]) -> int:
        return sum(f.stat().st_size for f in lang_files.values())

    sorted_dates = sorted(date_lang_files.items(), key=lambda kv: total_size(kv[1]), reverse=True)

    for date_key, lang_files in sorted_dates:
        if "de" not in lang_files:
            print(f"[{date_key}] Skipping — no DE anchor file found")
            continue

        present_langs = [l for l in ALL_LANGS if l in lang_files]
        other_langs   = [l for l in present_langs if l != "de"]

        out_path = OUTPUT_DIR / f"bert_gemini_aligned_{date_key}.jsonl"
        if out_path.exists():
            print(f"[{date_key}] Skipping — {out_path.name} already exists")
            continue

        print(f"\n{'='*60}")
        print(f"Date: {date_key}  |  Languages: {present_langs}")
        print(f"Pipeline: SwissBERT → Gemini correction (threshold={CORRECTION_THRESHOLD})")
        print(f"{'='*60}")

        de_lines = read_lines(lang_files["de"])
        print(f"  DE: {len(de_lines)} anchor lines")

        aligned_cols: dict[str, list[tuple[str, str]]] = {}
        for lang in other_langs:
            other_lines = read_lines(lang_files[lang])
            print(f"\n  {lang.upper()}: {len(other_lines)} fragments")
            pairs = align_lang_two_stage(de_lines, other_lines, lang)
            aligned_cols[lang] = pairs
            print(f"    → {len(pairs)} aligned rows")

        # Build rows as dicts
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

        # Write as JSONL (one JSON object per line)
        save_jsonl(out_path, rows)

        print(f"\n  Saved → {out_path}  ({len(rows)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
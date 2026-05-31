"""
align_languages_gemini.py
--------------------------
Aligns language text files against a DE anchor using Gemini 2.5 Flash Lite
via a LiteLLM proxy (LangChain ChatOpenAI interface).

Expected file names in INPUT_DIR:
    de_1985.txt   fr_1985.txt   it_1985.txt   rm_1985.txt   ...

Output TSVs go to OUTPUT_DIR:
    gemini_aligned_1985.tsv   (columns: de  fr  it  rm)

.env file (in the same folder as this script):
    GEMINI_API_KEY=your_key_here
    GENAI_BASE_URL=http://172.23.205.120:4000
"""

import json
import os
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

# ------------------------------------------------
# Config
# ------------------------------------------------
INPUT_DIR  = Path("texts")
OUTPUT_DIR = Path("aligned")
OUTPUT_DIR.mkdir(exist_ok=True)

ALL_LANGS  = ["de", "fr", "it", "rm"]
MODEL_NAME = "gemini-2.5-flash-lite"

# How many DE anchor lines to send per API call
BATCH_SIZE = 20

# Seconds to wait between API calls
RATE_LIMIT_DELAY = 1.0

# ------------------------------------------------
# LLM client
# ------------------------------------------------
llm = ChatOpenAI(
    model=MODEL_NAME,
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


# ------------------------------------------------
# Prompt
# ------------------------------------------------
PROMPT_TEMPLATE = """You are a multilingual text alignment expert.
You will receive:
  - A list of German (DE) anchor segments, numbered 1..N
  - A list of {lang_name} fragments, numbered 1..M (split aggressively at dots,
    so one DE segment may correspond to multiple fragments, or vice versa)

Your task: for each DE segment, find the {lang_name} fragment(s) that are its
translation and join them into one string. If no matching fragment exists,
return an empty string for that DE segment.

Rules:
  - Every DE segment must appear in the output exactly once.
  - A fragment can only be used once.
  - Consecutive fragments may be merged (joined with a space) to match one DE segment.
  - Do NOT translate or paraphrase — use the original fragment text verbatim.

Return ONLY a JSON array with exactly N objects, one per DE segment, in order:
[
  {{"de": "<de segment text>", "{lang_key}": "<matched fragment text or empty>"}},
  ...
]
No explanation, no markdown fences, no extra text — just the raw JSON array.

--- DE SEGMENTS ---
{de_segments}

--- {lang_key_upper} FRAGMENTS ---
{other_segments}"""


def build_prompt(de_batch: list[str], other_lines: list[str], lang: str) -> str:
    lang_name = {"fr": "French", "it": "Italian", "rm": "Romansh"}.get(lang, lang.upper())
    de_numbered    = "\n".join(f"{i+1}. {t}" for i, t in enumerate(de_batch))
    other_numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(other_lines))
    return PROMPT_TEMPLATE.format(
        lang_name=lang_name,
        lang_key=lang,
        lang_key_upper=lang.upper(),
        de_segments=de_numbered,
        other_segments=other_numbered,
    )


# ------------------------------------------------
# API call + parse
# ------------------------------------------------
def call_llm(prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            print(f"    API error (attempt {attempt+1}/{retries}): {e}")
            time.sleep(2 ** attempt)
    return "[]"


def parse_response(raw: str, de_batch: list[str], lang: str) -> list[tuple[str, str]]:
    # Strip markdown fences if model added them anyway
    raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw.strip())
    try:
        items = json.loads(raw)
        return [
            (normalize(item.get("de", "")), normalize(item.get(lang, "")))
            for item in items
        ]
    except json.JSONDecodeError:
        print(f"    Warning: could not parse JSON — falling back to empty matches")
        return [(de, "") for de in de_batch]


# ------------------------------------------------
# Align one language to DE in batches
# ------------------------------------------------
def align_lang(de_lines: list[str], other_lines: list[str], lang: str) -> list[tuple[str, str]]:
    all_pairs = []
    n_batches = (len(de_lines) + BATCH_SIZE - 1) // BATCH_SIZE

    for b in range(n_batches):
        de_batch = de_lines[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
        print(f"    Batch {b+1}/{n_batches} ({len(de_batch)} DE segments)...")

        prompt = build_prompt(de_batch, other_lines, lang)
        raw    = call_llm(prompt)
        pairs  = parse_response(raw, de_batch, lang)

        # Ensure exactly one pair per DE segment
        if len(pairs) != len(de_batch):
            print(f"    Warning: expected {len(de_batch)} pairs, got {len(pairs)} — padding")
            pairs = pairs[:len(de_batch)]
            while len(pairs) < len(de_batch):
                pairs.append((de_batch[len(pairs)], ""))

        all_pairs.extend(pairs)
        time.sleep(RATE_LIMIT_DELAY)

    return all_pairs


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
        print(f"{'='*60}")

        de_lines = read_lines(lang_files["de"])
        print(f"  DE: {len(de_lines)} anchor lines")

        aligned_cols: dict[str, list[tuple[str, str]]] = {}
        for lang in other_langs:
            other_lines = read_lines(lang_files[lang])
            print(f"\n  {lang.upper()}: {len(other_lines)} fragments — aligning via Gemini...")
            pairs = align_lang(de_lines, other_lines, lang)
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

        out_path = OUTPUT_DIR / f"gemini_aligned_{year}.tsv"
        df.to_csv(out_path, sep="\t", index=False)
        print(f"\n  Saved → {out_path}  ({len(df)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
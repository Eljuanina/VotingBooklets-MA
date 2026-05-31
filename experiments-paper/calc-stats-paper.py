import pandas as pd
from pathlib import Path
from collections import defaultdict

# Files for each vote
excel_files = {
    "1977": "gs-alignment77.xlsx",
    "1985": "gs-alignment85.xlsx",
    "2007": "gs-alignment07.xlsx"
}

# Initialize stats dictionaries
file_stats = defaultdict(dict)   # {year: {lang: token_count}}
lang_totals = defaultdict(int)   # {lang: total tokens across all files}

# Define column order
all_langs = ["de", "fr", "it", "rm"]

for year, path in excel_files.items():
    # Read without headers
    df = pd.read_excel(path, header=None)

    # Determine which languages are present
    num_cols = df.shape[1]
    langs = all_langs[:num_cols]

    for i, lang in enumerate(langs):
        # Count tokens in this column
        token_count = df[i].dropna().apply(lambda x: len(str(x).split())).sum()
        file_stats[year][lang] = token_count
        lang_totals[lang] += token_count

# Print per-file statistics
print("Tokens per language per file:")
for year, stats in file_stats.items():
    print(f"\n{year}:")
    for lang, count in stats.items():
        print(f"  {lang}: {count} tokens")

# Print totals across all files
print("\nTotal tokens per language across all files:")
for lang, count in lang_totals.items():
    print(f"{lang}: {count} tokens")

"""
# Tokens per language per file:

1977:
  de: 4016 tokens
  fr: 4997 tokens
  it: 4492 tokens

1985:
  de: 1621 tokens
  fr: 2199 tokens
  it: 1827 tokens
  rm: 2042 tokens

2007:
  de: 2017 tokens
  fr: 2612 tokens
  it: 2431 tokens
  rm: 2785 tokens

Total tokens per language across all files:
de: 7654 tokens
fr: 9808 tokens
it: 8750 tokens
rm: 4827 tokens
"""


import pandas as pd

# List of your Excel files
files = ["gs-alignment77.xlsx", "gs-alignment85.xlsx", "gs-alignment07.xlsx"]

# Loop over each file
for file in files:
    df = pd.read_excel(file, header=None)  # No header row
    print(f"\nFile: {file}")

    # Determine the languages present
    languages = df.shape[1]
    lang_labels = ["de", "fr", "it", "rm"][:languages]

    for i, lang in enumerate(lang_labels):
        # Count non-empty cells = paragraphs
        n_paragraphs = df[i].dropna().shape[0]

        # Count tokens per language
        n_tokens = df[i].dropna().apply(lambda x: len(str(x).split())).sum()

        print(f"  {lang}: {n_tokens} tokens, {n_paragraphs} paragraphs")

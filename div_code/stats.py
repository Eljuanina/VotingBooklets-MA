from pathlib import Path
from collections import defaultdict

# Root folder
root = Path("gold_files/gold_ocr/gold_2007")

# Languages you expect
languages = ["de", "fr", "it", "rm"]

# Stats
file_counts = defaultdict(int)   # {lang: number of files}
token_counts = defaultdict(int)  # {lang: total tokens}

# Loop through each language folder
for lang in languages:
    lang_path = root / lang

    if not lang_path.exists():
        print(f"Warning: {lang_path} does not exist")
        continue

    # Find all .txt files recursively
    txt_files = list(lang_path.rglob("*.txt"))

    file_counts[lang] = len(txt_files)

    for file in txt_files:
        try:
            text = file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Skipping {file}: {e}")
            continue

        tokens = text.split()
        token_counts[lang] += len(tokens)

# Print results
print("Stats per language:\n")

for lang in languages:
    print(f"{lang}:")
    print(f"  files:  {file_counts[lang]}")
    print(f"  tokens: {token_counts[lang]}")


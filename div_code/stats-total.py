from pathlib import Path
from collections import defaultdict

# Root folder
root = Path("gemini-ocr-2.5-flash-lite")

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


# import re
# from pathlib import Path
# from collections import defaultdict


# fr_path = root / "fr"
# it_path = root / "it"

# date_pattern = re.compile(r"\d{2}\.\d{2}\.\d{4}")

# def extract_date_counts(folder):
#     date_counts = defaultdict(list)
    
#     for file in folder.rglob("*.txt"):
#         match = date_pattern.search(file.name)
#         if match:
#             date = match.group()
#             date_counts[date].append(file.name)
#         else:
#             date_counts["NO_DATE"].append(file.name)
    
#     return date_counts

# fr_dates = extract_date_counts(fr_path)
# it_dates = extract_date_counts(it_path)

# all_dates = set(fr_dates) | set(it_dates)

# problems = []

# for date in sorted(all_dates):
#     fr_count = len(fr_dates.get(date, []))
#     it_count = len(it_dates.get(date, []))
    
#     if fr_count != it_count:
#         problems.append((date, fr_count, it_count))

# print(f"Dates with mismatched file counts: {len(problems)}\n")

# for date, fr_c, it_c in problems:
#     print(f"{date}: FR={fr_c}, IT={it_c}")
    
#     print("  FR files:")
#     for f in fr_dates.get(date, []):
#         print(f"    {f}")
    
#     print("  IT files:")
#     for f in it_dates.get(date, []):
#         print(f"    {f}")
    
#     print()


from pathlib import Path
from PyPDF2 import PdfReader
from collections import defaultdict

# Root folder containing language subfolders
root = Path("corpus/raw_voting_booklets")  # or wherever your PDFs are
languages = ["Deutsch", "Französisch", "Italienisch", "Rätoromanisch"]

page_counts = defaultdict(int)
file_counts = defaultdict(int)

for lang in languages:
    lang_path = root / lang
    if not lang_path.exists():
        print(f"Warning: {lang_path} does not exist")
        continue

    for pdf_file in lang_path.glob("*.pdf"):
        try:
            reader = PdfReader(str(pdf_file))
            num_pages = len(reader.pages)
            page_counts[lang] += num_pages
            file_counts[lang] += 1
        except Exception as e:
            print(f"Failed to read {pdf_file.name}: {e}")

# Print results
print("Pages per language:\n")
for lang in languages:
    print(f"{lang}:")
    print(f"  files: {file_counts[lang]}")
    print(f"  pages: {page_counts[lang]}")
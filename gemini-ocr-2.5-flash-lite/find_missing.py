import os
import re
from collections import defaultdict

base_dir = "."
langs = ["de", "it", "fr"]  # add "rm" if needed

def extract_date(fname):
    match = re.search(r'\((\d{2}\.\d{2}\.\d{4})\)', fname)
    return match.group(1) if match else None

dates_by_lang = {}
for lang in langs:
    lang_dir = os.path.join(base_dir, lang)
    dates_by_lang[lang] = set()
    for fname in os.listdir(lang_dir):
        if fname.endswith(".txt"):
            date = extract_date(fname)
            if date:
                dates_by_lang[lang].add(date)

all_dates = set.union(*dates_by_lang.values())

print("Missing dates per language:")
for lang in langs:
    missing = all_dates - dates_by_lang[lang]
    if missing:
        print(f"  {lang}: {sorted(missing)}")
    else:
        print(f"  {lang}: none missing")
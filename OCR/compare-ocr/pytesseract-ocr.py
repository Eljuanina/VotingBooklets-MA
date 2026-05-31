from pdf2image import convert_from_path
import pytesseract
import os
from pathlib import Path

# Input and output folders
input_folder = Path("data")
output_folder = Path("pytesseract")
output_folder.mkdir(exist_ok=True)

# Map filename prefix → Tesseract language code
LANG_MAP = {
    "de": "deu",                    # German
    "fr": "fra",                    # French
    "it": "ita",                    # Italian
    "rm": "deu+fra+ita+roh"         # Romansh fallback
}

# Loop through all PDFs in data folder
for pdf_path in input_folder.glob("*.pdf"):

    filename = pdf_path.stem

    # Detect language from filename prefix
    lang_prefix = filename.split("_")[0].strip().lower()
    lang_code = LANG_MAP.get(lang_prefix)

    if not lang_code:
        print(f"Unknown language prefix '{lang_prefix}', defaulting to deu+fra+ita+roh")
        lang_code = "deu+fra+ita+roh"

    print(f"\nProcessing {pdf_path.name} with language: {lang_code}")

    # Convert PDF pages to images
    pages = convert_from_path(str(pdf_path), dpi=300)

    all_text = ""

    for i, page in enumerate(pages):
        text = pytesseract.image_to_string(page, lang=lang_code)
        all_text += f"--- Page {i+1} ---\n{text}\n"

    output_path = output_folder / f"{filename}_extracted.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(all_text)

    print(f"Saved extracted text to {output_path}")

print("Extraction completed for all PDFs.")
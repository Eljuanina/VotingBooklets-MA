#!/usr/bin/env python3
from pathlib import Path
from extract_date import extract_date_from_title, BASE_DOWNLOAD_DIR
from ocr_gemini import run_gemini_ocr

LANG_FOLDERS = ["Deutsch", "Italienisch", "Französisch", "Rätoromanisch"]
OUTPUT_BASE = Path("gemini-ocr")

LANG_CODE = {
    "Deutsch": "de",
    "Französisch": "fr",
    "Italienisch": "it",
    "Rätoromanisch": "rm",
}

def get_output_path(pdf_path: Path) -> Path:
    try:
        relative_parent = pdf_path.parent.relative_to(BASE_DOWNLOAD_DIR)
    except ValueError:
        relative_parent = Path()

    lang_folder = relative_parent.parts[0] if relative_parent.parts else ""
    lang_code = LANG_CODE.get(lang_folder, lang_folder.lower())
    date = extract_date_from_title(pdf_path.stem)
    year = date.split(".")[-1] if date else pdf_path.stem

    output_dir = OUTPUT_BASE / relative_parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{lang_code}_{year}.txt"

def output_exists(pdf_path: Path) -> bool:
    return get_output_path(pdf_path).exists()

def find_pdfs_for_date(date_str: str):
    results = []
    for lang in LANG_FOLDERS:
        lang_dir = BASE_DOWNLOAD_DIR / lang
        if not lang_dir.exists():
            continue
        for pdf_path in lang_dir.rglob("*.pdf"):
            if pdf_path.name.startswith("."):
                continue
            if extract_date_from_title(pdf_path.stem) == date_str:
                results.append(pdf_path)
    return results

def process_dates(dates):
    for date in dates:
        print(f"\n=== Processing date: {date} ===")
        pdfs = find_pdfs_for_date(date)
        if not pdfs:
            print(f"[Skip] No files found for date {date}")
            continue
        for pdf_path in pdfs:
            out_path = get_output_path(pdf_path)
            if out_path.exists():
                print(f"[Skip] Already OCR'd: {out_path}")
                continue
            print(f"Found (new) PDF: {pdf_path} → {out_path.name}")
            try:
                run_gemini_ocr(pdf_path)
                # run_gemini_ocr saves as <stem>_extracted.txt — rename to lang_year.txt
                default_out = OUTPUT_BASE / pdf_path.parent.relative_to(BASE_DOWNLOAD_DIR) / f"{pdf_path.stem}_extracted.txt"
                if default_out.exists():
                    default_out.rename(out_path)
            except Exception as e:
                print(f"[Error] OCR failed for {pdf_path}: {e}")

if __name__ == "__main__":
    dates = ["12.06.1977", "01.12.1985", "11.03.2007"]
    process_dates(dates)
    print("\nAll dates processed.")
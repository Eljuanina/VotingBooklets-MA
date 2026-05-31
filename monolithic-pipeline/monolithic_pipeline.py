import sys
import os
import shutil

# ─── Step 1: Download ─────────────────────────────────────────────────────────
def run_download():
    print("\n" + "="*60)
    print("STEP 1: Downloading voting booklets")
    print("="*60)
    import download_voting_booklets
    download_voting_booklets.main()


# ─── Step 2: OCR (with check for existing files) ──────────────────────────────
DATES_TO_PROCESS = ["12.06.1977", "01.12.1985", "11.03.2007","01.13.1999"]
def check_existing_ocr():
    """Check if OCR files already exist in booklets_txt/"""
    source_dir = "booklets_txt"
    
    if not os.path.exists(source_dir):
        return False, []
    
    expected_files = []
    for date in DATES_TO_PROCESS:
        # Extract year from date
        year = date.split('.')[-1]
        for lang in ['de', 'fr', 'it', 'rm']:
            # Skip Romansh for 1977
            if year == '1977' and lang == 'rm':
                continue
            expected_files.append((lang, f"{lang}{year}.txt"))
    
    existing_files = []
    for lang, filename in expected_files:
        filepath = os.path.join(source_dir, lang, filename)
        if os.path.exists(filepath):
            existing_files.append((lang, filename))
    
    all_exist = len(existing_files) == len(expected_files)
    return all_exist, existing_files

def copy_existing_ocr():
    """Copy existing OCR files from booklets_txt/ to working directory"""
    source_dir = "booklets_txt"
    import extract_text_from_pdf
    # Adjust this based on where your OCR module outputs files
    dest_dir = getattr(extract_text_from_pdf, 'OUTPUT_DIR', 'output/ocr')
    
    os.makedirs(dest_dir, exist_ok=True)
    
    file_count = 0
    for lang in ['de', 'fr', 'it', 'rm']:
        lang_source = os.path.join(source_dir, lang)
        if not os.path.exists(lang_source):
            continue
            
        dest_lang_dir = os.path.join(dest_dir, lang)
        os.makedirs(dest_lang_dir, exist_ok=True)
        
        for file in os.listdir(lang_source):
            if file.endswith('.txt'):
                src_path = os.path.join(lang_source, file)
                dest_path = os.path.join(dest_lang_dir, file)
                shutil.copy2(src_path, dest_path)
                print(f"  Copied: {lang}/{file}")
                file_count += 1
    
    return file_count

from rotate import pdf_rotation
import extract_text_from_pdf

def rotate_pdfs_for_dates(dates):
    """Rotate PDFs that will be sent to OCR"""

    print("\nRotating PDFs before OCR...")

    rotated = 0

    for date in dates:
        pdfs = extract_text_from_pdf.find_pdfs_for_date(date)

        for pdf_path in pdfs:
            print(f"  Rotating: {pdf_path}")
            pdf_rotation(str(pdf_path))
            rotated += 1

    print(f"✓ Rotated {rotated} PDFs")

def run_ocr():
    print("\n" + "="*60)
    print("STEP 2: OCR Processing")
    print("="*60)
    
    all_exist, existing_files = check_existing_ocr()
    
    if all_exist:
        print("✓ All OCR files already exist in booklets_txt/")
        print(f"  Found {len(existing_files)} files:")
        for lang, filename in existing_files:
            print(f"    - {lang}/{filename}")
        print("\n→ Using existing OCR files (skipping Gemini OCR)")
        
        copied = copy_existing_ocr()
        print(f"\n✓ Copied {copied} OCR files to working directory")
    else:
        print("✗ OCR files not found or incomplete in booklets_txt/")

        rotate_pdfs_for_dates(DATES_TO_PROCESS)

        print("\n→ Running Gemini OCR...")

        import extract_text_from_pdf
        extract_text_from_pdf.process_dates(DATES_TO_PROCESS)

        if existing_files:
            print(f"  Found only {len(existing_files)} files")
        print("\n→ Running Gemini OCR...")
        
        import extract_text_from_pdf
        extract_text_from_pdf.process_dates(DATES_TO_PROCESS)
        
        print("\n✓ OCR complete")


# ─── Step 3: Clean ────────────────────────────────────────────────────────────
def run_clean():
    print("\n" + "="*60)
    print("STEP 3: Cleaning OCR output")
    print("="*60)
    import clean_ocr
    
    if not os.path.exists(clean_ocr.input_root):
        print(f"ERROR: Input directory '{clean_ocr.input_root}' not found!")
        print("Please run OCR step first.")
        sys.exit(1)
    
    cleaned_count = 0
    for root, dirs, files in os.walk(clean_ocr.input_root):
        for file in files:
            if not file.endswith(".txt"):
                continue
            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, clean_ocr.input_root)
            output_folder = os.path.join(
                clean_ocr.output_root,
                os.path.basename(clean_ocr.input_root),
                relative_path
            )
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, file)
            
            print(f"Cleaning: {input_path}")
            with open(input_path, "r", encoding="utf-8") as f:
                content = f.read()
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(clean_ocr.clean_text(content))
            cleaned_count += 1
    
    print(f"\n✓ Cleaned {cleaned_count} files")

# ─── Step 4: Align with sentence swiss bert ────────────────────────────────────────────────────────────
def run_align():
    print("\n" + "="*60)
    print("STEP 4: Aligning multilingual text")
    print("="*60)
    import align_with_ssb
    align_with_ssb.main()


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    steps = {
        "download": run_download,
        "ocr":      run_ocr,
        "clean":    run_clean,
        "align":    run_align,
    }
    
    # Usage:
    #   python pipeline.py              -> runs all steps
    #   python pipeline.py clean align  -> runs only clean + align
    #   python pipeline.py align        -> runs only align
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(steps.keys())
    
    for step in requested:
        if step not in steps:
            print(f"Unknown step '{step}'. Valid steps: {', '.join(steps)}")
            sys.exit(1)
    
    for step in requested:
        steps[step]()
    
    print("\n" + "="*60)
    print("Pipeline complete.")
    print("="*60)

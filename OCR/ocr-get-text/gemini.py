#!/usr/bin/env python3
import os
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Input and output folders
input_folder = Path("data")
output_folder = Path("gemini-ocr")
output_folder.mkdir(exist_ok=True)

load_dotenv()

OCR_PROMPT = """Extract all text from this document page image. Preserve the exact wording.
Return the text with each paragraph on its own line. Use a single newline between paragraphs.
Do not add labels, page numbers, or any extra text—only the extracted document text."""

def get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment.")
    return key

def extract_text_from_page_with_gemini(image: Image.Image, client: genai.Client) -> str:
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[OCR_PROMPT, image],
            config=types.GenerateContentConfig(
                temperature=0.0,
            ),
        )
        return response.text or ""
    except Exception as e:
        print(f"[Detail] API Call failed: {e}")
        return f"[Error: {e}]"

def run_gemini_ocr(pdf_path: Path, client: genai.Client, dpi: int = 200) -> None:

    if not pdf_path.is_file():
        print(f"PDF not found: {pdf_path}")
        return

    output_txt = pdf_path.stem + "_extracted.txt"
    output_path = output_folder / output_txt

    print(f"\n--- Processing {pdf_path.name} ---")
    print(f"Converting PDF (DPI: {dpi})")

    try:
        images = convert_from_path(str(pdf_path), dpi=dpi)
    except Exception as e:
        print(f"Failed to convert PDF. Make sure 'poppler' is installed. Error: {e}")
        return

    print(f"Found {len(images)} pages.")

    all_text = []
    for i, image in enumerate(images, start=1):
        print(f"  Processing Page {i}/{len(images)}...")
        page_text = extract_text_from_page_with_gemini(image, client)
        all_text.append(page_text)

    full_text = "\n\n".join(all_text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"Saved OCR text to: {output_path.resolve()}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--pdf", help="Optional single PDF override")
    args = parser.parse_args()

    api_key = get_api_key()
    client = genai.Client(api_key=api_key)

    if args.pdf:
        run_gemini_ocr(Path(args.pdf), client, dpi=args.dpi)
    else:
        # Process all PDFs inside data folder
        for pdf_path in input_folder.glob("*.pdf"):
            run_gemini_ocr(pdf_path, client, dpi=args.dpi)

    print("\nAll processing complete.")
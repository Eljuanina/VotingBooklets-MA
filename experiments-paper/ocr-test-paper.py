#!/usr/bin/env python3
import os
import time
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import base64
import io

input_folder = Path("../corpus/raw_voting_booklets/Rätoromanisch")
output_folder = Path("../new-test/gemini-25/rm")

load_dotenv()

llm = ChatOpenAI(
    model="gemini-2.5-flash",
    temperature=0,
    base_url="http://172.23.205.120:4000",
    extra_body={"drop_params": True},
)


OCR_PROMPT = """Extract all text from this document page image. Preserve the exact wording.
If the image shows a double-page spread (two pages side by side), extract the left page first in full, then the right page in full.
Return the text with each paragraph on its own line. Use a single newline between paragraphs.
Do not add labels, page numbers, or any extra text—only the extracted document text.
If the page is empty, return only ' '. """


MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds between retries

def pil_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


def extract_text_from_page_with_gemini(image: Image.Image) -> str | None:
    """
    Returns extracted text, or None if all retries failed.
    Never returns error strings — caller decides what to do on None.
    """
    image_b64 = pil_to_base64(image)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = llm.invoke(
                [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": OCR_PROMPT},
                        {"type": "image_url",
                         "image_url": {"url": image_b64}},
                    ],
                }]
            )

            return response.content or ""

        except Exception as e:
            print(f"[Attempt {attempt}/{MAX_RETRIES}] API call failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    return None

def run_gemini_ocr(pdf_path: Path, dpi: int = 200) -> bool:
    """
    Returns True if OCR succeeded and file was saved, False otherwise.
    Does NOT write the output file if any page failed.
    """
    if not pdf_path.is_file():
        print(f"PDF not found: {pdf_path}")
        return False

    try:
        relative_parent = pdf_path.parent.relative_to(input_folder)
    except ValueError:
        relative_parent = Path()

    target_output_dir = output_folder / relative_parent
    target_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_output_dir / f"{pdf_path.stem}_extracted.txt"

    print(f"\n--- Processing {pdf_path} ---")
    print(f"Output -> {output_path}")

    try:
        images = convert_from_path(str(pdf_path), dpi=dpi)
    except Exception as e:
        print(f"Failed to convert PDF: {e}")
        return False

    print(f"Found {len(images)} pages.")
    all_text = []

    for i, image in enumerate(images, start=1):
        print(f"  Processing page {i}/{len(images)}...")
        page_text = extract_text_from_page_with_gemini(image)
        if page_text is None:
            print(f"[Abort] Page {i} failed — not saving output for {pdf_path.name}")
            return False
        all_text.append(page_text)

    full_text = "\n\n".join(all_text)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"Saved OCR text to: {output_path.resolve()}")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--pdf", help="Optional single PDF override")
    args = parser.parse_args()

    if args.pdf:
        run_gemini_ocr(Path(args.pdf), dpi=args.dpi)
    else:
        for pdf_path in input_folder.rglob("*.pdf"):
            run_gemini_ocr(pdf_path, dpi=args.dpi)

    print("\nAll processing complete.")
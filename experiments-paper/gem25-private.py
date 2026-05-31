#!/usr/bin/env python3
import os
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
from dotenv import load_dotenv
from io import BytesIO
import base64
import google.generativeai as genai

input_folder = Path("corpus/raw_voting_booklets/Rätoromanisch")
output_folder = Path("gemini-ocr-2.5-flash-lite/rm")
output_folder.mkdir(parents=True, exist_ok=True)

load_dotenv()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash-lite",
    generation_config=genai.GenerationConfig(temperature=0.0),
)

OCR_PROMPT = """You are a professional OCR system. Extract ALL text from this PDF page with perfect accuracy.

=== CRITICAL RULES ===

1. HYPHENATED LINE BREAKS inside a sentence- Always rejoin the word:
   "vivise-\nzione" → "vivisezione"
   "medici-\nna" → "medicina"
   "ausschliesslich Er-\nzeugnisse" → "ausschliesslich Erzeugnisse"

2. NON-HYPHENATED LINE BREAKS between words in the same sentence - Join with a space:
   "della\nricerca" → "della ricerca"
   "des\nBundes" → "des Bundes"

3. OUTPUT FORMAT:
   - Headings: one line, followed by blank line
   - Sentences: one complete sentence per line
   - List items (1., 2., a., b., -): each complete item on ONE line
   - Blank line between paragraphs
   - NO line breaks inside sentences or list items

=== EXAMPLES ===

WRONG (Never create output like this):
```
L'iniziativa popolare «per la soppressione della vivise-
zione» esige che siano vietati in tutta la Svizzera.
2. gewerbsmässige Arbeiten an Waren, Bauwerken und Grundstücken, ausge-
nommen die Bebauung des Bodens für die Urproduktion;
```

CORRECT (always output like this):
```
L'iniziativa popolare «per la soppressione della vivisezione» esige che siano vietati in tutta la Svizzera.

2. gewerbsmässige Arbeiten an Waren, Bauwerken und Grundstücken, ausgenommen die Bebauung des Bodens für die Urproduktion;
```

=== WHAT TO EXCLUDE ===
- Page numbers (e.g., "Seite 3", "Page 5")
- Headers and footers that repeat on every page
- Boilerplate text not part of the main document

=== YOUR TASK ===
Extract the text from this page following these rules exactly.
Each sentence = one line.
Each list item = one line.
Rejoin ALL hyphenated words.
Remove ALL line breaks within sentences.
Do not add text that is not present in the image. Do not guess or infer missing text.

Begin extraction:"""

def image_to_pil_bytes(image: Image.Image) -> bytes:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return buffered.getvalue()

def extract_text_from_page_with_gemini(image: Image.Image) -> str:
    try:
        img_bytes = image_to_pil_bytes(image)
        image_part = {"mime_type": "image/png", "data": img_bytes}
        response = model.generate_content([OCR_PROMPT, image_part])
        return response.text or ""
    except Exception as e:
        print(f"[Detail] API Call failed: {e}")
        return f"[Error: {e}]"

def run_gemini_ocr(pdf_path: Path, dpi: int = 300) -> None:
    if not pdf_path.is_file():
        print(f"PDF not found: {pdf_path}")
        return

    output_path = output_folder / (pdf_path.stem + "_extracted.txt")
    if output_path.exists():
        print(f"Skipping (already exists): {output_path.name}")
        return

    print(f"\n--- Processing {pdf_path.name} ---")
    try:
        images = convert_from_path(str(pdf_path), dpi=dpi)
    except Exception as e:
        print(f"Failed to convert PDF: {e}")
        return

    print(f"Found {len(images)} pages.")
    all_text = []
    for i, image in enumerate(images, start=1):
        print(f"  Processing Page {i}/{len(images)}...")
        all_text.append(extract_text_from_page_with_gemini(image))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_text))
    print(f"Saved: {output_path.resolve()}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--pdf")
    args = parser.parse_args()

    if args.pdf:
        run_gemini_ocr(Path(args.pdf), dpi=args.dpi)
    else:
        for pdf_path in input_folder.glob("*.pdf"):
            run_gemini_ocr(pdf_path, dpi=args.dpi)
    print("\nAll processing complete.")
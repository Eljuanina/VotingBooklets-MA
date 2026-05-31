#!/usr/bin/env python3
import os
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from io import BytesIO
import base64

# Input and output folders
# input_folder = Path("../corpus/raw_voting_booklets/Rätoromanisch")
# output_folder = Path("../gemini-ocr-2.5-flash-lite/rm")
input_folder = Path("../corpus/raw_voting_booklets/Italienisch")
output_folder = Path("../gemini-ocr-2.5-flash-lite/it")
# input_folder = Path("../corpus/raw_voting_booklets/Französisch")
# output_folder = Path("../gemini-ocr-2.5-flash-lite/fr")
# input_folder = Path("../corpus/raw_voting_booklets/Deutsch")
# output_folder = Path("../gemini-ocr-2.5-flash-lite/de")
output_folder.mkdir(parents=True, exist_ok=True)


load_dotenv()

# Initialize LLM client (Gemini via LangChain)
llm = ChatOpenAI(
    model="gemini-2.5-flash-lite",
    temperature=0.0,
    base_url=os.environ.get("GENAI_BASE_URL", "http://172.23.205.120:4000"),
    model_kwargs={
        "extra_body": {"drop_params": True}  # ensures reasoning / extra params are dropped
    },
)

OCR_PROMPT = """You are a professional OCR system. Extract ALL text from this PDF page with perfect accuracy.
If a page is empty, return only ' '.
Repeated punctuation or decorative characters (e.g., dot leaders, horizontal rules, underscores) that appear 
in sequences longer than 5 characters must be collapsed into a single representative token. 
   For example:
      - '.............' becomes '.'
      - '-------------' becomes '-'
      - '_____________' becomes '_'
Do not reproduce the full sequence. Continue extracting the next meaningful text token immediately after.


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

def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def extract_text_from_page_with_gemini(image: Image.Image) -> str:
    try:
        from langchain_core.messages import HumanMessage
        
        img_b64 = image_to_base64(image)
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": OCR_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                }
            ]
        )
        
        response = llm.invoke([message])
        return response.content or ""
    except Exception as e:
        print(f"[Detail] API Call failed: {e}")
        return f"[Error: {e}]"

def run_gemini_ocr(pdf_path: Path, dpi: int = 300) -> None:
    """Convert PDF to images and extract text page by page"""
    if not pdf_path.is_file():
        print(f"PDF not found: {pdf_path}")
        return

    output_txt = pdf_path.stem + "_extracted.txt"
    output_path = output_folder / output_txt

    # Skip if already processed
    if output_path.exists():
        print(f"Skipping (already exists): {output_path.name}")
        return

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
        page_text = extract_text_from_page_with_gemini(image)
        all_text.append(page_text)

    full_text = "\n\n".join(all_text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"Saved OCR text to: {output_path.resolve()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--pdf", help="Optional single PDF override")
    args = parser.parse_args()

    if args.pdf:
        run_gemini_ocr(Path(args.pdf), dpi=args.dpi)
    else:
        for pdf_path in input_folder.glob("*.pdf"):
            run_gemini_ocr(pdf_path, dpi=args.dpi)

    print("\nAll processing complete.")
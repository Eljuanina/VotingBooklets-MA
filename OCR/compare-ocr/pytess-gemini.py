"""
OCR Post-Correction Script
--------------------------
Loops through transcription files in ./tesseract/, pairs each with its
corresponding PDF, calls Gemini to post-correct the OCR output, and
writes the corrected text to ./post-ocr-tesseract/.

Expected layout
---------------
tesseract/
    document_a.txt     <- OCR transcription
    document_b.txt
    ...

pdfs/                  <- PDFs with the same base name (configurable via PDF_DIR)
    document_a.pdf
    document_b.pdf
    ...

Output
------
post-ocr-tesseract/
    document_a.txt     <- corrected transcription
    document_b.txt
    ...
"""

import os
import base64
import pathlib
import sys
from dotenv import load_dotenv  

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TESSERACT_DIR = pathlib.Path("pytesseract")          # folder with .txt transcriptions
PDF_DIR       = pathlib.Path("data")               # folder with matching PDFs
OUTPUT_DIR    = pathlib.Path("post-ocr-tesseract") # corrected output

# ---------------------------------------------------------------------------
# Initialize Gemini via LangChain / LiteLLM proxy
# ---------------------------------------------------------------------------
llm = ChatOpenAI(
    model="gemini-2.5-flash-lite",
    temperature=0.0,
    base_url=os.environ.get("GENAI_BASE_URL", "http://172.23.205.120:4000"),
    model_kwargs={
        "extra_body": {"drop_params": True}  # drop reasoning / extra params
    },
)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert OCR post-correction assistant specialising in scanned historical and legal documents.

You will be given:
1. A scanned PDF document.
2. The raw OCR transcription of that PDF produced by Tesseract.

═══════════════════════════════════════════
TASK — POST-CORRECT THE OCR TRANSCRIPTION
═══════════════════════════════════════════

Work through the transcription sentence by sentence, comparing it against the
actual content visible in the PDF. Fix every OCR error you find.

─── COMMON OCR ERRORS TO FIX ───────────────
- Wrong characters / confusable glyphs
    – 0 ↔ O,  1 ↔ l ↔ I,  rn ↔ m,  cl ↔ d,  ii ↔ u,  vv ↔ w
    – accented / special characters mis-read (ä→a, ü→u, é→e, ß→ss, etc.)
- Broken or merged words
    – words split across line-breaks that should be joined:
      CRITICAL: hyphenated line-breaks must ALWAYS be rejoined into one word.
      Examples:  "hy-\nphen" → "hyphen",  "Treib-\nstoffe" → "Treibstoffe",
                 "ermässig-\nten" → "ermässigten"
      A soft hyphen at the end of a line followed by the rest of the word on
      the next line is NEVER kept — always merge and drop the hyphen.
    – two words fused into one (thedog → the dog)
- Line-break errors
    – Tesseract often inserts a hard newline in the middle of a sentence where
      the original PDF simply wrapped to the next line. These mid-sentence
      newlines must be removed so the sentence flows as one continuous line.
    – Only start a new line when a new sentence genuinely begins, or when a
      structural element (heading, list item, numbered paragraph) starts.
- Spacing errors
    – missing spaces between words or after punctuation
    – extra spaces inside a word  (w o r d → word)
- Punctuation errors
    – comma read as period or vice-versa
    – opening/closing quote mis-read (,, or '' instead of „" or "")
    – spurious punctuation inserted by scanner noise
- Word-order errors caused by multi-column layout misread
- Numbers and dates garbled (l997 → 1997,  §l → §1)
- Stray characters from scan artefacts (|, _, ~, * where none exist in PDF)

─── OUTPUT FORMAT ───────────────────────────
- Write ONE sentence per line — never break a sentence across multiple lines.
- A "sentence" ends at a terminal punctuation mark (. ; : ? !) that genuinely
  closes the thought. Semicolons in legal enumerations (a. … ; b. … ;) count
  as sentence-ending for this purpose — each list item on its own line.
- Separate paragraphs with a single blank line.
- Preserve headings, section numbers, and list labels (a. b. 1. 2.) exactly.
- Do NOT add, remove, summarise, or paraphrase any content.
- Do NOT add markdown, HTML, code fences, bullet symbols, or any formatting
  that is not present in the original PDF.
- Return ONLY the corrected plain text — no commentary, no explanations.

─── EXAMPLE (before → after) ────────────────
BEFORE (raw Tesseract):
  a. eine Steuer auf dem Umsatz von Waren und Leistungen sowie auf der Einfuhr. Das Gesetz bezeichnet
    die Umsätze von Waren und Leistungen, die der Steuer
  zum normalen oder zum ermässigten Satz unterliegen. Die Steuer beträgt höch-
  stens 10 Prozent des Entgelts;

AFTER (corrected):
  a. eine Steuer auf dem Umsatz von Waren und Leistungen sowie auf der Einfuhr. Das Gesetz bezeichnet die Umsätze von Waren und Leistungen, die der Steuer zum normalen oder zum ermässigten Satz unterliegen. Die Steuer beträgt höchstens 10 Prozent des Entgelts;

Key corrections applied in the example:
  1. Mid-sentence line-breaks removed → sentences flow on one line.
  2. "höch-\nstens" rejoined → "höchstens".
"""

def encode_pdf_as_base64(pdf_path: pathlib.Path) -> str:
    """Read a PDF and return its base64-encoded content."""
    with open(pdf_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def correct_transcription(pdf_path: pathlib.Path, ocr_text: str) -> str:
    """Send PDF + OCR text to Gemini and return the corrected transcription."""
    pdf_b64 = encode_pdf_as_base64(pdf_path)

    message = HumanMessage(
        content=[
            # System instruction embedded in the user turn (LiteLLM proxy friendly)
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
            },
            # The original PDF as a document (base64)
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": pdf_b64,
                },
            },
            # The raw OCR transcription
            {
                "type": "text",
                "text": (
                    "Below is the raw Tesseract OCR transcription for this PDF. "
                    "Please post-correct it:\n\n"
                    f"{ocr_text}"
                ),
            },
        ]
    )

    response = llm.invoke([message])
    return response.content.strip()


def main():
    # Validate input directory
    if not TESSERACT_DIR.exists():
        print(f"ERROR: Tesseract directory not found: {TESSERACT_DIR.resolve()}")
        sys.exit(1)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(TESSERACT_DIR.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {TESSERACT_DIR.resolve()}")
        sys.exit(0)

    print(f"Found {len(txt_files)} transcription(s) to process.\n")

    success_count = 0
    error_count   = 0

    for txt_path in txt_files:
        stem     = txt_path.stem                          # e.g. "de_1977_extracted"
        pdf_stem = stem.removesuffix("_extracted")        # e.g. "de_1977"
        pdf_path = PDF_DIR / f"{pdf_stem}.pdf"

        print(f"[{stem}]")

        # --- check PDF exists ---
        if not pdf_path.exists():
            print(f"  ⚠  PDF not found at {pdf_path} (looked for '{pdf_stem}.pdf') — skipping.\n")
            error_count += 1
            continue

        # --- read OCR text ---
        ocr_text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
        if not ocr_text:
            print(f"  ⚠  Empty transcription — skipping.\n")
            error_count += 1
            continue

        print(f"  PDF : {pdf_path}")
        print(f"  OCR : {len(ocr_text):,} chars")

        # --- call Gemini ---
        try:
            corrected = correct_transcription(pdf_path, ocr_text)
        except Exception as exc:
            print(f"  ✗  Gemini error: {exc}\n")
            error_count += 1
            continue

        # --- write corrected output ---
        out_path = OUTPUT_DIR / txt_path.name
        out_path.write_text(corrected, encoding="utf-8")
        print(f"  ✓  Saved corrected text → {out_path}  ({len(corrected):,} chars)\n")
        success_count += 1

    # Summary
    print("─" * 50)
    print(f"Done.  ✓ {success_count} corrected   ✗ {error_count} skipped/errored")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
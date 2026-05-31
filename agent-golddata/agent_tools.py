import os
import re
import base64
from io import BytesIO
import time
import pytesseract
from pdf2image import convert_from_path
from pypdf import PdfReader, PdfWriter
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import numpy as np
import unicodedata
import json
from llm_config import llm
from google import genai as google_genai
from google.genai import types as genai_types
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# ==========================
# CONFIG
# ==========================

FIXED_DIR = "data/fixed_pdfs"
TXT_DIR = "data/booklets_txt"

LANG_MAP = {
    "de": "deu",
    "fr": "fra",
    "it": "ita",
    "rm": "roh",
}

LANG_NAMES = {
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "rm": "Romansh",
}

gemini_llm = ChatOpenAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    base_url="http://172.23.205.120:4000",
    extra_body={"drop_params": True},
)

# ==========================
# HELPERS
# ==========================

def clean_tesseract_input(text: str) -> str:
    """
    Strip dot leaders and garbage from raw Tesseract output BEFORE sending to LLM.
    Prevents the LLM from seeing dots and reproducing/extending them.
    """
    text = re.sub(r"[.]{3,}", " ", text)
    text = re.sub(r"(\. ){3,}", " ", text)
    text = re.sub(r"-{3,}", "—", text)
    text = re.sub(r"_{3,}", "", text)
    text = re.sub(r"[|/\\]{2,}", "", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_hallucinations(text: str) -> str:
    """
    Remove LLM hallucinations that commonly appear in documents.
    """
    text = re.sub(r'\.{4,}', '', text)
    text = re.sub(r'-{4,}', '—', text)
    text = re.sub(r'_{4,}', '', text)
    lines = [l.rstrip() for l in text.splitlines()]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
 


def ensure_dirs():
    os.makedirs(FIXED_DIR, exist_ok=True)
    os.makedirs(TXT_DIR, exist_ok=True)


def detect_lang(pdf_path: str) -> str:
    """Returns tesseract lang string."""
    parent = os.path.basename(os.path.dirname(pdf_path))
    lang_code = LANG_MAP.get(parent)
    if lang_code:
        return lang_code
    for code, tess in LANG_MAP.items():
        if re.search(rf"\b{code}\b", os.path.basename(pdf_path), re.IGNORECASE):
            return tess
    return "deu+fra+ita"


def detect_lang_code(pdf_path: str) -> str:
    """Returns the short lang code (de/fr/it/rm)."""
    parent = os.path.basename(os.path.dirname(pdf_path))
    if parent in LANG_MAP:
        return parent
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    for code in LANG_MAP:
        if re.search(rf"(^|[_\-]){code}([_\-]|$)", stem, re.IGNORECASE):
            return code
    return "de"


def img_to_b64(img) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _read_paragraphs(path: str) -> list:
    """
    Read a corrected OCR txt file and return a list of paragraphs.
    Always splits on blank lines.
    This preserves paragraph-level granularity for alignment.
    """
    with open(path, encoding="utf-8-sig") as f:
        raw = f.read()

    def normalize(t):
        t = unicodedata.normalize("NFKC", t)
        # Collapse internal newlines within a paragraph into spaces
        t = re.sub(r"\n", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    return [normalize(p) for p in re.split(r"\n\s*\n", raw) if normalize(p)]
    
# ==========================
# TOOL 1: FIX ROTATION
# ==========================

@tool
def fix_pdf_rotation(pdf_path: str) -> str:
    """
    Checks the rotation metadata of each page in a PDF using pypdf.
    If any pages have a non-zero /Rotate value in their metadata,
    the rotation is applied into the page content and cleared,
    producing a visually correct, normalized PDF.

    Args:
        pdf_path: Path to the input PDF file.

    Returns:
        A status message with the path to the fixed PDF,
        or ROTATION_OK with the original path if no fix was needed.
    """
    print(f"🔄 Checking rotation: {pdf_path}")

    if not os.path.exists(pdf_path):
        return f"FILE_NOT_FOUND: {pdf_path}"

    ensure_dirs()

    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        return f"ERROR_READING_PDF: {e}"

    page_rotations = []
    for i, page in enumerate(reader.pages):
        rotation = page.rotation
        page_rotations.append(rotation)
        status = f"{rotation}°" if rotation != 0 else "OK"
        print(f"   📄 Page {i + 1}: /Rotate = {status}")

    if all(r == 0 for r in page_rotations):
        print("   ✅ All pages already correctly oriented.")
        return f"ROTATION_OK: {pdf_path}"

    writer = PdfWriter()
    fixed_pages = []

    for i, page in enumerate(reader.pages):
        existing = page.rotation
        if existing != 0:
            correction = (360 - existing) % 360
            page.rotate(correction)
            fixed_pages.append((i + 1, existing))
            print(f"   🔧 Page {i + 1}: was {existing}° → corrected by {correction}°")
        writer.add_page(page)

    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    out_path = os.path.join(FIXED_DIR, f"{basename}_fixed.pdf")

    with open(out_path, "wb") as f:
        writer.write(f)

    summary = ", ".join(f"p{p} was {r}°" for p, r in fixed_pages)
    print(f"   💾 Saved: {out_path}")
    return f"ROTATION_FIXED: {out_path} | Corrected: {summary}"


# ==========================
# TOOL 2: OCR WITH TESSERACT
# ==========================

@tool
def ocr_pdf_tesseract(pdf_path: str) -> str:
    """
    Performs OCR on a PDF file using pytesseract.
    Each page is rendered at 300 DPI and passed through Tesseract.
    Pages are separated by a blank line in the output text.
    The language is auto-detected from the file path (de/fr/it/rm).
    Saves the result as a .txt file and returns its path.
    Always run fix_pdf_rotation before this tool.

    Args:
        pdf_path: Path to the input PDF file (should already be rotation-corrected).

    Returns:
        Path to the generated .txt file, or an error message.
    """
    print(f"📄 OCR (tesseract): {pdf_path}")

    if not os.path.exists(pdf_path):
        return f"FILE_NOT_FOUND: {pdf_path}"

    ensure_dirs()

    lang_folder = os.path.basename(os.path.dirname(pdf_path))
    basename = os.path.basename(pdf_path)
    date_match = re.search(r"(\d{6,8})", basename)
    date_str = date_match.group(1) if date_match else os.path.splitext(basename)[0]
    out_file = os.path.join(TXT_DIR, f"{lang_folder}_{date_str}.txt")

    if os.path.exists(out_file):
        print(f"   ⏭️  Already exists: {out_file}")
        return out_file

    tess_lang = detect_lang(pdf_path)
    print(f"   🌐 Tesseract language: {tess_lang}")

    print("   📸 Rendering pages at 300 DPI...")
    try:
        images = convert_from_path(pdf_path, dpi=300)
    except Exception as e:
        return f"ERROR_RENDERING_PDF: {e}"

    print(f"   🔍 Running OCR on {len(images)} page(s)...")
    page_texts = []
    for i, img in enumerate(images):
        print(f"      Page {i + 1}/{len(images)}...")
        text = pytesseract.image_to_string(img, lang=tess_lang)
        page_texts.append(text.strip())

    full_text = "\n\n".join(page_texts)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(full_text)

    size = os.path.getsize(out_file)
    print(f"   💾 Saved: {out_file} ({size} bytes)")
    return out_file

# ==========================
# TOOL: GEMINI 2.5 PRO OCR
# ==========================

@tool
def ocr_pdf_gemini(pdf_path: str) -> str:
    """
    Performs OCR on a PDF file using Gemini 2.5 Pro vision model.

    The quality if the OCR is much better than Tesseract, but the cost to do it is higher. 
    Each page is rendered at 300 DPI, sent to Gemini as a JPEG image,
    and the extracted text is assembled page by page.
    The result is saved as a .txt file (same naming convention as
    ocr_pdf_tesseract) and its path is returned.
    Always run fix_pdf_rotation before this tool.

    Args:
        pdf_path: Path to the input PDF file (should already be rotation-corrected).

    Returns:
        Path to the generated .txt file, or an error message.
    """

    print(f"📄 OCR (Gemini 2.5 Pro): {pdf_path}")

    if not os.path.exists(pdf_path):
        return f"FILE_NOT_FOUND: {pdf_path}"

    ensure_dirs()

    lang_folder = os.path.basename(os.path.dirname(pdf_path))
    basename = os.path.basename(pdf_path)
    date_match = re.search(r"(\d{6,8})", basename)
    date_str = date_match.group(1) if date_match else os.path.splitext(basename)[0]
    out_file = os.path.join(TXT_DIR, f"{lang_folder}_{date_str}.txt")

    if os.path.exists(out_file):
        print(f"   ⏭️  Already exists: {out_file}")
        return out_file

    GEMINI_OCR_PROMPT = (
        "Extract all text from this document page image. Preserve the exact wording.\n"
        "If the image shows a double-page spread (two pages side by side), "
        "extract the left page first in full, then the right page in full.\n"
        "Return the text with each paragraph on its own block, separated by a blank line.\n"
        "Within a paragraph, keep all lines joined as a single block of text.\n"
        "Do not split paragraphs into individual sentences.\n"
        "Do not add labels, page numbers, or any extra text — only the extracted document text.\n"
        "NEVER generate long sequences of dots (......) or dashes (------). "
        "If the page shows a dotted leader line, replace it with a single space."
    )

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return "ERROR: GEMINI_API_KEY or GOOGLE_API_KEY not set in environment."

    client = google_genai.Client(api_key=api_key)

    print("   📸 Rendering pages at 300 DPI...")
    try:
        images = convert_from_path(pdf_path, dpi=300)
    except Exception as e:
        return f"ERROR_RENDERING_PDF: {e}"

    print(f"   🔍 Running Gemini OCR on {len(images)} page(s)...")
    page_texts = []
    MAX_RETRIES = 3
    RETRY_DELAY = 10

    for i, img in enumerate(images):
        print(f"      Page {i + 1}/{len(images)}...")
        b64 = img_to_b64(img)

        page_text = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=[
                        GEMINI_OCR_PROMPT,
                        google_genai.types.Part.from_bytes(
                            data=base64.b64decode(b64),
                            mime_type="image/jpeg",
                        ),
                    ],
                    config=google_genai.types.GenerateContentConfig(temperature=0.0),
                )
                page_text = response.text or ""
                break
            except Exception as e:
                print(f"      [Attempt {attempt}/{MAX_RETRIES}] API error: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)

        if page_text is None:
            print(f"   [Abort] Page {i + 1} failed after {MAX_RETRIES} attempts — not saving output.")
            return f"ERROR_OCR_FAILED: page {i + 1} of {pdf_path}"

        page_texts.append(page_text.strip())

    full_text = "\n\n".join(page_texts)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(full_text)

    size = os.path.getsize(out_file)
    print(f"   💾 Saved: {out_file} ({size} bytes)")
    return out_file


# ==========================
# TOOL: CLEAN OCR TEXT
# ==========================

@tool
def clean_ocr_text(txt_path: str) -> str:
    """
    Cleans a raw OCR .txt file using the standard Swiss voting booklet
    text normalisation pipeline.

    Applies the following cleaning steps in order:
    1. Unicode normalisation (NFKC)
    2. Remove invisible/zero-width characters and soft hyphens
    3. Normalise curly quotes → straight quotes, em/en-dashes → hyphen
    4. Remove layout artefacts: image placeholders, page markers, boilerplate
       metadata (e.g. "02-15_d 12.06.1977 10:30 Uhr Seite 3"), lone bullet symbols
    5. Rejoin hyphenated line-breaks (e.g. "Bun-\ndesrat" → "Bundesrat")
    6. Remove list markers (1. / a. / - at line start)
    7. Collapse multiple blank lines to a single blank line (paragraph separator)
    8. Normalise spacing around punctuation

    Run this tool AFTER ocr_pdf_tesseract or ocr_pdf_gemini, and BEFORE
    llm_post_correct_ocr or align_with_swissbert. It is safe to run on
    already-cleaned files (idempotent).

    The cleaned text overwrites the input file in-place.
    Paragraph boundaries (blank lines) are preserved throughout.

    Args:
        txt_path: Path to the OCR .txt file to clean.

    Returns:
        A status message with the file path and paragraph count after cleaning.
    """
    if not txt_path.endswith(".txt"):
        raise ValueError(f"Expected a .txt file, got: {txt_path}")
    with open(txt_path, encoding="utf-8", errors="replace") as f:
        text = f.read()


    print(f"🧹 Cleaning OCR text: {txt_path}")

    if not os.path.exists(txt_path):
        return f"FILE_NOT_FOUND: {txt_path}"

    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    original_paras = len([p for p in re.split(r"\n\s*\n", text) if p.strip()])

    # ── 1. Unicode normalisation ──────────────────────────────────────────────
    text = unicodedata.normalize("NFKC", text)

    # ── 2. Invisible / zero-width characters ─────────────────────────────────
    text = text.replace("\u00AD", "")                        # soft hyphen
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)        # zero-width chars

    # ── 3. Quotes and dashes ──────────────────────────────────────────────────
    text = text.replace("\u2018", "'").replace("\u2019", "'")  # curly single quotes
    text = text.replace("\u201C", '"').replace("\u201D", '"')  # curly double quotes
    text = re.sub(r"[\u2013\u2014\u2212]", "-", text)          # en-dash, em-dash, minus

    # ── 4. Layout artefacts ───────────────────────────────────────────────────
    text = re.sub(r"<!--\s*image\s*-->", "", text)
    text = re.sub(r"--\s*Page\s*\d+\s*--", "", text)
    text = re.sub(r"\s*--\s*", " ", text)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

    # Boilerplate metadata: "02-15 Deutsch 12.06.1977 10:30 Uhr Seite 3"
    text = re.sub(
        r"\b\d{2}-\d{2}\s+[A-Za-z]+\s+\d{1,2}\.\d{1,2}\.\d{4}\s+\d{1,2}:\d{2}\s+Uhr\s+Seite\s+\d+\s*[A-Za-z]*",
        "", text
    )
    text = re.sub(
        r"\b02-15_[dr]\s+.*?Seite\s+\d+.*?(?=\s|$)",
        "", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"\b\d{2}-\d{2}\s+[A-Za-z]+\s+Seite\s+\d+.*?(?=\s|$)",
        "", text
    )
    text = re.sub(
        r"\b[A-Za-z0-9_-]+\s+\d{1,2}\.\d{1,2}\.\d{4}\s+\d{1,2}:\d{2}\s+Uhr\s+Seite\s+[0-9]+",
        "", text
    )

    # Lone bullet symbols
    text = re.sub(r"[■•●▪]", "", text)

    # ── 5. Hyphenated line-breaks (within a paragraph only) ──────────────────
    # Only rejoin lines that are not separated by a blank line
    text = re.sub(r"-\s*\n(?!\s*\n)", "", text)

    # ── 6. List markers ───────────────────────────────────────────────────────
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[a-zA-Z]\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*-\s+", "", text, flags=re.MULTILINE)

    # ── 7. Collapse multiple blank lines → single blank line ─────────────────
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that are purely whitespace but keep blank-line separators
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines).strip()

    # ── 8. Spacing around punctuation ────────────────────────────────────────
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = text.strip()

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    cleaned_paras = len([p for p in re.split(r"\n\s*\n", text) if p.strip()])
    size = os.path.getsize(txt_path)
    print(f"   💾 Cleaned: {txt_path} ({original_paras} → {cleaned_paras} paragraphs, {size} bytes)")
    return (
        f"CLEANED: {txt_path} | "
        f"{original_paras} → {cleaned_paras} paragraphs | "
        f"{size} bytes"
    )


# ==========================
# TOOL: CHECK OCR QUALITY
# ==========================

OCR_QUALITY_PROMPT = """You are an expert OCR quality auditor for Swiss federal voting booklets.

You will receive a sample of extracted OCR text, and optionally a page image from the source PDF.

Evaluate the following dimensions:
1. CHARACTER_ERROR_RATE — estimate what fraction of characters look wrong (0.0–1.0)
2. GARBLING — are there sequences of nonsense characters or symbol soup? (none/minor/severe)
3. LANGUAGE_LEGIBILITY — can you still read and understand the text? (good/degraded/unreadable)
4. MISSING_CONTENT — does the text look suspiciously short or truncated? (ok/suspect/bad)
5. STRUCTURAL_INTEGRITY — are paragraphs and sentence boundaries preserved? (ok/broken/lost)
6. PAGE_ORDER — if a PDF page image is provided, compare the start of the OCR text against it:
   does the text begin with the LEFT side of the spread before the RIGHT side?
   If no image is provided, or the page is clearly single-sided, mark as ok.
   (ok/suspect/swapped)

Then give:
- OVERALL_SCORE: integer 0–100 (100 = perfect, 0 = completely unusable)
- NEEDS_REDO: true if OVERALL_SCORE < 60 or GARBLING=severe or LANGUAGE_LEGIBILITY=unreadable or PAGE_ORDER=swapped
- RECOMMENDED_ENGINE: which OCR engine to use if redo is needed
  * "tesseract" — for clean modern documents where Tesseract is likely sufficient
  * "gemini"    — for old/degraded/complex documents, Romansh, or when page order is swapped
                  (Tesseract cannot recover correct left/right order on its own)
- ISSUES: list of short strings describing the specific problems found

Respond ONLY with a JSON object, no markdown fences, no preamble:
{{
  "character_error_rate": <float>,
  "garbling": "<none|minor|severe>",
  "language_legibility": "<good|degraded|unreadable>",
  "missing_content": "<ok|suspect|bad>",
  "structural_integrity": "<ok|broken|lost>",
  "page_order": "<ok|suspect|swapped>",
  "overall_score": <int 0-100>,
  "needs_redo": <bool>,
  "recommended_engine": "<tesseract|gemini>",
  "issues": [<str>, ...]
}}

OCR TEXT SAMPLE:
{sample}"""


@tool
def check_ocr_quality(txt_path: str, pdf_path: str = None, sample_lines: int = 80) -> str:
    """
    Audits the quality of a raw OCR .txt file and reports whether it needs
    to be redone, and if so, with which engine.

    Sends a sample of the OCR text to Gemini for evaluation across five
    quality dimensions: character error rate, garbling, language legibility,
    missing content, and structural integrity. Returns a 0–100 overall score
    and a clear NEEDS_REDO verdict.

    If pdf_path is provided, the tool renders the first page of the PDF and
    sends it alongside the OCR text so Gemini can verify that double-page
    spreads were extracted left-side first. This is the only reliable way to
    catch left/right scrambling, since neither Tesseract nor heuristics can
    detect it from text alone.

    The agent should call this tool after ocr_pdf_tesseract or ocr_pdf_gemini,
    and act on the result autonomously:
    - If NEEDS_REDO=false → proceed to clean_ocr_text and llm_post_correct_ocr.
    - If NEEDS_REDO=true and recommended_engine="tesseract" → rerun ocr_pdf_tesseract.
    - If NEEDS_REDO=true and recommended_engine="gemini" → switch to ocr_pdf_gemini
      (higher quality, higher cost — use when Tesseract output is severely degraded).
    - If page_order=swapped → always rerun with ocr_pdf_gemini regardless of overall score,
      since Tesseract cannot recover correct order on its own.

    Args:
        txt_path:     Path to the OCR .txt file to audit.
        pdf_path:     Path to the original PDF. Strongly recommended — required for
                      reliable page order verification on double-page spreads.
        sample_lines: How many lines to sample for the quality check (default 80).

    Returns:
        A structured status string with the quality verdict and all dimension scores,
        plus the path to a saved JSON report.
    """
    print(f"🔬 OCR quality check: {txt_path}")

    if not os.path.exists(txt_path):
        return f"FILE_NOT_FOUND: {txt_path}"

    with open(txt_path, encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return "OCR_QUALITY_FAIL: file is empty — must redo OCR"

    total_lines = len(lines)

    if total_lines <= sample_lines:
        sample = lines
    else:
        third = sample_lines // 3
        mid_start = max(0, total_lines // 2 - third // 2)
        sample = (
            lines[:third]
            + lines[mid_start: mid_start + third]
            + lines[max(0, total_lines - third):]
        )

    sample_text = "".join(sample).strip()
    if not sample_text:
        return "OCR_QUALITY_FAIL: sampled text is blank — must redo OCR"

    # Build message — with or without PDF image
    if pdf_path and os.path.exists(pdf_path):
        print("   📸 Rendering first page for page-order verification...")
        try:
            images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=1)
            first_page_img = images[0]
            b64 = img_to_b64(first_page_img)
            has_image = True
        except Exception as e:
            print(f"   ⚠️  Could not render PDF for comparison: {e}")
            has_image = False
    else:
        has_image = False
        if not pdf_path:
            print("   ⚠️  No pdf_path provided — page order check will be text-only (unreliable)")

    prompt = OCR_QUALITY_PROMPT.format(sample=sample_text[:6000])

    if has_image:
        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": (
                "The image above is the first page of the source PDF. "
                "Use it to verify whether the OCR text starts with the LEFT side of the spread "
                "before the RIGHT side. If the page is single-sided, mark page_order as ok."
            )},
        ])
    else:
        message = HumanMessage(content=[{"type": "text", "text": prompt}])

    try:
        response = gemini_llm.invoke([message])
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
        result = json.loads(raw)
    except Exception as e:
        return f"OCR_QUALITY_ERROR: could not parse LLM response — {e}"

    report_path = txt_path.replace(".txt", "_ocr_quality.json")
    result["txt_path"] = txt_path
    result["pdf_path"] = pdf_path
    result["total_lines"] = total_lines
    result["sampled_lines"] = len(sample)
    result["image_verified"] = has_image
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    score = result.get("overall_score", -1)
    needs_redo = result.get("needs_redo", False)
    engine = result.get("recommended_engine", "gemini")
    garbling = result.get("garbling", "?")
    legibility = result.get("language_legibility", "?")
    missing = result.get("missing_content", "?")
    structure = result.get("structural_integrity", "?")
    page_order = result.get("page_order", "?")
    issues = result.get("issues", [])

    print(f"   📊 Overall score    : {score}/100")
    print(f"   Garbling            : {garbling}")
    print(f"   Language legibility : {legibility}")
    print(f"   Missing content     : {missing}")
    print(f"   Structural integrity: {structure}")
    print(f"   Page order          : {page_order} {'(verified against PDF)' if has_image else '(text-only, unreliable)'}")
    print(f"   Needs redo          : {needs_redo}")
    if needs_redo:
        print(f"   Recommended engine  : {engine}")
    if issues:
        print(f"   Issues              : {'; '.join(issues)}")
    print(f"   💾 Report saved     : {report_path}")

    verdict = "NEEDS_REDO" if needs_redo else "OK"
    issues_str = "; ".join(issues) if issues else "none"
    return (
        f"OCR_QUALITY_{verdict}: score={score}/100 | "
        f"garbling={garbling} | legibility={legibility} | "
        f"missing={missing} | structure={structure} | "
        f"page_order={page_order} ({'image-verified' if has_image else 'text-only'}) | "
        f"needs_redo={needs_redo} | recommended_engine={engine} | "
        f"issues=[{issues_str}] | "
        f"report={report_path}"
    )


# ==========================
# TOOL 4: LLM POST-CORRECTION
# ==========================

POST_CORRECTION_PROMPT = """You are an expert OCR post-correction specialist for Swiss federal voting booklets written in {lang_name}.

You are given:
1. A page image from the original PDF (the ground truth)
2. The raw OCR text extracted from that page

CRITICAL — PAGE ORDER:
If the image shows a double-page spread, the LEFT page must come first and the RIGHT page second.
If the current text starts with content from the right side, reorder it so the left side appears first.

Your correction task:
- Fix every misrecognized character, broken word, or garbled sequence
- Restore missing accents, hyphens, and punctuation (ä, ö, ü, é, à, ç, etc.)
- Correct wrongly merged or split words
- Remove page numbers
- Preserve ALL paragraph breaks (blank lines between paragraphs), headings, and numbering exactly as they appear in the image
- Do NOT split paragraphs into individual sentences
- Do NOT add, remove, or paraphrase any content — only fix OCR errors
- Output ONLY the corrected text, nothing else

CRITICAL RULES — violating these is worse than leaving the OCR error uncorrected:
- NEVER generate long sequences of dots (......), dashes (------), or any repeated characters
- If the page shows a dotted leader line, replace it with a single space or nothing
- NEVER invent or extend content that is not clearly legible in the image
- If a word or number is illegible, leave a single [?] placeholder — do not guess or fill in

RAW OCR TEXT:
{ocr_text}"""

@tool
def llm_post_correct_ocr(txt_path: str, pdf_path: str) -> str:
    """
    Post-corrects a raw OCR .txt file using an LLM (Gemini 2.5 Flash Lite).
    For each page, it sends the original PDF page image alongside the raw OCR
    text to the LLM, which compares both and fixes all OCR errors while preserving
    paragraph structure and content exactly. Paragraphs are separated by blank lines.

    The corrected text overwrites the original .txt file.

    Args:
        txt_path: Path to the raw OCR .txt file to correct.
        pdf_path: Path to the original PDF (used to render page images for comparison).

    Returns:
        Path to the corrected .txt file with a summary.
    """
    MAX_RETRIES = 3
    RETRY_DELAY = 10

    print(f"🤖 LLM post-correction: {txt_path}")

    if not os.path.exists(txt_path):
        return f"FILE_NOT_FOUND: {txt_path}"
    if not os.path.exists(pdf_path):
        return f"FILE_NOT_FOUND: {pdf_path}"

    lang_code = detect_lang_code(pdf_path)
    lang_name = LANG_NAMES.get(lang_code, "German")

    with open(txt_path, encoding="utf-8") as f:
        existing_text = f.read()
    ocr_pages = [p.strip() for p in existing_text.split("\n\n") if p.strip()]

    print("   📸 Rendering pages...")
    try:
        images = convert_from_path(pdf_path, dpi=200)
    except Exception as e:
        return f"ERROR_RENDERING_PDF: {e}"

    n_pages = len(images)
    print(f"   🔍 Correcting {n_pages} page(s) with Gemini 2.5 Flash Lite...")

    corrected_pages = []
    failed_pages = []

    for i, img in enumerate(images):
        ocr_text = ocr_pages[i] if i < len(ocr_pages) else ""
        print(f"      Page {i + 1}/{n_pages}...")

        b64 = img_to_b64(img)
        ocr_text_clean = clean_tesseract_input(ocr_text)
        prompt = POST_CORRECTION_PROMPT.format(
            lang_name=lang_name,
            ocr_text=ocr_text_clean,
        )
        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ])

        page_text = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = gemini_llm.invoke([message])
                page_text = clean_hallucinations(response.content.strip())
                break
            except Exception as e:
                print(f"      [Attempt {attempt}/{MAX_RETRIES}] API error: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)

        if page_text is None:
            # Fall back to the cleaned OCR text for this page
            print(f"      ⚠️  Page {i + 1} failed — using cleaned OCR text as fallback")
            page_text = ocr_text_clean
            failed_pages.append(i + 1)

        corrected_pages.append(page_text)

    full_text = "\n\n".join(corrected_pages)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    size = os.path.getsize(txt_path)

    if failed_pages:
        print(f"   ⚠️  {len(failed_pages)} page(s) used OCR fallback: {failed_pages}")
        return (
            f"CORRECTED_PARTIAL: {txt_path} | {n_pages} pages | "
            f"fallback pages (OCR only): {failed_pages}"
        )

    print(f"   💾 Saved corrected OCR: {txt_path} ({size} bytes)")
    return f"CORRECTED: {txt_path} | {n_pages} pages post-corrected by Gemini"

# ==========================
# TOOL 5: SWISSBERT ALIGNMENT
# ==========================
PARALLEL_DIR    = "data/parallel_data"
MODEL_NAME      = "jgrosjean-mathesis/sentence-swissbert"
SWISSBERT_BATCH = 32

LANG_CODE_MAP = {
    "de": "de_CH",
    "fr": "fr_CH",
    "it": "it_CH",
    "rm": "rm_CH",
}

_GAP_PENALTY  = -0.4
_MERGE_BONUS  = -0.4
_POS_WEIGHT   =  0.8
_MAX_MERGE    =  4

_swissbert_cache = {}


def _load_swissbert():
    if "model" not in _swissbert_cache:
        print(f"   🧠 Loading {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        _swissbert_cache["tokenizer"] = tokenizer
        _swissbert_cache["model"] = model
        _swissbert_cache["device"] = device
        print(f"   ✅ Model ready on {device}")
    return _swissbert_cache["model"], _swissbert_cache["tokenizer"], _swissbert_cache["device"]


def _encode(sentences: list, lang: str) -> np.ndarray:
    model, tokenizer, device = _load_swissbert()
    lang_code = LANG_CODE_MAP.get(lang, "de_CH")
    if hasattr(model, "set_default_language"):
        model.set_default_language(lang_code)
    all_emb = []
    for start in range(0, len(sentences), SWISSBERT_BATCH):
        batch = sentences[start: start + SWISSBERT_BATCH]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            return_tensors="pt", max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        tok_emb = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        emb = (
            (tok_emb * mask).sum(dim=1) /
            mask.sum(dim=1).clamp(min=1e-9)
        ).cpu().numpy()
        all_emb.append(emb)
    return np.vstack(all_emb)

def _dp_align(
    a_emb: np.ndarray,
    b_emb: np.ndarray,
    position_weight: float = _POS_WEIGHT,
) -> list:
    n, m = len(a_emb), len(b_emb)
    # no division by 0 errors
    if n == 0 or m == 0:
        return []
    pw = position_weight

    sim = cosine_similarity(a_emb, b_emb)
    for i in range(n):
        for j in range(m):
            pos_diff = abs(i / n - j / m)
            sim[i, j] += pw * (1.0 - pos_diff)

    merged_a: dict = {}
    merged_b: dict = {}

    def get_a(i, l):
        if (i, l) not in merged_a:
            merged_a[(i, l)] = np.mean(a_emb[i: i + l], axis=0)
        return merged_a[(i, l)]

    def get_b(j, l):
        if (j, l) not in merged_b:
            merged_b[(j, l)] = np.mean(b_emb[j: j + l], axis=0)
        return merged_b[(j, l)]

    def seg_sim(i, la, j, lb):
        va = get_a(i, la)
        vb = get_b(j, lb)
        dot = np.dot(va, vb)
        norm = np.linalg.norm(va) * np.linalg.norm(vb)
        raw = float(dot / norm) if norm > 0 else 0.0
        pos_a = (i + (la - 1) / 2) / n
        pos_b = (j + (lb - 1) / 2) / m
        return raw + pw * (1.0 - abs(pos_a - pos_b))

    INF = float("-inf")
    dp = np.full((n + 1, m + 1), INF)
    back = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(n + 1):
        for j in range(m + 1):
            cur = dp[i][j]
            if cur == INF:
                continue
            for la in range(0, _MAX_MERGE + 1):
                if i + la > n:
                    break
                for lb in range(0, _MAX_MERGE + 1):
                    if la == 0 and lb == 0:
                        continue
                    if j + lb > m:
                        break
                    ni, nj = i + la, j + lb
                    if la == 0:
                        score = cur + _GAP_PENALTY * lb
                    elif lb == 0:
                        score = cur + _GAP_PENALTY * la
                    else:
                        penalty = _MERGE_BONUS * (la + lb - 2)
                        score = cur + seg_sim(i, la, j, lb) + penalty
                    if score > dp[ni][nj]:
                        dp[ni][nj] = score
                        back[ni][nj] = (i, j, la, lb)

    pairs_raw = []
    i, j = n, m
    while i > 0 or j > 0:
        prev = back[i][j]
        if prev is None:
            break
        pi, pj, la, lb = prev
        pairs_raw.append((tuple(range(pi, pi + la)), tuple(range(pj, pj + lb))))
        i, j = pi, pj
    pairs_raw.reverse()

    pairs = []
    seen_a: set = set()
    seen_b: set = set()
    for a_idxs, b_idxs in pairs_raw:
        if set(a_idxs) & seen_a or set(b_idxs) & seen_b:
            continue
        pairs.append((a_idxs, b_idxs))
        seen_a.update(a_idxs)
        seen_b.update(b_idxs)
    return pairs

def _build_rows(de: list, langs: dict) -> list:
    de_groups: dict = {}
    for lang, (texts, align) in langs.items():
        for a_idxs, b_idxs in align:
            key = tuple(sorted(a_idxs))
            if key not in de_groups:
                de_groups[key] = {l: [] for l in langs}
            de_groups[key][lang].extend(list(b_idxs))

    for key in list(de_groups.keys()):
        for lang in langs:
            de_groups[key].setdefault(lang, [])

    de_groups.pop((), None)
    matched_keys = sorted([k for k in de_groups if k], key=lambda k: k[0])

    index_to_keys: dict = {}
    for k in matched_keys:
        for idx in k:
            index_to_keys.setdefault(idx, []).append(k)

    keys_to_remove: set = set()
    for idx, keys in index_to_keys.items():
        if len(keys) <= 1:
            continue
        primary = max(keys, key=len)
        for other in keys:
            if other == primary or other in keys_to_remove:
                continue
            for lang in langs:
                de_groups[primary][lang].extend(de_groups[other].get(lang, []))
            keys_to_remove.add(other)
    for k in keys_to_remove:
        del de_groups[k]

    matched_keys = sorted([k for k in de_groups if k], key=lambda k: k[0])

    rows: list = []
    used_lang: dict = {lang: set() for lang in langs}
    lang_idx_to_row: dict = {lang: {} for lang in langs}

    for ri, key in enumerate(matched_keys):
        group = de_groups[key]
        de_text = " ".join(de[i] for i in key)
        lang_cols: dict = {}
        for lang, (texts, _) in langs.items():
            indices = sorted(set(group[lang]))
            lang_cols[lang] = " ".join(texts[j] for j in indices)
            used_lang[lang].update(indices)
            for j in indices:
                lang_idx_to_row[lang][j] = ri
        rows.append([float(key[0]), de_text, lang_cols])

    extra_rows: list = []
    for lang, (texts, _) in langs.items():
        unmatched_js = sorted(j for j in range(len(texts)) if j not in used_lang[lang])
        if not unmatched_js:
            continue
        groups: list = []
        grp = [unmatched_js[0]]
        for j in unmatched_js[1:]:
            if j == grp[-1] + 1:
                grp.append(j)
            else:
                groups.append(grp)
                grp = [j]
        groups.append(grp)

        matched_js_sorted = sorted(used_lang[lang])
        for grp in groups:
            grp_text = " ".join(texts[j] for j in grp)
            empty = {l: "" for l in langs}
            empty[lang] = grp_text
            next_j = next((j for j in matched_js_sorted if j > grp[-1]), None)
            prev_j = next((j for j in reversed(matched_js_sorted) if j < grp[0]), None)
            if next_j is not None:
                pos = rows[lang_idx_to_row[lang][next_j]][0] - 0.5
            elif prev_j is not None:
                pos = rows[lang_idx_to_row[lang][prev_j]][0] + 0.5
            else:
                pos = -1.0
            extra_rows.append([pos, "", empty])

    all_rows = rows + extra_rows
    all_rows.sort(key=lambda r: r[0])
    return [(de_text, lang_cols) for _, de_text, lang_cols in all_rows]

@tool
def align_with_swissbert(txt_files: list) -> str:
    """
    Aligns corrected OCR text files from different languages into a multilingual
    parallel corpus using sentence-swissbert (jgrosjean-mathesis/sentence-swissbert).

    Uses X-MOD language adapters (de_CH, fr_CH, it_CH, rm_CH) for accurate
    cross-lingual paragraph embeddings. German is used as the anchor language.
    Input units are paragraphs separated by blank lines — sentences are NOT split.

    Alignment supports 1:1, 1:N and N:1 paragraph merges (up to 3).
    Output is a JSONL file — one aligned paragraph group per line.

    Args:
        txt_files: List of dicts with "path" (str) and "lang" (str) keys.
                   Example: [{"path": "data/booklets_txt/booklets_de_1977.txt", "lang": "de"},
                              {"path": "data/booklets_txt/booklets_fr_1977.txt", "lang": "fr"},
                              {"path": "data/booklets_txt/booklets_it_1977.txt", "lang": "it"}]

    Returns:
        Path to the output JSONL file.
    """

    print(f"🔗 SwissBERT alignment: {len(txt_files)} language(s)...")
    os.makedirs(PARALLEL_DIR, exist_ok=True)

    lang_texts = {}
    year = None

    for item in txt_files:
        path = item.get("path", "") if isinstance(item, dict) else item
        lang = item.get("lang") if isinstance(item, dict) else None

        if not os.path.exists(path):
            print(f"   ⚠️  Missing file: {path}")
            continue
        if not lang:
            m = re.match(r"([a-z]{2})_", os.path.basename(path))
            lang = m.group(1) if m else "unk"
        if year is None:
            ym = re.search(r"(\d{4})", os.path.basename(path))
            if ym:
                year = ym.group(1)

        units = _read_paragraphs(path)
        lang_texts[lang] = units
        print(f"   📄 {lang}: {len(units)} paragraphs from {os.path.basename(path)}")

    if len(lang_texts) < 2:
        return "ERROR: need at least 2 language files to align"

    ref_lang = "de" if "de" in lang_texts else list(lang_texts.keys())[0]
    other_langs = [l for l in ["fr", "it", "rm"] if l in lang_texts and l != ref_lang]
    de_texts = lang_texts[ref_lang]

    print(f"   📌 Anchor: {ref_lang} ({len(de_texts)} paragraphs)")
    print(f"   🔢 Encoding {ref_lang}...")
    de_emb = _encode(de_texts, ref_lang)

    langs_for_build: dict = {}
    for lang in other_langs:
        other_texts = lang_texts[lang]
        print(f"   🔀 {ref_lang} ↔ {lang} ({len(other_texts)} paragraphs)...")
        other_emb = _encode(other_texts, lang)
        pos_weight = 0.1 if lang == "rm" else _POS_WEIGHT
        pairs = _dp_align(de_emb, other_emb, position_weight=pos_weight)
        langs_for_build[lang] = (other_texts, pairs)
        print(f"      → {len(pairs)} aligned pairs")

    rows = _build_rows(de_texts, langs_for_build)

    lang_order = [l for l in ["fr", "it", "rm"] if l in lang_texts]
    out_file = os.path.join(PARALLEL_DIR, f"{year or 'unknown'}_parallel.jsonl")

    with open(out_file, "w", encoding="utf-8") as f:
        for de_text, lang_cols in rows:
            record = {"de": de_text}
            for lang in lang_order:
                record[lang] = lang_cols.get(lang, "")
            if year:
                record["date"] = year
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"   💾 Saved: {out_file} ({len(rows)} rows)")
    return f"ALIGNED: {out_file} | {len(rows)} rows | langs: {[ref_lang] + other_langs}"


# ==========================
# TOOL 6: LLM ALIGNMENT REFINEMENT
# ==========================

REFINE_SYSTEM_PROMPT = """You are an expert multilingual alignment editor specialising in Swiss federal voting booklets.

You will receive a batch of aligned paragraph groups from a multilingual corpus.
Each group is a JSON object with language codes as keys (e.g. "de", "fr", "it", "rm") and aligned paragraph text as values.

Your task is to refine each group so the result is a clean, paragraph-level parallel alignment:
- MERGE consecutive groups if they are clearly fragments of the same paragraph (e.g. a heading split from its body, or a paragraph split across two groups).
- FIX obvious OCR leftovers (garbled chars, broken words) that survived post-correction.
- DISCARD any group where no language has meaningful content (e.g. all values are empty or just punctuation).
- Do NOT split a group into smaller units — keep each group as a complete paragraph.
- NEVER add, translate, or paraphrase content — only reorganise and clean existing text.

Output ONLY a JSON array of refined group objects, one per paragraph-level unit.
No preamble, no markdown fences, no explanation — just the raw JSON array.

IMPORTANT: preserve all language keys present in the input. If a language is missing for a group, use an empty string "".

Example input batch:
[
  {"de": "Der Bundesrat", "fr": "Le Conseil fédéral", "it": "Il Consiglio federale"},
  {"de": "empfiehlt die Annahme der Vorlage.", "fr": "recommande l'adoption du projet.", "it": "raccomanda l'adozione del progetto."}
]

Example output:
[
  {"de": "Der Bundesrat empfiehlt die Annahme der Vorlage.", "fr": "Le Conseil fédéral recommande l'adoption du projet.", "it": "Il Consiglio federale raccomanda l'adozione del progetto."}
]
"""


def _call_refine_llm(batch: list, lang_keys: list) -> list:
    """Send a batch of aligned groups to the LLM for refinement. Returns refined groups."""
    batch_json = json.dumps(batch, ensure_ascii=False, indent=2)
    message = HumanMessage(content=[
        {"type": "text", "text": REFINE_SYSTEM_PROMPT},
        {"type": "text", "text": f"Refine this alignment batch:\n{batch_json}"},
    ])
    response = gemini_llm.invoke([message])
    raw = response.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw).strip()

    try:
        refined = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"      ⚠️  JSON parse error: {e} — keeping original batch")
        return batch

    if not isinstance(refined, list):
        print("      ⚠️  LLM returned non-list — keeping original batch")
        return batch

    cleaned = []
    for group in refined:
        if not isinstance(group, dict):
            continue
        for lang in lang_keys:
            if lang not in group:
                group[lang] = ""
        if all(not v.strip() for k, v in group.items() if k in lang_keys):
            continue
        cleaned.append(group)

    return cleaned


@tool
def refine_alignment(jsonl_path: str, batch_size: int = 20) -> str:
    """
    Refines a multilingual parallel alignment JSONL file using an LLM.

    Reads the aligned groups produced by align_with_swissbert, sends them in
    batches to Gemini for quality checking, and rewrites the file with clean,
    paragraph-level alignments.

    The LLM will:
    - MERGE paragraph fragments across consecutive groups where needed
    - FIX residual OCR errors in the aligned text
    - DISCARD empty or meaningless groups
    - PRESERVE all language columns (de/fr/it/rm)
    - NOT split paragraphs into sentences

    Args:
        jsonl_path: Path to the JSONL file produced by align_with_swissbert.
        batch_size: Number of alignment groups to send per LLM call (default 20).

    Returns:
        Path to the refined JSONL file (overwrites in-place) with a summary.
    """
    print(f"✨ Refining alignment: {jsonl_path}")

    if not os.path.exists(jsonl_path):
        return f"FILE_NOT_FOUND: {jsonl_path}"

    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"   ⚠️  Skipping malformed line: {line[:80]}")

    if not rows:
        return f"ERROR: No valid rows found in {jsonl_path}"

    sample = rows[0]
    lang_keys = [k for k in sample if k in ("de", "fr", "it", "rm")]
    meta_keys = [k for k in sample if k not in lang_keys]
    print(f"   🌐 Languages: {lang_keys}  |  Meta: {meta_keys}  |  {len(rows)} input groups")

    meta = {k: rows[0][k] for k in meta_keys if k in rows[0]}
    lang_rows = [{k: r.get(k, "") for k in lang_keys} for r in rows]

    refined_rows = []
    n_batches = (len(lang_rows) + batch_size - 1) // batch_size
    print(f"   📦 Processing {n_batches} batch(es) of up to {batch_size} groups...")

    for i in range(0, len(lang_rows), batch_size):
        batch = lang_rows[i: i + batch_size]
        batch_num = i // batch_size + 1
        print(f"      Batch {batch_num}/{n_batches} ({len(batch)} groups)...")
        refined_batch = _call_refine_llm(batch, lang_keys)
        refined_rows.extend(refined_batch)
        print(f"         → {len(batch)} → {len(refined_batch)} groups")

    final_rows = []
    for row in refined_rows:
        merged = dict(row)
        merged.update(meta)
        final_rows.append(merged)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    size = os.path.getsize(jsonl_path)
    delta = len(final_rows) - len(rows)
    delta_str = f"+{delta}" if delta >= 0 else str(delta)
    print(f"   💾 Saved: {jsonl_path} ({size} bytes)")
    print(f"   📊 {len(rows)} → {len(final_rows)} groups ({delta_str})")
    return (
        f"REFINED: {jsonl_path} | "
        f"{len(rows)} → {len(final_rows)} sentence groups ({delta_str}) | "
        f"langs: {lang_keys}"
    )


# ==========================
# TOOL 7: ALIGNMENT QUALITY CHECKER
# ==========================

ALIGN_CHECK_PROMPT = """You are an expert multilingual alignment auditor for Swiss federal voting booklets.

You will receive a batch of aligned sentence groups from a parallel corpus.
Each group is a JSON object with language codes as keys (e.g. "de", "fr", "it") and aligned text as values.

For each group, evaluate:
1. SEMANTIC MATCH — do all language versions say the same thing? (0–100)
2. COMPLETENESS — is any language version suspiciously short, truncated, or empty? (flag=true/false)
3. OCR_RESIDUE — are there obvious OCR artifacts remaining (garbled chars, broken words)? (flag=true/false)

Respond ONLY with a JSON array, one object per input group, in the same order:
[
  {{
    "idx": <int, 0-based>,
    "semantic_score": <int 0-100>,
    "completeness_ok": <bool>,
    "ocr_residue": <bool>,
    "issue": "<one-line description or empty string if no issue>"
  }},
  ...
]

No preamble, no markdown fences — raw JSON array only.

ALIGNMENT BATCH:
{batch_json}"""


@tool
def check_alignment_quality(jsonl_path: str, batch_size: int = 30, score_threshold: int = 70) -> str:
    """
    Audits a multilingual parallel alignment JSONL file for quality issues.

    Sends batches of aligned groups to Gemini, which scores each group on:
    - Semantic match across languages (0-100)
    - Completeness (no truncated/missing translations)
    - OCR residue (leftover garbled characters)

    Produces a detailed JSON report and prints a summary with flagged rows.
    Does NOT modify the JSONL file — use refine_alignment to fix issues.

    If the report shows that one language has systematically empty or truncated
    translations across many rows (>10% of groups), this indicates an upstream
    OCR failure for that language — not an alignment issue. In that case:
    1. Use delete_ocr_file to remove the cached .txt for the affected language
    2. Re-run ocr_pdf_gemini for that language PDF
    3. Re-run clean_ocr_text and llm_post_correct_ocr for that language
    4. Re-run align_with_swissbert with all languages including the re-processed one
    Do NOT call refine_alignment to fix systematic source data gaps — it cannot
    recover content that was never extracted.

    Args:
        jsonl_path:      Path to the JSONL file from align_with_swissbert or refine_alignment.
        batch_size:      Groups per LLM call (default 30).
        score_threshold: Semantic score below which a group is flagged (default 70).

    Returns:
        Summary string with overall stats and path to the quality report JSON.
    """

    print(f"🔬 Alignment quality check: {jsonl_path}")

    if not os.path.exists(jsonl_path):
        return f"FILE_NOT_FOUND: {jsonl_path}"

    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not rows:
        return f"ERROR: No valid rows in {jsonl_path}"

    # Convergence guard — refuse to run if improvement has stalled
    quality_report_path = jsonl_path.replace(".jsonl", "_alignment_quality.json")
    if os.path.exists(quality_report_path):
        with open(quality_report_path, encoding="utf-8") as f:
            last_report = json.load(f)
        last_flagged = last_report.get("n_flagged_total", 0)
        last_total = last_report.get("n_groups", 1)
        flag_rate = last_flagged / last_total
        if flag_rate < 0.15:
            return (
                f"REFINE_SKIPPED: flag rate is {flag_rate:.1%} — below 15% threshold, "
                f"corpus is good enough. No further refinement needed."
            )

    sample = rows[0]
    lang_keys = [k for k in sample if k in ("de", "fr", "it", "rm")]
    print(f"   🌐 Languages: {lang_keys}  |  {len(rows)} groups to audit")

    lang_rows = [{k: r.get(k, "") for k in lang_keys} for r in rows]

    all_results = []
    n_batches = (len(lang_rows) + batch_size - 1) // batch_size
    print(f"   📦 Auditing {n_batches} batch(es) of up to {batch_size} groups...")

    for i in range(0, len(lang_rows), batch_size):
        batch = lang_rows[i: i + batch_size]
        batch_num = i // batch_size + 1
        print(f"      Batch {batch_num}/{n_batches} ({len(batch)} groups)...", end=" ")

        batch_with_idx = [{"_idx": j, **b} for j, b in enumerate(batch)]
        batch_json = json.dumps(batch_with_idx, ensure_ascii=False, indent=2)
        prompt = ALIGN_CHECK_PROMPT.format(batch_json=batch_json)

        message = HumanMessage(content=[{"type": "text", "text": prompt}])
        response = gemini_llm.invoke([message])
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()

        try:
            batch_results = json.loads(raw)
            if not isinstance(batch_results, list):
                raise ValueError("not a list")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"⚠️  parse error: {e}")
            batch_results = [
                {"idx": j, "semantic_score": -1, "completeness_ok": True,
                 "ocr_residue": False, "issue": "audit failed"}
                for j in range(len(batch))
            ]

        for local_j, result in enumerate(batch_results):
            global_idx = i + local_j
            result["global_idx"] = global_idx
            if global_idx < len(rows):
                result["row"] = {k: rows[global_idx].get(k, "") for k in lang_keys}
            all_results.append(result)

        n_flagged_batch = sum(
            1 for r in batch_results
            if r.get("semantic_score", 100) < score_threshold
            or not r.get("completeness_ok", True)
            or r.get("ocr_residue", False)
        )
        print(f"{n_flagged_batch} flagged")

    valid = [r for r in all_results if r.get("semantic_score", -1) >= 0]
    scores = [r["semantic_score"] for r in valid]
    avg_score = round(sum(scores) / len(scores), 1) if scores else -1

    flagged_semantic   = [r for r in all_results if 0 <= r.get("semantic_score", 100) < score_threshold]
    flagged_incomplete = [r for r in all_results if not r.get("completeness_ok", True)]
    flagged_ocr        = [r for r in all_results if r.get("ocr_residue", False)]
    all_flagged_idx    = sorted(set(
        [r["global_idx"] for r in flagged_semantic] +
        [r["global_idx"] for r in flagged_incomplete] +
        [r["global_idx"] for r in flagged_ocr]
    ))

    report = {
        "jsonl_path": jsonl_path,
        "langs": lang_keys,
        "n_groups": len(rows),
        "avg_semantic_score": avg_score,
        "score_threshold": score_threshold,
        "n_flagged_total": len(all_flagged_idx),
        "n_flagged_semantic": len(flagged_semantic),
        "n_flagged_incomplete": len(flagged_incomplete),
        "n_flagged_ocr_residue": len(flagged_ocr),
        "flagged_indices": all_flagged_idx,
        "all_results": all_results,
    }

    report_path = jsonl_path.replace(".jsonl", "_alignment_quality.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n   📊 Alignment Quality Report — {os.path.basename(jsonl_path)}")
    print(f"   Avg semantic score    : {avg_score}/100")
    print(f"   Groups audited        : {len(rows)}")
    print(f"   Flagged (score<{score_threshold})  : {len(flagged_semantic)}")
    print(f"   Flagged (incomplete)  : {len(flagged_incomplete)}")
    print(f"   Flagged (OCR residue) : {len(flagged_ocr)}")
    print(f"   Total flagged rows    : {len(all_flagged_idx)}")

    if all_flagged_idx:
        print(f"\n   ⚠️  Flagged rows (showing up to 20):")
        for idx in all_flagged_idx[:20]:
            r = all_results[idx]
            row = r.get("row", {})
            de_snippet = row.get("de", "")[:60].replace("\n", " ")
            print(f"      Row {idx:>4} | score={r.get('semantic_score','?'):>3} | "
                  f"complete={'Y' if r.get('completeness_ok') else 'N'} | "
                  f"ocr={'Y' if r.get('ocr_residue') else 'N'} | "
                  f"de: {de_snippet!r}")
            if r.get("issue"):
                print(f"              issue: {r['issue']}")
        if len(all_flagged_idx) > 20:
            print(f"      ... and {len(all_flagged_idx) - 20} more — see report JSON")

    quality_pct = round(100 * (len(rows) - len(all_flagged_idx)) / len(rows), 1) if rows else 0
    print(f"\n   ✅ Clean rows: {len(rows) - len(all_flagged_idx)}/{len(rows)} ({quality_pct}%)")
    print(f"   💾 Report saved: {report_path}")

    verdict = "NEEDS_REFINEMENT" if len(all_flagged_idx) > 0 else "OK"
    return (
        f"ALIGNMENT_QUALITY_{verdict}: avg_score={avg_score} | "
        f"{len(all_flagged_idx)}/{len(rows)} groups flagged "
        f"(semantic={len(flagged_semantic)}, incomplete={len(flagged_incomplete)}, ocr={len(flagged_ocr)}) | "
        f"report={report_path}"
    )

# ==========================
# TOOL: DELETE OCR FILE
# ==========================

@tool
def delete_ocr_file(txt_path: str) -> str:
    """
    Deletes a cached OCR .txt file so it can be re-processed.

    Use this before re-running ocr_pdf_tesseract or ocr_pdf_gemini when
    check_alignment_quality reveals systematic gaps in one language that
    indicate an upstream OCR failure. Both OCR tools skip files that already
    exist — you must delete the file first to force a fresh extraction.

    Args:
        txt_path: Path to the .txt file to delete.

    Returns:
        Confirmation or error message.
    """
    if not os.path.exists(txt_path):
        return f"FILE_NOT_FOUND: {txt_path}"
    os.remove(txt_path)
    print(f"   🗑️  Deleted: {txt_path}")
    return f"DELETED: {txt_path} — ready for re-OCR"



# ==========================
# TOOLS LIST
# ==========================
TOOLS = [
    fix_pdf_rotation,
    ocr_pdf_tesseract,
    ocr_pdf_gemini,       
    check_ocr_quality,  
    clean_ocr_text,       
    llm_post_correct_ocr,
    align_with_swissbert,
    check_alignment_quality,
    refine_alignment,
    delete_ocr_file
]
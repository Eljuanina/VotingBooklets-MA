from pypdf import PdfReader, PdfWriter
import os

def pdf_rotation(pdf_path):
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

    directory = os.path.dirname(pdf_path)
    basename = os.path.basename(pdf_path)
    out_path = os.path.join(directory, basename)

    with open(out_path, "wb") as f:
        writer.write(f)

    summary = ", ".join(f"p{p} was {r}°" for p, r in fixed_pages)
    print(f"   💾 Saved: {out_path}")
    return f"ROTATION_FIXED: {out_path} | Corrected: {summary}"


if __name__ == "__main__":
    pdf_rotation("de_1985.pdf")
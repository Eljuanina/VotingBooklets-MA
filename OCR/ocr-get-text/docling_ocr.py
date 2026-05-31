import os
from docling.document_converter import DocumentConverter

# Input and output folders
input_folder = "data"
output_folder = "docling_output"

os.makedirs(output_folder, exist_ok=True)

# Initialize converter once
converter = DocumentConverter()

# Loop through all PDF files in input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(input_folder, filename)
        name_without_ext = os.path.splitext(filename)[0]

        print(f"\nProcessing {filename}...")

        # Convert PDF (OCR if scanned, or native extraction)
        result = converter.convert(pdf_path)

        # Export text
        markdown_text = result.document.export_to_markdown()

        # Optional: remove markdown headers if you just want plain text
        plain_text = "\n".join(
            line.lstrip("#").strip() for line in markdown_text.splitlines()
        )

        # Save to output file
        output_path = os.path.join(
            output_folder, f"{name_without_ext}_extracted.txt"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(plain_text)

        print(f"Saved extracted text to {output_path}")

print("All PDFs processed.")
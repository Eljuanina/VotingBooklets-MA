"""
count_pdf_pages.py
------------------
Count the number of pages in every PDF file within a folder.

Usage:
    python count_pdf_pages.py                        # scans current directory
    python count_pdf_pages.py /path/to/folder        # scans specified folder
    python count_pdf_pages.py /path/to/folder --recursive  # includes subfolders

Requirements:
    pip install pypdf


this script was used exclusively during my thesis to count the number of pages in the pdfs for the statistics
"""

import sys
import argparse
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:
    print("Error: pypdf is not installed. Run:  pip install pypdf")
    sys.exit(1)


def count_pages(pdf_path: Path) -> int | None:
    """Return the page count for a single PDF, or None on error."""
    try:
        reader = PdfReader(str(pdf_path))
        return len(reader.pages)
    except Exception as e:
        print(f"  [WARNING] Could not read '{pdf_path.name}': {e}")
        return None


def scan_folder(folder: Path, recursive: bool = False) -> None:
    """Scan a folder for PDFs and print a page-count report."""
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = sorted(folder.glob(pattern))

    if not pdf_files:
        print(f"No PDF files found in '{folder}'.")
        return

    print(f"\n{'PDF File':<55} {'Pages':>6}")
    print("-" * 63)

    total_pages = 0
    error_count = 0

    for pdf_path in pdf_files:
        pages = count_pages(pdf_path)
        display_name = str(pdf_path.relative_to(folder))

        if pages is not None:
            print(f"{display_name:<55} {pages:>6}")
            total_pages += pages
        else:
            print(f"{display_name:<55} {'ERROR':>6}")
            error_count += 1

    print("-" * 63)
    print(f"{'TOTAL  —  ' + str(len(pdf_files)) + ' file(s)':<55} {total_pages:>6}")
    if error_count:
        print(f"\n{error_count} file(s) could not be read (see warnings above).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count pages in all PDF files within a folder."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="Path to the folder to scan (default: current directory)",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Also scan subfolders recursively",
    )
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        print(f"Error: '{folder}' is not a valid directory.")
        sys.exit(1)

    print(f"Scanning: {folder}{'  (recursive)' if args.recursive else ''}")
    scan_folder(folder, recursive=args.recursive)


if __name__ == "__main__":
    main()
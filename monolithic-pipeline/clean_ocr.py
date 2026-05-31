import os
import re
import unicodedata


input_root = "gemini-ocr"

output_root = "cleaned_ocr_txt"


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def remove_invisible_chars(text: str) -> str:
    text = text.replace("\u00AD", "")
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
    return text


def normalize_quotes_and_dashes(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"[–—−]", "-", text)
    return text


def remove_layout_artifacts(text: str) -> str:
    # bestehende Regeln …
    text = re.sub(r"<!--\s*image\s*-->", "", text)
    text = re.sub(r"--\s*Page\s*\d+\s*--", "", text)
    text = re.sub(r"\s*--\s*", " ", text)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

    # 🔥 BOILERPLATE METADATA nach dem Join
    text = re.sub(
        r"\b\d{2}-\d{2}\s+[A-Za-z]+\s+\d{1,2}\.\d{1,2}\.\d{4}\s+\d{1,2}:\d{2}\s+Uhr\s+Seite\s+\d+\s*[A-Za-z]*",
        "",
        text
    )

    text = re.sub(
        r"\b02-15_[dr]\s+.*?Seite\s+\d+.*?(?=\s|$)",
        "",
        text,
        flags=re.IGNORECASE
    )

    # falls Varianten ohne Uhr vorkommen
    text = re.sub(
        r"\b\d{2}-\d{2}\s+[A-Za-z]+\s+Seite\s+\d+.*?(?=\s|$)",
        "",
        text
    )

    text = re.sub(
        r"\b[A-Za-z0-9_-]+\s+\d{1,2}\.\d{1,2}\.\d{4}\s+\d{1,2}:\d{2}\s+Uhr\s+Seite\s+[0-9]+",
        "",
        text
    )

    # 🔥 remove lone bullet symbols
    text = re.sub(r"[■•●▪]", "", text)
    

    return text


def fix_line_break_artifacts(text: str) -> str:
    text = re.sub(r"-\s*\n\s*", "", text)
    # text = re.sub(r"\n+", " ", text) # remove paragraph structure 
    return text


def normalize_spacing(text: str) -> str:
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    # text = re.sub(r"\s+", " ", text) # remove paragraph structure 
    return text.strip()


def remove_list_markers(text: str) -> str:
    # Remove numeric lists: 1. 2. 3.
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Remove alphabetic lists: a. b. c.
    text = re.sub(r'^\s*[a-zA-Z]\.\s+', '', text, flags=re.MULTILINE)

    # Remove leading dash bullets
    text = re.sub(r'^\s*-\s+', '', text, flags=re.MULTILINE)

    return text

def remove_empty_lines(text: str) -> str:
    lines = text.splitlines()
    lines = [line.strip() for line in lines if line.strip() != ""]
    return "\n".join(lines)


def clean_text(text: str) -> str:
    text = normalize_unicode(text)
    text = remove_invisible_chars(text)
    text = normalize_quotes_and_dashes(text)
    text = remove_layout_artifacts(text)
    text = fix_line_break_artifacts(text)      # join lines first
    text = remove_list_markers(text)    
    text = remove_empty_lines(text)
    text = normalize_spacing(text)
    return text


if __name__ == "__main__":

    engine_name = os.path.basename(input_root)

    for root, dirs, files in os.walk(input_root):

        for file in files:
            if file.endswith(".txt"):

                input_path = os.path.join(root, file)

                relative_path = os.path.relpath(root, input_root)

                output_folder = os.path.join(
                        output_root,
                        engine_name,
                        relative_path
                )

                os.makedirs(output_folder, exist_ok=True)

                output_path = os.path.join(output_folder, file)

                print(f"Processing: {input_path}")

                with open(input_path, "r", encoding="utf-8") as f:
                    content = f.read()

                cleaned_content = clean_text(content)

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_content)

    print("All files cleaned successfully.")
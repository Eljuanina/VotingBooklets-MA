import re
from pathlib import Path

BASE_DOWNLOAD_DIR = Path("voting booklets")


MONTHS = {
    # German
    "januar": "01", "februar": "02", "märz": "03", "maerz": "03",
    "april": "04", "mai": "05", "juni": "06", "juli": "07",
    "august": "08", "september": "09", "oktober": "10",
    "november": "11", "dezember": "12",

    # French
    "janvier": "01", "février": "02", "fevrier": "02", "mars": "03",
    "avril": "04", "mai": "05", "juin": "06", "juillet": "07",
    "août": "08", "aout": "08", "septembre": "09",
    "octobre": "10", "novembre": "11", "décembre": "12", "decembre": "12",

    # Italian
    "gennaio": "01", "febbraio": "02", "marzo": "03",
    "aprile": "04", "maggio": "05", "giugno": "06",
    "luglio": "07", "agosto": "08", "settembre": "09",
    "ottobre": "10", "novembre": "11", "dicembre": "12",

    # Romansh
    "schaner": "01", "favrer": "02", "mars": "03",
    "avrigl": "04", "matg": "05", "zercladur": "06",
    "fanadur": "07", "avust": "08", "settember": "09",
    "october": "10", "november": "11", "december": "12",
}


def extract_date_from_title(title: str):
    """
    Extracts date patterns:

    1) (DD.MM.YYYY)
    2) day_month_year (in any language)
    3) any DD.MM.YYYY (fallback)
    """

    # --- 1) Parentheses date ---
    match = re.search(r"\((\d{2}\.\d{2}\.\d{4})\)", title)
    if match:
        return match.group(1)

    # --- 2) day_month_year pattern (any language) ---
    match = re.search(r"(\d{1,2})_da_([a-zA-Z]+)_(\d{4})", title)  # Romansh
    if not match:
        match = re.search(r"(\d{1,2})_([a-zA-Z]+)_(\d{4})", title)  # German/French/Italian style

    if match:
        day = match.group(1).zfill(2)
        month_name = match.group(2).lower()
        year = match.group(3)

        month = MONTHS.get(month_name)
        if month:
            return f"{day}.{month}.{year}"

    # --- 3) fallback: any DD.MM.YYYY ---
    match = re.search(r"\d{2}\.\d{2}\.\d{4}", title)
    return match.group(0) if match else None


def scan_downloads():
    for path in BASE_DOWNLOAD_DIR.rglob("*"):
        if not path.is_file():
            continue

        # 👉 ignore system/hidden files
        if path.name.startswith("."):
            continue

        title = path.stem  # filename without extension
        date = extract_date_from_title(title)

        if date:
            print(f"{path.name} → date: {date}")
        else:
            print(f"{path.name} → no date found")



if __name__ == "__main__":
    scan_downloads()





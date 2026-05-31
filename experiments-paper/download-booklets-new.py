#!/usr/bin/env python3

import os
import re
import requests
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

BASE_URL = "https://www.bk.admin.ch/bk/de/home/dokumentation/abstimmungsbuechlein.html"  # <-- CHANGE THIS

BASE_DOWNLOAD_DIR = Path("corpus/raw_voting_booklets")
BASE_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0"
})


def sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name


def get_language_links(url: str):
    """Extract language links from nav.nav-lang."""
    response = session.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    languages = []

    for a in soup.select("nav.nav-lang ul li a"):
        href = a.get("href")
        lang_label = a.get("aria-label")

        if not href or not lang_label:
            continue

        full_url = urljoin(url, href)
        languages.append((lang_label.strip(), full_url))

    return languages


def get_download_links(url: str):
    """Extract file links from download list."""
    response = session.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    links = []

    for a in soup.select("div.mod.mod-downloadlist ul li a"):
        href = a.get("href")
        title = a.get_text(strip=True)

        if not href:
            continue

        full_url = urljoin(url, href)
        links.append((full_url, title))

    return links


def download_file(url: str, title: str, language: str):
    """Download file into language subfolder only if not already downloaded."""
    lang_folder = BASE_DOWNLOAD_DIR / sanitize_filename(language)
    lang_folder.mkdir(exist_ok=True)

    parsed = urlparse(url)
    ext = os.path.splitext(parsed.path)[1]

    if not ext:
        ext = ".pdf"

    filename = sanitize_filename(title) + ext
    filepath = lang_folder / filename

    # ✅ CHECK FIRST — no request if file exists
    if filepath.exists():
        # print(f"Skipping (already exists): {filepath.name}")
        return

    print(f"Downloading [{language}]: {title}")

    response = session.get(url, stream=True)
    response.raise_for_status()

    with open(filepath, "wb") as f:
        for chunk in response.iter_content(8192):
            f.write(chunk)


def main():
    print(f"Scanning base page for German booklets: {BASE_URL}")

    downloaded_any = False

    # --- 1) Download from landing page (German) ---
    try:
        links = get_download_links(BASE_URL)
        print(f"Found {len(links)} files on base page.")

        for file_url, title in links:
            try:
                if download_file(file_url, title, "Deutsch"):
                    downloaded_any = True
            except Exception as e:
                print(f"Download failed: {e}")

    except Exception as e:
        print(f"Failed to process base page: {e}")

    # --- 2) Language pages ---
    languages = get_language_links(BASE_URL)
    print(f"Found {len(languages)} language versions.")

    for language, lang_url in languages:
        print(f"\n=== Processing {language} ===")
        print(f"URL: {lang_url}")

        try:
            links = get_download_links(lang_url)
            print(f"Found {len(links)} files.")

            for file_url, title in links:
                try:
                    if download_file(file_url, title, language):
                        downloaded_any = True
                except Exception as e:
                    print(f"Download failed: {e}")

        except Exception as e:
            print(f"Failed to process language page: {e}")

    # --- 3) Final report ---
    if downloaded_any:
        print("\n✅ New files downloaded.")
    else:
        print("\n✔ All up to date – nothing new.")

    print("All downloads complete.")


if __name__ == "__main__":
    main()
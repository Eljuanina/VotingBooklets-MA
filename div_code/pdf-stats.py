import re
from pathlib import Path
import fitz  # PyMuPDF
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Configuration ─────────────────────────────────────────────
LANG_FOLDERS = {
    "de": "corpus/raw_voting_booklets/Deutsch",
    "fr": "corpus/raw_voting_booklets/Französisch",
    "it": "corpus/raw_voting_booklets/Italienisch",
    "rm": "corpus/raw_voting_booklets/Rätoromanisch",
}

OUTPUT_CSV  = "booklet_stats.csv"
OUTPUT_PLOT = "booklets_over_time.png"

LANGUAGE_COLORS = {
    "de": "#C8A951",  # yellowish gold
    "fr": "#2E5FA3",  # blue
    "it": "#4A7C3F",  # green
    "rm": "#8B1A1A",  # bordeaux
}

LANGUAGE_LABELS = {
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "rm": "Romansh",
}

# ── Multiple filename patterns to handle all naming conventions ──
PATTERNS = [
    # (DD.MM.YYYY) — standard format
    re.compile(r"\((\d{2})\.(\d{2})\.(\d{4})\)"),
    # (DD.MM.YY) — two-digit year e.g. (09.02.25)
    re.compile(r"\((\d{2})\.(\d{2})\.(\d{2})\)"),
    # DD.MM.YYYY appearing after a dash e.g. -_30.11.2014
    re.compile(r"[\-_](\d{2})\.(\d{2})\.(\d{4})"),
    # YYYY-MM-DD e.g. RM_votaziun_2021-03-07
    re.compile(r"(\d{4})-(\d{2})-(\d{2})"),
    # DD_MM_YY at start e.g. 02_Erlaeuterungen_rg_30-11-08 -> 30-11-08
    re.compile(r"(\d{2})-(\d{2})-(\d{2})(?!\d)"),
    # YYYY_MM_DD e.g. 2016_02_28
    re.compile(r"(\d{4})_(\d{2})_(\d{2})"),
    # Bund_YYYY_MM_DD e.g. Bund_2014_02_09
    re.compile(r"(\d{4})_(\d{2})_(\d{2})"),
    # DDMMYY glued e.g. 090613 in Erlaeuterungen_090613
    re.compile(r"_(\d{2})(\d{2})(\d{2})(?!\d)"),
    # DDmonthYYYY written out e.g. 8maerz2015
    # handled separately below
]

MONTH_MAP = {
    "januar": "01", "februar": "02", "maerz": "03", "april": "04",
    "mai": "05", "juni": "06", "juli": "07", "august": "08",
    "september": "09", "oktober": "10", "november": "11", "dezember": "12",
}
WRITTEN_MONTH_PATTERN = re.compile(
    r"(\d{1,2})(" + "|".join(MONTH_MAP.keys()) + r")(\d{4})",
    re.IGNORECASE
)


def parse_date(filename: str):
    """Try all patterns and return (day, month, year) as ints or None."""
    name = filename.lower()

    # Written month e.g. 8maerz2015
    m = WRITTEN_MONTH_PATTERN.search(name)
    if m:
        d, mon, y = m.groups()
        return int(d), int(MONTH_MAP[mon.lower()]), int(y)

    # (DD.MM.YYYY) — four digit year
    m = re.search(r"\((\d{2})\.(\d{2})\.(\d{4})\)", filename)
    if m:
        d, mo, y = m.groups()
        return int(d), int(mo), int(y)

    # DD.MM.YYYY after separator
    m = re.search(r"[\-_ ](\d{2})\.(\d{2})\.(\d{4})", filename)
    if m:
        d, mo, y = m.groups()
        return int(d), int(mo), int(y)

    # DD.MM.YYYY anywhere (e.g. after dash in rm filenames)
    m = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", filename)
    if m:
        d, mo, y = m.groups()
        return int(d), int(mo), int(y)

    # YYYY-MM-DD
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", filename)
    if m:
        y, mo, d = m.groups()
        return int(d), int(mo), int(y)

    # YYYY_MM_DD
    m = re.search(r"(\d{4})_(\d{2})_(\d{2})", filename)
    if m:
        y, mo, d = m.groups()
        return int(d), int(mo), int(y)

    # (DD.MM.YY) — two digit year, assume 20xx
    m = re.search(r"\((\d{2})\.(\d{2})\.(\d{2})\)", filename)
    if m:
        d, mo, y = m.groups()
        return int(d), int(mo), 2000 + int(y)

    # DD-MM-YY at end e.g. 30-11-08
    m = re.search(r"(\d{2})-(\d{2})-(\d{2})(?!\d)", filename)
    if m:
        d, mo, y = m.groups()
        return int(d), int(mo), 2000 + int(y)

    # DDMMYY glued after underscore e.g. _090613
    m = re.search(r"_(\d{2})(\d{2})(\d{2})(?!\d)", filename)
    if m:
        d, mo, y = m.groups()
        return int(d), int(mo), 2000 + int(y)

    return None


# ── Step 1: Parse PDFs ────────────────────────────────────────
records = []

for lang, folder in LANG_FOLDERS.items():
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"  Warning: folder not found: {folder_path}")
        continue

    pdfs = sorted(folder_path.glob("*.pdf"))
    print(f"  Found {len(pdfs)} PDFs in {folder_path}")

    skipped = []
    for pdf_path in pdfs:
        result = parse_date(pdf_path.name)
        if result is None:
            skipped.append(pdf_path.name)
            continue

        day, month, year = result

        # Sanity check
        if not (1970 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31):
            skipped.append(pdf_path.name)
            continue

        date_str = f"{year:04d}-{month:02d}-{day:02d}"

        try:
            doc = fitz.open(pdf_path)
            n_pages = doc.page_count
            doc.close()
        except Exception as e:
            print(f"    Error reading {pdf_path.name}: {e}")
            continue

        records.append({
            "filename": pdf_path.name,
            "date":     date_str,
            "year":     year,
            "month":    month,
            "day":      day,
            "language": lang,
            "pages":    n_pages,
        })

    if skipped:
        print(f"  [{lang}] Could not parse date for {len(skipped)} files:")
        for s in skipped:
            print(f"    {s}")

df = pd.DataFrame(records)

if df.empty:
    print("\nNo PDFs matched. Check folder paths.")
    raise SystemExit

# ── Step 2: Summary statistics ────────────────────────────────
print("\n── Summary ──────────────────────────────────────────────")
print(f"Total booklets : {len(df)}")
print(f"Year range     : {df['year'].min()} – {df['year'].max()}")
print()

summary = (
    df.groupby("language")["pages"]
    .agg(booklets="count", total_pages="sum",
         avg_pages="mean", min_pages="min", max_pages="max")
    .round(1)
)
print(summary.to_string())

yearly = (
    df.groupby(["year", "language"])["pages"]
    .agg(booklets="count", total_pages="sum")
    .reset_index()
)

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nFull data saved to {OUTPUT_CSV}")

# ── Step 3: Plot ──────────────────────────────────────────────
languages = ["de", "fr", "it", "rm"]
year_min  = df["year"].min()
year_max  = df["year"].max()
all_years = list(range(year_min, year_max + 1))

pivot_pages = (
    yearly.pivot(index="year", columns="language", values="total_pages")
    .reindex(index=all_years, columns=languages)
    .fillna(0)
)

pivot_booklets = (
    yearly.pivot(index="year", columns="language", values="booklets")
    .reindex(index=all_years, columns=languages)
    .fillna(0)
)

LINE_STYLES = {
    "de": (2.0, "--"),
    "fr": (2.0, ":"),
    "it": (2.0, "-."),
    "rm": (2.5, "-"),
}

fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1]})
fig.patch.set_facecolor("white")

# ── Panel A: individual lines per language ────────────────────
ax1 = axes[0]

# shade years where Romansh is missing
rm_missing = [y for y in all_years if pivot_pages.loc[y, "rm"] == 0]
for y in rm_missing:
    ax1.axvspan(y - 0.5, y + 0.5, color="#8B1A1A", alpha=0.08, zorder=0)

for lang in languages:
    lw, ls = LINE_STYLES[lang]
    ax1.plot(
        all_years,
        pivot_pages[lang].values,
        color=LANGUAGE_COLORS[lang],
        linewidth=lw,
        linestyle=ls,
        marker="o",
        markersize=3.5,
        label=LANGUAGE_LABELS[lang],
        zorder=3,
    )

# annotation bottom right of panel, well clear of legend
ax1.annotate(
    "shaded = Romansh missing",
    xy=(0.98, 0.04),
    xycoords="axes fraction",
    fontsize=14,
    color="#8B1A1A",
    alpha=0.8,
    ha="right",
)

ax1.set_ylabel("Total pages per year", fontsize=12)
ax1.set_title("Swiss Federal Voting Booklets 1977–2026",
              fontsize=16, fontweight="bold", pad=14)

# legend outside plot area, top right
ax1.legend(
    fontsize=14,
    framealpha=0.9,
    loc="upper left",
    bbox_to_anchor=(1.01, 1),
    borderaxespad=0,
)

ax1.set_xlim(year_min, year_max)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# add some headroom above max value so lines are not clipped
ax1.set_ylim(0, pivot_pages.values.max() * 1.15)
ax1.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_facecolor("#FAFAFA")

# ── Panel B: booklets per year (grouped bars) ─────────────────
ax2 = axes[1]
bar_width = 0.18
offsets   = [-1.5, -0.5, 0.5, 1.5]

for i, lang in enumerate(languages):
    sub = yearly[yearly["language"] == lang].sort_values("year")
    if sub.empty:
        continue
    ax2.bar(
        sub["year"] + offsets[i] * bar_width,
        sub["booklets"],
        width=bar_width,
        color=LANGUAGE_COLORS[lang],
        label=LANGUAGE_LABELS[lang],
        alpha=0.9,
        zorder=3,
    )

ax2.legend(
    title="Language",
    fontsize=12,
    title_fontsize=14,
    framealpha=0.9,
    loc="upper left",
    bbox_to_anchor=(1.01, 1),
    borderaxespad=0,
)

ax2.set_ylabel("Booklets per year", fontsize=14)
ax2.set_xlabel("Year", fontsize=14)
ax2.set_xlim(year_min - 0.5, year_max + 0.5)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax2.set_xticks(range(year_min, year_max + 1, 5))
ax2.set_xticklabels(range(year_min, year_max + 1, 5),
                    rotation=45, ha="right", fontsize=12)
ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax2.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
ax2.spines[["top", "right"]].set_visible(False)
ax2.set_facecolor("#FAFAFA")

plt.tight_layout(h_pad=1.0)
plt.savefig(OUTPUT_PLOT, dpi=180, bbox_inches="tight")
print(f"Plot saved to {OUTPUT_PLOT}")
plt.show()
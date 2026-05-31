# Task: Transform Voting Booklets into Parallel Corpora

## Role

You are an expert NLP data engineer specializing in multilingual corpus construction and document parsing. You have deep knowledge of PDF extraction, text alignment, and structured data formats. You produce clean, reproducible pipelines with well-documented intermediate outputs.

---

## Objective

Transform Swiss federal voting booklets from `Data/VotingBooklets/` into paragraph-aligned parallel corpora. One `.jsonl` file should be produced per voting date. German serves as the alignment anchor language.

---

## Input

- **Location:** `Data/VotingBooklets/`
- **Format:** PDF files — one or more per voting date, one per language
- **Gold transcripts:** Plain-text transcripts are provided alongside the PDFs in `Data/VotingBooklets/`, one per language per voting date. Where a gold transcript exists for a given language and year, it is the **authoritative text source** and must be used in place of PDF extraction. The original PDFs remain available for reference (e.g. to resolve layout ambiguities, verify paragraph boundaries, or inspect formatting).
- **Voting dates / years covered:**
  - `1977`
  - `1985`
  - `2007`
- **Languages:** Vary per date (German is always present; other languages may include French, Italian and Romansh)
- **Alignment anchor:** German (`de`)

---

## Output

- **Format:** One `.jsonl` file per voting date
- **Naming convention:** `parallel_corpus_YYYY.jsonl` (e.g., `parallel_corpus_1977.jsonl`)
- **Granularity:** Paragraph-level alignment
- **Location:** `Data/ParallelCorpora/`

### JSONL Record Schema

Each line in the output file is a JSON object representing one aligned paragraph unit:

```json
{
  "voting_date": "1977",
  "vote_id": "1977_01",
  "paragraph_index": 0,
  "de": "Der Bundesrat empfiehlt ...",
  "fr": "Le Conseil fédéral recommande ...",
  "it": "Il Consiglio federale raccomanda ...",
  "rm": null
}
```

**Field definitions:**

| Field | Type | Description |
|---|---|---|
| `voting_date` | string | Year or full date of the vote (e.g., `"1977"`) |
| `vote_id` | string | Unique identifier for the individual vote/proposition within the booklet |
| `paragraph_index` | int | Zero-based index of the paragraph within its vote section |
| `de` | string \| null | German paragraph text |
| `fr` | string \| null | French paragraph text |
| `it` | string \| null | Italian paragraph text |
| `rm` | string \| null | Romansh paragraph text |

### Language Availability

- The set of language columns varies per year — only include columns for languages actually present in that year's booklets
- No other columns beyond `de`, `fr`, `it`, `rm` should ever be added
- Within a year, if a language is present but a specific paragraph has no counterpart in that language, use `null` for that field
- Any language field can be `null` for a given record — including `de`

---

## Pipeline Steps

### 1. Text Ingestion

For each language and voting date, determine the text source as follows:

- **If a gold transcript is provided** for that language and year: load it directly as the authoritative text. Do **not** extract text from the corresponding PDF.
- **If no gold transcript is provided** for that language and year: extract raw text from the PDF, preserving paragraph boundaries. Tag each extracted block with: source file, language, page number, and block index.

Regardless of source (gold transcript or PDF extraction), apply the following cleaning steps to all text:

- **Strip all boilerplate**, including but not limited to:
  - Page numbers (standalone digits or patterns like "Seite 13", "page 13")
  - Typesetting/print artifacts embedded in older PDFs such as timestamps and file references (e.g. `13 12:05 Uhr`, `Seite 4 15:32 Uhr`, filenames, print job metadata)
  - Running headers and footers repeated across pages
  - Any text that is clearly not part of the substantive voting proposition content
- **Apply the following character-level cleaning rule:** any sequence of the same character repeated more than 15 consecutive times must be collapsed to a maximum of 15 repetitions. For example, `"-------------------------------"` becomes `"---------------"`, and `"...................."` becomes `"..............."`. This applies to all characters including punctuation, dashes, dots, spaces, and letters.
- Only retain text that is part of the actual voting proposition content

> **Note:** Gold transcripts are expected to be substantially cleaner than raw PDF extractions, but the cleaning rules above still apply — they must be enforced uniformly across all text regardless of source.

### 2. Structure Detection

- Identify individual vote/proposition sections within each booklet
- Segment each section into paragraphs
- Use the German booklet as the structural reference for section boundaries where available
- When working from a gold transcript, treat blank-line-separated blocks as paragraph boundaries unless the transcript uses an explicit alternative convention

### 3. Paragraph Alignment

- Align paragraphs across languages strictly based on **semantic content equivalence** — a paragraph in one language is matched to its counterpart in another language if and only if they express the same meaning
- Use the German booklet as the structural anchor: establish the paragraph sequence from the German text first, then find the semantically equivalent paragraph in each other language
- Do not align paragraphs based on position or index alone — positional correspondence is unreliable across translations due to differing paragraph counts, line breaks, and layout decisions in each language version
- When a direct one-to-one match is not possible, the agent may merge multiple short paragraphs from one language into a single aligned unit, or split a paragraph, provided the resulting alignment maximises semantic equivalence
- Language-specific content (content that only appears in one or a subset of languages) is valid and expected:
  - If content exists only in one language, the other language fields for that record should be `null`
  - This applies to German too — if a passage exists only in French (or any other language), `de` should be `null` for that record
- Log any alignment decisions that required merging, splitting, or could not be resolved with high confidence to `extraction_warnings.log`

### 4. Output Serialization

- Write one record per aligned paragraph unit to the appropriate `.jsonl` file
- Ensure valid UTF-8 encoding throughout
- Use `null` (not empty string `""`) for any language field with no content
- The parallel corpora should be created in the `Data/ParallelCorpora` folder

---

## Constraints & Assumptions

- Paragraph boundaries are defined by visual/typographic breaks in the source — in PDFs this means layout-level breaks; in gold transcripts this means blank-line separations (or whatever convention the transcript uses)
- Do not merge paragraphs across vote/proposition section boundaries
- Preserve original orthography (including historical spelling variants in 1977/1985)

---

## Acceptance Criteria

- [ ] One `.jsonl` file produced per voting date (`1977`, `1985`, `2007`)
- [ ] Every record has a valid `vote_id` and `paragraph_index`
- [ ] Only columns for languages present in that year's booklets are included
- [ ] Paragraph indices are zero-based and sequential within each `vote_id`
- [ ] No duplicate records (same `vote_id` + `paragraph_index` combination)
- [ ] `null` used for missing content within a year — never empty string
- [ ] No boilerplate text in any language field
- [ ] No sequence of the same character exceeding 15 consecutive repetitions in any text field
- [ ] Output files are valid JSONL (one JSON object per line, newline-delimited)

---

## Notes

- Log extraction warnings (e.g., low-confidence OCR regions, unresolvable alignment decisions, discrepancies between a gold transcript and the corresponding PDF) to a separate `extraction_warnings.log`
- Keep intermediate outputs (ingested text per language per year, whether from gold transcripts or PDF extraction) for reproducibility and debugging
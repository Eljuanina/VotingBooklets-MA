# From PDFs to Parallel Corpus: Automated Text Extraction and Alignment across the Four Swiss National Languages

This repository contains all code, data, and experimental files for the master thesis on OCR and alignment of Swiss federal voting booklets across multiple languages.

---

## Repository Structure

```
VotingBooklets-MA/
├── OCR/
├── Voting Booklets PDF/
├── gold_files/
├── gemini-ocr/
├── gemini-ocr-2.5-flash-lite/
├── corpus/
├── experiments-paper/
├── agent/
├── agent-golddata/
├── antigravity/
├── ocr-results-antigravity/
├── monolithic-pipeline/
├── finepdf/
└── div_code/
```

---

## Folder Descriptions

### `OCR/`
Code and output files for OCR experiments, including Pytesseract, Gemini-based OCR, and post-OCR correction of Pytesseract output using Gemini 2.5 Flash Lite.

### `Voting Booklets PDF/`
Raw PDF files of the Swiss federal voting booklets used across all experiments.

### `gold_files/`
Gold-standard annotations for OCR and alignment, covering three voting dates: `1977-06-12`, `1985-12-01`, and `2007-03-11`.

### `gemini-ocr/`
Code and output files for OCR of voting booklets using Gemini.

### `gemini-ocr-2.5-flash-lite/`
Gemini 2.5 Flash Lite OCR output for all voting booklets in the corpus, organized by language.

### `corpus/`
Raw PDFs of all voting booklets in the corpus, together with the voting booklet content aligned per voting date across languages.

### `experiments-paper/`
Code for the alignment experiments conducted to identify the best cross-lingual alignment methods.

### `agent/`
Code for the LangChain-based agentic pipeline for processing voting booklets.

### `agent-golddata/`
Code and output files for running the LangChain agentic pipeline on gold-standard data.

### `antigravity/`
Code and output files for the agentic pipeline using Antigravity as the agent tool.

### `ocr-results-antigravity/`
OCR results produced by the Antigravity-based agentic pipeline.

### `monolithic-pipeline/`
Code for the monolithic pipeline approach for OCR and alignment.

### `finepdf/`
Files and code to download voting booklet URLs from FinePDF.

### `div_code/`
Evaluation code for OCR, including scripts for calculating statistics and other analysis.
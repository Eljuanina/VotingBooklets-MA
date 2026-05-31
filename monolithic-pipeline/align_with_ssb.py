import os
import numpy as np
import torch
import json
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR   = "cleaned_ocr_txt/gemini-ocr"
DE_DIR     = os.path.join(BASE_DIR, "Deutsch")
FR_DIR     = os.path.join(BASE_DIR, "Französisch")
IT_DIR     = os.path.join(BASE_DIR, "Italienisch")
RM_DIR     = os.path.join(BASE_DIR, "Rätoromanisch")
OUTPUT_DIR = "parallel_corpus"

LANG_CODE_MAP = {"de": "de_CH", "fr": "fr_CH", "it": "it_CH", "rm": "rm_CH"}
BATCH_SIZE = 32

model_name = "jgrosjean-mathesis/sentence-swissbert"
swissbert  = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
swissbert.eval()

MAX_MERGE       = 4
POSITION_WEIGHT = 0.8


def read_paragraphs(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def encode(sentences, lang_code):
    """Mean-pool SwissBERT; returns (N, H) numpy array."""
    swissbert.set_default_language(lang_code)
    all_emb = []
    for start in range(0, len(sentences), BATCH_SIZE):
        batch  = sentences[start: start + BATCH_SIZE]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = swissbert(**inputs)
        tok_emb = outputs.last_hidden_state
        mask    = inputs["attention_mask"].unsqueeze(-1).float()
        emb     = (torch.sum(tok_emb * mask, 1) /
                   torch.clamp(mask.sum(1), min=1e-9)).cpu().numpy()
        all_emb.append(emb)
    return np.vstack(all_emb)


def dp_align(a_emb, b_emb, gap=-0.4, merge_bonus=-0.4, position_weight=None):
    n, m = len(a_emb), len(b_emb)
    sim = cosine_similarity(a_emb, b_emb)

    pw = position_weight if position_weight is not None else POSITION_WEIGHT
    for i in range(n):
        for j in range(m):
            pos_diff = abs(i / n - j / m)
            sim[i, j] += pw * (1.0 - pos_diff)

    merged_a = {}
    merged_b = {}

    def get_a(i, l):
        if (i, l) not in merged_a:
            merged_a[(i, l)] = np.mean(a_emb[i:i + l], axis=0)
        return merged_a[(i, l)]

    def get_b(j, l):
        if (j, l) not in merged_b:
            merged_b[(j, l)] = np.mean(b_emb[j:j + l], axis=0)
        return merged_b[(j, l)]

    def seg_sim(i, la, j, lb):
        va = get_a(i, la)
        vb = get_b(j, lb)
        dot  = np.dot(va, vb)
        norm = np.linalg.norm(va) * np.linalg.norm(vb)
        if norm == 0:
            return 0.0
        raw_sim = float(dot / norm)
        pos_a   = (i + (la - 1) / 2) / n
        pos_b   = (j + (lb - 1) / 2) / m
        pos_diff = abs(pos_a - pos_b)
        return raw_sim + pw * (1.0 - pos_diff)

    INF  = float('-inf')
    dp   = np.full((n + 1, m + 1), INF)
    back = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(n + 1):
        for j in range(m + 1):
            if dp[i][j] == INF:
                continue
            for la in range(0, MAX_MERGE + 1):
                if i + la > n:
                    break
                for lb in range(0, MAX_MERGE + 1):
                    if la == 0 and lb == 0:
                        continue
                    if j + lb > m:
                        break
                    ni, nj = i + la, j + lb
                    if la == 0:
                        score = dp[i][j] + gap * lb
                    elif lb == 0:
                        score = dp[i][j] + gap * la
                    else:
                        penalty = merge_bonus * (la + lb - 2)
                        score   = dp[i][j] + seg_sim(i, la, j, lb) + penalty
                    if score > dp[ni][nj]:
                        dp[ni][nj]   = score
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

    pairs  = []
    seen_a = set()
    seen_b = set()
    for a_idxs, b_idxs in pairs_raw:
        if set(a_idxs) & seen_a or set(b_idxs) & seen_b:
            continue
        pairs.append((a_idxs, b_idxs))
        seen_a.update(a_idxs)
        seen_b.update(b_idxs)
    return pairs


def build_rows(de, langs):
    de_groups = {}
    for lang, (texts, align) in langs.items():
        for a_idxs, b_idxs in align:
            key = tuple(sorted(a_idxs))
            if key not in de_groups:
                de_groups[key] = {l: [] for l in langs}
            de_groups[key][lang].extend(b_idxs)

    for key in list(de_groups.keys()):
        for lang in langs:
            de_groups[key].setdefault(lang, [])

    de_groups.pop((), None)
    matched_keys = sorted([k for k in de_groups if k], key=lambda k: k[0])

    index_to_keys = {}
    for k in matched_keys:
        for idx in k:
            index_to_keys.setdefault(idx, []).append(k)

    keys_to_remove = set()
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

    rows            = []
    used_lang       = {lang: set() for lang in langs}
    lang_idx_to_row = {lang: {} for lang in langs}

    for ri, key in enumerate(matched_keys):
        group     = de_groups[key]
        de_text   = " ".join(de[i] for i in key)
        lang_cols = {}
        for lang, (texts, _) in langs.items():
            indices         = sorted(set(group[lang]))
            lang_cols[lang] = " ".join(texts[j] for j in indices)
            used_lang[lang].update(indices)
            for j in indices:
                lang_idx_to_row[lang][j] = ri
        rows.append([float(key[0]), de_text, lang_cols])

    extra_rows = []
    for lang, (texts, _) in langs.items():
        unmatched_js = sorted(j for j in range(len(texts)) if j not in used_lang[lang])
        if not unmatched_js:
            continue

        groups = []
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
            empty    = {l: "" for l in langs}
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


def align_multilingual(de, lang_texts):
    de_emb = encode(de, LANG_CODE_MAP["de"])

    langs = {}
    for lang, texts in lang_texts.items():
        lang_emb   = encode(texts, LANG_CODE_MAP[lang])
        pos_weight = 0.1 if lang == "rm" else POSITION_WEIGHT
        align      = dp_align(de_emb, lang_emb, position_weight=pos_weight)
        langs[lang] = (texts, align)

    return build_rows(de, langs)


def save_rows(rows, path, lang_order):
    with open(path, "w", encoding="utf-8") as f:
        for de_text, lang_cols in rows:
            record = {"de": de_text}
            for lang in lang_order:
                record[lang] = lang_cols.get(lang, "")
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename in os.listdir(DE_DIR):
        if not filename.startswith("de_") or not filename.endswith(".txt"):
            continue

        year = filename.replace("de_", "").replace(".txt", "")

        lang_dirs = {
            "fr": (FR_DIR, f"fr_{year}.txt"),
            "it": (IT_DIR, f"it_{year}.txt"),
            "rm": (RM_DIR, f"rm_{year}.txt"),
        }

        lang_texts = {}
        for lang, (lang_dir, lang_file) in lang_dirs.items():
            lang_path = os.path.join(lang_dir, lang_file)
            if os.path.exists(lang_path):
                lang_texts[lang] = read_paragraphs(lang_path)

        if not lang_texts:
            print(f"Skipping {year}: no other languages found")
            continue

        present = ["de"] + list(lang_texts.keys())
        print(f"Aligning {year} — languages: {', '.join(present)}")

        de_par = read_paragraphs(os.path.join(DE_DIR, filename))
        rows   = align_multilingual(de_par, lang_texts)

        lang_order = [l for l in ["fr", "it", "rm"] if l in lang_texts]
        out_file = os.path.join(OUTPUT_DIR, f"mono_aligned_{year}.jsonl")
        save_rows(rows, out_file, lang_order)
        print(f"Saved: {out_file} ({len(rows)} rows, languages: {', '.join(present)})")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# Library reliance:
# pip install pandas torch fasttext transformers

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import re
import time
import torch
import fasttext
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import csv

MAX_TOKENS = 524
PAUSE = 0.2
SMART_SPLIT_ENABLED = True
MIN_LEN = 30       # Chunks shorter than this length are only merged within the same script
DEBUG = False       # If True, print chunking/splitting/fallback/repair information
LATIN_RATIO_THRESHOLD = 0.9
LATIN_INLINE_THRESHOLD = 10
BATCH_SIZE = 2

# ---------------------------------------
# Non-Latin script detection pattern: Han, Hangul, Thai, Arabic, Bengali
# ---------------------------------------
NON_LATIN_PATTERN = re.compile(
    r'[\u4e00-\u9fff\uac00-\ud7af\u0e00-\u0e7f\u0600-\u06ff\u0980-\u09ff]'
)

# ---------------------------------------
# Load NLLB-200-3.3B model
# ---------------------------------------
print("Loading NLLB-200-3.3B model...")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", use_fast=True)
model     = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

tokenizer.model_max_length = MAX_TOKENS
model.config.max_length     = MAX_TOKENS
forced_bos_token_id         = tokenizer.convert_tokens_to_ids("eng_Latn")

# ---------------------------------------
# Load fastText language identification model
# ---------------------------------------
print("Loading fastText language identification model...")
lid_model = fasttext.load_model("lid.176.bin")

language_pool = {
    "Chinese":    "zho_Hans",
    "Italian":    "ita_Latn",
    "Vietnamese": "vie_Latn",
    "Arabic":     "arb_Arab",
    "Korean":     "kor_Hang",
    "Thai":       "tha_Thai",
    "Bengali":    "ben_Beng",
    "Swahili":    "swh_Latn",
    "Javanese":   "jav_Latn"
}

ISO2_TO_NLLB = {
    'zh': 'zho_Hans', 'it': 'ita_Latn', 'vi': 'vie_Latn', 'ar': 'arb_Arab',
    'ko': 'kor_Hang', 'th': 'tha_Thai', 'bn': 'ben_Beng', 'sw': 'swh_Latn', 'jv': 'jav_Latn'
}

# ---------------------------------------
# Text Cleaning
# ---------------------------------------
def clean_text(text):
    if not text or not isinstance(text, str):
        return text
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'^\s*[\*\-]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n+', ' ', text)
    text = text.replace('：', ':').replace('；', ';').replace('，', ',')\
                .replace('。', '.').replace('？', '?').replace('！', '!')\
                .replace('…', '...')
    text = re.sub(r'([\.!\?…;:])([^\s])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------------------------------------
# Script Detection & Block Splitting
# ---------------------------------------
def detect_script(ch):
    if re.match(r'[\u4e00-\u9fff]', ch): return "Han"
    if re.match(r'[\uac00-\ud7af]', ch): return "Hangul"
    if re.match(r'[\u0e00-\u0e7f]', ch): return "Thai"
    if re.match(r'[\u0600-\u06ff]', ch): return "Arabic"
    if re.match(r'[\u0980-\u09ff]', ch): return "Bengali"
    if re.match(r'[A-Za-z]',    ch): return "Latin"
    return "Neutral"

def split_script_blocks(text):
    blocks = []
    if not text:
        return blocks
    cur_script = detect_script(text[0])
    cur_chunk  = text[0]
    for ch in text[1:]:
        scr = detect_script(ch)
        if scr == cur_script or scr == "Neutral":
            cur_chunk += ch
        else:
            blocks.append((cur_script, cur_chunk))
            cur_script, cur_chunk = scr, ch
    blocks.append((cur_script, cur_chunk))
    return blocks

# ---------------------------------------
# Sentence Split Rules per Script
# (based on punctuation + spaces, including colon/semicolon)
# ---------------------------------------
rules = {
    "Han":     r'(?<=[。！？…\.!\?；;:])[ \t]+',
    "Latin":   r'(?<=[\.!\?…;:])[ \t]+',
    "Hangul":  r'(?<=[\.!\?…;:])[ \t]+',
    "Thai":    r'(?<=[\.!\?ฯ;:])[ \t]+',
    "Arabic":  r'(?<=[\.!\؟؛;:])[ \t]+',
    "Bengali": r'(?<=[।॥\?\!:])[ \t]+',
}

def split_by_token_limit(text, tokenizer, max_tokens):
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= max_tokens:
        return [text]

    words = text.split()
    chunks, buffer = [], []
    for word in words:
        buffer.append(word)
        if len(tokenizer.tokenize(" ".join(buffer))) > max_tokens:
            buffer.pop()
            if buffer:
                chunks.append(" ".join(buffer))
            buffer = [word]
    if buffer:
        chunks.append(" ".join(buffer))
    return chunks


# ---------------------------------------
# Strict sentence splitting: merge only short segments within the same script
# ---------------------------------------
def split_sentences_by_script(blocks):
    # Step 1: merge short Latin segments sandwiched between identical non-Latin scripts
    merged_blocks = []
    i = 0
    while i < len(blocks):
        script, chunk = blocks[i]
        if (script == "Latin"
            and len(chunk.strip()) < LATIN_INLINE_THRESHOLD
            and 0 < i < len(blocks) - 1):
            prev_script, prev_chunk = blocks[i - 1]
            next_script, next_chunk = blocks[i + 1]
            if prev_script == next_script and prev_script != "Latin":
                # Merge three blocks into one
                merged = prev_chunk.strip() + " " + chunk.strip() + " " + next_chunk.strip()
                merged_blocks[-1] = (prev_script, merged)  # replace the previous block
                i += 2  # skip current and next block
                continue
        merged_blocks.append((script, chunk))
        i += 1

    # Step 2: split into sentences and merge short sentences
    out_pairs = []
    for script, chunk in merged_blocks:
        if script == "Latin" and re.match(r'https?://', chunk):
            out_pairs.append((chunk.strip(), script))
            continue

        pattern = rules.get(script)
        segments = [s.strip() for s in re.split(pattern, chunk) if s.strip()] if pattern else [chunk.strip()]

        for seg in segments:
            if out_pairs and out_pairs[-1][1] == script and len(seg) < MIN_LEN:
                combined = out_pairs[-1][0] + " " + seg
                if len(tokenizer.tokenize(combined)) <= MAX_TOKENS:
                    out_pairs[-1] = (combined, script)
                    continue
            out_pairs.append((seg, script))

    return out_pairs


# ---------------------------------------
# Fallback sentence splitting: merge by length only, ignore script boundaries
# ---------------------------------------
def split_sentences_fallback(blocks):
    parts = []
    for script, chunk in blocks:
        if script == "Latin" and re.match(r'https?://', chunk):
            parts.append(chunk.strip())
            continue
        pattern = rules.get(script)
        segments = [s.strip() for s in re.split(pattern, chunk) if s.strip()] if pattern else [chunk.strip()]
        parts.extend(segments)

    out = []
    for seg in parts:
        if out and len(seg) < MIN_LEN:
            combined = out[-1] + " " + seg
            if len(tokenizer.tokenize(combined)) <= MAX_TOKENS:
                out[-1] = combined
                continue
        out.append(seg)
    return out


# ---------------------------------------
# Batch translation: translate a list of texts in one model call
# ---------------------------------------
def translate_batch(text_list):
    inputs = tokenizer(
        text_list,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_TOKENS
    ).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_TOKENS,
            forced_bos_token_id=forced_bos_token_id,
            do_sample=False
        )
    decoded = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    time.sleep(PAUSE)
    return decoded


def batch_translate_fixed(sentences, batch_size=BATCH_SIZE):
    """
    Divide sentences into fixed-size batches and call translate_batch for each batch.
    """
    results = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        results.extend(translate_batch(batch))
    return results


# ---------------------------------------
# Repair remaining non-Latin fragments: translate each unique non-Latin substring
# ---------------------------------------
def fix_non_latin_in_text(final_text):
    segments = re.findall(r'[^\x00-\x7F]+', final_text)
    seen = set()
    for seg in segments:
        if seg in seen or len(seg.strip()) < 2:
            continue
        seen.add(seg)
        translation = translate_batch([seg])[0]
        final_text = final_text.replace(seg, translation)
        if DEBUG:
            print(f"Repairing non-Latin fragment: {seg} → {translation}")
    return final_text


# ---------------------------------------
# Main translation logic (with fallback, repair, and DEBUG logging)
# ---------------------------------------
def translate_text(text, col_name=None, row_idx=None):
    if not text or not isinstance(text, str):
        return ""
    cleaned = clean_text(text)
    blocks = split_script_blocks(cleaned)
    prefix = f"[Column: {col_name}, Row: {row_idx}]"

    content_only = re.sub(r'[^A-Za-z\u4e00-\u9fff\u0e00-\u0e7f\u0600-\u06ff\u0980-\u09ff\uac00-\ud7af]', '', cleaned)
    latin_chars = re.findall(r'[A-Za-z]', content_only)
    latin_ratio = len(latin_chars) / max(len(content_only), 1)

    if DEBUG:
        print(f"{prefix} After cleaning: {cleaned}")
        print(f"{prefix} Script blocks: {[(b[0], b[1]) for b in blocks]}")
        print(f"{prefix} Latin ratio: {len(latin_chars)} / {len(content_only)} → {latin_ratio:.2f}")

    if latin_ratio >= LATIN_RATIO_THRESHOLD:
        # fastText-based path: identify language and batch-translate full text
        predictions = lid_model.predict(cleaned.replace('\n', ' '), k=20)
        label_scores = zip(predictions[0], predictions[1])

        filtered = [(lbl.replace("__label__", "").lower(), score)
                    for lbl, score in label_scores
                    if lbl.replace("__label__", "").lower() in ISO2_TO_NLLB]

        if filtered:
            best_lang, prob = filtered[0]
            tokenizer.src_lang = ISO2_TO_NLLB[best_lang]
            if DEBUG:
                print(f"{prefix} fastText detected whitelisted language: {best_lang} ({prob:.2f}), src_lang={tokenizer.src_lang}")
        else:
            tokenizer.src_lang = list(ISO2_TO_NLLB.values())[0]
            if DEBUG:
                print(f"{prefix} No whitelisted language match, using default src_lang={tokenizer.src_lang}")

        sentences = [s for s, _ in split_sentences_by_script(blocks)]
        results = batch_translate_fixed(sentences)
    else:
        # Non-Latin path: set src_lang based on script for each sentence
        script_to_langname = {
            "Han": "Chinese",
            "Hangul": "Korean",
            "Thai": "Thai",
            "Arabic": "Arabic",
            "Bengali": "Bengali",
            "Latin": "English"  # fallback
        }
        sentence_pairs = split_sentences_by_script(blocks)
        results = []

        for sentence, script in sentence_pairs:
            lang_key = script_to_langname.get(script)
            if lang_key and lang_key in language_pool:
                tokenizer.src_lang = language_pool[lang_key]
            else:
                tokenizer.src_lang = "eng_Latn"

            try:
                result = translate_batch([sentence])[0]
            except Exception as e:
                print(f"{prefix} Translation failed [{script}] → {sentence[:20]}...: {e}")
                result = ""
            results.append(result)

    if DEBUG:
        print(f"{prefix} First-pass translation results: {results}")

    # Fallback mechanism
    joined = " ".join(results)
    if NON_LATIN_PATTERN.search(joined):
        if DEBUG:
            print(f"{prefix} Non-Latin characters remain after first pass, entering fallback stage")

        fallback_segments = split_sentences_fallback(blocks)
        results = batch_translate_fixed(fallback_segments)
        joined = " ".join(results)

        if NON_LATIN_PATTERN.search(joined):
            if DEBUG:
                print(f"{prefix} Still non-Latin after second pass, performing full-text fallback")
            if len(tokenizer.tokenize(cleaned)) > MAX_TOKENS:
                # Split by punctuation for oversized text
                segments = re.split(r'(?<=[\.\!\?。！？；;])\s+', cleaned)
                buffer, chunks = "", []
                for seg in segments:
                    trial = (buffer + " " + seg).strip()
                    if len(tokenizer.tokenize(trial)) <= MAX_TOKENS:
                        buffer = trial
                    else:
                        if buffer:
                            chunks.append(buffer)
                        buffer = seg
                if buffer:
                    chunks.append(buffer)
                results = batch_translate_fixed(chunks)
                joined = " ".join(results)
            else:
                joined = translate_batch([cleaned])[0]

            joined = fix_non_latin_in_text(joined)

    if DEBUG:
        print(f"{prefix} Final translation: {joined}")
    return joined


# ---------------------------------------
# Main execution flow
# ---------------------------------------
def main():
    input_files = [
        "translated_type1_outputs.csv",
        "translated_type2_output.csv"
    ]
    columns = [
        "Gemma2_IT output",
        "Mistral7B output",
        "Llama32_3BInstr output",
        "Llama32_3B output",
        "Llama2_7B output",
        "Tulu3_8B output"
    ]

    for input_file in input_files:
        output_file = input_file.replace(".csv", "_back.csv")
        print(f"\nProcessing file: {input_file}")
        df = pd.read_csv(input_file).head(10)

        for col in columns:
            if col not in df.columns:
                print(f"Column not found: {col} (skipped)")
                continue
            print(f"Translating column: {col}")
            texts = df[col].fillna("").tolist()
            translated = []
            for idx, txt in enumerate(texts, start=1):
                print(f" • row {idx}/{len(texts)}", end="\r")
                translated.append(translate_text(txt, col_name=col, row_idx=idx))

            # Append the translated column after the original
            new_col = col.replace(" output", "") + " translated"
            df[new_col] = translated

            # Move the new column next to the original column
            col_idx = df.columns.get_loc(col)
            cols = list(df.columns)
            cols.insert(col_idx + 1, cols.pop(cols.index(new_col)))
            df = df[cols]

            print()

        df.to_csv(output_file, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
        print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
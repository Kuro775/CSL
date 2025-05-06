#Library reliance：
# pip install transformers torch spacy requests evaluate pandas absl-py unbabel-comet bert_score
# python -m spacy download en_core_web_sm


import pandas as pd
import time
import random
import spacy
import math
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from evaluate import load



# --------------------------
# Load NLLB translation model
# --------------------------
print("Loading NLLB model, please wait...")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")
translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang="eng_Latn", max_length=512)

# --------------------------
# Load spaCy English model
# --------------------------
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

# --------------------------
# Load evaluation metrics: COMET and BERTScore
# --------------------------
comet_metric = load("comet", config_name="Unbabel/wmt22-comet-da")
bertscore_metric = load("bertscore")

def evaluate_translation(source, reference, hypothesis):
    try:
        if not hypothesis.strip() or hypothesis.strip().startswith("[ERROR]"):
            return {"comet": 0.0, "bertscore": 0.0}

        results = {}

        # COMET: try multiple field names
        comet = comet_metric.compute(
            sources=[source],
            predictions=[hypothesis],
            references=[reference]
        )
        if "score" in comet:
            results["comet"] = round(comet["score"], 4)
        elif "scores" in comet:
            results["comet"] = round(comet["scores"][0], 4)
        elif "comet" in comet:
            results["comet"] = round(comet["comet"][0], 4)
        else:
            results["comet"] = 0.0

        # BERTScore: standard use
        bertscore = bertscore_metric.compute(
            predictions=[hypothesis],
            references=[reference],
            lang="en"  # Assume English output
        )
        if isinstance(bertscore["f1"], list):
            results["bertscore"] = round(bertscore["f1"][0], 4)
        else:
            results["bertscore"] = round(bertscore["f1"], 4)

        return results

    except Exception as e:
        print(f"[Eval Error] {e}")
        return {"comet": 0.0, "bertscore": 0.0}



# --------------------------
# Supported languages (Name → NLLB code)
# --------------------------
language_pool = {
    "Chinese": "zho_Hans",
    "Italian": "ita_Latn",
    "Vietnamese": "vie_Latn",
    "Arabic": "arb_Arab",
    "Korean": "kor_Hang",
    "Thai": "tha_Thai",
    "Bengali": "ben_Beng",
    "Swahili": "swh_Latn",
    "Javanese": "jav_Latn"
}

# --------------------------
# Type 2: POS-based partial translation with ratio control
# --------------------------
def word_level_code_switch(text, lang_code, ratio):
    try:
        doc = nlp(text)
        translated_tokens = []
        pos_to_translate = {"NOUN", "VERB", "ADJ"}
        target_indices = [i for i, token in enumerate(doc) if token.pos_ in pos_to_translate and token.is_alpha]
        n_to_translate = math.ceil(len(target_indices) * ratio)
        selected_indices = set(random.sample(target_indices, min(n_to_translate, len(target_indices))))

        for i, token in enumerate(doc):
            if i in selected_indices:
                try:
                    result = translator(token.text, src_lang="eng_Latn", tgt_lang=lang_code)
                    translated_word = result[0]["translation_text"]
                    translated_tokens.append(translated_word)
                except:
                    translated_tokens.append(token.text)
            else:
                translated_tokens.append(token.text)

        return spacy.tokens.Doc(doc.vocab, words=translated_tokens).text
    except Exception as e:
        return f"[ERROR] {str(e)}"

# # --------------------------
# # Dispatch translator for Type 1 or Type 2
# # --------------------------
# def code_switch_translation(text):
#     lang_name, lang_code = random.choice(list(language_pool.items()))
#     ratio = round(random.uniform(0.1, 0.9), 2)

#     try:
#         if random.random() < 0.5:
#             result = translator(text, src_lang="eng_Latn", tgt_lang=lang_code)
#             return lang_name, ratio, result[0]["translation_text"], "Type 1"
#         else:
#             mixed_result = word_level_code_switch(text, lang_code, ratio)
#             return lang_name, ratio, mixed_result, "Type 2"
#     except Exception as e:
#         return lang_name, ratio, f"[ERROR] {str(e)}", "ERROR"

# Mode 1: Type 1 only
def code_switch_translation_type1(text):
    lang_name, lang_code = random.choice(list(language_pool.items()))
    ratio = round(random.uniform(0.1, 0.9), 2)
    try:
        result = translator(text, src_lang="eng_Latn", tgt_lang=lang_code)
        return lang_name, ratio, result[0]["translation_text"], "Type 1"
    except Exception as e:
        return lang_name, ratio, f"[ERROR] {str(e)}", "ERROR"

# Mode 2: Type 2 only
def code_switch_translation_type2(text):
    lang_name, lang_code = random.choice(list(language_pool.items()))
    ratio = round(random.uniform(0.1, 0.9), 2)
    try:
        mixed_result = word_level_code_switch(text, lang_code, ratio)
        return lang_name, ratio, mixed_result, "Type 2"
    except Exception as e:
        return lang_name, ratio, f"[ERROR] {str(e)}", "ERROR"

# Mode 3: Random mode
def code_switch_translation_random(text):
    if random.random() < 0.5:
        return code_switch_translation_type1(text)
    else:
        return code_switch_translation_type2(text)


# --------------------------
# Main logic
# --------------------------
input_file = "sampled2.csv"
df = pd.read_csv(input_file)
# df = df.head(10)
total_rows = len(df)
print(f"There are {total_rows} prompts in the CSV file.")

def process_translations(translation_function, output_filename):
    questions = []
    target_languages = []
    ratios = []
    translated_results = []
    translation_types = []
    comet_scores, bertscore_scores = [], []
    batch_size = 8

    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = df.iloc[start_idx:end_idx]
        batch_prompts = batch_df["prompt"].tolist()

        print(f"Processing prompts {start_idx+1} ~ {end_idx} for {output_filename}...", flush=True)

        for prompt in batch_prompts:
            lang, ratio, translated, trans_type = translation_function(prompt)
            metrics = evaluate_translation(prompt, prompt, translated)

            questions.append(prompt)
            target_languages.append(lang)
            ratios.append(ratio)
            translated_results.append(translated)
            translation_types.append(trans_type)
            comet_scores.append(metrics["comet"])
            bertscore_scores.append(metrics["bertscore"])

            time.sleep(0.5)

    output_df = pd.DataFrame({
        "question": questions,
        "target_language": target_languages,
        "translated_ratio": ratios,
        "translated_result": translated_results,
        "translation_type": translation_types,
        "comet": comet_scores,
        "bertscore": bertscore_scores
    })

    output_df.to_csv(output_filename, index=False)
    print(f"✅ Translation and evaluation for {output_filename} completed. Results saved.")


process_translations(code_switch_translation_type1, "translated_type1.csv")

process_translations(code_switch_translation_type2, "translated_type2.csv")

process_translations(code_switch_translation_random, "translated_random.csv")
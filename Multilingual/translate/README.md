
# Multilingual Jailbreak Prompt Construction Project

## Overview

These codes are designed to **simulate multilingual jailbreak prompts** following and innovating the methodology proposed in the paper **“Multilingual Jailbreak Challenges in Large Language Models”**. It includes:

- **Prompt transformation and type classification (Type 1 & Type 2)**
- **Word-level translation based on POS tagging**
- **Ratio-controlled code-switching mechanism**
- **Three main scripts**:
  - `translate.py` – Multilingual translation generator using NLLB
  - `model.py` – generating response using Mistral
  - `combine.py` – End-to-end prompt transformation pipeline (NLLB + Mistral)

---

## Methodology

### Prompt Types (based on the paper)

| Type | Description |
|------|-------------|
| **Type 1** | **Full sentence translation** of the English prompt into a randomly selected target language. This directly evaluates the model’s multilingual safety boundaries. |
| **Type 2** | **Partial code-switching** translation where only selected **nouns, verbs, and adjectives** are translated, while keeping the rest of the sentence in English. This simulates realistic evasion of LLM safety filters. |

---

## POS-Based Code Switching Strategy (Type 2)

To make prompts more linguistically deceptive:

- We tokenize each sentence using **spaCy** and extract words tagged as:
  - `NOUN`, `VERB`, or `ADJ`
- We randomly select a portion of these words to translate, controlled by a **`ratio`** parameter (e.g., 40% of the eligible words).
- Translated words are replaced **in-place**, preserving the sentence structure and grammar.

---

## Ratio Mechanism (Type 2)

The **`ratio`** parameter determines **how many eligible words (by POS)** are translated:

- A random ratio is generated between `0.1` and `0.9` for each prompt.
- If a sentence has 10 eligible words and ratio = `0.4`, then **4 of them will be translated** into the selected target language.
- This allows fine-grained control over **how multilingual the prompt becomes**, simulating different levels of adversarial behavior.

---

### Evaluation Metrics

- In our updated pipeline, we use **COMET** and **BERTScore** as evaluation metrics. Unlike BLEU or ROUGE-L, these metrics capture semantic similarity using pre-trained embeddings, which is especially beneficial in partial translations (code-switching scenarios).
- **COMET** evaluates the overall quality and adequacy by measuring how well the semantic meaning is preserved.
- **BERTScore** computes an F1-based score that reflects the similarity at the token-embedding level.
- (Optionally, character-based metrics such as CHRF could be retained if morphological details are of interest.)

### Workflow Diagram (Conceptual)

1. **Input CSV:**
   The input file (e.g., `sampled2.csv`) contains original English prompts.

2. **Translation Module (NLLB):**
   - **Type 1:** Full sentence translation into a target language.
   - **Type 2:** Partial (code-switched) translation using POS tagging and ratio control.

3. **Integration Module (combine.py):**
   - Translated output from NLLB is used as material input.
   - A crafted prompt (describing a historical/cultural transformation) is built.
   - The transformation prompt is sent to the locally deployed Mistral model (via Ollama API) to generate 15 stylistic variants.

4. **Evaluation Module:**
   - The transformation output is evaluated against the NLLB translation using COMET and BERTScore.

5. **Output CSV:**
   - Contains original question, target language, translated result, translation type, transformation output, and evaluation scores.

---

## Code Structure

- Please use the following commands to install all required libraries:
  - pip install transformers torch spacy requests evaluate pandas absl-py unbabel-comet bert_score
  - python -m spacy download en_core_web_sm


### `model.py`

- **Purpose**: Performs generation using the base Mistral model.
- **Functions**:
  - Loads Mistral via Ollama or local model interface
  - Accepts prompts (in English or code-switched form)
  - Produces model outputs for evaluation or further processing

> Use this if you only need to **generate outputs** without translation.
### `translate.py`

- **Purpose**: Performs multilingual prompt transformation using NLLB only.
- **Functions**:
  - Initializes the NLLB translation pipeline and spaCy POS tagger
  - Translates prompts from a CSV file (`test.csv`) using:
    - Type 1 (full translation)
    - Type 2 (POS-based ratio-controlled code-switching)
  - Outputs a file `translated_with_type.csv`

> Useful for testing multilingual translation or generating standalone code-switched prompts.

---

### `combine.py`

- **Purpose**: End-to-end pipeline integrating NLLB + Mistral via Ollama.
- **Functions**:
  - Translates prompts using the same Type 1 / Type 2 logic
  - Uses **NLLB’s output** as input to **Mistral**, with prompts crafted for historical/cultural context rewriting
  - Saves all intermediate and final outputs to `final_output.csv`

### How to Modify the Prompt

You can modify the Mistral prompt template directly in:

```
combine.py → lines 126-138 (inside `build_prompt()` function)
```

This block controls how the prompt is wrapped and contextualized before being sent to the Mistral model.

---

## Output Format

translate.py scripts export a structured CSV with fields like:

| Field | Description |
|-------|-------------|
| `question` | Raw English input |
| `target_language` | Translated language name |
| `translated_ratio` | Ratio of eligible words translated (only for Type 2) |
| `translated_result` | The multilingual prompt |
| `translation_type` | Type 1 or Type 2 |
| `comet` | COMET scores |
| `bertscore` | BERTScore scores |

combine.py will have an extra column:
| `transformation_output` | Final model response from Mistral |

---

## Summary

This project not only replicates the core multilingual jailbreak prompt types from the research, but also:

- **Enhances control** with a `ratio`-based translation design
- **Implements realistic prompt evasion** through POS-level manipulation
- Offers a **flexible and extendable codebase** for multilingual safety testing

Feel free to build on this for paraphrase-based (Type 3) experiments or red teaming use cases.

---

## Script Responsibilities Overview

This project involves three main Python scripts, each serving a distinct role:

### `model.py`
- **Purpose**: Acts as the **original Mistral model interface**, responsible for generating outputs.
- **Usage**: If you only need to **generate results** (e.g., cultural/historical rewriting based on a prompt), use this script.

### `translate.py`
- **Purpose**: Handles all multilingual translation work.
- **Usage**: Use this script if you only want to **translate prompts**, either fully or partially (Type 1 / Type 2 translation).

### `combine.py`
- **Purpose**: Combines the power of **translation and generation** in a full pipeline.
- **Usage**: This is the recommended entry point if you're following the project objective: first **translate** the input (via `translate.py`), then **generate** model responses (via `model.py`).

> In short:
> - **Translate only** → `translate.py`
> - **Generate only** → `model.py`
> - **Translate + Generate** → `combine.py`

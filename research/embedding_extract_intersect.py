import random
import torch
import json
import pandas as pd
import csv
from transformers import AutoModel, AutoTokenizer

FILENAME = "intersect.csv"
OUTPUT_FILENAME = "context/intersect/intersect_embeddings.json"
CONTEXT_COLUMNS = [
    "input_original",
    "input_satirical",
    "input_cultural",
    "input_research",
    "input_imaginary"
]

def get_last_token_embeddings(model_name, texts):
    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    all_layers = []

    # tokenize + forward once if you want speed, but here we do it per text
    for txt in texts:
        inputs = tokenizer(txt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        hidden_states = outputs.hidden_states
        last_i = inputs.input_ids.shape[1] - 1
        # get last-token embedding from each layer
        layer_embs = [
            layer[:, last_i, :].squeeze(0).cpu().tolist()
            for layer in hidden_states
        ]
        all_layers.append(layer_embs)
    return all_layers

def load_rows_from_csv(filename, sample_size=None):
    rows = []
    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # you could filter out rows missing any of the context columns here
            rows.append(r)
    if sample_size:
        rows = random.sample(rows, min(sample_size, len(rows)))
    return rows

def main():
    rows = load_rows_from_csv(FILENAME)

    model_name = "meta-llama/Meta-Llama-2-7b-hf"
    embeddings_list = []

    for row in rows:
        entry = {
            "prompt": row["prompt"],    # or whatever your “base” prompt column is
            "embeddings": {}
        }

        for col in CONTEXT_COLUMNS:
            text = row[col]
            # get embeddings for this one example in that context
            embs = get_last_token_embeddings(model_name, [text])[0]
            # store under a shorter key, e.g. "original", "satirical", etc.
            key = col.replace("input_", "")
            entry["embeddings"][key] = embs

        embeddings_list.append(entry)

    # write out
    with open(OUTPUT_FILENAME, "w") as out:
        json.dump(embeddings_list, out, indent=2)

if __name__ == "__main__":
    main()

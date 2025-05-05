import json
from pathlib import Path

MODEL = "mistral_7b_ins_v02"
INPUT_FILE  = f"./context/intersection/{MODEL}/intersect_embeddings.json"
OUTPUT_DIR  = f"./context/intersection/{MODEL}/"        
CONTEXT_KEYS = ["original", "satirical", "cultural",
                "research", "imaginary"]                    

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)

def split_embeddings(combined, contexts):
    per_ctx = {ctx: [] for ctx in contexts}

    for record in combined:                 # one entry per prompt‑id
        for ctx in contexts:
            per_ctx[ctx].append({
                "prompt":     record["prompts"][ctx],
                "embeddings": record["embeddings"][ctx]
            })
    return per_ctx

def main():
    combined = load_json(INPUT_FILE)
    split    = split_embeddings(combined, CONTEXT_KEYS)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    for ctx, lst in split.items():
        outfile = Path(OUTPUT_DIR) / f"input_{ctx}_embeddings.json"
        save_json(lst, outfile)

if __name__ == "__main__":
    main()

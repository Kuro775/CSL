import json
import torch
import torch.nn.functional as F
from itertools import combinations

FILENAME         = "./context/intersect/intersect_embeddings.json"
OUTPUT_FILENAME  = "./context/intersect/intersect_similarities.json"

def load_embeddings_from_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_context_cosine_similarities(embeddings_list):
    results = []
    for entry in embeddings_list:
        prompt = entry["prompt"]
        ctx_embs = entry["embeddings"]  # dict: context_name → [layer_embs…]
        contexts = list(ctx_embs.keys())

        prompt_result = {
            "prompt": prompt,
            "context_similarities": []
        }

        # For every (context A, context B) pair
        for ctx_a, ctx_b in combinations(contexts, 2):
            layers_a = ctx_embs[ctx_a]
            layers_b = ctx_embs[ctx_b]
            if len(layers_a) != len(layers_b):
                raise ValueError(f"Layer count mismatch for {ctx_a} vs {ctx_b} in prompt '{prompt}'")

            # Compute per-layer cosine
            sims = []
            for layer_vec_a, layer_vec_b in zip(layers_a, layers_b):
                t1 = torch.tensor(layer_vec_a, dtype=torch.float32)
                t2 = torch.tensor(layer_vec_b, dtype=torch.float32)
                sims.append(
                    F.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0)).item()
                )

            prompt_result["context_similarities"].append({
                "pair": [ctx_a, ctx_b],
                "similarities": sims
            })

        results.append(prompt_result)

    return results

def save_similarities_to_json(results, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

def main():
    embeddings = load_embeddings_from_json(FILENAME)
    sims       = compute_context_cosine_similarities(embeddings)
    save_similarities_to_json(sims, OUTPUT_FILENAME)
    print(f"Saved context-wise cosine similarities for {len(sims)} prompts to {OUTPUT_FILENAME!r}")

if __name__ == "__main__":
    main()

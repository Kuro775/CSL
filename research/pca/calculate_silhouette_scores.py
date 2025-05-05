import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

# --- CONFIGURATION ----------------------------------------------------------

MODELS = {
    "gemma_2_9b_it":        "gemma_2_9b_it",
    "mistral_7b_instruct":  "mistral_7b_ins_v02",
    "llama_3.2_3b": "llama_32_3b",
    "llama_3.2_3b_instruct": "llama_32_3b_ins",
    "llama_3.1_tulu_3_8b": "llama_31_tulu_3_8b",
    "llama_2_7b_chat": "llama_2_7b_chat",
}

CONTEXTS = ["original", "satirical", "cultural", "research", "imaginary"]
BASE_DIR = "./context/intersection"

# ---------------------------------------------------------------------------

def load_embeddings(path):
    with open(path, "r") as f:
        return json.load(f)

def extract_layer(data, layer_idx):
    return np.vstack([
        entry["embeddings"][layer_idx] if layer_idx >= 0
        else entry["embeddings"][-1]
        for entry in data
    ])

def find_best_layer_and_context_scores(folder_name):
    # build file-paths and load data
    fps = [os.path.join(BASE_DIR, folder_name, f"input_{ctx}_embeddings.json")
           for ctx in CONTEXTS]
    all_data = [load_embeddings(fp) for fp in fps]
    
    # determine number of layers
    n_layers = len(all_data[0][0]["embeddings"])
    
    best_overall = {"layer": None, "score": -1.0}
    # track per-layer overall scores if you want
    for L in range(n_layers):
        groups = [extract_layer(dat, L) for dat in all_data]
        X = np.vstack(groups)
        labels = np.concatenate([np.full(g.shape[0], i) for i, g in enumerate(groups)])
        
        # PCA to 2D
        X2 = PCA(n_components=2).fit_transform(X)
        try:
            score = silhouette_score(X2, labels)
        except ValueError:
            score = -1.0
        
        if score > best_overall["score"]:
            best_overall.update(layer=L, score=score)
    
    # now compute per-context at best layer
    L = best_overall["layer"]
    groups = [extract_layer(dat, L) for dat in all_data]
    X = np.vstack(groups)
    labels = np.concatenate([np.full(g.shape[0], i) for i, g in enumerate(groups)])
    X2 = PCA(n_components=2).fit_transform(X)
    sample_scores = silhouette_samples(X2, labels)
    
    context_avgs = {}
    for i, ctx in enumerate(CONTEXTS):
        mask = (labels == i)
        context_avgs[ctx] = float(np.mean(sample_scores[mask]))
    
    return {
        "best_layer": best_overall["layer"],
        "overall_silhouette": best_overall["score"],
        **{f"{ctx}_silhouette": context_avgs[ctx] for ctx in CONTEXTS}
    }

def main():
    rows = []
    for model_key, folder in MODELS.items():
        res = find_best_layer_and_context_scores(folder)
        rows.append({"model": model_key, **res})
    
    df = pd.DataFrame(rows)
    # reorder columns
    cols = ["model", "best_layer", "overall_silhouette"] + [f"{ctx}_silhouette" for ctx in CONTEXTS]
    print(df[cols].to_string(index=False))

if __name__ == "__main__":
    main()

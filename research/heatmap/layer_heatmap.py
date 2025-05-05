"""
layer_heatmap.py
Visualise layer‑wise silhouette scores for each context as a heat map.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

# ------------------------------------------------------------
# CONFIG
MODEL_FOLDER   = "./context/intersection/"  # change per model
CONTEXTS       = ["satirical", "cultural", "research", "imaginary"]
EMB_TEMPLATE   = "input_{ctx}_embeddings.json"                # file naming pattern
MODEL_NAMES     = ["gemma_2_9b_it", "llama_2_7b_chat", "llama_31_tulu_3_8b",
                  "llama_32_3b", "llama_32_3b_ins", "mistral_7b_ins_v02"]
# ------------------------------------------------------------

def load_embeddings(path):
    with open(path) as f:
        return json.load(f)

def extract_layer(mat, layer):
    """Return (n_prompts, embed_dim) for a given layer."""
    return np.vstack([e["embeddings"][layer] for e in mat])

def compute_layer_context_matrix(folder, contexts):
    """Return (n_layers, n_contexts) silhouette matrix."""
    paths  = [os.path.join(folder, EMB_TEMPLATE.format(ctx=c)) for c in contexts]
    mats   = [load_embeddings(p) for p in paths]
    nlay   = len(mats[0][0]["embeddings"])
    matrix = np.zeros((nlay, len(contexts)))

    for L in range(nlay):
        # stack prompts & build label vector
        groups = [extract_layer(m, L) for m in mats]
        X      = np.vstack(groups)
        labels = np.concatenate([np.full(g.shape[0], i) for i, g in enumerate(groups)])

        # quick 2‑D PCA to cut noise
        X2 = PCA(n_components=2).fit_transform(X)
        s  = silhouette_samples(X2, labels)     # individual scores
        # average per context (column)
        start = 0
        for j, g in enumerate(groups):
            matrix[L, j] = s[start:start+len(g)].mean()
            start += len(g)

    return matrix

def plot_heatmap(mat, contexts, model_name):
    fig, ax = plt.subplots(figsize=(len(contexts)*1.6, 6))
    im = ax.imshow(mat, aspect="auto", origin="lower")  # lower = layer 0 at bottom

    # axis labels
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(np.arange(mat.shape[0]))
    ax.set_ylabel("Layer index")

    ax.set_xticks(np.arange(len(contexts)))
    ax.set_xticklabels(contexts, rotation=45, ha="right")

    # colour bar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Silhouette score")

    ax.set_title("Layer‑wise context separation (silhouette)")
    plt.tight_layout()
    out_png = os.path.join(NEW_MODEL_FOLDER, f"{model_name}_heatmap.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved heat‑map to {out_png}")

if __name__ == "__main__":
    for MODEL_NAME in MODEL_NAMES:
        NEW_MODEL_FOLDER = os.path.join(MODEL_FOLDER, MODEL_NAME)
        M = compute_layer_context_matrix(NEW_MODEL_FOLDER, CONTEXTS)
        plot_heatmap(M, CONTEXTS, MODEL_NAME)

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def load_embeddings(fn):
    with open(fn) as f:
        return json.load(f)

def extract_layer_embeddings(data, layer_idx):
    return np.array([
        entry["embeddings"][layer_idx] if layer_idx >= 0 else entry["embeddings"][-1]
        for entry in data
    ])

def visualize_3d_pca(file_paths, layer_idx, out_fn):
    # load & stack
    groups = [ extract_layer_embeddings(load_embeddings(fp), layer_idx)
               for fp in file_paths ]
    X = np.vstack(groups)
    labels = np.concatenate([
        np.full(g.shape[0], i) for i, g in enumerate(groups)
    ])
    
    # PCA to 3 components
    pca = PCA(n_components=3)
    X3 = pca.fit_transform(X)

    # plot
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap("tab10")

    start = 0
    for i, pts in enumerate(groups):
        n = pts.shape[0]
        coords = X3[start:start+n]
        ax.scatter(
            coords[:,0], coords[:,1], coords[:,2],
            s=20, color=cmap(i), alpha=0.7,
            label=os.path.basename(file_paths[i]).replace('_embeddings.json','')
        )
        start += n

    ax.set_title(f"3-Component PCA of Layer {layer_idx}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.savefig(out_fn, dpi=150)
    plt.close(fig)
    print(f"Saved 3D PCA plot to {out_fn}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_names", nargs="+")
    parser.add_argument("--layer", type=int, default=-1, help="Which layer to plot")
    parser.add_argument("--output_dir", default="pca_3d", help="Where to save plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fps = [
      os.path.join("context", "intersection", "mistral_7b_ins_v03", f"input_{n}_embeddings.json")
      for n in args.dataset_names
    ]

    out_path = os.path.join(args.output_dir, f"layer{args.layer}_3d.png")
    visualize_3d_pca(fps, args.layer, out_path)

# python visualize_multiple_embeddings_with_3d_pca.py original satirical cultural research imaginary --layer 13 --output_dir ./pca_3d  

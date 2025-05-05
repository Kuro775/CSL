import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
from matplotlib.patches import Polygon

def load_embeddings(fn):
    with open(fn) as f:
        return json.load(f)

def extract_layer_embeddings(data, layer_idx):
    return np.array([
        entry["embeddings"][layer_idx] if layer_idx >= 0 else entry["embeddings"][-1]
        for entry in data
    ])

def visualize_with_smooth_boundaries(file_paths, layer_idx, out_fn):
    groups = [ extract_layer_embeddings(load_embeddings(fp), layer_idx)
               for fp in file_paths ]
    X = np.vstack(groups)
    coords = PCA(n_components=2).fit_transform(X)

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(10,8))

    start = 0
    for i, pts in enumerate(groups):
        n = pts.shape[0]
        seg = coords[start:start+n]
        color = cmap(i % 10)
        label = os.path.basename(file_paths[i]).replace('_embeddings.json','')

        ax.scatter(seg[:,0], seg[:,1], s=20, color=color, alpha=0.6, label=label)

        if n >= 3:
            hull = ConvexHull(seg)
            hull_pts = seg[hull.vertices]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])

            deltas = np.sqrt(((hull_pts[1:] - hull_pts[:-1])**2).sum(axis=1))
            u = np.concatenate([[0], np.cumsum(deltas)])
            u /= u[-1]
            tck, _ = splprep([hull_pts[:,0], hull_pts[:,1]], u=u, s=0)
            u_fine = np.linspace(0,1,200)
            x_smooth, y_smooth = splev(u_fine, tck)

            poly = Polygon(np.vstack([x_smooth, y_smooth]).T,
                           closed=True,
                           facecolor=color,
                           edgecolor=color,
                           alpha=0.2,
                           linewidth=1.5)
            ax.add_patch(poly)

        start += n

    ax.set_title(f"PCA Layer {layer_idx} with Smooth Boundaries")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize="small")
    ax.grid(True)
    fig.savefig(out_fn, dpi=150)
    plt.close(fig)
    print(f"Saved {out_fn}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_names", nargs="+")
    parser.add_argument("--output_dir", default="layer_smooth")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fps = [
      os.path.join("context", "intersection", "gemma_2_9b_it",
                   f"input_{n}_embeddings.json")
      for n in args.dataset_names
    ]

    sample = load_embeddings(fps[0])
    num_layers = len(sample[0]["embeddings"])
    needed_layers = [2, num_layers // 2, num_layers - 1]
    for L in needed_layers:
        out_path = os.path.join(args.output_dir, f"layer{L}_smooth.png")
        visualize_with_smooth_boundaries(fps, L, out_path)



# python .\research\pca\visualize_multiple_embeddings_with_pca.py original satirical cultural research imaginary --output_dir context/intersection/gemma_2_2b_it/pca
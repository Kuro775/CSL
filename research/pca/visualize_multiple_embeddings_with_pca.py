import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_embeddings(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data

def extract_layer_embeddings(embeddings_data, layer_index=-1):
    embedding_list = []
    for entry in embeddings_data:
        layer_embedding = entry["embeddings"][-1] if layer_index == -1 else entry["embeddings"][layer_index]
        embedding_list.append(layer_embedding)
    return np.array(embedding_list)

def visualize_multiple_embeddings_with_pca(file_paths, layer_index, output_filename):
    all_embeddings = []
    dataset_indices = []
    dataset_labels = []

    start_index = 0
    for file_path in file_paths:
        embeddings_data = load_embeddings(file_path)
        embeddings = extract_layer_embeddings(embeddings_data, layer_index=layer_index)
        num_points = embeddings.shape[0]
        all_embeddings.append(embeddings)
        dataset_indices.append((start_index, start_index + num_points))
        start_index += num_points
        label = os.path.splitext(os.path.basename(file_path))[0].replace('_embeddings', '')
        dataset_labels.append(label)

    combined_embeddings = np.vstack(all_embeddings)
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(combined_embeddings)

    plt.figure(figsize=(10, 8))
    for (start, end), label in zip(dataset_indices, dataset_labels):
        plt.scatter(reduced_embeddings[start:end, 0],
                    reduced_embeddings[start:end, 1],
                    alpha=0.7, label=label)

    plt.title(f"PCA Visualization of Layer {layer_index}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_filename)
    plt.close()
    print(f"Layer {layer_index} PCA visualization saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize multiple embedding datasets using PCA for all layers."
    )
    parser.add_argument(
        "dataset_names",
        nargs="+",
        help="List of dataset names (e.g., original satirical cultural research imaginary)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pca_outputs",
        help="Output directory to save PCA plots for each layer."
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    file_paths = [
        os.path.join(".", "context", "intersection", "mistral_7b_ins_v03", f"input_{name}_embeddings.json")
        for name in args.dataset_names
    ]

    # Assume the number of layers is obtained from the first file
    sample_data = load_embeddings(file_paths[0])
    num_layers = len(sample_data[0]["embeddings"])

    for layer_idx in range(num_layers):
        output_file = os.path.join(args.output_dir, f"pca_layer_{layer_idx}.png")
        visualize_multiple_embeddings_with_pca(file_paths, layer_idx, output_file)

# python .\research\pca\visualize_multiple_embeddings_with_pca.py original satirical cultural research imaginary --output_dir context/intersection/gemma_2_2b_it/pca
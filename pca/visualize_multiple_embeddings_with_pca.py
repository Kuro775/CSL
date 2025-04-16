import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_embeddings(filename):
    """Load embeddings from a JSON file."""
    with open(filename, "r") as f:
        data = json.load(f)
    return data

def extract_layer_embeddings(embeddings_data, layer_index=-1):
    """
    Extract embeddings from a specific layer.
    
    Each entry in embeddings_data is expected to be a dictionary with:
      - "prompt": the input text (optional)
      - "embeddings": a list of layer embeddings (each is a list or vector)
      
    Args:
        embeddings_data (list): List of embedding data entries.
        layer_index (int): Index of the layer to extract (default: -1 for the last layer).
    
    Returns:
        np.array: Array of extracted embeddings.
    """
    embedding_list = []
    for entry in embeddings_data:
        # Select the embedding from the desired layer (default is the last layer)
        layer_embedding = entry["embeddings"][-1] if layer_index == -1 else entry["embeddings"][layer_index]
        embedding_list.append(layer_embedding)
    
    return np.array(embedding_list)

def visualize_multiple_embeddings_with_pca(file_paths, layer_index, output_filename):
    """
    Load and visualize multiple embeddings from different JSON files.
    Each dataset is plotted in a different color on a single scatter plot.
    
    Args:
        file_paths (list): List of file paths to embedding JSON files.
        layer_index (int): Layer index to extract (default is -1, the last layer).
        output_filename (str): Path for the output plot image.
    """
    all_embeddings = []
    dataset_indices = []  # List of (start, end) indices for each dataset
    dataset_labels = []   # Use dataset names as labels
    
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
    
    # Concatenate all embeddings along rows
    combined_embeddings = np.vstack(all_embeddings)
    
    # Apply PCA to the combined embeddings
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(combined_embeddings)
    
    # Create scatter plot with each dataset in a different color
    plt.figure(figsize=(10, 8))
    for (start, end), label in zip(dataset_indices, dataset_labels):
        plt.scatter(reduced_embeddings[start:end, 0],
                    reduced_embeddings[start:end, 1],
                    alpha=0.7, label=label)
    
    plt.title("PCA Visualization of Multiple Embeddings")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.legend()
    plt.savefig(output_filename)
    plt.close()
    print(f"PCA visualization saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize multiple embedding datasets using PCA. "
                    "Each dataset name corresponds to './context/<name>/<name>_embeddings.json'."
    )
    parser.add_argument(
        "dataset_names",
        nargs="+",
        help="List of dataset names (e.g., cultural research satirical)."
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Layer index to extract embeddings from (default: -1, which is the last layer)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="multi_embedding_pca_plot.png",
        help="Output file name for the PCA scatter plot."
    )
    args = parser.parse_args()
    
    # Build file paths based on dataset names
    file_paths = []
    for name in args.dataset_names:
        path = os.path.join(".", "context", name, f"{name}_embeddings.json")
        if not os.path.exists(path):
            print(f"Warning: {path} does not exist.")
        file_paths.append(path)
    
    # Visualize the multiple embeddings with PCA
    visualize_multiple_embeddings_with_pca(file_paths, args.layer, args.output)
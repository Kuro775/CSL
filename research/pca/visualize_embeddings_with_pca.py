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
    """
    Extract embeddings from a specific layer.
    
    Each entry in embeddings_data is expected to be a dictionary with:
        - "prompt": the input text
        - "embeddings": a list of layer embeddings (each is a list or vector)
    
    Args:
        embeddings_data (list): List of embedding data entries.
        layer_index (int): Index of the layer to extract (default is -1: the last layer).
    
    Returns:
        tuple: A numpy array of embeddings and a corresponding list of prompt texts.
    """
    embedding_list = []
    prompt_list = []
    for entry in embeddings_data:
        prompt = entry.get("prompt", "No Prompt")
        # Select the embedding from the desired layer (default is the last layer)
        layer_embedding = entry["embeddings"][-1] if layer_index == -1 else entry["embeddings"][layer_index]
        embedding_list.append(layer_embedding)
        prompt_list.append(prompt)
    
    return np.array(embedding_list), prompt_list

def visualize_embeddings_with_pca(embeddings, output_filename="pca_embedding_visualization.png"):
    """
    Reduce embedding dimensions with PCA and visualize them in a scatter plot.
    The scatter plot is saved to the output_filename without point annotation labels.
    
    Args:
        embeddings (np.array): Array of high-dimensional embeddings.
        output_filename (str): File name for saving the plot.
    """
    # Perform PCA to reduce embeddings to 2 dimensions
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7)
    
    plt.title("PCA Visualization of Embeddings")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    
    # Save the plot to the specified file
    plt.savefig(output_filename)
    plt.close()
    print(f"PCA visualization saved to {output_filename}")

def derive_output_filename(input_path):
    """
    Create an output filename by appending '_pca_plot.png' to the base of the input file.
    
    For example, if the input file is 'data/embeddings.json', the output file will be
    'data/embeddings_pca_plot.png'.
    """
    dir_name, base_name = os.path.split(input_path)
    name, _ = os.path.splitext(base_name)
    output_filename = os.path.join(dir_name, name + "_pca_plot.png")
    return output_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Embedding Data using PCA.")
    parser.add_argument("embedding_path", type=str, help="Path to the embeddings JSON file.")
    parser.add_argument("--layer", type=int, default=-1, help="Layer index to extract (default: last layer).")
    args = parser.parse_args()
    
    # Load embeddings from the specified file
    embeddings_data = load_embeddings(args.embedding_path)
    
    # Extract embeddings (and prompt texts) from the specified layer
    embeddings, _ = extract_layer_embeddings(embeddings_data, layer_index=args.layer)
    
    # Determine output filename based on input file name
    output_filename = derive_output_filename(args.embedding_path)
    
    # Create PCA visualization without per-point annotation labels
    visualize_embeddings_with_pca(embeddings, output_filename=output_filename)

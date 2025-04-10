import json
import matplotlib.pyplot as plt
import numpy as np

FILENAME = "./context/satirical/satirical_statistics.json"
OUTPUT_FILENAME = "./context/satirical/satirical_plot.png"

def load_statistics_from_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def plot_similarity_statistics(statistics, output_filename="cosine_similarity_plot.png"):
    layers = [stat["layer"] for stat in statistics]
    means = [round(stat["mean"], 3) for stat in statistics]
    lower_bounds = [round(stat["confidence_interval"][0], 3) for stat in statistics]
    upper_bounds = [round(stat["confidence_interval"][1], 3) for stat in statistics]
    print(lower_bounds)
    print(upper_bounds)
    
    plt.figure(figsize=(10, 5))
    plt.plot(layers, means, marker='o', linestyle='-', color='b', label='Mean Similarity')
    plt.fill_between(layers, lower_bounds, upper_bounds, color='b', alpha=0.2, label='95% Confidence Interval')
    
    plt.xlabel("Layer")
    plt.ylabel("Cosine Similarity")
    plt.ylim(0,1.1)
    plt.title("Cosine Similarity Across Layers with Confidence Interval")
    plt.legend()
    plt.grid(True)

    plt.savefig(output_filename)
    plt.close()

# Example usage
statistics = load_statistics_from_json(FILENAME)
plot_similarity_statistics(statistics, output_filename=OUTPUT_FILENAME)

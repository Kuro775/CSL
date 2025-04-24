import json
import matplotlib.pyplot as plt

def load_statistics_from_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def plot_multiple_statistics(statistics_dict, output_filename="multiple_stats_plot.png"):
    """
    statistics_dict is expected to be a dictionary where:
      key: a label for the statistic (e.g., 'Cosine Similarity')
      value: a list of dictionaries for that statistic with keys such as 'layer', 'mean', 
             and optionally 'confidence_interval'
    """
    plt.figure(figsize=(10, 5))
    
    for label, statistics in statistics_dict.items():
        # Extract common x values
        layers = [stat["layer"] for stat in statistics]
        means = [round(stat["mean"], 3) for stat in statistics]
        plt.plot(layers, means, marker='o', linestyle='-', label=f'{label} Mean')
        
        # Check if confidence interval is provided and plot it
        if statistics and "confidence_interval" in statistics[0]:
            lower_bounds = [round(stat["confidence_interval"][0], 3) for stat in statistics]
            upper_bounds = [round(stat["confidence_interval"][1], 3) for stat in statistics]
            plt.fill_between(layers, lower_bounds, upper_bounds, alpha=0.2, label=f'{label} 95% CI')
    
    plt.xlabel("Layer")
    plt.ylabel("Value")
    plt.title("Comparison of Multiple Statistics Across Layers")
    plt.ylim(0,1.1)
    plt.legend()
    plt.grid(True)
    plt.savefig(output_filename)
    plt.close()
    print(f"Plot saved to {output_filename}")

# Example usage:
# Load separate statistics files (make sure these JSON files exist in your working directory)
cultural_stats = load_statistics_from_json("./context/cultural/cultural_statistics.json")
imaginary_stats = load_statistics_from_json("./context/imaginary/imaginary_statistics.json")
research_stats = load_statistics_from_json("./context/research/research_statistics.json")
satirical_stats = load_statistics_from_json("./context/satirical/satirical_statistics.json")

stats_data = {
    "Cultural": cultural_stats,
    "Imaginary": imaginary_stats,
    "Research": research_stats,
    "Satirical": satirical_stats,
}

plot_multiple_statistics(stats_data, output_filename="multi_dataset_plot.png")

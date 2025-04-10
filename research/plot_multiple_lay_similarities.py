import json
import matplotlib.pyplot as plt

def load_statistics_from_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def plot_multiple_stats_single_file(statistics, stat_keys=["mean", "median"], output_filename="multiple_stats_single_file_plot.png"):
    """
    statistics: a list of dictionaries, each containing at least 'layer' and the keys to plot
    stat_keys: list of statistic keys to plot (e.g., "mean", "median")
    """
    layers = [stat["layer"] for stat in statistics]
    plt.figure(figsize=(10, 5))
    
    # Define a list of markers or colors for variety
    markers = ["o", "s", "D", "^", "v"]
    
    for idx, key in enumerate(stat_keys):
        # Extract values for the metric key
        values = [round(stat.get(key, 0), 3) for stat in statistics]
        plt.plot(layers, values, marker=markers[idx % len(markers)], linestyle='-', label=key.title())
    
    plt.xlabel("Layer")
    plt.ylabel("Value")
    plt.title("Multiple Statistics Across Layers")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_filename)
    plt.close()
    print(f"Plot saved to {output_filename}")

# Example usage:
statistics = load_statistics_from_json("similarity_statistics.json")
# Assuming the JSON contains keys "mean" and "median" for each layer
plot_multiple_stats_single_file(statistics, stat_keys=["mean", "median"], output_filename="multi_metric_plot.png")

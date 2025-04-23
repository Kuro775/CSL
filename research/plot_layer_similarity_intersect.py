import json
import matplotlib.pyplot as plt

FILENAME        = "./context/intersect/intersect_similarities_statistics.json"
OUTPUT_FILENAME = "./context/intersect/intersect_similarities_plot.png"

def load_statistics_from_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def plot_context_pair_statistics(statistics, output_filename):
    plt.figure(figsize=(12, 6))

    for entry in statistics:
        pair = entry["pair"]                    # e.g. ["original", "satirical"]
        stats = entry["statistics"]            # list of {layer, mean, confidence_interval}

        layers = [s["layer"] for s in stats]
        means  = [s["mean"] for s in stats]
        lower  = [s["confidence_interval"][0] for s in stats]
        upper  = [s["confidence_interval"][1] for s in stats]

        label = f"{pair[0]} vs {pair[1]}"
        plt.plot(layers, means, marker='o', linestyle='-', label=label)
        plt.fill_between(layers, lower, upper, alpha=0.2)

    plt.xlabel("Layer")
    plt.ylabel("Mean Cosine Similarity")
    plt.ylim(0, 1.05)
    plt.title("Context‐Pair Cosine Similarity Across Layers")
    plt.legend(title="Context Pairs", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_filename)
    plt.close()

if __name__ == "__main__":
    stats = load_statistics_from_json(FILENAME)
    plot_context_pair_statistics(stats, OUTPUT_FILENAME)
    print(f"Saved plot to {OUTPUT_FILENAME!r}")

import json
import numpy as np
import scipy.stats as stats

FILENAME = "./context/intersection/llama_32_3b_ins/intersect_similarities.json"
OUTPUT_FILENAME = "./context/intersection/llama_32_3b_ins/intersect_statistics.json"

def load_similarities_from_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

#[{pair: [original, satirical], statistic: {layer, mean, confidence_interval} }, ... ]

def compute_statistics(similarities):
    if not similarities:
        raise ValueError("No similarity data available.")

    num_layers = len(similarities[0]["context_similarities"][0]["similarities"])

    statistics = []
    for pair_index in range(len(similarities[0]["context_similarities"])):
        all_layer_similarities = [[] for _ in range(num_layers)]

        for entry in similarities:
            entry = entry["context_similarities"][pair_index]
            for layer_idx, similarity in enumerate(entry["similarities"]):
                all_layer_similarities[layer_idx].append(similarity)

        stats_results = []
        for layer_idx, layer_sims in enumerate(all_layer_similarities):
            mean = np.mean(layer_sims)
            std_err = stats.sem(layer_sims) if len(layer_sims) > 1 else 0  # Avoid NaN for single data points
            
            if len(layer_sims) > 1:
                confidence_interval = stats.t.interval(0.95, len(layer_sims)-1, loc=mean, scale=std_err)
            else:
                confidence_interval = (mean, mean)  # If only one value, CI is just the mean
            
            stats_results.append({
                "layer": layer_idx,
                "mean": mean,
                "confidence_interval": confidence_interval
            })

        statistics.append({
            "pair": similarities[0]["context_similarities"][pair_index]["pair"],
            "statistics": stats_results
        })
        
    return statistics

def save_statistics_to_json(stats_results, filename="similarity_statistics.json"):
    with open(filename, "w") as f:
        json.dump(stats_results, f, indent=4)

# Example usage
similarities = load_similarities_from_json(FILENAME)
statistics = compute_statistics(similarities)

# Save results to a JSON file
save_statistics_to_json(statistics, filename=OUTPUT_FILENAME)

# Print results
# for stat in statistics:
    # print(f"Layer {stat['layer']}: Mean={stat['mean']}, Confidence Interval={stat['confidence_interval']}")
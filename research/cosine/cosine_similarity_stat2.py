import json
import numpy as np
import scipy.stats as stats

FILENAME = "./context/safe2/safe2_similarities.json"
OUTPUT_FILENAME = "./context/safe2/safe2_statistics.json"

def load_similarities_from_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def compute_statistics(similarities):
    if not similarities:
        raise ValueError("No similarity data available.")
    
    num_layers = len(similarities[0]["similarities"])
    all_layer_similarities = [[] for _ in range(num_layers)]
    
    # Filter out pairs where the first layer similarity is less than 0.9
    filtered_similarities = [entry for entry in similarities if entry["similarities"][0] >= 0.9]
    
    # Now process the filtered list
    for entry in filtered_similarities:
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
    
    return stats_results

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

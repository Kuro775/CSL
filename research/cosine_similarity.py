import json
import torch
import torch.nn.functional as F
from itertools import combinations

FILENAME = "./context/safe2/safe2_embeddings.json"
OUTPUT_FILENAME = "./context/safe2/safe2_similarities.json"

def load_embeddings_from_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def compute_all_cosine_similarities(embeddings):
    num_embeddings = len(embeddings)
    if num_embeddings < 2:
        raise ValueError("At least two sets of embeddings are required to compute cosine similarity.")
    
    results = []
    for (idx1, emb1), (idx2, emb2) in combinations(enumerate(embeddings), 2):
        if len(emb1["embeddings"]) != len(emb2["embeddings"]):
            raise ValueError("Embeddings must have the same number of layers.")
        
        similarities = []
        for layer_emb1, layer_emb2 in zip(emb1["embeddings"], emb2["embeddings"]):
            tensor1 = torch.tensor(layer_emb1, dtype=torch.float32)
            tensor2 = torch.tensor(layer_emb2, dtype=torch.float32)
            similarity = F.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0)).item()
            similarities.append(similarity)
        
        results.append({"pair": (emb1["prompt"], emb2["prompt"]), "similarities": similarities})
    
    return results

def save_similarities_to_json(results, filename="similarities.json"):
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

# Example usage
filename = FILENAME
embeddings = load_embeddings_from_json(filename = FILENAME)
pairwise_cosine_similarities = compute_all_cosine_similarities(embeddings)

# Save results to a JSON file
save_similarities_to_json(pairwise_cosine_similarities, filename=OUTPUT_FILENAME)

# Print results
# for result in pairwise_cosine_similarities:
#     print(f"Pair {result['pair']}: {result['similarities']}")

import json

INPUT_DIR = "context/intersection/llama_32_3b_ins/"
CONTEXT_COLUMNS = ["input_original", "input_satirical", "input_cultural", "input_research", "input_imaginary"]

def load_embeddings_from_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)
    
def save_embeddings_to_json(embeddings, filename="embeddings.json"):
    with open(filename, "w") as f:
        json.dump(embeddings, f, indent=4)
    
embeddings_list = [{"prompts": dict(), "embeddings": dict()} for _ in range(45)]

for con in CONTEXT_COLUMNS:
    filename = INPUT_DIR + f"{con}_embeddings.json"
    embeddings = load_embeddings_from_json(filename)
    key = con.replace("input_", "")

    for i in range(len(embeddings)):
        embeddings_list[i]['prompts'][key] = embeddings[i]['prompt']
        embeddings_list[i]['embeddings'][key] = embeddings[i]['embeddings']

save_embeddings_to_json(embeddings_list, filename="./context/intersection/llama_32_3b_ins/intersect_embeddings.json")